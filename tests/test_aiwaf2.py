import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-2" / "code"
sys.path.insert(0, str(CODE_DIR))

from loaddata import NORMAL, SQL_INJECTION, XSS, loaddata_sqli, loaddata_xss
from predict import consensus_label, label_name
from vecmodel import FeaturePipeline


class DataLoadingTests(unittest.TestCase):
    def test_sqli_labels_are_mapped_to_normal_and_sql_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sqli.csv"
            with filename.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(
                    [
                        ["payload", "label"],
                        ["/products?page=1", "0"],
                        ["' union select password from users", "1"],
                    ]
                )

            _, labels = loaddata_sqli(filename)

        self.assertEqual([NORMAL, SQL_INJECTION], labels)

    def test_xss_labels_are_mapped_to_normal_and_xss(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "xss.csv"
            with filename.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(
                    [
                        ["id", "payload", "label"],
                        ["1", "/search?q=hello", "0"],
                        ["2", "<script>alert(1)</script>", "1"],
                    ]
                )

            _, labels = loaddata_xss(filename)

        self.assertEqual([NORMAL, XSS], labels)


class FeaturePipelineTests(unittest.TestCase):
    def test_both_representations_have_expected_shape(self):
        payloads = [
            "/products?page=1",
            "/products?page=2",
            "<script>alert(1)</script>",
            "' union select password from users",
        ]
        pipeline = FeaturePipeline(max_features=100, max_vocab=100, sequence_length=16)
        pipeline.fit(payloads)

        self.assertEqual((4, 16), pipeline.transform_sequence(payloads).shape)
        self.assertEqual(4, pipeline.transform_tfidf(payloads).shape[0])
        self.assertGreater(pipeline.vocab_size, 1)

    def test_label_names_accept_integer_model_output(self):
        self.assertEqual("正常", label_name(0))
        self.assertEqual("XSS攻击", label_name(1))
        self.assertEqual("SQL注入攻击", label_name(2))

    def test_consensus_uses_majority_vote(self):
        results = {
            "rf": "正常",
            "knn": "SQL注入攻击",
            "svm": "SQL注入攻击",
            "cnn": "SQL注入攻击",
            "gru": "正常",
        }
        self.assertEqual("SQL注入攻击", consensus_label(results))

    def test_consensus_reports_tie(self):
        self.assertEqual(
            "模型意见不一致",
            consensus_label({"rf": "正常", "svm": "SQL注入攻击"}),
        )


if __name__ == "__main__":
    unittest.main()
