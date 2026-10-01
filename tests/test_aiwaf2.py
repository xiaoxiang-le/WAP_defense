import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-2" / "code"
sys.path.insert(0, str(CODE_DIR))

from loaddata import NORMAL, SQL_INJECTION, XSS, loaddata_sqli, loaddata_xss
from artifacts import atomic_json_dump, sha256_file
import predict
from predict import consensus_label, label_name
from splitdata import _deduplicate
from staticfeature import normalize_payload
import trainmain
from trainmain import validate_training_options
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

    def test_deduplication_uses_the_model_normalization(self):
        payloads, labels = _deduplicate(
            ["/item?id=1", "/ITEM?id=2", "/safe", "/SAFE"],
            [SQL_INJECTION, SQL_INJECTION, NORMAL, XSS],
        )
        self.assertEqual(
            [normalize_payload("/item?id=1")],
            [normalize_payload(payload) for payload in payloads],
        )
        self.assertEqual([SQL_INJECTION], labels)


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


class ArtifactValidationTests(unittest.TestCase):
    def test_manifest_accepts_matching_pipeline_and_model_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            pipeline = model_dir / "feature_pipeline.pkl"
            model = model_dir / "mult_rf.pkl"
            pipeline.write_bytes(b"pipeline")
            model.write_bytes(b"model")
            manifest = model_dir / "model_manifest.json"
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "labels": ["正常", "XSS攻击", "SQL注入攻击"],
                    "pipeline": {
                        "filename": pipeline.name,
                        "sha256": sha256_file(pipeline),
                    },
                    "models": {
                        "rf": {
                            "filename": model.name,
                            "sha256": sha256_file(model),
                        }
                    },
                },
                manifest,
            )

            with patch.object(predict, "MODEL_DIR", model_dir), patch.object(
                predict, "MANIFEST_PATH", manifest
            ):
                self.assertEqual(pipeline, predict.validate_artifacts({"rf"}))

    def test_manifest_rejects_a_modified_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            pipeline = model_dir / "feature_pipeline.pkl"
            model = model_dir / "mult_rf.pkl"
            pipeline.write_bytes(b"pipeline")
            model.write_bytes(b"original")
            manifest = model_dir / "model_manifest.json"
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "labels": ["正常", "XSS攻击", "SQL注入攻击"],
                    "pipeline": {
                        "filename": pipeline.name,
                        "sha256": sha256_file(pipeline),
                    },
                    "models": {
                        "rf": {
                            "filename": model.name,
                            "sha256": sha256_file(model),
                        }
                    },
                },
                manifest,
            )
            model.write_bytes(b"modified")

            with patch.object(predict, "MODEL_DIR", model_dir), patch.object(
                predict, "MANIFEST_PATH", manifest
            ), self.assertRaisesRegex(RuntimeError, "校验失败"):
                predict.validate_artifacts({"rf"})


class TrainingOptionTests(unittest.TestCase):
    def test_all_cannot_be_combined_with_a_specific_model(self):
        with self.assertRaisesRegex(ValueError, "不能与具体模型"):
            validate_training_options(["all", "rf"], 3, 42, 100, 100, 20)

    def test_cnn_rejects_a_sequence_shorter_than_its_kernels(self):
        with self.assertRaisesRegex(ValueError, "不能小于 7"):
            validate_training_options(["cnn"], 3, 42, 100, 100, 6)

    def test_programmatic_training_rejects_nonpositive_sizes(self):
        with self.assertRaisesRegex(ValueError, "必须大于 0"):
            validate_training_options(["svm"], 3, 42, 0, 100, 20)


class ArtifactPromotionTests(unittest.TestCase):
    def test_staged_artifacts_replace_existing_files_only_when_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            staging_dir = root / "staging"
            model_dir.mkdir()
            staging_dir.mkdir()
            pipeline_path = model_dir / "feature_pipeline.pkl"
            model_path = model_dir / "mult_rf.pkl"
            pipeline_path.write_bytes(b"old-pipeline")
            model_path.write_bytes(b"old-model")
            (staging_dir / pipeline_path.name).write_bytes(b"new-pipeline")
            (staging_dir / model_path.name).write_bytes(b"new-model")

            self.assertEqual(b"old-model", model_path.read_bytes())
            with patch.object(trainmain, "MODEL_DIR", model_dir), patch.object(
                trainmain, "PIPELINE_PATH", pipeline_path
            ):
                trainmain._promote_artifacts(
                    staging_dir, {"rf"}, include_pipeline=True
                )

            self.assertEqual(b"new-pipeline", pipeline_path.read_bytes())
            self.assertEqual(b"new-model", model_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
