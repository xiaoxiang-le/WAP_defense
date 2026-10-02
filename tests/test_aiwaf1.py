import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scapy.all import Ether, IP, Raw, TCP
from scapy.layers.inet6 import IPv6


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-1" / "code"
sys.path.insert(0, str(CODE_DIR))

from train_url import Train, _deduplicate_samples, normalize_url, split_word
from detection_log import JsonlDetectionLogger
import geturl


def load_rule_module():
    spec = importlib.util.spec_from_file_location(
        "aiwaf1_attack_type", CODE_DIR / "type.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttackRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rule_module()

    def test_xss_rule_is_case_insensitive(self):
        tokens = split_word("/?q=<SCRIPT>alert(1)</SCRIPT>")
        result = self.rules.find_type(tokens)
        self.assertIn("XSS", result)
        self.assertIn("严重", result)

    def test_sql_rule_recognizes_case_and_when_as_separate_tokens(self):
        tokens = split_word("/?id=1 union select case when 1=1")
        result = self.rules.find_type(tokens)
        self.assertIn("SQL注入", result)
        self.assertIn("中级", result)

    def test_clean_request_is_unknown(self):
        result = self.rules.find_type(split_word("/products/list"))
        self.assertIn("未知", result)

    def test_training_deduplicates_normalized_urls_and_drops_conflicts(self):
        payloads, labels = _deduplicate_samples(
            ["/item?id=1", "/ITEM?id=2", "/safe", "/SAFE"],
            [1, 1, 0, 1],
        )
        self.assertEqual([normalize_url("/item?id=1")], [normalize_url(x) for x in payloads])
        self.assertEqual([1], labels)


class PredictionThresholdTests(unittest.TestCase):
    class FakeVectorizer:
        @staticmethod
        def transform(urls):
            return urls

    class FakeClassifier:
        classes_ = [0, 1]

        @staticmethod
        def predict_proba(vectors):
            return [[0.3, 0.7] if value == "attack" else [0.8, 0.2] for value in vectors]

    def setUp(self):
        self.detector = Train()
        self.detector.vectorizer = self.FakeVectorizer()
        self.detector.classifier = self.FakeClassifier()

    def test_threshold_controls_classification_and_reports_probability(self):
        details = self.detector.predict_details(["safe", "attack"], threshold=0.75)

        self.assertEqual([0, 0], [item["label"] for item in details])
        self.assertEqual(0.7, details[1]["malicious_probability"])
        self.assertEqual(
            "url为恶意攻击", self.detector.predict(["attack"], threshold=0.6)
        )

    def test_invalid_threshold_and_empty_prediction_are_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.predict_details(["payload"], threshold=1.1)
        with self.assertRaises(ValueError):
            self.detector.predict([])


class CaptureConfigurationTests(unittest.TestCase):
    def test_custom_port_is_bound_and_filtered(self):
        with patch.object(geturl, "bind_layers") as bind_layers_mock, patch.object(
            geturl, "sniff"
        ) as sniff_mock:
            records = geturl.sniff_requests(
                interface="test-interface", port=18080, timeout=2, count=1
            )

        self.assertEqual([], records)
        self.assertEqual(2, bind_layers_mock.call_count)
        options = sniff_mock.call_args.kwargs
        self.assertEqual("tcp port 18080", options["filter"])
        self.assertEqual("test-interface", options["iface"])
        self.assertTrue(callable(options["lfilter"]))
        self.assertIn("session", options)

    def test_post_body_is_included_and_limited(self):
        packet = (
            Ether()
            / IP(src="127.0.0.1", dst="127.0.0.1")
            / TCP(sport=12345, dport=8080)
            / geturl.http.HTTP()
            / geturl.http.HTTPRequest(
                Method=b"POST",
                Host=b"example.test",
                Path=b"/login",
                Content_Type=b"application/x-www-form-urlencoded",
            )
            / Raw(load=b"name=admin&query=<script>alert(1)</script>")
        )

        record = geturl.packet_to_record(packet, max_body_bytes=20)

        self.assertEqual("example.test/login", record["url"])
        self.assertEqual("name=admin&query=<sc", record["body"])
        self.assertIn(record["body"], record["payload"])

    def test_ipv6_addresses_are_included_in_the_record(self):
        packet = (
            IPv6(src="2001:db8::1", dst="2001:db8::2")
            / TCP(sport=12345, dport=8080)
            / geturl.http.HTTP()
            / geturl.http.HTTPRequest(
                Method=b"GET",
                Host=b"example.test",
                Path=b"/status",
            )
        )

        record = geturl.packet_to_record(packet)

        self.assertIn("IP_Src：2001:db8::1", record["display"])
        self.assertIn("IP_Dst：2001:db8::2", record["display"])


class DetectionLogTests(unittest.TestCase):
    def test_log_rotation_keeps_configured_number_of_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlDetectionLogger(path, max_bytes=1, backup_count=2)

            for index in range(3):
                logger.write({"url": "/{}".format(index)}, "result-{}".format(index), "risk")

            current = json.loads(path.read_text(encoding="utf-8"))
            previous = json.loads(Path("{}.1".format(path)).read_text(encoding="utf-8"))
            oldest = json.loads(Path("{}.2".format(path)).read_text(encoding="utf-8"))

        self.assertEqual("result-2", current["classification"])
        self.assertEqual("result-1", previous["classification"])
        self.assertEqual("result-0", oldest["classification"])

    def test_log_rotation_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlDetectionLogger(path, max_bytes=0)
            logger.write({"url": "/one"}, "one", "risk")
            logger.write({"url": "/two"}, "two", "risk")

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(2, len(lines))
        self.assertFalse(Path("{}.1".format(path)).exists())

    def test_log_omits_request_body_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlDetectionLogger(path)
            logger.write(
                {
                    "source_ip": "127.0.0.1",
                    "target_ip": "127.0.0.1",
                    "method": "POST",
                    "host": "example.test",
                    "path": "/login",
                    "url": "example.test/login",
                    "content_type": "application/json",
                    "body": '{"password":"secret"}',
                },
                "url为恶意攻击",
                "攻击类型：SQL注入  攻击等级：严重",
                malicious_probability=0.9,
            )
            event = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("body", event)
        self.assertEqual("POST", event["method"])
        self.assertEqual("url为恶意攻击", event["classification"])
        self.assertEqual(0.9, event["malicious_probability"])

    def test_log_can_include_request_body_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlDetectionLogger(path, include_body=True)
            logger.write(
                {"url": "example.test/login", "body": "id=1 union select"},
                "url为恶意攻击",
                "攻击类型：SQL注入  攻击等级：中级",
            )
            event = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual("id=1 union select", event["body"])


if __name__ == "__main__":
    unittest.main()
