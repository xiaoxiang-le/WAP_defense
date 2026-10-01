import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scapy.all import Ether, IP, Raw, TCP


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-1" / "code"
sys.path.insert(0, str(CODE_DIR))

from train_url import _deduplicate_samples, normalize_url, split_word
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


if __name__ == "__main__":
    unittest.main()
