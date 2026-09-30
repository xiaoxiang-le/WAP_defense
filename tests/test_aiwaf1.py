import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "AiWaf-1" / "code"
sys.path.insert(0, str(CODE_DIR))

from train_url import split_word
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


if __name__ == "__main__":
    unittest.main()
