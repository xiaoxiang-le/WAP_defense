import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class JsonlDetectionLogger:
    def __init__(self, filename, include_body=False):
        self.path = Path(filename).expanduser().resolve()
        self.include_body = include_body
        self._lock = threading.Lock()

    def write(self, record, classification, risk, malicious_probability=None):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": record.get("source_ip", ""),
            "target_ip": record.get("target_ip", ""),
            "method": record.get("method", ""),
            "host": record.get("host", ""),
            "path": record.get("path", ""),
            "url": record.get("url", ""),
            "content_type": record.get("content_type", ""),
            "classification": classification,
            "risk": risk,
        }
        if malicious_probability is not None:
            event["malicious_probability"] = float(malicious_probability)
        if self.include_body:
            event["body"] = record.get("body", "")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
