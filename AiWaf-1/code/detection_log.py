import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class JsonlDetectionLogger:
    def __init__(
        self, filename, include_body=False, max_bytes=10 * 1024 * 1024, backup_count=5
    ):
        if max_bytes < 0:
            raise ValueError("日志大小上限不能为负数")
        if backup_count < 0:
            raise ValueError("日志历史文件数量不能为负数")
        self.path = Path(filename).expanduser().resolve()
        self.include_body = include_body
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()

    def _backup_path(self, index):
        return Path("{}.{}".format(self.path, index))

    def _rotate_if_needed(self, incoming_bytes):
        if (
            not self.max_bytes
            or not self.path.exists()
            or self.path.stat().st_size == 0
            or self.path.stat().st_size + incoming_bytes <= self.max_bytes
        ):
            return

        if self.backup_count == 0:
            self.path.unlink()
            return

        oldest = self._backup_path(self.backup_count)
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.exists():
                source.replace(self._backup_path(index + 1))
        self.path.replace(self._backup_path(1))

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
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self._rotate_if_needed(len(line.encode("utf-8")))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
