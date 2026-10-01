import hashlib
import json
import os
from pathlib import Path

import joblib


MANIFEST_SCHEMA_VERSION = 1


def sha256_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path):
    path = Path(path)
    return path.with_name("{}.tmp{}".format(path.stem, path.suffix))


def atomic_joblib_dump(value, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    joblib.dump(value, temporary)
    os.replace(str(temporary), str(path))


def atomic_keras_save(model, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    model.save(temporary)
    os.replace(str(temporary), str(path))


def atomic_json_dump(value, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def load_json(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))
