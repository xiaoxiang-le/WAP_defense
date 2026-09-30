import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

NORMAL = 0
XSS = 1
SQL_INJECTION = 2
LABEL_NAMES = ("正常", "XSS攻击", "SQL注入攻击")


def _read_dataset(filename, data_column, label_column, attack_label):
    data = []
    labels = []
    with Path(filename).open(encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) <= max(data_column, label_column):
                continue
            payload = row[data_column].strip()
            source_label = row[label_column].strip()
            if not payload or source_label not in {"0", "1"}:
                continue
            data.append(payload)
            labels.append(NORMAL if source_label == "0" else attack_label)
    return data, labels


def loaddata_xss(filename=None):
    path = Path(filename) if filename else DATA_DIR / "XSS_dataset.csv"
    return _read_dataset(path, data_column=1, label_column=2, attack_label=XSS)


def loaddata_sqli(filename=None):
    path = Path(filename) if filename else DATA_DIR / "SQLiV3.csv"
    return _read_dataset(path, data_column=0, label_column=1, attack_label=SQL_INJECTION)


def load_all_data():
    sqli_data, sqli_labels = loaddata_sqli()
    xss_data, xss_labels = loaddata_xss()
    return sqli_data + xss_data, sqli_labels + xss_labels
