import os
import pickle
import re
from pathlib import Path
from urllib.parse import unquote

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
GOOD_URL_PATH = BASE_DIR / "data" / "good_fromE.txt"
BAD_URL_PATH = BASE_DIR / "data" / "badqueries.txt"
MODEL_PATH = BASE_DIR / "model" / "lg.pickle"
MODEL_VERSION = 2


def get_url():
    with GOOD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        good_urls = [line.strip() for line in handle if line.strip()]
    with BAD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        bad_urls = [line.strip() for line in handle if line.strip()]
    return good_urls, bad_urls


def normalize_url(url):
    url = unquote(unquote(str(url))).lower()
    url = re.sub(r"\d+", "0", url)
    return re.sub(r"(http|https)://[a-z0-9\.@&/#!#\?]+", "http://u", url)


def split_word(url):
    url = normalize_url(url)
    pattern = r"""(?x)[\w\.]+?\(|\)|"\w+?"|'\w+?'|http://\w|</\w+>|<\w+>|<\w+|\w+=|>|[\w\.]+"""
    return [
        token
        for token in nltk.regexp_tokenize(url, pattern)
        if token not in {"0", "http://u", ""}
    ]


class Train:
    def __init__(self):
        self.model_version = MODEL_VERSION
        self.vectorizer = None
        self.classifier = None

    def model_train(self, seed=42):
        good_urls, bad_urls = get_url()
        payloads, labels = _deduplicate_samples(
            good_urls + bad_urls,
            [0] * len(good_urls) + [1] * len(bad_urls),
        )
        train_data, test_data, train_labels, test_labels = train_test_split(
            payloads,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=20000,
            min_df=2,
            sublinear_tf=True,
            preprocessor=normalize_url,
        )
        train_vectors = self.vectorizer.fit_transform(train_data)
        test_vectors = self.vectorizer.transform(test_data)
        self.classifier = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        self.classifier.fit(train_vectors, train_labels)
        accuracy = self.classifier.score(test_vectors, test_labels)
        print("AiWaf-1 模型准确率：{:.2%}".format(accuracy))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(self, handle)
        os.replace(str(temporary_path), str(MODEL_PATH))
        return accuracy

    @classmethod
    def load_or_train(cls):
        try:
            with MODEL_PATH.open("rb") as handle:
                model = pickle.load(handle)
            if (
                getattr(model, "model_version", None) != MODEL_VERSION
                or model.vectorizer is None
                or model.classifier is None
            ):
                raise ValueError("旧模型格式不兼容")
            return model
        except (FileNotFoundError, AttributeError, EOFError, pickle.UnpicklingError, ValueError):
            model = cls()
            model.model_train()
            return model

    def predict_labels(self, urls):
        vectors = self.vectorizer.transform([str(url) for url in urls])
        return self.classifier.predict(vectors)

    def predict_details(self, urls, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("恶意判定阈值必须在 0 到 1 之间")
        urls = [str(url) for url in urls]
        if not urls:
            return []
        vectors = self.vectorizer.transform(urls)
        probabilities = self.classifier.predict_proba(vectors)
        malicious_index = list(self.classifier.classes_).index(1)
        details = []
        for row in probabilities:
            probability = float(row[malicious_index])
            label = int(probability >= threshold)
            details.append(
                {
                    "label": label,
                    "malicious_probability": probability,
                    "message": "url为正常请求" if label == 0 else "url为恶意攻击",
                }
            )
        return details

    def predict(self, urls, threshold=0.5):
        details = self.predict_details(urls, threshold=threshold)
        if not details:
            raise ValueError("至少需要一个待检测 URL")
        return details[0]["message"]


def _deduplicate_samples(payloads, labels):
    samples = {}
    conflicts = set()
    for payload, label in zip(payloads, labels):
        key = normalize_url(payload).strip()
        if not key:
            continue
        previous = samples.get(key)
        if previous is not None and previous[1] != label:
            conflicts.add(key)
        else:
            samples[key] = (payload, label)
    for key in conflicts:
        samples.pop(key, None)
    ordered = [samples[key] for key in sorted(samples)]
    return [item[0] for item in ordered], [item[1] for item in ordered]

