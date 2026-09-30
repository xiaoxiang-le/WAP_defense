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


def get_url():
    with GOOD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        good_urls = [line.strip() for line in handle if line.strip()]
    with BAD_URL_PATH.open(encoding="utf-8", errors="ignore") as handle:
        bad_urls = [line.strip() for line in handle if line.strip()]
    return good_urls, bad_urls


def split_word(url):
    url = unquote(unquote(str(url))).lower()
    url = re.sub(r"\d+", "0", url)
    url = re.sub(r"(http|https)://[a-zA-Z0-9\.@&/#!#\?]+", "http://u", url)
    pattern = r"""(?x)[\w\.]+?\(|\)|"\w+?"|'\w+?'|http://\w|</\w+>|<\w+>|<\w+|\w+=|>|[\w\.]+"""
    return [
        token
        for token in nltk.regexp_tokenize(url, pattern)
        if token not in {"0", "http://u", ""}
    ]


class Train:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None

    def model_train(self, seed=42):
        good_urls, bad_urls = get_url()
        payloads = good_urls + bad_urls
        labels = [0] * len(good_urls) + [1] * len(bad_urls)
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
        with MODEL_PATH.open("wb") as handle:
            pickle.dump(self, handle)
        return accuracy

    @classmethod
    def load_or_train(cls):
        try:
            with MODEL_PATH.open("rb") as handle:
                model = pickle.load(handle)
            if model.vectorizer is None or model.classifier is None:
                raise ValueError("旧模型格式不兼容")
            return model
        except (FileNotFoundError, AttributeError, ValueError):
            model = cls()
            model.model_train()
            return model

    def predict_labels(self, urls):
        vectors = self.vectorizer.transform(
            [unquote(str(url)).lower() for url in urls]
        )
        return self.classifier.predict(vectors)

    def predict(self, urls):
        label = int(self.predict_labels(urls)[0])
        return "url为正常请求" if label == 0 else "url为恶意攻击"

