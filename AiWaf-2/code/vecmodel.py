from pathlib import Path

import joblib
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from staticfeature import GeneSeg, normalize_payload


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_PATH = BASE_DIR / "model" / "feature_pipeline.pkl"


class FeaturePipeline:
    def __init__(self, max_features=10000, max_vocab=20000, sequence_length=200):
        self.max_vocab = max_vocab
        self.sequence_length = sequence_length
        self.tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            preprocessor=normalize_payload,
        )
        self.tokenizer = Tokenizer(
            num_words=max_vocab,
            lower=False,
            filters="",
            oov_token="<OOV>",
        )

    def fit(self, payloads):
        self.tfidf.fit(payloads)
        self.tokenizer.fit_on_texts([GeneSeg(payload) for payload in payloads])
        return self

    def transform_tfidf(self, payloads):
        return self.tfidf.transform(payloads)

    def transform_sequence(self, payloads):
        tokenized = [GeneSeg(payload) for payload in payloads]
        sequences = self.tokenizer.texts_to_sequences(tokenized)
        return sequence.pad_sequences(
            sequences,
            maxlen=self.sequence_length,
            padding="post",
            truncating="post",
        )

    @property
    def vocab_size(self):
        return min(self.max_vocab, len(self.tokenizer.word_index) + 1)

    def save(self, filename=DEFAULT_PIPELINE_PATH):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filename=DEFAULT_PIPELINE_PATH):
        return joblib.load(filename)


def payload2vec(sentence, representation="sequence"):
    pipeline = FeaturePipeline.load()
    if representation == "tfidf":
        return pipeline.transform_tfidf([sentence])
    if representation == "sequence":
        return pipeline.transform_sequence([sentence])
    raise ValueError("representation 必须是 'tfidf' 或 'sequence'")
