from pathlib import Path

import joblib
from sklearn.neighbors import KNeighborsClassifier

from evaluate import evaluate_predictions


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "mult_knn.pkl"


class KNNModel:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train(self):
        model = KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        print("[KNN] 开始训练")
        model.fit(self.train_data, self.train_labels)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        predictions = model.predict(self.test_data)
        return evaluate_predictions("KNN", self.test_labels, predictions)

