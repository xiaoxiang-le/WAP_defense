from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

from artifacts import atomic_joblib_dump
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
        atomic_joblib_dump(model, MODEL_PATH)
        predictions = model.predict(self.test_data)
        return evaluate_predictions("KNN", self.test_labels, predictions)

