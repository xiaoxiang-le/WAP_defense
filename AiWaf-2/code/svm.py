from pathlib import Path

import joblib
from sklearn.svm import LinearSVC

from evaluate import evaluate_predictions


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "mult_svm.pkl"


class SVMModel:
    def __init__(self, train_data, train_labels, test_data, test_labels, seed=42):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.seed = seed

    def train(self):
        model = LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=self.seed,
            dual="auto",
        )
        print("[SVM] 开始训练")
        model.fit(self.train_data, self.train_labels)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        predictions = model.predict(self.test_data)
        return evaluate_predictions("SVM", self.test_labels, predictions)
