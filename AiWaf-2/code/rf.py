from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from artifacts import atomic_joblib_dump
from evaluate import evaluate_predictions


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "mult_rf.pkl"


class RFModel:
    def __init__(
        self, train_data, train_labels, test_data, test_labels, seed=42, model_path=MODEL_PATH
    ):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.seed = seed
        self.model_path = Path(model_path)

    def train(self):
        model = RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )
        print("[RF] 开始训练")
        model.fit(self.train_data, self.train_labels)
        atomic_joblib_dump(model, self.model_path)
        predictions = model.predict(self.test_data)
        return evaluate_predictions("RF", self.test_labels, predictions)
