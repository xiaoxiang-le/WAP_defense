import argparse
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from tensorflow import keras

from loaddata import LABEL_NAMES
from vecmodel import FeaturePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_NAMES = ("rf", "knn", "svm", "cnn", "gru")
DEFAULT_PAYLOAD = (
    "http://honywen.com/index.cgi?year=<script>alert(1)</script>"
)


def configure_console_encoding():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def label_name(label):
    return LABEL_NAMES[int(label)]


def consensus_label(model_results):
    counts = Counter(model_results.values())
    if not counts:
        raise ValueError("至少需要一个模型结果")
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "模型意见不一致"
    return ranked[0][0]


def predict(payload, selected_models=("all",)):
    pipeline = FeaturePipeline.load()
    selected = set(MODEL_NAMES if "all" in selected_models else selected_models)
    results = {}

    if selected.intersection({"rf", "knn", "svm"}):
        tfidf = pipeline.transform_tfidf([payload])
        model_files = {
            "rf": "mult_rf.pkl",
            "knn": "mult_knn.pkl",
            "svm": "mult_svm.pkl",
        }
        for model_name, filename in model_files.items():
            if model_name in selected:
                model = joblib.load(MODEL_DIR / filename)
                results[model_name] = label_name(model.predict(tfidf)[0])

    if selected.intersection({"cnn", "gru"}):
        payload_sequence = pipeline.transform_sequence([payload])
        model_files = {
            "cnn": "mult_cnn.keras",
            "gru": "gru.keras",
        }
        for model_name, filename in model_files.items():
            if model_name in selected:
                model = keras.models.load_model(MODEL_DIR / filename, compile=False)
                probabilities = model.predict(payload_sequence, verbose=0)[0]
                results[model_name] = label_name(np.argmax(probabilities))

    if len(results) > 1:
        results["ensemble"] = consensus_label(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="使用 AiWaf-2 模型检测请求 Payload")
    parser.add_argument("payload", nargs="?", default=DEFAULT_PAYLOAD, help="待检测的 URL 或 Payload")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + MODEL_NAMES,
        default=["all"],
        help="参与预测的模型，默认使用全部模型",
    )
    return parser.parse_args()


def main():
    configure_console_encoding()
    args = parse_args()
    results = predict(args.payload, args.models)
    for model_name, result in results.items():
        print("{} 模型预测结果：{}".format(model_name.upper(), result))


if __name__ == "__main__":
    main()

