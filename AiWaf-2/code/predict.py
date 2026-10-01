import argparse
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from tensorflow import keras

from artifacts import MANIFEST_SCHEMA_VERSION, load_json, sha256_file
from loaddata import LABEL_NAMES
from vecmodel import FeaturePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
MODEL_NAMES = ("rf", "knn", "svm", "cnn", "gru")
MODEL_FILES = {
    "rf": "mult_rf.pkl",
    "knn": "mult_knn.pkl",
    "svm": "mult_svm.pkl",
    "cnn": "mult_cnn.keras",
    "gru": "gru.keras",
}
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


def validate_artifacts(selected_models):
    if not MANIFEST_PATH.exists():
        raise RuntimeError("缺少 model_manifest.json，请先全量训练模型")
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("模型 manifest 版本不受支持，请重新训练模型")
    if tuple(manifest.get("labels", ())) != tuple(LABEL_NAMES):
        raise RuntimeError("模型标签定义与当前代码不一致，请重新训练模型")

    pipeline_entry = manifest.get("pipeline", {})
    pipeline_path = MODEL_DIR / pipeline_entry.get("filename", "")
    if not pipeline_path.is_file() or sha256_file(pipeline_path) != pipeline_entry.get("sha256"):
        raise RuntimeError("特征管线缺失或校验失败，请重新训练模型")

    model_entries = manifest.get("models", {})
    for model_name in selected_models:
        entry = model_entries.get(model_name, {})
        expected_filename = MODEL_FILES[model_name]
        model_path = MODEL_DIR / entry.get("filename", "")
        if entry.get("filename") != expected_filename or not model_path.is_file():
            raise RuntimeError("{} 模型缺失，请重新训练该模型".format(model_name.upper()))
        if sha256_file(model_path) != entry.get("sha256"):
            raise RuntimeError("{} 模型校验失败，请重新训练该模型".format(model_name.upper()))
    return pipeline_path


def predict(payload, selected_models=("all",)):
    selected = set(MODEL_NAMES if "all" in selected_models else selected_models)
    pipeline_path = validate_artifacts(selected)
    pipeline = FeaturePipeline.load(pipeline_path)
    results = {}

    if selected.intersection({"rf", "knn", "svm"}):
        tfidf = pipeline.transform_tfidf([payload])
        for model_name in ("rf", "knn", "svm"):
            if model_name in selected:
                model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
                results[model_name] = label_name(model.predict(tfidf)[0])

    if selected.intersection({"cnn", "gru"}):
        payload_sequence = pipeline.transform_sequence([payload])
        for model_name in ("cnn", "gru"):
            if model_name in selected:
                model = keras.models.load_model(
                    MODEL_DIR / MODEL_FILES[model_name], compile=False
                )
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
    try:
        results = predict(args.payload, args.models)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit("预测失败：{}".format(error))
    for model_name, result in results.items():
        print("{} 模型预测结果：{}".format(model_name.upper(), result))


if __name__ == "__main__":
    main()

