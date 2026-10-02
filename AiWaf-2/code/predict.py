import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import joblib
import numpy as np

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


def normalize_model_selection(selected_models):
    if "all" in selected_models:
        if len(selected_models) != 1:
            raise ValueError("all 不能与具体模型同时使用")
        return set(MODEL_NAMES)
    selected = set(selected_models)
    if not selected:
        raise ValueError("至少需要选择一个模型")
    unknown = selected.difference(MODEL_NAMES)
    if unknown:
        raise ValueError("未知模型：{}".format(", ".join(sorted(unknown))))
    return selected


class PredictionEngine:
    def __init__(self, selected_models=("all",)):
        self.selected = normalize_model_selection(selected_models)
        pipeline_path = validate_artifacts(self.selected)
        self.pipeline = FeaturePipeline.load(pipeline_path)
        self.models = {}

        for model_name in ("rf", "knn", "svm"):
            if model_name in self.selected:
                self.models[model_name] = joblib.load(
                    MODEL_DIR / MODEL_FILES[model_name]
                )

        if self.selected.intersection({"cnn", "gru"}):
            from tensorflow import keras

            for model_name in ("cnn", "gru"):
                if model_name in self.selected:
                    self.models[model_name] = keras.models.load_model(
                        MODEL_DIR / MODEL_FILES[model_name], compile=False
                    )

    def predict_many(self, payloads):
        payloads = list(payloads)
        if not payloads:
            return []
        results = [{} for _ in payloads]

        traditional = self.selected.intersection({"rf", "knn", "svm"})
        if traditional:
            tfidf = self.pipeline.transform_tfidf(payloads)
            for model_name in ("rf", "knn", "svm"):
                if model_name in traditional:
                    labels = self.models[model_name].predict(tfidf)
                    for result, label in zip(results, labels):
                        result[model_name] = label_name(label)

        neural = self.selected.intersection({"cnn", "gru"})
        if neural:
            sequences = self.pipeline.transform_sequence(payloads)
            for model_name in ("cnn", "gru"):
                if model_name in neural:
                    probabilities = self.models[model_name].predict(
                        sequences, verbose=0
                    )
                    labels = np.argmax(probabilities, axis=1)
                    for result, label in zip(results, labels):
                        result[model_name] = label_name(label)

        if len(self.selected) > 1:
            for result in results:
                result["ensemble"] = consensus_label(result)
        return results


def predict(payload, selected_models=("all",)):
    return PredictionEngine(selected_models).predict_many([payload])[0]


def load_payloads(payload=None, input_file=None):
    if input_file:
        lines = Path(input_file).read_text(encoding="utf-8").splitlines()
        payloads = [line for line in lines if line.strip()]
        if not payloads:
            raise ValueError("输入文件中没有可预测的 Payload")
        return payloads
    return [payload if payload is not None else DEFAULT_PAYLOAD]


def parse_args():
    parser = argparse.ArgumentParser(description="使用 AiWaf-2 模型检测请求 Payload")
    parser.add_argument("payload", nargs="?", help="待检测的 URL 或 Payload")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="批量输入文件，每行一个 Payload；不能与位置参数同时使用",
    )
    parser.add_argument(
        "--json", action="store_true", help="以 JSON 格式输出预测结果"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + MODEL_NAMES,
        default=["all"],
        help="参与预测的模型，默认使用全部模型",
    )
    args = parser.parse_args()
    if args.input_file and args.payload is not None:
        parser.error("payload 位置参数不能与 --input-file 同时使用")
    try:
        normalize_model_selection(args.models)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    configure_console_encoding()
    args = parse_args()
    try:
        payloads = load_payloads(args.payload, args.input_file)
        predictions = PredictionEngine(args.models).predict_many(payloads)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise SystemExit("预测失败：{}".format(error))
    if args.json:
        output = [
            {"payload": payload, "results": results}
            for payload, results in zip(payloads, predictions)
        ]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    for index, (payload, results) in enumerate(zip(payloads, predictions), start=1):
        if len(payloads) > 1:
            print("\n[{}] Payload：{}".format(index, payload))
        for model_name, result in results.items():
            print("{} 模型预测结果：{}".format(model_name.upper(), result))


if __name__ == "__main__":
    main()

