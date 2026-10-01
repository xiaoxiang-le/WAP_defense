import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from artifacts import (
    MANIFEST_SCHEMA_VERSION,
    atomic_json_dump,
    load_json,
    sha256_file,
)
from cnn import CNNModel
from gru import GRUModel
from knn import KNNModel
from rf import RFModel
from splitdata import splitmain
from svm import SVMModel
from vecmodel import FeaturePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "model" / "training_metrics.json"
MANIFEST_PATH = BASE_DIR / "model" / "model_manifest.json"
PIPELINE_PATH = BASE_DIR / "model" / "feature_pipeline.pkl"
MODEL_NAMES = ("rf", "knn", "svm", "cnn", "gru")
MODEL_FILES = {
    "rf": "mult_rf.pkl",
    "knn": "mult_knn.pkl",
    "svm": "mult_svm.pkl",
    "cnn": "mult_cnn.keras",
    "gru": "gru.keras",
}


def configure_console_encoding():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="训练并评估 AiWaf-2 分类模型")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + MODEL_NAMES,
        default=["all"],
        help="需要训练的模型，默认训练全部模型",
    )
    parser.add_argument("--epochs", type=int, default=3, help="CNN 和 GRU 的训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-features", type=int, default=10000, help="TF-IDF 最大特征数")
    parser.add_argument("--max-vocab", type=int, default=20000, help="神经网络最大词表大小")
    parser.add_argument("--sequence-length", type=int, default=200, help="神经网络输入序列长度")
    return parser.parse_args()


def train_models(
    selected_models,
    epochs=3,
    seed=42,
    max_features=10000,
    max_vocab=20000,
    sequence_length=200,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    (
        train_payloads,
        train_labels,
        validation_payloads,
        validation_labels,
        test_payloads,
        test_labels,
        dataset_metadata,
    ) = splitmain(seed=seed, return_metadata=True)

    selected = set(MODEL_NAMES if "all" in selected_models else selected_models)
    configuration = {
        "max_features": max_features,
        "max_vocab": max_vocab,
        "sequence_length": sequence_length,
    }
    full_retrain = selected == set(MODEL_NAMES)

    if full_retrain:
        pipeline = FeaturePipeline(**configuration)
        print("正在拟合 TF-IDF 和序列词表...")
        pipeline.fit(train_payloads)
        pipeline.save(PIPELINE_PATH)
        metrics = {}
        manifest_models = {}
    else:
        pipeline, previous_manifest = _load_reusable_pipeline(
            dataset_metadata=dataset_metadata,
            seed=seed,
            configuration=configuration,
        )
        metrics = load_json(METRICS_PATH) if METRICS_PATH.exists() else {}
        manifest_models = dict(previous_manifest.get("models", {}))

    traditional = selected.intersection({"rf", "knn", "svm"})
    if traditional:
        train_tfidf = pipeline.transform_tfidf(train_payloads)
        test_tfidf = pipeline.transform_tfidf(test_payloads)

        if "rf" in selected:
            metrics["rf"] = RFModel(
                train_tfidf, train_labels, test_tfidf, test_labels, seed=seed
            ).train()
        if "knn" in selected:
            metrics["knn"] = KNNModel(
                train_tfidf, train_labels, test_tfidf, test_labels
            ).train()
        if "svm" in selected:
            metrics["svm"] = SVMModel(
                train_tfidf, train_labels, test_tfidf, test_labels, seed=seed
            ).train()

    neural = selected.intersection({"cnn", "gru"})
    if neural:
        train_sequence = pipeline.transform_sequence(train_payloads)
        validation_sequence = pipeline.transform_sequence(validation_payloads)
        test_sequence = pipeline.transform_sequence(test_payloads)

        neural_arguments = (
            train_sequence,
            train_labels,
            validation_sequence,
            validation_labels,
            test_sequence,
            test_labels,
            pipeline.vocab_size,
            pipeline.sequence_length,
        )
        if "cnn" in selected:
            metrics["cnn"] = CNNModel(*neural_arguments, epochs=epochs).train()
        if "gru" in selected:
            metrics["gru"] = GRUModel(*neural_arguments, epochs=epochs).train()

    for model_name in selected:
        filename = MODEL_FILES[model_name]
        manifest_models[model_name] = {
            "filename": filename,
            "sha256": sha256_file(BASE_DIR / "model" / filename),
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_metadata,
        "seed": seed,
        "labels": ["正常", "XSS攻击", "SQL注入攻击"],
        "feature_configuration": configuration,
        "pipeline": {
            "filename": PIPELINE_PATH.name,
            "sha256": sha256_file(PIPELINE_PATH),
        },
        "models": manifest_models,
    }
    atomic_json_dump(metrics, METRICS_PATH)
    atomic_json_dump(manifest, MANIFEST_PATH)
    print("\n训练完成，评估结果已保存至 {}".format(METRICS_PATH))
    return metrics


def _load_reusable_pipeline(dataset_metadata, seed, configuration):
    if not MANIFEST_PATH.exists():
        raise RuntimeError("缺少模型 manifest；请先使用 --models all 完成全量训练")

    manifest = load_json(MANIFEST_PATH)
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "seed": seed,
        "feature_configuration": configuration,
    }
    actual = {
        "schema_version": manifest.get("schema_version"),
        "dataset_fingerprint": manifest.get("dataset", {}).get("dataset_fingerprint"),
        "seed": manifest.get("seed"),
        "feature_configuration": manifest.get("feature_configuration"),
    }
    if actual != expected:
        raise RuntimeError(
            "数据、随机种子或特征参数已变化；请使用 --models all 重新生成全部模型"
        )

    pipeline_entry = manifest.get("pipeline", {})
    if (
        pipeline_entry.get("filename") != PIPELINE_PATH.name
        or not PIPELINE_PATH.exists()
        or sha256_file(PIPELINE_PATH) != pipeline_entry.get("sha256")
    ):
        raise RuntimeError("特征管线缺失或校验失败；请使用 --models all 重新训练")
    return FeaturePipeline.load(PIPELINE_PATH), manifest


def main():
    configure_console_encoding()
    args = parse_args()
    train_models(
        selected_models=args.models,
        epochs=args.epochs,
        seed=args.seed,
        max_features=args.max_features,
        max_vocab=args.max_vocab,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
