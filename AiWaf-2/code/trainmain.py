import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from artifacts import (
    MANIFEST_SCHEMA_VERSION,
    atomic_json_dump,
    load_json,
    sha256_file,
)
from knn import KNNModel
from rf import RFModel
from splitdata import splitmain
from svm import SVMModel
from vecmodel import FeaturePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
PIPELINE_PATH = MODEL_DIR / "feature_pipeline.pkl"
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


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是大于或等于 0 的整数")
    return number


def validate_training_options(
    selected_models, epochs, seed, max_features, max_vocab, sequence_length
):
    if "all" in selected_models and len(selected_models) != 1:
        raise ValueError("all 不能与具体模型同时使用")
    if not selected_models:
        raise ValueError("至少需要选择一个模型")
    if epochs <= 0 or max_features <= 0 or max_vocab <= 0 or sequence_length <= 0:
        raise ValueError("训练轮数和特征规模必须大于 0")
    if seed < 0:
        raise ValueError("随机种子不能为负数")
    selected = set(MODEL_NAMES if "all" in selected_models else selected_models)
    unknown = selected.difference(MODEL_NAMES)
    if unknown:
        raise ValueError("未知模型：{}".format(", ".join(sorted(unknown))))
    if "cnn" in selected and sequence_length < 7:
        raise ValueError("CNN 的序列长度不能小于 7")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="训练并评估 AiWaf-2 分类模型")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + MODEL_NAMES,
        default=["all"],
        help="需要训练的模型，默认训练全部模型",
    )
    parser.add_argument(
        "--epochs", type=positive_integer, default=3, help="CNN 和 GRU 的训练轮数"
    )
    parser.add_argument("--seed", type=nonnegative_integer, default=42, help="随机种子")
    parser.add_argument(
        "--max-features", type=positive_integer, default=10000, help="TF-IDF 最大特征数"
    )
    parser.add_argument(
        "--max-vocab", type=positive_integer, default=20000, help="神经网络最大词表大小"
    )
    parser.add_argument(
        "--sequence-length", type=positive_integer, default=200, help="神经网络输入序列长度"
    )
    args = parser.parse_args()
    try:
        validate_training_options(
            args.models,
            args.epochs,
            args.seed,
            args.max_features,
            args.max_vocab,
            args.sequence_length,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def train_models(
    selected_models,
    epochs=3,
    seed=42,
    max_features=10000,
    max_vocab=20000,
    sequence_length=200,
):
    selected = validate_training_options(
        selected_models, epochs, seed, max_features, max_vocab, sequence_length
    )
    random.seed(seed)
    np.random.seed(seed)
    if selected.intersection({"cnn", "gru"}):
        import tensorflow as tf

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

    configuration = {
        "max_features": max_features,
        "max_vocab": max_vocab,
        "sequence_length": sequence_length,
    }
    full_retrain = selected == set(MODEL_NAMES)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=str(MODEL_DIR)) as directory:
        staging_dir = Path(directory)
        if full_retrain:
            pipeline = FeaturePipeline(**configuration)
            print("正在拟合 TF-IDF 和序列词表...")
            pipeline.fit(train_payloads)
            pipeline.save(staging_dir / PIPELINE_PATH.name)
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

        _train_selected_models(
            selected=selected,
            pipeline=pipeline,
            staging_dir=staging_dir,
            train_payloads=train_payloads,
            train_labels=train_labels,
            validation_payloads=validation_payloads,
            validation_labels=validation_labels,
            test_payloads=test_payloads,
            test_labels=test_labels,
            epochs=epochs,
            seed=seed,
            metrics=metrics,
        )

        for model_name in selected:
            filename = MODEL_FILES[model_name]
            manifest_models[model_name] = {
                "filename": filename,
                "sha256": sha256_file(staging_dir / filename),
            }

        pipeline_candidate = (
            staging_dir / PIPELINE_PATH.name if full_retrain else PIPELINE_PATH
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_metadata,
            "seed": seed,
            "labels": ["正常", "XSS攻击", "SQL注入攻击"],
            "feature_configuration": configuration,
            "pipeline": {
                "filename": PIPELINE_PATH.name,
                "sha256": sha256_file(pipeline_candidate),
            },
            "models": manifest_models,
        }
        _promote_artifacts(staging_dir, selected, include_pipeline=full_retrain)
        atomic_json_dump(metrics, METRICS_PATH)
        atomic_json_dump(manifest, MANIFEST_PATH)
    print("\n训练完成，评估结果已保存至 {}".format(METRICS_PATH))
    return metrics


def _train_selected_models(
    selected,
    pipeline,
    staging_dir,
    train_payloads,
    train_labels,
    validation_payloads,
    validation_labels,
    test_payloads,
    test_labels,
    epochs,
    seed,
    metrics,
):
    traditional = selected.intersection({"rf", "knn", "svm"})
    if traditional:
        train_tfidf = pipeline.transform_tfidf(train_payloads)
        test_tfidf = pipeline.transform_tfidf(test_payloads)
        if "rf" in selected:
            metrics["rf"] = RFModel(
                train_tfidf,
                train_labels,
                test_tfidf,
                test_labels,
                seed=seed,
                model_path=staging_dir / MODEL_FILES["rf"],
            ).train()
        if "knn" in selected:
            metrics["knn"] = KNNModel(
                train_tfidf,
                train_labels,
                test_tfidf,
                test_labels,
                model_path=staging_dir / MODEL_FILES["knn"],
            ).train()
        if "svm" in selected:
            metrics["svm"] = SVMModel(
                train_tfidf,
                train_labels,
                test_tfidf,
                test_labels,
                seed=seed,
                model_path=staging_dir / MODEL_FILES["svm"],
            ).train()

    neural = selected.intersection({"cnn", "gru"})
    if neural:
        from cnn import CNNModel
        from gru import GRUModel

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
            metrics["cnn"] = CNNModel(
                *neural_arguments,
                epochs=epochs,
                model_path=staging_dir / MODEL_FILES["cnn"],
            ).train()
        if "gru" in selected:
            metrics["gru"] = GRUModel(
                *neural_arguments,
                epochs=epochs,
                model_path=staging_dir / MODEL_FILES["gru"],
            ).train()


def _promote_artifacts(staging_dir, selected, include_pipeline):
    if include_pipeline:
        os.replace(str(staging_dir / PIPELINE_PATH.name), str(PIPELINE_PATH))
    for model_name in sorted(selected):
        filename = MODEL_FILES[model_name]
        os.replace(str(staging_dir / filename), str(MODEL_DIR / filename))


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
