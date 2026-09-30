import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn import CNNModel
from gru import GRUModel
from knn import KNNModel
from rf import RFModel
from splitdata import splitmain
from svm import SVMModel
from vecmodel import FeaturePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "model" / "training_metrics.json"
MODEL_NAMES = ("rf", "knn", "svm", "cnn", "gru")


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
    ) = splitmain(seed=seed)

    pipeline = FeaturePipeline(
        max_features=max_features,
        max_vocab=max_vocab,
        sequence_length=sequence_length,
    )
    print("正在拟合 TF-IDF 和序列词表...")
    pipeline.fit(train_payloads)
    pipeline.save()

    selected = set(MODEL_NAMES if "all" in selected_models else selected_models)
    metrics = {}

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

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n训练完成，评估结果已保存至 {}".format(METRICS_PATH))
    return metrics


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
