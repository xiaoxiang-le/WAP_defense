import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from loaddata import LABEL_NAMES


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"


def plot_confusion_matrix(
    model_name, matrix, class_names=("Normal", "XSS", "SQL injection")
):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float),
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    axis.set(
        title="{} Confusion Matrix".format(model_name),
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")

    threshold = normalized.max() / 2 if normalized.size else 0
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            axis.text(
                column,
                row,
                "{:.2f}".format(normalized[row, column]),
                ha="center",
                va="center",
                color="white" if normalized[row, column] > threshold else "black",
            )
    figure.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name)
    figure.savefig(IMAGE_DIR / "confusion_matrix_{}.png".format(safe_name), dpi=150)
    plt.close(figure)


def evaluate_predictions(model_name, true_labels, predicted_labels):
    labels = list(range(len(LABEL_NAMES)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    print("\n{} 混淆矩阵：\n{}".format(model_name, matrix))
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    plot_confusion_matrix(model_name, matrix)
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
