from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Calculate the standard finance-oriented classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, object]:
    """Evaluate a classifier and return predictions, probabilities, and metrics."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {
        "model_name": model_name,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "classification_report": report,
    }


def print_evaluation_report(result: Dict[str, object]) -> None:
    """Print a readable model evaluation block."""

    metrics = result["metrics"]
    print(f"\n{'=' * 60}")
    print(f"MODEL: {result['model_name']}")
    print(f"{'=' * 60}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:>10}: {metric_value:.4f}")
    print("\nClassification Report:")
    print(result["classification_report"])


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str | Path) -> None:
    """Save the confusion matrix heatmap."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(models: Dict[str, object], X_test, y_test, save_path: str | Path) -> None:
    """Compare ROC curves for all trained models."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def find_optimal_threshold(y_true, y_prob) -> Dict[str, float]:
    """Choose a decision threshold that maximizes F1 score."""

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)
    f1_scores = 2 * (precisions * recalls) / np.clip(precisions + recalls, 1e-9, None)
    best_index = int(np.nanargmax(f1_scores))
    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precisions[best_index]),
        "recall": float(recalls[best_index]),
        "f1": float(f1_scores[best_index]),
    }


def save_metrics_report(results: Dict[str, Dict[str, object]], save_path: str | Path) -> None:
    """Persist evaluation metrics and reports to JSON."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for model_name, result in results.items():
        serializable[model_name] = {
            "metrics": result["metrics"],
            "classification_report": result["classification_report"],
        }

    with save_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)
