"""
utils/metrics.py
================
Performance metrics computation for the streetlight detection model.
Computes: Precision, Recall, F1-Score, Accuracy, mAP.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Class names used throughout the project
CLASS_NAMES = ["functional", "non_functional"]


def compute_classification_metrics(y_true: list,
                                    y_pred: list,
                                    save_dir: str = "outputs/logs") -> dict:
    """
    Compute classification metrics from ground-truth and predicted labels.

    Args:
        y_true   : List of true class indices
        y_pred   : List of predicted class indices
        save_dir : Directory to save confusion matrix plot

    Returns:
        Dictionary with precision, recall, f1, accuracy per class
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Overall Accuracy ──
    acc = accuracy_score(y_true, y_pred)

    # ── Per-class Precision, Recall, F1 ──
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # ── Print formatted report ──
    print("\n" + "=" * 55)
    print("       📊  CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 55)
    print(f"  Overall Accuracy : {acc * 100:.2f}%\n")

    for i, cls in enumerate(CLASS_NAMES):
        print(f"  Class: {cls.upper()}")
        print(f"    Precision : {precision[i]:.4f}")
        print(f"    Recall    : {recall[i]:.4f}")
        print(f"    F1-Score  : {f1[i]:.4f}")
        print(f"    Support   : {support[i]}")
        print()

    print("  Full Classification Report:")
    print("  " + "-" * 45)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    # Indent for neat printing
    for line in report.split("\n"):
        print("  " + line)
    print("=" * 55)

    # ── Confusion Matrix Plot ──
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _plot_confusion_matrix(cm, save_dir)

    return {
        "accuracy": acc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": cm.tolist()
    }


def _plot_confusion_matrix(cm: np.ndarray, save_dir: str):
    """Plot and save a styled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title("Confusion Matrix — Streetlight Detection", fontsize=13, pad=12)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [INFO] Confusion matrix saved → {save_path}")


def iou(box1: tuple, box2: tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2 : Tuples (x1, y1, x2, y2)

    Returns:
        IoU score (0.0–1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_map(predictions: list,
                ground_truths: list,
                iou_threshold: float = 0.5) -> float:
    """
    Compute mean Average Precision (mAP) at a given IoU threshold.

    Args:
        predictions   : List of dicts {bbox, class_id, confidence}
        ground_truths : List of dicts {bbox, class_id}
        iou_threshold : IoU threshold for a True Positive (default 0.5)

    Returns:
        mAP score (0.0–1.0)
    """
    aps = []

    for cls_id in [0, 1]:
        cls_preds = [p for p in predictions if p["class_id"] == cls_id]
        cls_gts   = [g for g in ground_truths if g["class_id"] == cls_id]

        if not cls_gts:
            continue

        # Sort predictions by descending confidence
        cls_preds.sort(key=lambda x: x["confidence"], reverse=True)

        matched = set()
        tp_list, fp_list = [], []

        for pred in cls_preds:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(cls_gts):
                if i in matched:
                    continue
                score = iou(pred["bbox"], gt["bbox"])
                if score > best_iou:
                    best_iou, best_idx = score, i

            if best_iou >= iou_threshold and best_idx not in matched:
                tp_list.append(1)
                fp_list.append(0)
                matched.add(best_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision-recall curve
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall_curve    = tp_cum / (len(cls_gts) + 1e-6)

        # Area under P-R curve (AP) using trapezoidal rule
        ap = np.trapz(precision_curve, recall_curve)
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0
