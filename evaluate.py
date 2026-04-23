"""
evaluate.py
===========
Evaluate the trained YOLOv8 model on the test set.
Computes and displays:
  - Precision, Recall, F1-Score per class
  - Overall Accuracy
  - mAP@0.5 and mAP@0.5:0.95
  - Confusion matrix plot
  - Inference speed (ms/image)

Usage:
    python evaluate.py
    python evaluate.py --weights models/best.pt --data configs/streetlight.yaml
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from colorama import Fore, Style, init

from utils.metrics import compute_classification_metrics, compute_map

init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Streetlight Detection Model"
    )
    parser.add_argument(
        "--weights", type=str, default="models/best.pt",
        help="Path to trained YOLOv8 weights"
    )
    parser.add_argument(
        "--data", type=str, default="configs/streetlight.yaml",
        help="Dataset YAML config"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for evaluation (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for mAP computation (default: 0.5)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs/logs",
        help="Directory to save evaluation results"
    )
    return parser.parse_args()


def run_yolo_validation(model: YOLO, args) -> dict:
    """
    Run YOLOv8's built-in validation to get mAP and other metrics.

    Returns:
        Dictionary of YOLOv8 validation metrics
    """
    print(f"\n{Fore.CYAN}[INFO] Running YOLOv8 validation...{Style.RESET_ALL}")

    metrics = model.val(
        data    = args.data,
        split   = args.split,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        verbose = True,
        save_json = True,
        project = args.save_dir,
        name    = "eval_results"
    )

    return metrics


def run_per_image_evaluation(model: YOLO, args) -> dict:
    """
    Run per-image inference on the test set and collect
    predicted vs ground-truth labels for classification metrics.

    Returns:
        Dictionary with y_true, y_pred, predictions, ground_truths
    """
    import yaml

    # Load dataset config to find test image paths
    with open(args.data, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("path", "."))
    test_img_dir = data_root / cfg.get(args.split, f"images/{args.split}")
    test_lbl_dir = data_root / "labels" / args.split

    if not test_img_dir.exists():
        print(f"{Fore.YELLOW}[WARN] Test image dir not found: {test_img_dir}{Style.RESET_ALL}")
        return {}

    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in test_img_dir.iterdir() if f.suffix.lower() in supported]

    if not image_files:
        print(f"{Fore.YELLOW}[WARN] No test images found in {test_img_dir}{Style.RESET_ALL}")
        return {}

    print(f"[INFO] Evaluating on {len(image_files)} test images...")

    y_true_all = []
    y_pred_all = []
    predictions_all  = []
    ground_truths_all = []

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # ── Ground truth labels ──
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    y_true_all.append(cls_id)
                    ground_truths_all.append({
                        "bbox": (x1, y1, x2, y2),
                        "class_id": cls_id
                    })

        # ── Model predictions ──
        results = model.predict(
            source  = image,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            verbose = False
        )[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            y_pred_all.append(cls_id)
            predictions_all.append({
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "confidence": conf
            })

    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "predictions": predictions_all,
        "ground_truths": ground_truths_all
    }


def print_yolo_metrics(metrics):
    """Print key metrics from YOLOv8 validation results."""
    print("\n" + "=" * 60)
    print(f"{Fore.CYAN}  📊  YOLO VALIDATION METRICS{Style.RESET_ALL}")
    print("=" * 60)

    try:
        box = metrics.box
        print(f"  mAP@0.5       : {Fore.GREEN}{box.map50:.4f}{Style.RESET_ALL}")
        print(f"  mAP@0.5:0.95  : {Fore.GREEN}{box.map:.4f}{Style.RESET_ALL}")
        print(f"  Mean Precision : {box.mp:.4f}")
        print(f"  Mean Recall    : {box.mr:.4f}")

        if hasattr(box, "ap_class_index"):
            print(f"\n  Per-Class AP@0.5:")
            class_names = ["functional", "non_functional"]
            for i, ap in enumerate(box.ap50):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                print(f"    {name:20s} : {ap:.4f}")
    except Exception as e:
        print(f"  [Note] Some metrics unavailable: {e}")

    print("=" * 60)


def main():
    args = parse_args()

    # ── Check weights ──
    if not os.path.exists(args.weights):
        print(f"{Fore.RED}[ERROR] Weights not found: {args.weights}{Style.RESET_ALL}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"{Fore.CYAN}  🚦  STREETLIGHT DETECTION — MODEL EVALUATION{Style.RESET_ALL}")
    print(f"{'=' * 60}")
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print(f"  Split   : {args.split}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")

    # ── Load model ──
    print(f"\n[INFO] Loading model...")
    model = YOLO(args.weights)

    # ── YOLOv8 built-in validation (mAP) ──
    yolo_metrics = run_yolo_validation(model, args)
    print_yolo_metrics(yolo_metrics)

    # ── Per-image classification metrics ──
    print(f"\n{Fore.CYAN}[INFO] Computing per-image classification metrics...{Style.RESET_ALL}")
    eval_data = run_per_image_evaluation(model, args)

    if eval_data and eval_data["y_true"] and eval_data["y_pred"]:
        compute_classification_metrics(
            y_true   = eval_data["y_true"],
            y_pred   = eval_data["y_pred"],
            save_dir = args.save_dir
        )

        # ── mAP from our own implementation ──
        if eval_data["predictions"] and eval_data["ground_truths"]:
            our_map = compute_map(
                predictions   = eval_data["predictions"],
                ground_truths = eval_data["ground_truths"],
                iou_threshold = args.iou
            )
            print(f"\n  Custom mAP@{args.iou} : {Fore.GREEN}{our_map:.4f}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[WARN] Not enough data for classification metrics.{Style.RESET_ALL}")
        print(f"       Ensure test images and labels are in place.")

    print(f"\n{Fore.GREEN}[INFO] Evaluation complete! Results saved → {args.save_dir}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
