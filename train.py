"""
train.py
========
YOLOv8 Training Script for Streetlight Outage Detection.
Trains a YOLOv8 model to classify streetlights as:
  - functional     (class 0) → light is ON
  - non_functional (class 1) → light is OFF

Usage:
    python train.py
    python train.py --epochs 100 --batch 32 --model yolov8s.pt
    python train.py --epochs 50 --batch 16 --imgsz 640 --device cuda
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from colorama import Fore, Style, init

init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Streetlight Outage Detection"
    )

    # ── Model selection ──
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help=(
            "YOLOv8 base model. Options:\n"
            "  yolov8n.pt — nano   (fastest, least accurate)\n"
            "  yolov8s.pt — small  (good balance)\n"
            "  yolov8m.pt — medium (more accurate)\n"
            "  yolov8l.pt — large\n"
            "  yolov8x.pt — extra-large (most accurate)\n"
            "Default: yolov8n.pt"
        )
    )

    # ── Dataset config ──
    parser.add_argument(
        "--data", type=str, default="configs/streetlight.yaml",
        help="Path to dataset YAML config (default: configs/streetlight.yaml)"
    )

    # ── Training hyperparameters ──
    parser.add_argument("--epochs",  type=int,   default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--lr0",     type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf",     type=float, default=0.01,
                        help="Final LR as fraction of lr0 (default: 0.01)")

    # ── Hardware ──
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: '' (auto), '0' (GPU 0), 'cpu' (default: auto)"
    )

    # ── Output ──
    parser.add_argument(
        "--project", type=str, default="models",
        help="Directory to save training results (default: models/)"
    )
    parser.add_argument(
        "--name", type=str, default="streetlight_yolov8",
        help="Run name (subfolder inside --project)"
    )

    # ── Augmentation flags ──
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation (default: True)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15 epochs)")

    # ── Resume training ──
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 65)
    print(f"{Fore.CYAN}  🚦  STREETLIGHT OUTAGE DETECTION — MODEL TRAINING{Style.RESET_ALL}")
    print("=" * 65)


def check_gpu():
    """Print GPU/CPU status."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"{Fore.GREEN}[GPU] ✔ CUDA available — {gpu} ({vram:.1f} GB VRAM){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[CPU] No GPU detected — training on CPU (slower){Style.RESET_ALL}")
        print(f"      Tip: Use yolov8n.pt for faster CPU training.")


def train(args):
    print_banner()
    check_gpu()

    print(f"\n[CONFIG] Model     : {args.model}")
    print(f"[CONFIG] Dataset   : {args.data}")
    print(f"[CONFIG] Epochs    : {args.epochs}")
    print(f"[CONFIG] Batch     : {args.batch}")
    print(f"[CONFIG] Image Size: {args.imgsz}x{args.imgsz}")
    print(f"[CONFIG] Save Dir  : {args.project}/{args.name}\n")

    # ── Validate dataset config exists ──
    if not os.path.exists(args.data):
        print(f"{Fore.RED}[ERROR] Dataset config not found: {args.data}{Style.RESET_ALL}")
        print("        Make sure configs/streetlight.yaml exists.")
        return

    # ── Load pretrained YOLOv8 model ──
    print(f"[INFO] Loading base model: {args.model}")
    print("       (Will auto-download from Ultralytics if not present)\n")
    model = YOLO(args.model)

    # ── Start training ──
    print(f"{Fore.CYAN}[INFO] Starting training...{Style.RESET_ALL}\n")

    results = model.train(
        data        = args.data,
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        lr0         = args.lr0,
        lrf         = args.lrf,
        device      = args.device if args.device else None,
        project     = args.project,
        name        = args.name,
        patience    = args.patience,
        resume      = args.resume,
        augment     = args.augment,

        # ── Augmentation settings (helps with varied lighting) ──
        hsv_h       = 0.015,   # Hue augmentation (lighting variation)
        hsv_s       = 0.7,     # Saturation augmentation
        hsv_v       = 0.4,     # Value (brightness) augmentation — key for streetlights
        flipud      = 0.0,     # No vertical flip (streetlights are upright)
        fliplr      = 0.5,     # Horizontal flip (OK for streets)
        mosaic      = 1.0,     # Mosaic augmentation
        mixup       = 0.1,     # MixUp augmentation
        degrees     = 5.0,     # Small rotation augmentation

        # ── Other settings ──
        save        = True,
        save_period = 10,      # Save checkpoint every 10 epochs
        plots       = True,    # Save training plots (loss, mAP curves)
        verbose     = True,
        seed        = 42,
        workers     = 4,
    )

    # ── Post-training summary ──
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights = Path(args.project) / args.name / "weights" / "last.pt"

    print("\n" + "=" * 65)
    print(f"{Fore.GREEN}  ✅  TRAINING COMPLETE!{Style.RESET_ALL}")
    print("=" * 65)

    if best_weights.exists():
        print(f"  Best weights : {best_weights}")
        # Copy best weights to models/ root for easy access
        import shutil
        shutil.copy2(best_weights, "models/best.pt")
        print(f"  Copied to    : models/best.pt")

    if last_weights.exists():
        print(f"  Last weights : {last_weights}")

    print(f"\n  To run inference:")
    print(f"    python detect_image.py --source data/images/test/ --weights models/best.pt")
    print(f"    python detect_video.py --source 0 --weights models/best.pt")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
