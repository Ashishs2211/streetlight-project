"""
scripts/split_dataset.py
========================
Splits the raw dataset (data/raw/) into train / val / test sets
and organizes images + labels into the YOLO folder structure.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train 0.7 --val 0.2 --test 0.1
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def split_dataset(raw_dir: str,
                  output_dir: str,
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.20,
                  test_ratio:  float = 0.10,
                  seed: int = 42):
    """
    Split images and labels from raw_dir into train/val/test.

    Expected raw_dir structure:
        data/raw/
            images/
                img001.jpg
                img002.jpg
                ...
            labels/
                img001.txt
                img002.txt
                ...

    Args:
        raw_dir     : Path to raw dataset
        output_dir  : Path to data/ directory
        train_ratio : Fraction for training
        val_ratio   : Fraction for validation
        test_ratio  : Fraction for testing
        seed        : Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    raw_images = Path(raw_dir) / "images"
    raw_labels = Path(raw_dir) / "labels"

    # Collect all image files
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [
        f for f in raw_images.iterdir()
        if f.suffix.lower() in supported
    ]

    if not image_files:
        print(f"[ERROR] No images found in {raw_images}")
        print("        Place your images in data/raw/images/")
        return

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # Remaining goes to test
    splits = {
        "train": image_files[:n_train],
        "val":   image_files[n_train : n_train + n_val],
        "test":  image_files[n_train + n_val:]
    }

    print(f"\n[INFO] Total images found : {n}")
    print(f"[INFO] Train : {len(splits['train'])}")
    print(f"[INFO] Val   : {len(splits['val'])}")
    print(f"[INFO] Test  : {len(splits['test'])}")
    print()

    output = Path(output_dir)

    for split_name, files in splits.items():
        img_out = output / "images" / split_name
        lbl_out = output / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Copying {split_name} set...")
        for img_path in tqdm(files, desc=f"  {split_name:5s}"):
            # Copy image
            shutil.copy2(img_path, img_out / img_path.name)

            # Copy corresponding label (if exists)
            lbl_path = raw_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            else:
                # Create empty label file for images with no annotations
                (lbl_out / (img_path.stem + ".txt")).touch()

    print("\n[✔] Dataset split complete!")
    print(f"    Images → {output / 'images'}")
    print(f"    Labels → {output / 'labels'}")


def main():
    parser = argparse.ArgumentParser(
        description="Split raw dataset into train/val/test for YOLOv8"
    )
    parser.add_argument("--raw_dir",    default="data/raw",
                        help="Path to raw dataset directory")
    parser.add_argument("--output_dir", default="data",
                        help="Output data directory (default: data/)")
    parser.add_argument("--train", type=float, default=0.70,
                        help="Train split ratio (default: 0.70)")
    parser.add_argument("--val",   type=float, default=0.20,
                        help="Validation split ratio (default: 0.20)")
    parser.add_argument("--test",  type=float, default=0.10,
                        help="Test split ratio (default: 0.10)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
