"""
detect_image.py
===============
Run YOLOv8 inference on a single image or a folder of images.
Applies low-light enhancement automatically for dark images.
Saves annotated results and faulty frame crops.

Usage:
    python detect_image.py --source data/images/test/street1.jpg
    python detect_image.py --source data/images/test/
    python detect_image.py --source data/images/test/ --weights models/best.pt --conf 0.4
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from colorama import Fore, Style, init

# Local utilities
from utils.preprocessing import preprocess_for_inference, is_low_light
from utils.alert import alert_outage, alert_functional, alert_summary

init(autoreset=True)

# ─────────────────────────────────────────────
# Class Configuration
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "non_functional", 1: "functional"}

# Red = faulty, Green = working
CLASS_COLORS = {
    0: (0, 0, 220),      # non-functional
    1: (0, 200, 0),      # functional
}

LABEL_BG = {
    0: (0, 0, 200),
    1: (0, 180, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streetlight Outage Detection — Image Inference"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image file or folder of images"
    )
    parser.add_argument(
        "--weights", type=str, default="models/best.pt",
        help="Path to trained YOLOv8 weights (default: models/best.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold for detections (default: 0.35)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs",
        help="Directory to save annotated images (default: outputs/)"
    )
    parser.add_argument(
        "--no_enhance", action="store_true",
        help="Disable automatic low-light enhancement"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated image in a window"
    )
    return parser.parse_args()


def draw_detections(image: np.ndarray,
                    detections,
                    conf_threshold: float) -> tuple:
    """
    Draw bounding boxes, labels, and confidence scores on the image.

    Args:
        image          : BGR image
        detections     : YOLO results object
        conf_threshold : Minimum confidence to display

    Returns:
        (annotated_image, functional_count, non_functional_count)
    """
    annotated = image.copy()
    functional_count    = 0
    non_functional_count = 0

    # Iterate over all detections in the result
    for box in detections.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        label  = CLASS_NAMES.get(cls_id, "unknown")
        color  = CLASS_COLORS.get(cls_id, (128, 128, 128))
        bg     = LABEL_BG.get(cls_id, (100, 100, 100))

        # Bounding box coordinates (pixel)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── Draw bounding box ──
        thickness = 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # ── Label text ──
        icon = "✅" if cls_id == 1 else "⚠️"
        text  = f"{label.upper()}  {conf:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )

        # Label background pill
        pad = 4
        cv2.rectangle(
            annotated,
            (x1 - 1, y1 - th - baseline - pad * 2),
            (x1 + tw + pad * 2, y1),
            bg, -1
        )

        # Label text
        cv2.putText(
            annotated, text,
            (x1 + pad, y1 - baseline - pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        if cls_id == 1:
            functional_count += 1
        else:
            non_functional_count += 1

    # ── Status bar at top of image ──
    bar_text = (
        f"  Functional: {functional_count}  |  "
        f"Non-Functional: {non_functional_count}  |  "
        f"Conf ≥ {conf_threshold:.2f}"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(
        annotated, bar_text, (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (200, 200, 200), 1, cv2.LINE_AA
    )

    return annotated, functional_count, non_functional_count


def process_image(image_path: Path,
                  model: YOLO,
                  args,
                  save_dir: Path,
                  faulty_dir: Path) -> tuple:
    """
    Run full inference pipeline on a single image.

    Returns:
        (functional_count, non_functional_count)
    """
    # ── Load image ──
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"{Fore.YELLOW}[WARN] Could not read: {image_path}{Style.RESET_ALL}")
        return 0, 0

    img_name = image_path.name
    print(f"\n[INFO] Processing: {img_name}", end="")

    # ── Low-light enhancement ──
    if not args.no_enhance:
        processed = preprocess_for_inference(image, auto_enhance=True)
        was_enhanced = is_low_light(image)
        if was_enhanced:
            print(f" {Fore.YELLOW}[low-light enhanced]{Style.RESET_ALL}", end="")
    else:
        processed = image

    print()

    # ── YOLO Inference ──
    results = model.predict(
        source    = processed,
        conf      = args.conf,
        iou       = args.iou,
        imgsz     = args.imgsz,
        verbose   = False
    )[0]

    # ── Draw detections ──
    annotated, func_count, nonfunc_count = draw_detections(
        processed, results, args.conf
    )

    # ── Save annotated image ──
    out_path = save_dir / f"detected_{img_name}"
    cv2.imwrite(str(out_path), annotated)

    # ── Handle non-functional detections ──
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id == 0 and conf >= args.conf:
            # Save faulty frame
            faulty_path = faulty_dir / f"FAULTY_{img_name}"
            cv2.imwrite(str(faulty_path), annotated)

            bbox = tuple(map(int, box.xyxy[0]))
            alert_outage(
                source     = str(image_path),
                frame_id   = img_name,
                confidence = conf,
                bbox       = bbox,
                saved_path = str(faulty_path)
            )

    # Print functional detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id == 1 and conf >= args.conf:
            alert_functional(str(image_path), img_name, conf)

    # ── Show image window (optional) ──
    if args.show:
        cv2.imshow(f"Streetlight Detection — {img_name}", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"FAULT_COUNT={nonfunc_count}")

    return func_count, nonfunc_count


def main():
    args = parse_args()

    # ── Check weights ──
    if not os.path.exists(args.weights):
        print(f"{Fore.RED}[ERROR] Weights not found: {args.weights}{Style.RESET_ALL}")
        print("        Train first with: python train.py")
        print("        Or specify path:  --weights path/to/best.pt")
        return

    # ── Setup output directories ──
    save_dir  = Path(args.save_dir) / "annotated_images"
    faulty_dir = Path(args.save_dir) / "faulty_frames"
    save_dir.mkdir(parents=True, exist_ok=True)
    faulty_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"\n{Fore.CYAN}[INFO] Loading model: {args.weights}{Style.RESET_ALL}")
    model = YOLO(args.weights)

    # ── Collect image files ──
    source = Path(args.source)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if source.is_file() and source.suffix.lower() in supported:
        image_files = [source]
    elif source.is_dir():
        image_files = [
            f for f in source.iterdir()
            if f.suffix.lower() in supported
        ]
        if not image_files:
            print(f"{Fore.RED}[ERROR] No images found in {source}{Style.RESET_ALL}")
            return
    else:
        print(f"{Fore.RED}[ERROR] Invalid source: {source}{Style.RESET_ALL}")
        return

    print(f"[INFO] Found {len(image_files)} image(s) to process")
    print(f"[INFO] Confidence threshold : {args.conf}")
    print(f"[INFO] Output dir           : {save_dir}")

    # ── Process all images ──
    total_functional    = 0
    total_non_functional = 0

    for img_path in image_files:
        f, nf = process_image(img_path, model, args, save_dir, faulty_dir)
        total_functional     += f
        total_non_functional += nf

    # ── Summary ──
    alert_summary(
        total_frames        = len(image_files),
        functional_count    = total_functional,
        non_functional_count = total_non_functional,
        source              = str(source)
    )

    print(f"{Fore.GREEN}[INFO] Annotated images saved → {save_dir}{Style.RESET_ALL}")


# ======================
# CAMERA CAPTURE DETECT
# ======================
@app.route("/capture_detect", methods=["POST"])
def capture_detect():

    if not logged_in():
        return redirect("/login")

    success, frame = camera.read()

    if not success:
        return "Camera Error"

    filename = "camera_capture.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    # Remove old outputs
    old = glob.glob("outputs/annotated_images/*")
    for f in old:
        try:
            os.remove(f)
        except:
            pass

    cmd = [
        "python",
        "detect_image.py",
        "--source",
        filepath,
        "--weights",
        MODEL_PATH
    ]

    subprocess.run(cmd)

    generated = glob.glob("outputs/annotated_images/*")

    if generated:
        latest = generated[0]
        shutil.copy(latest, os.path.join(RESULT_FOLDER, filename))

    return redirect("/camera_result")
if __name__ == "__main__":
    main()
