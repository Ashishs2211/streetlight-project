"""
detect_video.py
===============
Real-time streetlight outage detection from:
  - Webcam (live feed)
  - Video file (.mp4, .avi, etc.)
  - CCTV/RTSP stream

Features:
  - Low-light enhancement on each frame
  - Bounding boxes with class labels + confidence
  - Auto-saving of faulty frames
  - On-screen alert overlay
  - FPS counter
  - Press 'q' to quit, 's' to take screenshot

Usage:
    python detect_video.py --source 0                         # Webcam
    python detect_video.py --source footage/night_street.mp4  # Video file
    python detect_video.py --source rtsp://ip:port/stream      # CCTV
"""

import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from colorama import Fore, Style, init

from utils.preprocessing import preprocess_for_inference, is_low_light
from utils.alert import alert_outage, alert_functional, alert_summary

init(autoreset=True)

# ─────────────────────────────────────────────
# Class Configuration
# ─────────────────────────────────────────────
CLASS_NAMES  = {0: "Functional", 1: "Non-Functional"}
CLASS_COLORS = {0: (0, 210, 0), 1: (0, 0, 230)}     # Green / Red in BGR
LABEL_BG     = {0: (0, 160, 0), 1: (0, 0, 180)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streetlight Outage Detection — Real-Time Video"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0=webcam, path to video file, or RTSP URL"
    )
    parser.add_argument(
        "--weights", type=str, default="models/best.pt",
        help="Path to trained YOLOv8 weights (default: models/best.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="Confidence threshold (default: 0.40)"
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
        help="Directory to save faulty frames (default: outputs/)"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Save annotated output as a video file"
    )
    parser.add_argument(
        "--no_enhance", action="store_true",
        help="Disable automatic low-light enhancement"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=1,
        help="Process every Nth frame (1=all, 2=every other, etc.)"
    )
    parser.add_argument(
        "--alert_cooldown", type=int, default=30,
        help="Min frames between repeated alerts for same outage (default: 30)"
    )
    return parser.parse_args()


def draw_frame(frame: np.ndarray,
               results,
               conf_threshold: float,
               fps: float,
               frame_id: int,
               enhanced: bool) -> tuple:
    """
    Annotate a video frame with detection results.

    Returns:
        (annotated_frame, functional_count, non_functional_count,
         outage_detected, max_outage_confidence, outage_bbox)
    """
    annotated = frame.copy()
    func_count = 0
    nonfunc_count = 0
    outage_detected = False
    max_conf = 0.0
    outage_bbox = None

    h, w = annotated.shape[:2]

    # ── Draw each detection ──
    for box in results.boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_id, (128, 128, 128))
        bg    = LABEL_BG.get(cls_id, (80, 80, 80))
        label = CLASS_NAMES.get(cls_id, "unknown")

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw corner accents for visual polish
        corner_len = 12
        corner_thick = 3
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy),            color, corner_thick)
            cv2.line(annotated, (cx, cy), (cx,            cy + dy * corner_len), color, corner_thick)

        # Label
        text = f"{label}  {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pad = 4
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - th - bl - pad * 2)),
            (x1 + tw + pad * 2, y1),
            bg, -1
        )
        cv2.putText(
            annotated, text,
            (x1 + pad, y1 - bl - pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        if cls_id == 0:
            func_count += 1
        else:
            nonfunc_count += 1
            outage_detected = True
            if conf > max_conf:
                max_conf   = conf
                outage_bbox = (x1, y1, x2, y2)

    # ── HUD — Top status bar ──
    bar_h = 30
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    hud_text = (
        f"FPS: {fps:.1f}  |  "
        f"Frame: {frame_id}  |  "
        f"{'[LOW-LIGHT ENHANCED]  ' if enhanced else ''}"
        f"✅ Functional: {func_count}   ⚠️ Non-Functional: {nonfunc_count}"
    )
    cv2.putText(
        annotated, hud_text, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (220, 220, 220), 1, cv2.LINE_AA
    )

    # ── ALERT overlay when outage detected ──
    if outage_detected:
        pulse = int(abs(np.sin(frame_id * 0.15)) * 100) + 100
        alert_color = (0, 0, pulse + 100)

        # Bottom alert banner
        banner_h = 36
        overlay2 = annotated.copy()
        cv2.rectangle(overlay2, (0, h - banner_h), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay2, 0.8, annotated, 0.2, 0, annotated)

        alert_text = f"⚠  STREETLIGHT OUTAGE DETECTED  |  Confidence: {max_conf:.2f}"
        (atw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ax = (w - atw) // 2

        cv2.putText(
            annotated, alert_text,
            (ax, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    return annotated, func_count, nonfunc_count, outage_detected, max_conf, outage_bbox


def main():
    args = parse_args()

    # ── Validate weights ──
    if not os.path.exists(args.weights):
        print(f"{Fore.RED}[ERROR] Weights not found: {args.weights}{Style.RESET_ALL}")
        print("        Train first: python train.py")
        return

    # ── Setup output directories ──
    faulty_dir = Path(args.save_dir) / "faulty_frames"
    faulty_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"\n{Fore.CYAN}[INFO] Loading YOLOv8 model: {args.weights}{Style.RESET_ALL}")
    model = YOLO(args.weights)
    print(f"[INFO] Model loaded ✔")

    # ── Open video source ──
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"{Fore.RED}[ERROR] Cannot open video source: {args.source}{Style.RESET_ALL}")
        print("        Check your webcam index or video file path.")
        return

    # ── Get video properties ──
    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_name = str(args.source)

    print(f"[INFO] Source       : {src_name}")
    print(f"[INFO] Resolution   : {width}x{height}")
    print(f"[INFO] Input FPS    : {fps_in:.1f}")
    if total_fr > 0:
        print(f"[INFO] Total Frames : {total_fr}")
    print(f"\n{Fore.GREEN}[INFO] Detection started. Press 'q' to quit, 's' for screenshot.{Style.RESET_ALL}\n")

    # ── Video writer (optional) ──
    video_writer = None
    if args.save_video:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_out_path = Path(args.save_dir) / f"output_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_out_path), fourcc, fps_in, (width, height)
        )
        print(f"[INFO] Saving video → {video_out_path}")

    # ── Tracking variables ──
    frame_id         = 0
    func_total       = 0
    nonfunc_total    = 0
    last_alert_frame = -999    # For cooldown logic
    fps_display      = 0.0
    prev_time        = time.time()

    # ── Main detection loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n[INFO] Stream ended or video finished.")
            break

        frame_id += 1

        # Skip frames for performance (configurable)
        if frame_id % args.skip_frames != 0:
            continue

        # ── FPS calculation ──
        curr_time = time.time()
        fps_display = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # ── Low-light enhancement ──
        enhanced = False
        if not args.no_enhance and is_low_light(frame):
            frame = preprocess_for_inference(frame, auto_enhance=True)
            enhanced = True

        # ── YOLO inference ──
        results = model.predict(
            source  = frame,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            verbose = False
        )[0]

        # ── Annotate frame ──
        annotated, fc, nfc, outage, max_conf, bbox = draw_frame(
            frame, results, args.conf, fps_display, frame_id, enhanced
        )

        func_total    += fc
        nonfunc_total += nfc

        # ── Alert logic (with cooldown) ──
        if outage and (frame_id - last_alert_frame) >= args.alert_cooldown:
            # Save faulty frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            save_path = faulty_dir / f"frame_{frame_id:06d}_{ts}.jpg"
            cv2.imwrite(str(save_path), annotated)

            alert_outage(
                source     = src_name,
                frame_id   = f"frame_{frame_id:06d}",
                confidence = max_conf,
                bbox       = bbox,
                saved_path = str(save_path)
            )
            last_alert_frame = frame_id

        # ── Display ──
        cv2.imshow("Streetlight Outage Detection  |  q=quit  s=screenshot", annotated)

        # ── Write to video output ──
        if video_writer:
            video_writer.write(annotated)

        # ── Keyboard controls ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] Quit key pressed.")
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ss_path = Path(args.save_dir) / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(ss_path), annotated)
            print(f"{Fore.GREEN}[INFO] Screenshot saved → {ss_path}{Style.RESET_ALL}")

    # ── Cleanup ──
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # ── Session summary ──
    alert_summary(
        total_frames        = frame_id,
        functional_count    = func_total,
        non_functional_count = nonfunc_total,
        source              = src_name
    )

    print(f"{Fore.GREEN}[INFO] Faulty frames saved → {faulty_dir}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
