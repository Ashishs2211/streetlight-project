"""
utils/alert.py
==============
Alert generation system for streetlight outage detection.
Prints formatted alerts to terminal and logs them to file.
Can be extended to send SMS/email notifications.
"""

import os
import csv
import logging
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# ─────────────────────────────────────────────
# Logger Setup
# ─────────────────────────────────────────────
os.makedirs("outputs/logs", exist_ok=True)

logging.basicConfig(
    filename="outputs/logs/detections.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("StreetlightAlert")


# ─────────────────────────────────────────────
# CSV Log File Setup
# ─────────────────────────────────────────────
CSV_LOG_PATH = "outputs/logs/alert_log.csv"

def _init_csv_log():
    """Initialize CSV log file with header if it doesn't exist."""
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Source", "Frame/Image",
                "Class", "Confidence", "BBox", "Saved_Path"
            ])

_init_csv_log()


# ─────────────────────────────────────────────
# Core Alert Functions
# ─────────────────────────────────────────────
def alert_outage(source: str,
                 frame_id,
                 confidence: float,
                 bbox: tuple = None,
                 saved_path: str = None):
    """
    Fire an alert when a non-functional streetlight is detected.

    Args:
        source     : Source name (filename or 'webcam')
        frame_id   : Frame number or image name
        confidence : Detection confidence score (0.0–1.0)
        bbox       : Bounding box tuple (x1, y1, x2, y2)
        saved_path : Path where faulty frame was saved
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bbox_str = f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})" if bbox else "N/A"

    # ── Terminal alert (colored) ──
    print(
        f"{Fore.RED}[ALERT] ⚠️  NON-FUNCTIONAL STREETLIGHT DETECTED{Style.RESET_ALL} | "
        f"Source: {Fore.YELLOW}{source}{Style.RESET_ALL} | "
        f"Frame: {Fore.CYAN}{frame_id}{Style.RESET_ALL} | "
        f"Confidence: {Fore.MAGENTA}{confidence:.2f}{Style.RESET_ALL}"
    )

    if saved_path:
        print(f"{Fore.GREEN}[SAVED] Frame saved → {saved_path}{Style.RESET_ALL}")

    # ── File log ──
    logger.warning(
        f"OUTAGE DETECTED | Source={source} | Frame={frame_id} | "
        f"Conf={confidence:.2f} | BBox={bbox_str} | Saved={saved_path}"
    )

    # ── CSV log ──
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, source, frame_id,
            "non_functional", f"{confidence:.4f}",
            bbox_str, saved_path or ""
        ])


def alert_functional(source: str, frame_id, confidence: float):
    """
    Log a functional streetlight detection (info level only).

    Args:
        source     : Source name
        frame_id   : Frame or image identifier
        confidence : Detection confidence score
    """
    print(
        f"{Fore.GREEN}[OK] ✅  Functional Streetlight{Style.RESET_ALL} | "
        f"Frame: {frame_id} | Conf: {confidence:.2f}"
    )
    logger.info(
        f"FUNCTIONAL | Source={source} | Frame={frame_id} | Conf={confidence:.2f}"
    )


def alert_summary(total_frames: int,
                  functional_count: int,
                  non_functional_count: int,
                  source: str):
    """
    Print and log a session summary after processing completes.

    Args:
        total_frames        : Total frames/images processed
        functional_count    : Number of functional detections
        non_functional_count: Number of non-functional detections
        source              : Input source
    """
    print("\n" + "=" * 60)
    print(f"{Fore.CYAN}  📊  DETECTION SESSION SUMMARY{Style.RESET_ALL}")
    print("=" * 60)
    print(f"  Source         : {source}")
    print(f"  Total Processed: {total_frames}")
    print(f"  ✅  Functional  : {Fore.GREEN}{functional_count}{Style.RESET_ALL}")
    print(f"  ⚠️   Non-Functional: {Fore.RED}{non_functional_count}{Style.RESET_ALL}")
    if total_frames > 0:
        outage_rate = (non_functional_count / total_frames) * 100
        print(f"  Outage Rate    : {Fore.YELLOW}{outage_rate:.1f}%{Style.RESET_ALL}")
    print(f"  Log saved at   : {CSV_LOG_PATH}")
    print("=" * 60 + "\n")

    logger.info(
        f"SESSION SUMMARY | Source={source} | Total={total_frames} | "
        f"Functional={functional_count} | NonFunctional={non_functional_count}"
    )
