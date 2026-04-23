# 🚦 Computer Vision-Based Detection of Streetlight Outages in Urban Areas
### B.Tech Final Year Project | Deep Learning + Computer Vision

---

## 📁 Project Structure

```
streetlight_detection/
├── data/
│   ├── images/
│   │   ├── train/          # Training images
│   │   ├── val/            # Validation images
│   │   └── test/           # Test images
│   ├── labels/
│   │   ├── train/          # YOLO format labels for training
│   │   ├── val/            # YOLO format labels for validation
│   │   └── test/           # YOLO format labels for testing
│   └── raw/                # Raw collected images (before split)
├── models/                 # Saved trained model weights
├── outputs/
│   ├── faulty_frames/      # Auto-saved frames where outage detected
│   └── logs/               # Training and inference logs
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # Image enhancement & low-light handling
│   ├── metrics.py          # Precision, Recall, Accuracy computation
│   └── alert.py            # Alert generation system
├── configs/
│   └── streetlight.yaml    # YOLOv8 dataset configuration
├── scripts/
│   └── split_dataset.py    # Train/val/test split utility
├── train.py                # Model training script
├── detect_image.py         # Inference on single/batch images
├── detect_video.py         # Real-time detection from video/webcam
├── evaluate.py             # Performance evaluation script
├── requirements.txt        # All dependencies
└── README.md               # This file
```

---

## ⚙️ Installation

```bash
# 1. Clone / download the project folder
cd streetlight_detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

### Option A — Collect Your Own Dataset
1. Capture night-time street images using a camera/CCTV/smartphone.
2. Ensure both ON (functional) and OFF (non-functional) streetlights are present.
3. Place all raw images inside `data/raw/`.
4. Annotate using **LabelImg** or **Roboflow** (see annotation guide below).
5. Run the dataset split script:
   ```bash
   python scripts/split_dataset.py
   ```

### Option B — Use a Public Dataset
- **Roboflow Universe**: https://universe.roboflow.com  
  Search "streetlight detection" and export in YOLOv8 format.
- **Open Images Dataset**: Filter for street/night images.
- After download, place images in `data/images/` and labels in `data/labels/`.

### 🏷️ Annotation Guide (LabelImg)
```bash
pip install labelImg
labelImg
```
- Open image folder → draw bounding boxes around each streetlight.
- Assign class:
  - `0` → `functional`  (light is ON / glowing)
  - `1` → `non_functional` (light is OFF / dark)
- Save annotations in **YOLO format** (.txt files).

### YOLO Label Format (each .txt file):
```
<class_id> <x_center> <y_center> <width> <height>
```
Example:
```
0 0.512 0.423 0.045 0.120   ← functional streetlight
1 0.231 0.387 0.042 0.115   ← non-functional streetlight
```

---

## 🚀 Training the Model

```bash
python train.py --epochs 50 --batch 16 --imgsz 640
```

Options:
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Input image size |
| `--model` | yolov8n.pt | Base YOLOv8 model (n/s/m/l/x) |
| `--device` | auto | cuda / cpu |

---

## 🖼️ Inference on Images

```bash
# Single image
python detect_image.py --source data/images/test/street1.jpg

# Folder of images
python detect_image.py --source data/images/test/

# Specify custom model weights
python detect_image.py --source data/images/test/ --weights models/best.pt
```

---

## 🎥 Real-Time Video / Webcam Detection

```bash
# Webcam (default camera, index 0)
python detect_video.py --source 0

# Video file
python detect_video.py --source path/to/video.mp4

# CCTV stream (RTSP)
python detect_video.py --source rtsp://username:password@ip:port/stream
```

---

## 📊 Evaluate Model Performance

```bash
python evaluate.py --weights models/best.pt
```

Outputs: Precision, Recall, F1-Score, mAP@0.5

---

## 👁️ Sample Output

```
[ALERT] ⚠️  NON-FUNCTIONAL STREETLIGHT DETECTED at frame 142 | Confidence: 0.91
[ALERT] ⚠️  NON-FUNCTIONAL STREETLIGHT DETECTED at frame 143 | Confidence: 0.88
[INFO]  Frame saved → outputs/faulty_frames/frame_0142.jpg
```

---

## 📌 Notes for B.Tech Project

- Minimum recommended dataset: **500+ images** (balanced classes)
- For GPU training: install `torch` with CUDA support
- YOLOv8n (nano) works well on CPU; use YOLOv8s/m for better accuracy
- Low-light enhancement is applied automatically during inference
