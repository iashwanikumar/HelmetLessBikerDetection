<div align="center">

# 🪖 Helmet-less Motorcyclist Detection System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x%2F2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN%20Module-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Ready-yellow?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

<br/>

**A two-stage deep learning pipeline that detects motorcyclists and automatically flags helmet-use violations — designed for traffic surveillance and road safety enforcement.**

</div>

---

## 📌 Overview

This system integrates two state-of-the-art object detection models in a sequential pipeline:

- A **Faster R-CNN** model locates and crops bike riders from a frame or video stream
- A custom-trained **YOLOv3** model then classifies each cropped rider as `Helmet` or `No Helmet`
- Detected violations are **automatically logged** as frame snapshots into a dedicated output directory

---

## 🚀 System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT FRAME / VIDEO                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1 — Motorcyclist Detection                    │
│                    [ Faster R-CNN Model ]                        │
│           Detects and crops individual bike riders               │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Cropped rider image
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2 — Helmet Classification                     │
│               [ Custom-Trained YOLOv3 Model ]                   │
│         Classifies rider as "Helmet" or "No Helmet"             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          ✅ HELMET OK              ❌ NO HELMET — VIOLATION
                                  Frame snapshot exported
                                   to /output/ directory
```

| Step | Action | Model Used |
|------|--------|-----------|
| 1 | Detect and crop motorcyclists from frame | Faster R-CNN |
| 2 | Classify each crop for helmet compliance | YOLOv3 (custom) |
| 3 | Export violation snapshots automatically | — |

---

## 🛠️ Tech Stack & Requirements

| Component | Details |
|-----------|---------|
| **Runtime Environment** | Google Colab *(GPU Accelerator Recommended)* |
| **Detection Model 1** | Faster R-CNN via TensorFlow Object Detection API |
| **Detection Model 2** | YOLOv3 (custom-trained) via OpenCV DNN Module |
| **Core Frameworks** | TensorFlow v1.x / v2.x (compatibility layers), OpenCV |
| **Supporting Libraries** | NumPy, `tf_slim`, Protobuf Compiler (`protoc`) |

---

## 📂 Project Directory Structure

Ensure your Google Drive folder — `/MyDrive/HelmetDetection/` — matches this layout **before** running the notebook:

```
HelmetDetection/
│
├── rcnn/
│   ├── frozen_inference_graph.pb     # Pre-trained Faster R-CNN weights
│   └── label_map.pbtxt               # Label mappings for R-CNN
│
├── yolo/
│   ├── yolov3_custom.cfg             # YOLOv3 network configuration
│   ├── yolov3_custom_4000.weights    # Custom-trained YOLOv3 weights
│   └── obj.names                     # Class targets: 'Helmet', 'No Helmet'
│
├── input/
│   ├── images/                       # Static input images
│   └── videos/                       # Video samples for processing
│
└── output/                           # ⚠️ Violation snapshots logged here
```

---

## 📖 Step-by-Step Execution Guide

> **Platform:** Google Colab &nbsp;|&nbsp; **Recommended:** GPU Runtime

---

### Step 1 — Open and Initialize the Notebook

- Copy the project code into a clean **Google Colab Notebook**
- Navigate to **Runtime → Change runtime type** and set the Hardware Accelerator to **GPU**

---

### Step 2 — Mount Google Drive

Run the initialization cells to authenticate and mount your personal storage:

```python
from google.colab import drive
drive.mount('/content/drive')
```

> ⚠️ Ensure your models, weights, and config files are uploaded under `/MyDrive/HelmetDetection/` before proceeding.

---

### Step 3 — Compile Dependencies & Protobufs

Run the environment setup cells to:
- Clone the **TensorFlow Models repository**
- Execute the `protoc` compiler

This registers the **Object Detection API** utilities required for image processing.

---

### Step 4 — Configure Data Processing Paths

Before running detection cells, update the input file paths in the configuration block:

```python
# Update these strings with your exact filenames before processing
IMAGE_NAME = 'input/images/your_sample_photo.jpeg'
VIDEO_NAME = 'input/videos/your_traffic_clip.mp4'
```

---

### Step 5 — Execute and Retrieve Logs

Run all pipeline cells **sequentially**. Output behavior:

| Input Type | Output |
|-----------|--------|
| **Static Images** | Processed frame displayed inline in real-time |
| **Video Streams** | Annotated `.mp4` + cropped violation images saved to `/output/` in Google Drive |

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

© 2026 Helmet Detection Project &nbsp;|&nbsp; Built for Road Safety 🛣️

</div>
