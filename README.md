# Face Recognition with Python & OpenCV

A Python-based face detection and recognition project that explores multiple image processing techniques — including Haar cascades, Local Binary Pattern (LBP), and Zero Mean Standardization — to identify and recognize faces in real time via webcam.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)

---

## Overview

This project implements real-time face detection and recognition using OpenCV. It serves as a practical exploration of classic computer vision techniques:

- **Haar Cascade** — Fast, pre-trained face detection
- **Local Binary Pattern (LBP)** — Texture-based feature extraction for recognition
- **Zero Mean Standardization** — Preprocessing to normalize lighting conditions

The project includes standalone demo scripts as well as a complete face recognition pipeline that trains on a custom dataset captured from a webcam.

---

## Features

- Real-time face detection from webcam using Haar cascades
- Visualization of LBP texture features on detected faces
- Face preprocessing with Zero Mean Standardization
- Full face recognition pipeline:
  - Capture training images per person
  - Train an LBPH (Local Binary Patterns Histograms) model
  - Recognize faces live with name and confidence score
- Folder-based known-faces recognition using custom LBP + Chi-Squared histogram comparison

---

## Architecture

```
Webcam Input
     │
     ▼
Grayscale Conversion
     │
     ▼
Haar Cascade Face Detection
     │
     ▼
Face Region of Interest (ROI)
     │
     ├──► Basic Detection (cc_default.py)
     │
     ├──► Zero Mean Standardization (szm-default.py)
     │
     ├──► LBP Visualization (lbp-default.py)
     │
     └──► Face Recognition
               ├── LBPH Model (face_recognize/)
               │       Training: create_data.py → datasets/
               │       Inference: face_recognize.py
               │
               └── Custom LBP + Histogram Matching (Local Binary Patern/)
                       Training data: known_faces/<person_name>/
                       Inference: local_binary_pattern.py
```

Each approach is independently runnable and demonstrates a different stage or technique in the face recognition pipeline.

---

## Folder Structure

```
face_recognition/
│
├── cc_default.py                          # Basic Haar cascade face detection demo
├── lbp-default.py                         # Face detection with LBP texture visualization
├── szm-default.py                         # Face detection with Zero Mean Standardization
│
├── face_recognize/                        # Full LBPH face recognition pipeline
│   ├── create_data.py                     # Captures training face images via webcam
│   ├── face_recognize.py                  # Trains LBPH model and recognizes faces live
│   ├── haarcascade_frontalface_default.xml # Haar cascade model file
│   └── datasets/                          # Captured training images (one subfolder per person)
│       ├── Ica/
│       ├── Rey/
│       └── Sye/
│
└── Local Binary Patern/                   # Custom LBP face recognition pipeline
    ├── local_binary_pattern.py            # Recognizes faces using LBP histograms
    └── known_faces/                       # Reference images (one subfolder per person)
        └── <person_name>/
            └── *.jpg / *.png
```

---

## Requirements

- Python 3.7+
- [OpenCV](https://opencv.org/) with contrib modules
- NumPy

Install dependencies with pip:

```bash
pip install opencv-python opencv-contrib-python numpy
```

> **Note:** `opencv-contrib-python` is required for `cv2.face.LBPHFaceRecognizer_create()` used in the `face_recognize/` pipeline.

---

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/reyimanuel/face_recognition.git
   cd face_recognition
   ```

2. **Install dependencies:**

   ```bash
   pip install opencv-python opencv-contrib-python numpy
   ```

3. **Ensure your webcam is connected** (device index `0` is used by default in all scripts).

---

## Usage

### 1. Basic Face Detection

Detects faces in real time using a Haar cascade classifier.

```bash
python cc_default.py
```

Press **`q`** to quit.

---

### 2. Face Detection with LBP Visualization

Detects faces and displays the Local Binary Pattern texture of each detected face region.

```bash
python lbp-default.py
```

Press **`q`** to quit.

---

### 3. Face Detection with Zero Mean Standardization

Detects faces and normalizes each face region using zero mean standardization before display.

```bash
python szm-default.py
```

Press **`q`** to quit.

---

### 4. Full LBPH Face Recognition Pipeline

This two-step pipeline trains a model on captured images and then recognizes faces live.

**Step 1 — Capture training data:**

Edit `sub_data` in `create_data.py` to set the person's name, then run:

```bash
cd face_recognize
python create_data.py
```

The script captures 30 face images and saves them to `datasets/<name>/`. Press **`Esc`** to stop early.

**Step 2 — Train and recognize:**

```bash
python face_recognize.py
```

Detected faces are labeled with the recognized name and a confidence score. A score below `500` is considered a match. Press **`Esc`** to quit.

---

### 5. Custom LBP Histogram Face Recognition

Recognizes faces by comparing LBP histograms against a library of known faces.

**Prepare known faces:**

Create a directory structure under `Local Binary Patern/known_faces/`:

```
known_faces/
└── Alice/
    ├── photo1.jpg
    └── photo2.jpg
└── Bob/
    └── photo1.png
```

**Run recognition:**

```bash
cd "Local Binary Patern"
python local_binary_pattern.py
```

Faces are labeled with the matched name if the Chi-Squared distance is below `0.5`, otherwise labeled `Unknown`. Press **`q`** to quit.

---

## How It Works

### Haar Cascade Detection

OpenCV's pre-trained `haarcascade_frontalface_default.xml` model uses Haar-like features and a cascade of classifiers to rapidly detect frontal faces in grayscale images.

### Local Binary Pattern (LBP)

For each pixel, the 8 surrounding neighbors are compared to the center pixel. The result is encoded as an 8-bit binary number, producing a texture descriptor that is robust to monotonic illumination changes.

```
Neighbors > center → bit = 1, else bit = 0
LBP code = binary encoding of all 8 comparisons
```

Histograms of LBP codes are compared using **Chi-Squared distance** for face matching.

### LBPH Face Recognizer (`cv2.face.LBPHFaceRecognizer`)

OpenCV's built-in LBPH recognizer divides the face into a grid of cells, computes LBP histograms per cell, and concatenates them. The model is trained on labeled image sets and returns a prediction with a confidence score during inference.

### Zero Mean Standardization

Normalizes each face region by subtracting the mean pixel value and dividing by the standard deviation, reducing the effect of uneven lighting across captures.

```
face_standardized = (face - mean(face)) / std(face)
```

---

## Results

**Face detection without preprocessing:**

![Basic Face Detection](https://github.com/reyimanuel/face_recognition/assets/110801278/6ea2c93c-d47d-4968-9e33-53879c2b8f62)

**Face detection with Zero Mean Standardization:**

![Zero Mean Standardization](https://github.com/reyimanuel/face_recognition/assets/110801278/a9bc568b-d3da-4770-b176-8907368568ac)

**Face detection with Local Binary Pattern:**

![LBP Face Detection](https://github.com/reyimanuel/face_recognition/assets/110801278/b574f97a-be53-4833-8a5b-23d6e79dcb21)

**Face recognition from folder dataset:**

![Dataset-based Face Recognition](https://github.com/reyimanuel/face_recognition/assets/110801278/2c33e8f9-691b-4d18-a5a9-9a2f42d10a02)
