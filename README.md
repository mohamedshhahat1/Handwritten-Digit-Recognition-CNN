# 🧠 Handwritten Digit & Text Recognition — Deep Learning with PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-green?style=for-the-badge)](http://yann.lecun.com/exdb/mnist/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.36%25-brightgreen?style=for-the-badge)](saved_models/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A comprehensive deep learning project featuring **two recognition systems**:

1. **Digit Recognition (CNN)** — Classifies handwritten digits (0–9) with **99.36% accuracy** on MNIST
2. **Text Recognition (CRNN + CTC)** — Reads handwritten words and sentences in **English, Arabic, and digits**

Both models are served through a **FastAPI REST API** with an interactive **Web UI** for real-time predictions.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Branches](#-branches)
- [Installation](#-installation)
- [Usage](#-usage)
  - [CLI Interface](#cli-interface-mainpy)
  - [Training](#training)
  - [OCR Training](#ocr-training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
  - [FastAPI Server](#-fastapi-server)
  - [Web UI](#-web-ui)
  - [Docker](#-docker)
  - [Model Comparison](#-model-comparison-cnn-vs-mlp)
  - [PDF Report](#-generate-pdf-report)
  - [TensorBoard](#-tensorboard)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Configuration](#%EF%B8%8F-configuration)
- [API Reference](#-api-reference)
- [Technologies Used](#-technologies-used)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## ✨ Features

### Core ML
- 🏗️ **CNN** for digit classification (99.36% accuracy on MNIST)
- 🔤 **CRNN (CNN + BiLSTM + CTC)** for handwritten text/OCR recognition
- 🧠 **MLP baseline** for model comparison and educational purposes
- 📊 Automated CNN vs MLP benchmarking with visualization

### Training & Evaluation
- ⚡ Early stopping to prevent overfitting
- 🎨 Data augmentation (rotation, shift, scale, Gaussian noise)
- 📡 TensorBoard integration for real-time training visualization
- 💾 Model versioning with timestamped checkpoints (best + last N)
- 🧪 Centralized configuration system (`config.py`)
- 📈 Per-digit precision/recall/F1 classification reports
- 🔢 Confusion matrix generation and heatmap visualization

### Deployment & API
- 🚀 **FastAPI REST API** — file upload & base64 endpoints for both digit and OCR
- 🌐 **Web UI** — interactive drawing canvas with live predictions
- ✍️ **OCR Web UI** — handwriting pad for full text recognition
- 📦 **Docker & Docker Compose** for one-command deployment
- 🔄 CORS-enabled for cross-origin requests

### Tooling
- 📊 Auto-generated PDF training reports
- 🖼️ Augmentation visualization utilities
- 📝 Comprehensive CLI with argparse (`main.py`, `train.py`, `predict.py`)

---

## 🎬 Demo

### Digit Recognition (Web UI)
Draw a digit on the canvas → CNN predicts with confidence and probability bars for all 10 classes.

### OCR Mode (Web UI)
Write a word or sentence → CRNN reads the text (supports English A–Z/a–z, Arabic ا–ي, and digits 0–9).

> Start the server with `python api/run_server.py` and open `http://localhost:8000`

---

## 📁 Project Structure

```
Handwritten-Digit-Recognition-CNN/
│
├── api/                           # FastAPI REST API
│   ├── __init__.py
│   ├── app.py                     # API endpoints (digit + OCR)
│   └── run_server.py              # Uvicorn server launcher
│
├── web/                           # Frontend Web UIs
│   ├── index.html                 # Digit drawing canvas
│   └── ocr.html                   # OCR handwriting pad
│
├── model/                         # Neural network architectures
│   ├── __init__.py
│   ├── cnn_model.py               # CNN for digit classification
│   ├── mlp_model.py               # MLP baseline for comparison
│   ├── crnn_model.py              # CRNN (CNN + BiLSTM) for OCR
│   ├── ocr_dataset.py             # Synthetic OCR data generator
│   ├── iam_dataset.py             # IAM & RIMES real handwriting datasets
│   └── ocr_utils.py               # Charset, CTC decoding, CER/WER metrics
│
├── data/                          # Data loading & preprocessing
│   ├── __init__.py
│   └── data_loader.py             # MNIST loader with augmentation
│
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── early_stopping.py          # Early stopping callback
│   ├── logger.py                  # TensorBoard logging wrapper
│   ├── model_manager.py           # Model versioning & checkpoint manager
│   └── visualize.py               # Plotting utilities
│
├── saved_models/                  # Trained model weights (auto-managed)
├── outputs/                       # Generated plots, reports, confusion matrices
├── runs/                          # TensorBoard event logs
│
├── config.py                      # ⚙️ Centralized hyperparameter configuration
├── main.py                        # 🎯 Unified CLI entry point
├── train.py                       # 🏋️ CNN training pipeline
├── train_ocr.py                   # 🔤 CRNN/OCR training pipeline
├── evaluate.py                    # 📊 Model evaluation & metrics
├── predict.py                     # 🔮 Single-image prediction CLI
├── compare_models.py              # 🧠 CNN vs MLP benchmarking
├── generate_report.py             # 📄 PDF report generation
├── quantize.py                    # ⚡ Model quantization for edge deployment
├── export_onnx.py                 # 🔄 ONNX export for cross-platform inference
│
├── Dockerfile                     # 🐳 Container build
├── docker-compose.yml             # 🐳 Compose orchestration
├── requirements.txt               # 📦 Python dependencies
├── .gitignore
└── README.md
```

---

## 🌿 Branches

| Branch | Description | Status |
|--------|-------------|--------|
| `main` | Initial complete CNN project | ✅ Stable |
| `feature/enhancements` | Professional ML upgrades: config system, TensorBoard, model versioning, early stopping, data augmentation | ✅ Stable |
| `feature/deployment-upgrades` | Full deployment stack: FastAPI, Web UI, Docker, model comparison, PDF reports, **OCR mode** | ✅ Active (latest) |

### Branch History

```
73cf2c4  first commit
    │
d619bac  Add complete Handwritten Digit Recognition CNN project [main]
    │
5e15800  Add professional ML enhancements [feature/enhancements]
    │
a894e85  Add deployment upgrades: FastAPI, Web UI, Docker, model comparison, PDF report
    │
8ea2520  Add trained model metadata (99.36% accuracy on MNIST)
    │
715a06d  Add Real OCR Mode: CRNN (CNN + BiLSTM + CTC) [feature/deployment-upgrades] ← HEAD
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mohamedshhahat1/Handwritten-Digit-Recognition-CNN.git
cd Handwritten-Digit-Recognition-CNN

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework & datasets |
| `fastapi` / `uvicorn` | REST API server |
| `matplotlib` / `seaborn` | Visualization & plotting |
| `scikit-learn` | Classification metrics |
| `Pillow` | Image processing |
| `numpy` | Numerical computing |
| `tqdm` | Progress bars |
| `tensorboard` | Training visualization |
| `reportlab` | PDF report generation |
| `arabic-reshaper` / `python-bidi` | Arabic text support |

---

## 🎯 Usage

### CLI Interface (`main.py`)

The unified entry point for all operations:

```bash
# Train the CNN model
python main.py --mode train

# Train with custom hyperparameters
python main.py --mode train --epochs 20 --batch-size 128 --lr 0.0005

# Train without data augmentation
python main.py --mode train --no-augment

# Evaluate on the MNIST test set
python main.py --mode evaluate

# Predict a single image
python main.py --mode predict --image path/to/digit.png --verbose

# Run demo on random test images
python main.py --mode demo

# Quantize model for edge deployment
python main.py --mode quantize

# Quantize with static method and compare all
python main.py --mode quantize --quantize-mode static
python main.py --mode quantize --compare
```

---

### Training

#### CNN (Digit Recognition)

```bash
# Using defaults from config.py
python train.py

# Custom training run
python train.py --epochs 20 --lr 0.0005 --batch-size 128

# Without data augmentation
python train.py --no-augment
```

**What happens during training:**
1. Downloads MNIST dataset (60,000 training + 10,000 test images)
2. Applies data augmentation (rotation, shift, scale, noise)
3. Trains with Adam optimizer + weight decay
4. Logs metrics to TensorBoard in real-time
5. Saves versioned checkpoints + best model
6. Early stopping if validation accuracy plateaus
7. Saves training history to JSON

---

### OCR Training

#### CRNN (Text Recognition)

```bash
# Default: 20 epochs, 5000 synthetic samples
python train_ocr.py

# Custom configuration
python train_ocr.py --epochs 30 --batch-size 32 --lr 0.001 --samples 10000

# Train on IAM Handwriting Database (real English handwriting)
python train_ocr.py --dataset iam --data-dir ./data/iam

# Train on RIMES Dataset (real French handwriting)
python train_ocr.py --dataset rimes --data-dir ./data/rimes

# Train on real data with more epochs
python train_ocr.py --dataset iam --data-dir ./data/iam --epochs 50 --batch-size 64
```

**Supported Datasets:**

| Dataset | Language | Samples | Source |
|---------|----------|---------|--------|
| `synthetic` | English + Arabic + Digits | Generated on-the-fly | No download needed |
| `iam` | English | ~13,000 text lines | [IAM Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) |
| `rimes` | French | ~12,000 text lines | [RIMES Database](http://www.a2ialab.com/doku.php?id=rimes_database) |

**Dataset setup (IAM):**
```
data/iam/
├── lines/              # Line images (a01-000u-00.png, ...)
│   ├── a01/
│   │   ├── a01-000u/
│   │   │   └── *.png
│   │   └── ...
│   └── ...
└── lines.txt           # Annotations file
```

**How it works:**
- Supports both synthetic generation and real handwriting datasets
- Automatic fallback to synthetic if real dataset not found
- Data augmentation for real datasets (rotation, blur, noise)
- Uses CTC loss for variable-length sequence alignment
- Reports Character Error Rate (CER) during validation
- Saves best model based on lowest CER

---

### Evaluation

```bash
# Evaluate best model on MNIST test set
python evaluate.py

# Evaluate a specific model checkpoint
python evaluate.py --model-path saved_models/mnist_cnn_v5_20260530_214310.pth
```

**Outputs:**
- Overall accuracy
- Per-digit accuracy breakdown
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap (saved to `outputs/confusion_matrix.png`)

---

### Prediction

```bash
# Predict a single image
python predict.py --image my_digit.png

# With full probability breakdown
python predict.py --image my_digit.png --verbose

# Demo mode with random test images
python predict.py --demo --num-samples 10

# Use a specific model
python predict.py --image my_digit.png --model-path saved_models/mnist_cnn_best.pth
```

---

### 🚀 FastAPI Server

Start the prediction API:

```bash
python api/run_server.py
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict digit from uploaded image (multipart) |
| `POST` | `/predict/base64` | Predict digit from base64-encoded image |
| `POST` | `/predict/ocr` | OCR text recognition from uploaded image |
| `POST` | `/predict/ocr/base64` | OCR text recognition from base64 image |
| `GET`  | `/health` | Health check with model status |
| `GET`  | `/` | Digit recognition Web UI |
| `GET`  | `/ocr` | OCR Web UI |
| `GET`  | `/docs` | Interactive Swagger API documentation |

#### Example API Calls

```bash
# Digit prediction from file
curl -X POST "http://localhost:8000/predict" \
  -F "file=@digit.png"

# Digit prediction from base64
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'

# OCR text recognition
curl -X POST "http://localhost:8000/predict/ocr" \
  -F "file=@handwritten_text.png"

# Health check
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true, "ocr_loaded": true}
```

---

### 🌐 Web UI

With the API server running, open your browser:

- **`http://localhost:8000`** — Digit Recognition: draw a digit (0–9) on the canvas
- **`http://localhost:8000/ocr`** — OCR Mode: write words/sentences for text recognition

Features:
- Responsive drawing canvas (mouse + touch support)
- Real-time prediction with confidence scores
- Probability bar chart for all 10 digit classes
- Switch between Digit mode and OCR mode

---

### 📦 Docker

```bash
# Build and run
docker build -t digit-recognition .
docker run -p 8000:8000 digit-recognition

# Or use Docker Compose (includes health checks & volume mounts)
docker-compose up --build
```

The Docker container:
- Serves the FastAPI app on port 8000
- Mounts `saved_models/` as a volume for model persistence
- Includes health checks (`/health` endpoint)
- Auto-restarts on failure

---

### 🧠 Model Comparison (CNN vs MLP)

```bash
python compare_models.py
```

Trains both CNN and MLP on the same data with identical hyperparameters, then generates:
- `outputs/comparison_accuracy.png` — Training/test accuracy curves
- `outputs/comparison_loss.png` — Training/test loss curves
- `outputs/comparison_summary.png` — Bar chart comparing final metrics
- Terminal comparison table

---

### 📊 Generate PDF Report

```bash
python generate_report.py
```

Creates a professional multi-page PDF at `outputs/training_report.pdf` including:
- Model architecture diagram
- Training curves (loss & accuracy)
- Confusion matrix
- Per-class metrics
- Configuration summary

---

### ⚡ Model Quantization (Edge Deployment)

```bash
# Dynamic quantization (quick, no calibration needed)
python quantize.py

# Static quantization (maximum compression)
python quantize.py --mode static

# Quantization-aware training (best accuracy)
python quantize.py --mode qat --epochs 3

# Quantize the ResNet model
python quantize.py --model resnet

# Compare all methods side-by-side
python quantize.py --compare
```

Reduces model size for deployment on edge devices (mobile, Raspberry Pi, etc.):

| Method | Accuracy Loss | Size Reduction | Speedup |
|--------|:---:|:---:|:---:|
| Dynamic | 0.00% | 3.5× smaller | 1.7× faster |
| Static | ~0.1–0.5% | ~4× smaller | 2–3× faster |
| QAT | ~0% | ~4× smaller | 2–3× faster |

Output: `saved_models/mnist_cnn_quantized_{dynamic,static,qat}.pth`

---

### 🔄 ONNX Export (Cross-Platform Deployment)

```bash
# Export CNN to ONNX
python export_onnx.py

# Export ResNet model
python export_onnx.py --model resnet

# Export with validation (compare PyTorch vs ONNX outputs)
python export_onnx.py --validate

# Enable dynamic batch size
python export_onnx.py --dynamic-batch

# Via main.py
python main.py --mode export
python main.py --mode export --validate
```

Deploy the exported `.onnx` model on any platform:

| Platform | Runtime |
|----------|---------|
| Desktop/Server | ONNX Runtime (C++, Python, C#, Java) |
| NVIDIA GPU | TensorRT |
| Intel CPU/GPU | OpenVINO |
| Apple devices | CoreML (via onnx-coreml) |
| Android/iOS | ONNX Runtime Mobile |
| Web browsers | ONNX.js / ort-web |

**Inference example (Python):**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("saved_models/mnist_cnn.onnx")
result = session.run(None, {"input": image_array})  # image: (1, 1, 28, 28) float32
predicted_digit = np.argmax(result[0])
```

Output: `saved_models/mnist_{cnn,resnet}.onnx`

---

### 📡 TensorBoard

```bash
# Start TensorBoard (after training)
tensorboard --logdir=./runs

# Open http://localhost:6006 in your browser
```

Visualizes:
- Training loss per batch and epoch
- Training & validation accuracy
- Sample predictions at intervals
- Model computation graph

---

## 🏛️ Model Architectures

### 1. CNN — Digit Classification

```
Input (1×28×28) grayscale image
       │
       ▼
┌────────────────────────────┐
│  Conv2D(1→32, 3×3, pad=1)  │  → 32 feature maps
│  ReLU + MaxPool2D(2×2)     │  → 32×14×14
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Conv2D(32→64, 3×3, pad=1) │  → 64 feature maps
│  ReLU + MaxPool2D(2×2)     │  → 64×7×7
└────────────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Flatten (3136)             │
│  FC(3136→128) + ReLU        │
│  Dropout(0.25)              │
│  FC(128→10)                 │
└────────────────────────────┘
       │
       ▼
  Output: 10 class logits
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D(1→32, 3×3, pad=1) + ReLU | 32 × 28 × 28 | 320 |
| MaxPool2D(2×2) | 32 × 14 × 14 | 0 |
| Conv2D(32→64, 3×3, pad=1) + ReLU | 64 × 14 × 14 | 18,496 |
| MaxPool2D(2×2) | 64 × 7 × 7 | 0 |
| Flatten → FC(3136→128) + ReLU | 128 | 401,536 |
| Dropout(0.25) → FC(128→10) | 10 | 1,290 |
| **Total** | | **421,642** |

---

### 2. CRNN — OCR Text Recognition

```
Input (1×32×W) grayscale image, height=32, variable width
       │
       ▼
┌─────────────────────────────────────┐
│  CNN Backbone (7 conv layers)        │
│  64 → 128 → 256 → 256 → 512 → 512  │
│  BatchNorm + ReLU + MaxPool          │
│  Output: (512, 1, W/4)              │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Map-to-Sequence                     │
│  Reshape: (W/4, batch, 512)         │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  BiLSTM (2 layers, hidden=256)      │
│  Output: (W/4, batch, 512)          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Linear(512 → num_classes)          │
│  Log-Softmax                         │
└─────────────────────────────────────┘
       │
       ▼
  CTC Decoding → Text output
```

**Character Set:** 94 classes (A–Z + a–z + 0–9 + 28 Arabic letters + space/punctuation + CTC blank)

---

### 3. MLP — Baseline Comparison

```
Input (1×28×28) → Flatten (784)
  → FC(784→512) + ReLU + Dropout(0.2)
  → FC(512→256) + ReLU + Dropout(0.2)
  → FC(256→128) + ReLU + Dropout(0.2)
  → FC(128→10)
```

**Total Parameters:** ~535,818 (more than CNN, but lower accuracy — demonstrates why CNNs are preferred for images)

---

## 📊 Results

### CNN Performance (MNIST)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.36%** |
| Training Accuracy | 98.18% |
| Best Epoch | 9 / 15 |
| Training Loss | 0.0588 |
| Parameters | 421,642 |
| Training Time | ~3 min (CPU) |

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 15 (with early stopping, patience=5) |
| Augmentation | Rotation ±10°, translate ±10%, scale 0.9–1.1× |
| Dropout | 0.25 |

### CNN vs MLP Comparison

| Metric | CNN | MLP |
|--------|-----|-----|
| Test Accuracy | **~99.3%** | ~97.8% |
| Parameters | 421K | 535K |
| Spatial Awareness | ✅ Yes | ❌ No |
| Weight Sharing | ✅ Yes | ❌ No |

> CNNs exploit spatial locality and weight sharing, achieving higher accuracy with fewer parameters.

---

## ⚙️ Configuration

All hyperparameters are centralized in **`config.py`**:

```python
# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
WEIGHT_DECAY = 1e-4

# Model
NUM_CLASSES = 10
DROPOUT_RATE = 0.25

# Early Stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Data Augmentation
AUGMENTATION_ENABLED = True
ROTATION_DEGREES = 10
TRANSLATE_RANGE = (0.1, 0.1)
SCALE_RANGE = (0.9, 1.1)

# Paths
DATA_DIR = "./data/mnist"
MODEL_DIR = "./saved_models"
LOG_DIR = "./runs"
OUTPUT_DIR = "./outputs"
```

Run `python config.py` to print the full configuration in a formatted display.

---

## 📡 API Reference

### `POST /predict`
Predict digit from an uploaded image file.

**Request:** `multipart/form-data` with field `file`  
**Response:**
```json
{
  "digit": 7,
  "confidence": 0.9834,
  "probabilities": {
    "0": 0.0001, "1": 0.0003, "2": 0.0012,
    "3": 0.0008, "4": 0.0002, "5": 0.0005,
    "6": 0.0001, "7": 0.9834, "8": 0.0089,
    "9": 0.0045
  }
}
```

### `POST /predict/base64`
Predict digit from base64-encoded image (useful for canvas data).

**Request:**
```json
{"image": "data:image/png;base64,iVBOR..."}
```

### `POST /predict/ocr/base64`
Recognize handwritten text from base64 image.

**Response:**
```json
{
  "text": "hello world",
  "confidence": 0.8742,
  "model_loaded": true
}
```

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ocr_loaded": true
}
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| ![Python](https://img.shields.io/badge/-Python_3.8+-3776AB?style=flat&logo=python&logoColor=white) | Programming language |
| ![PyTorch](https://img.shields.io/badge/-PyTorch_2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep learning framework |
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | REST API framework |
| ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat&logo=docker&logoColor=white) | Containerization |
| ![TensorBoard](https://img.shields.io/badge/-TensorBoard-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Training visualization |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) | Plotting & visualization |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Evaluation metrics |
| ![Pillow](https://img.shields.io/badge/-Pillow-3776AB?style=flat) | Image processing |
| ![HTML/CSS/JS](https://img.shields.io/badge/-HTML%2FCSS%2FJS-E34F26?style=flat&logo=html5&logoColor=white) | Web UI frontend |

---

## 🔮 Roadmap

- [x] ~~CNN for digit classification~~
- [x] ~~Data augmentation (rotation, scaling, shifting)~~
- [x] ~~Web interface for interactive predictions~~
- [x] ~~Drawing canvas with live predictions~~
- [x] ~~Confusion matrix visualization~~
- [x] ~~FastAPI REST API~~
- [x] ~~Docker containerization~~
- [x] ~~Model comparison (CNN vs MLP)~~
- [x] ~~TensorBoard integration~~
- [x] ~~Early stopping~~
- [x] ~~PDF report generation~~
- [x] ~~OCR / text recognition (CRNN + CTC)~~
- [x] ~~Arabic language support~~
- [x] ~~Learning rate scheduling (cosine annealing, warm restarts)~~
- [x] ~~Batch normalization in the CNN~~
- [x] ~~ResNet-style skip connections for deeper architectures~~
- [x] ~~Model quantization for edge deployment~~
- [x] ~~ONNX export for cross-platform inference~~
- [x] ~~Real handwriting dataset training (IAM, RIMES)~~
- [x] ~~Beam search CTC decoding for better OCR accuracy~~
- [ ] Language model integration for OCR post-processing

---

## 👤 Author

**Mohamed Shhahat**  
GitHub: [@mohamedshhahat1](https://github.com/mohamedshhahat1)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Mohamed Shhahat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Made with ❤️ and PyTorch by <a href="https://github.com/mohamedshhahat1">Mohamed Shhahat</a>
</p>
