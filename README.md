# 🧠 Handwritten Digit Recognition CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-green?style=for-the-badge)](http://yann.lecun.com/exdb/mnist/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A deep learning project that implements a **Convolutional Neural Network (CNN)** using PyTorch to recognize handwritten digits (0-9) from the MNIST dataset with high accuracy.

---

## 📋 Project Description

This project demonstrates the power of convolutional neural networks for image classification tasks. The model is trained on the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing), each sized 28x28 pixels. The CNN architecture is designed to efficiently extract spatial features from the input images and classify them into one of 10 digit classes.

---

## ✨ Features

- 🏗️ Clean, modular CNN architecture built with PyTorch
- 📊 Training pipeline with loss and accuracy tracking
- 📈 Visualization utilities for training metrics and predictions
- 🔍 Single image prediction support
- 💾 Model checkpointing and versioning
- 📉 Dropout regularization to prevent overfitting
- 🖼️ Automatic MNIST dataset downloading and preprocessing
- 📝 Comprehensive evaluation with detailed metrics
- 🚀 **FastAPI REST API** with `/predict` endpoint
- 🌐 **Web UI** — draw digits and get live predictions
- 📦 **Docker** support for easy deployment
- 🧠 **CNN vs MLP comparison** with accuracy benchmarks
- 📊 **Auto-generated PDF reports** of training results
- ⚡ Early stopping to prevent overfitting
- 🧪 Centralized configuration system
- 📡 TensorBoard integration for live monitoring
- 🎨 Data augmentation (rotation, shift, scale, noise)

---

## 📁 Project Structure

```
Handwritten-Digit-Recognition-CNN/
├── api/
│   ├── __init__.py
│   ├── app.py                 # FastAPI prediction API
│   └── run_server.py          # Server startup script
├── web/
│   └── index.html             # Drawing canvas Web UI
├── model/
│   ├── __init__.py
│   ├── cnn_model.py           # CNN architecture
│   └── mlp_model.py           # MLP architecture (for comparison)
├── data/
│   ├── __init__.py
│   └── data_loader.py         # Data loading & augmentation
├── utils/
│   ├── __init__.py
│   ├── visualize.py           # Plotting utilities
│   ├── logger.py              # TensorBoard logging
│   ├── model_manager.py       # Model versioning
│   └── early_stopping.py      # Early stopping utility
├── saved_models/              # Trained model weights
├── outputs/                   # Generated plots and reports
├── config.py                  # Centralized configuration
├── main.py                    # CLI entry point
├── train.py                   # Training pipeline
├── evaluate.py                # Model evaluation
├── predict.py                 # Image prediction & CLI
├── compare_models.py          # CNN vs MLP comparison
├── generate_report.py         # PDF report generation
├── Dockerfile                 # Docker containerization
├── docker-compose.yml         # Docker Compose config
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**

```bash
git clone https://github.com/mohamedshhahat1/Handwritten-Digit-Recognition-CNN.git
cd Handwritten-Digit-Recognition-CNN
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
# venv\Scripts\activate         # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### Using the CLI (main.py)

You can use `main.py` as a unified entry point:

```bash
# Train the model
python main.py --mode train

# Train with custom hyperparameters
python main.py --mode train --epochs 20 --batch-size 128 --lr 0.0005

# Evaluate on the test set
python main.py --mode evaluate

# Predict a single image
python main.py --mode predict --image path/to/digit.png

# Run a demo on random test images
python main.py --mode demo
```

### Training the Model

Train the CNN model on the MNIST dataset:

```bash
python train.py
```

This will:
- Download the MNIST dataset automatically (if not already present)
- Train the model for the configured number of epochs
- Save the trained model weights to `saved_models/`
- Generate training loss and accuracy plots in `outputs/`

### Evaluating the Model

Evaluate the trained model on the test dataset:

```bash
python evaluate.py
```

This will output:
- Overall test accuracy
- Per-class accuracy
- Confusion matrix

### Making Predictions

Run inference on a single image:

```bash
python predict.py
```

The script will load a sample image and display the predicted digit along with the model's confidence score.

### 🚀 FastAPI Server

Start the prediction API:

```bash
python api/run_server.py
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then use the API:

```bash
# Predict from image file
curl -X POST "http://localhost:8000/predict" -F "file=@digit.png"

# Response:
# {"digit": 7, "confidence": 0.98, "probabilities": {"0": 0.001, ...}}
```

### 🌐 Web UI

Open `http://localhost:8000` in your browser (with the API server running) to access the drawing canvas. Draw a digit and click "Predict" to see results!

### 📦 Docker

```bash
# Build and run with Docker
docker build -t digit-recognition .
docker run -p 8000:8000 digit-recognition

# Or use Docker Compose
docker-compose up --build
```

### 🧠 Model Comparison (CNN vs MLP)

```bash
python compare_models.py
```

Trains both models and generates comparison plots in `outputs/`.

### 📊 Generate PDF Report

```bash
python generate_report.py
```

Creates a professional PDF report at `outputs/training_report.pdf`.

---

## 🏛️ Model Architecture

The CNN architecture consists of two convolutional blocks followed by a fully connected classifier:

```
Input (1x28x28)
       │
       ▼
┌─────────────────────┐
│  Conv2D(1→32, 3x3)  │
│  ReLU Activation     │
│  MaxPool2D(2x2)     │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Conv2D(32→64, 3x3) │
│  ReLU Activation     │
│  MaxPool2D(2x2)     │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Flatten             │
│  FC(64*7*7 → 128)   │
│  ReLU Activation     │
│  Dropout(0.25)       │
│  FC(128 → 10)       │
└─────────────────────┘
       │
       ▼
Output (10 classes)
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | 1 x 28 x 28 | 0 |
| Conv2D(1→32, 3x3, pad=1) + ReLU | 32 x 28 x 28 | 320 |
| MaxPool2D(2x2) | 32 x 14 x 14 | 0 |
| Conv2D(32→64, 3x3, pad=1) + ReLU | 64 x 14 x 14 | 18,496 |
| MaxPool2D(2x2) | 64 x 7 x 7 | 0 |
| Flatten | 3136 | 0 |
| FC(3136→128) + ReLU | 128 | 401,536 |
| Dropout(0.25) | 128 | 0 |
| FC(128→10) | 10 | 1,290 |

**Total Parameters:** ~421,642

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99.2% |
| Test Accuracy | ~99.0% |
| Training Epochs | 10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |

> *Results may vary slightly between runs due to random initialization.*

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) | Programming language |
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep learning framework |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) | Data visualization |
| ![torchvision](https://img.shields.io/badge/-torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Dataset and transforms |

---

## 🔮 Future Improvements

- [ ] Add data augmentation (rotation, scaling, shifting) to improve robustness
- [ ] Implement learning rate scheduling for better convergence
- [ ] Add batch normalization layers for faster training
- [ ] Build a web interface using Flask or Streamlit for interactive predictions
- [ ] Extend to recognize handwritten letters (EMNIST dataset)
- [ ] Add support for custom image input from a drawing canvas
- [ ] Implement model quantization for deployment on edge devices
- [ ] Add confusion matrix visualization
- [ ] Experiment with deeper architectures (ResNet-style skip connections)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.



---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shhahat</a>
</p>
