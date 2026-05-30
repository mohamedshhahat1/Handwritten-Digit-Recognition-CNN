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
- 💾 Model checkpointing and saving
- 📉 Dropout regularization to prevent overfitting
- 🖼️ Automatic MNIST dataset downloading and preprocessing
- 📝 Comprehensive evaluation with detailed metrics

---

## 📁 Project Structure

```
Handwritten-Digit-Recognition-CNN/
├── model/
│   ├── __init__.py
│   └── cnn_model.py          # CNN architecture definition
├── data/
│   ├── __init__.py
│   └── data_loader.py        # Data loading & preprocessing
├── utils/
│   ├── __init__.py
│   └── visualize.py          # Plotting utilities
├── saved_models/             # Trained model weights
├── outputs/                  # Generated plots and figures
├── main.py                   # CLI entry point (train/evaluate/predict/demo)
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation script
├── predict.py                # Single image prediction
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
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

```
MIT License

Copyright (c) 2024

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
  Made with ❤️ and PyTorch
</p>
