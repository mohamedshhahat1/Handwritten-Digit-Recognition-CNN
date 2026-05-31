"""
Model package for Handwritten Digit Recognition.

This package provides CNN, ResNet, CRNN, and MLP model architectures used for
classifying handwritten digits and recognizing text from the MNIST dataset.

Available models:
    - CNN: Standard 2-layer CNN with batch normalization (lightweight, fast)
    - ResNetCNN: Deeper ResNet-style architecture with skip connections (higher accuracy)
    - CRNN: CNN + BiLSTM for OCR text recognition
    - MLP: Multi-layer perceptron baseline (no spatial awareness)

Available datasets:
    - SyntheticOCRDataset: Generated synthetic handwriting images
    - IAMDataset: IAM Handwriting Database (real English handwriting)
    - RIMESDataset: RIMES Database (real French handwriting)
"""

from model.cnn_model import CNN, ResNetCNN, ResidualBlock
from model.mlp_model import MLP
from model.crnn_model import CRNN
from model.ocr_utils import OCRCharset
from model.iam_dataset import IAMDataset, RIMESDataset, get_ocr_dataset
from model.language_model import LanguageModel

# Alias for backward compatibility
MNISTNet = CNN

__all__ = [
    "CNN", "ResNetCNN", "ResidualBlock",
    "CRNN", "OCRCharset",
    "MLP", "MNISTNet",
    "IAMDataset", "RIMESDataset", "get_ocr_dataset",
    "LanguageModel",
]
