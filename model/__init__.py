"""
Model package for Handwritten Digit Recognition.

This package provides CNN, ResNet, and MLP model architectures used for
classifying handwritten digits from the MNIST dataset.

Available models:
    - CNN: Standard 2-layer CNN with batch normalization (lightweight, fast)
    - ResNetCNN: Deeper ResNet-style architecture with skip connections (higher accuracy)
    - MLP: Multi-layer perceptron baseline (no spatial awareness)
"""

from model.cnn_model import CNN, ResNetCNN, ResidualBlock
from model.mlp_model import MLP

# Alias for backward compatibility
MNISTNet = CNN

__all__ = ["CNN", "ResNetCNN", "ResidualBlock", "MLP", "MNISTNet"]
