"""
Model package for Handwritten Digit Recognition.

This package provides both CNN and MLP model architectures used for
classifying handwritten digits from the MNIST dataset.
"""

from model.cnn_model import CNN
from model.mlp_model import MLP

# Alias for backward compatibility
MNISTNet = CNN

__all__ = ["CNN", "MLP", "MNISTNet"]
