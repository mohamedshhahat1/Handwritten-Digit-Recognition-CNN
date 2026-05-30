"""
Model package for Handwritten Digit Recognition CNN.

This package provides the CNN model architecture used for classifying
handwritten digits from the MNIST dataset.
"""

from model.cnn_model import CNN

# Alias for backward compatibility
MNISTNet = CNN

__all__ = ["CNN", "MNISTNet"]
