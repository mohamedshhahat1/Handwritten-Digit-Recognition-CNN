"""
Data package for Handwritten Digit Recognition.

Provides convenient access to the MNIST data loading utilities.
"""

from .data_loader import get_data_loaders

__all__ = ['get_data_loaders']
