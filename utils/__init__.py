"""
Utils package for Handwritten Digit Recognition CNN.

This package provides utility functions for:
- Visualization (training curves, predictions, confusion matrix)
- TensorBoard logging
- Model version management
- Early stopping
"""

from utils.visualize import (
    plot_training_loss,
    plot_training_accuracy,
    plot_loss_and_accuracy,
    visualize_predictions,
    plot_confusion_matrix,
)

from utils.logger import TBLogger
from utils.model_manager import ModelManager
from utils.early_stopping import EarlyStopping, create_early_stopping

__all__ = [
    "plot_training_loss",
    "plot_training_accuracy",
    "plot_loss_and_accuracy",
    "visualize_predictions",
    "plot_confusion_matrix",
    "TBLogger",
    "ModelManager",
    "EarlyStopping",
    "create_early_stopping",
]
