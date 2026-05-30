"""
TensorBoard logging utility for the Handwritten Digit Recognition CNN.

This module provides a clean wrapper around TensorBoard's SummaryWriter
for tracking training metrics, model architecture, and sample predictions.

Usage:
    from utils.logger import TBLogger

    # As a context manager (recommended)
    with TBLogger(log_dir="./runs", experiment_name="my_experiment") as logger:
        logger.log_scalar("loss", 0.5, step=1)
        logger.log_epoch_metrics(epoch=1, train_loss=0.3, train_acc=90.0, val_acc=88.5, lr=0.001)

    # Or manually
    logger = TBLogger()
    logger.log_scalar("loss", 0.5, step=1)
    logger.close()

To view logs, run the following command in your terminal:
    tensorboard --logdir=runs

Then open http://localhost:6006 in your browser.
"""

from datetime import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """A wrapper around TensorBoard's SummaryWriter for structured experiment logging.

    Provides methods for logging scalars, histograms, model graphs, and
    sample prediction images during training of the digit recognition CNN.

    Attributes:
        writer (SummaryWriter): The underlying TensorBoard SummaryWriter instance.
        log_path (str): Full path where TensorBoard logs are saved.
    """

    def __init__(self, log_dir="./runs", experiment_name=None):
        """Initialize the TBLogger.

        Args:
            log_dir (str): Base directory for TensorBoard log files.
                Defaults to "./runs".
            experiment_name (str, optional): Name for this experiment run.
                If None, a timestamped name is generated automatically
                (e.g., "run_2024-01-15_14-30-00").
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"run_{timestamp}"

        self.log_path = f"{log_dir}/{experiment_name}"
        self.writer = SummaryWriter(log_dir=self.log_path)

        print(f"[TBLogger] TensorBoard logs will be saved to: {self.log_path}")
        print(f"[TBLogger] To view logs, run: tensorboard --logdir={log_dir}")

    def log_scalar(self, tag, value, step):
        """Log a single scalar value.

        Args:
            tag (str): Identifier for the scalar (e.g., "Loss/train").
            value (float): The scalar value to log.
            step (int): The global step or iteration number.
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars on the same plot for comparison.

        Args:
            main_tag (str): The parent tag that groups the scalars
                (e.g., "Accuracy").
            tag_scalar_dict (dict): Dictionary mapping sub-tag names to
                scalar values (e.g., {"train": 0.9, "val": 0.85}).
            step (int): The global step or iteration number.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_training_step(self, loss, batch_idx, epoch, total_batches):
        """Log per-batch training loss.

        Computes a global step from the epoch and batch index so that
        the loss curve is continuous across epochs.

        Args:
            loss (float): The training loss for the current batch.
            batch_idx (int): Index of the current batch within the epoch.
            epoch (int): The current epoch number.
            total_batches (int): Total number of batches per epoch.
        """
        global_step = epoch * total_batches + batch_idx
        self.writer.add_scalar("Loss/train_step", loss, global_step)

    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_acc, lr):
        """Log all epoch-level metrics at once.

        This logs training loss, training accuracy, validation accuracy,
        and the current learning rate, all indexed by epoch.

        Args:
            epoch (int): The current epoch number.
            train_loss (float): Average training loss for the epoch.
            train_acc (float): Training accuracy for the epoch (percentage).
            val_acc (float): Validation accuracy for the epoch (percentage).
            lr (float): Current learning rate.
        """
        self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)
        self.writer.add_scalar("Learning_Rate", lr, epoch)

        # Also log train and val accuracy on the same plot for easy comparison
        self.writer.add_scalars(
            "Accuracy/comparison",
            {"train": train_acc, "val": val_acc},
            epoch,
        )

    def log_model_graph(self, model, input_tensor):
        """Log the model architecture graph for visualization.

        Args:
            model (torch.nn.Module): The model to visualize.
            input_tensor (torch.Tensor): A sample input tensor with the
                correct shape (e.g., torch.randn(1, 1, 28, 28) for MNIST).
        """
        self.writer.add_graph(model, input_tensor)

    def log_predictions(self, images, predictions, labels, epoch, num_samples=8):
        """Log sample prediction images with their predicted and true labels.

        Creates a grid of images annotated with prediction results,
        useful for visually inspecting model performance.

        Args:
            images (torch.Tensor): Batch of input images (N, C, H, W).
            predictions (torch.Tensor): Predicted class indices.
            labels (torch.Tensor): Ground truth class indices.
            epoch (int): The current epoch number.
            num_samples (int): Number of samples to include in the grid.
                Defaults to 8.
        """
        # Take only the specified number of samples
        num_samples = min(num_samples, len(images))
        sample_images = images[:num_samples]
        sample_preds = predictions[:num_samples]
        sample_labels = labels[:num_samples]

        # Create an image grid from the sample images
        img_grid = torchvision.utils.make_grid(sample_images, nrow=num_samples, normalize=True)

        # Log the image grid with a descriptive tag showing predictions vs labels
        pred_str = ", ".join(
            [f"{p.item()}({'✓' if p == l else '✗'})" for p, l in zip(sample_preds, sample_labels)]
        )
        self.writer.add_image(
            f"Predictions/epoch_{epoch}",
            img_grid,
            epoch,
        )

        # Add text summary of predictions for this epoch
        text = f"Predictions: {sample_preds.tolist()}\nLabels:      {sample_labels.tolist()}"
        self.writer.add_text(f"Predictions_Text/epoch_{epoch}", text, epoch)

    def log_histogram(self, tag, values, step):
        """Log a histogram of values (e.g., weights or gradients).

        Useful for monitoring weight distributions and gradient flow
        during training.

        Args:
            tag (str): Identifier for the histogram (e.g., "conv1/weights").
            values (torch.Tensor or numpy.ndarray): The values to create
                a histogram from.
            step (int): The global step or iteration number.
        """
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """Flush pending events and close the SummaryWriter.

        Should be called when logging is complete to ensure all data
        is written to disk.
        """
        self.writer.flush()
        self.writer.close()
        print(f"[TBLogger] Writer closed. Logs saved to: {self.log_path}")

    def __enter__(self):
        """Enter the context manager, returning this logger instance."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, closing the writer."""
        self.close()
        return False
