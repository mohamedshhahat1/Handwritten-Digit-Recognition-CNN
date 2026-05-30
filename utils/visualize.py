"""
Visualization module for Handwritten Digit Recognition CNN.

This module provides functions to visualize training progress and model
predictions. All plots are saved to the 'outputs/' directory.

Dependencies:
    - matplotlib
    - seaborn
    - numpy
    - torch (for tensor handling)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Directory where all plots will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _ensure_output_dir():
    """Create the outputs directory if it doesn't already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_loss(losses, save_filename="training_loss.png"):
    """
    Plot the training loss curve over epochs.

    Parameters
    ----------
    losses : list of float
        Training loss values for each epoch. For example, if you trained
        for 10 epochs, this should be a list with 10 values.
    save_filename : str, optional
        Name of the file to save the plot (default: 'training_loss.png').

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    _ensure_output_dir()

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate epoch numbers starting from 1
    epochs = range(1, len(losses) + 1)

    # Plot the loss curve
    ax.plot(epochs, losses, marker="o", color="red", linewidth=2, markersize=5, label="Training Loss")

    # Add labels and title
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Make sure x-axis shows integer epoch numbers
    ax.set_xticks(list(epochs))

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training loss plot saved to: {save_path}")
    return save_path


def plot_training_accuracy(accuracies, save_filename="training_accuracy.png"):
    """
    Plot the training accuracy curve over epochs.

    Parameters
    ----------
    accuracies : list of float
        Training accuracy values for each epoch (as percentages, e.g., 95.5
        means 95.5% accuracy).
    save_filename : str, optional
        Name of the file to save the plot (default: 'training_accuracy.png').

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    _ensure_output_dir()

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate epoch numbers starting from 1
    epochs = range(1, len(accuracies) + 1)

    # Plot the accuracy curve
    ax.plot(epochs, accuracies, marker="s", color="blue", linewidth=2, markersize=5, label="Training Accuracy")

    # Add labels and title
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Make sure x-axis shows integer epoch numbers
    ax.set_xticks(list(epochs))

    # Set y-axis limits for accuracy (0-100%)
    ax.set_ylim(0, 105)

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training accuracy plot saved to: {save_path}")
    return save_path


def plot_loss_and_accuracy(losses, accuracies, save_filename="loss_and_accuracy.png"):
    """
    Plot both training loss and accuracy on the same figure using dual y-axes.

    This is useful to see how loss decreases while accuracy increases during
    training, all in one view.

    Parameters
    ----------
    losses : list of float
        Training loss values for each epoch.
    accuracies : list of float
        Training accuracy values for each epoch (as percentages).
    save_filename : str, optional
        Name of the file to save the plot (default: 'loss_and_accuracy.png').

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    _ensure_output_dir()

    # Create figure with one set of axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Generate epoch numbers starting from 1
    epochs = range(1, len(losses) + 1)

    # --- Left y-axis: Loss ---
    color_loss = "red"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color=color_loss)
    line1 = ax1.plot(
        epochs, losses,
        marker="o", color=color_loss, linewidth=2, markersize=5,
        label="Training Loss"
    )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_xticks(list(epochs))

    # --- Right y-axis: Accuracy ---
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color_acc = "blue"
    ax2.set_ylabel("Accuracy (%)", fontsize=12, color=color_acc)
    line2 = ax2.plot(
        epochs, accuracies,
        marker="s", color=color_acc, linewidth=2, markersize=5,
        label="Training Accuracy"
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 105)

    # Combine legends from both axes into one
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11)

    # Add title and grid
    ax1.set_title("Training Loss and Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Loss and accuracy plot saved to: {save_path}")
    return save_path


def visualize_predictions(images, predicted_labels, actual_labels, num_samples=16,
                          save_filename="predictions.png"):
    """
    Visualize model predictions on sample images in a grid layout.

    Displays each image with its predicted label and actual label. Correct
    predictions are shown in green, incorrect ones in red.

    Parameters
    ----------
    images : numpy.ndarray or torch.Tensor
        Batch of images. Expected shape: (N, 1, 28, 28) or (N, 28, 28).
        Values should be in [0, 1] range for proper display.
    predicted_labels : list or numpy.ndarray
        Predicted digit labels (integers 0-9).
    actual_labels : list or numpy.ndarray
        True/actual digit labels (integers 0-9).
    num_samples : int, optional
        Number of sample images to display (default: 16). Will be arranged
        in a grid as close to square as possible.
    save_filename : str, optional
        Name of the file to save the plot (default: 'predictions.png').

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    _ensure_output_dir()

    # Convert torch tensors to numpy arrays if needed
    if hasattr(images, "cpu"):
        images = images.cpu().detach().numpy()
    if hasattr(predicted_labels, "cpu"):
        predicted_labels = predicted_labels.cpu().numpy()
    if hasattr(actual_labels, "cpu"):
        actual_labels = actual_labels.cpu().numpy()

    # Ensure we don't try to show more samples than available
    num_samples = min(num_samples, len(images))

    # Calculate grid dimensions (try to make it as square as possible)
    num_cols = int(np.ceil(np.sqrt(num_samples)))
    num_rows = int(np.ceil(num_samples / num_cols))

    # Create the figure and grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 3))

    # Flatten axes array for easy iteration (handles both 1D and 2D cases)
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_samples):
        ax = axes[idx]

        # Get the image and remove the channel dimension if present
        img = images[idx]
        if img.ndim == 3:
            # Shape is (1, 28, 28) -> squeeze to (28, 28)
            img = img.squeeze(0)

        # Display the image in grayscale
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")  # Hide axis ticks

        # Get prediction and actual label
        pred = int(predicted_labels[idx])
        actual = int(actual_labels[idx])

        # Color the title green if correct, red if incorrect
        is_correct = pred == actual
        title_color = "green" if is_correct else "red"

        ax.set_title(
            f"Pred: {pred}\nActual: {actual}",
            fontsize=10,
            color=title_color,
            fontweight="bold"
        )

    # Hide any unused subplot spaces
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    # Add an overall title
    fig.suptitle("Model Predictions on Sample Images", fontsize=14, fontweight="bold", y=1.02)

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Predictions visualization saved to: {save_path}")
    return save_path


def plot_confusion_matrix(confusion_mat, class_names=None, save_filename="confusion_matrix.png"):
    """
    Plot a confusion matrix as a heatmap.

    The confusion matrix shows how often each digit is classified as each
    other digit. The diagonal represents correct predictions.

    Parameters
    ----------
    confusion_mat : numpy.ndarray
        A 2D confusion matrix of shape (num_classes, num_classes).
        Element [i, j] represents how many times class i was predicted as
        class j. You can generate this using sklearn.metrics.confusion_matrix.
    class_names : list of str, optional
        Labels for each class. Defaults to ['0', '1', ..., '9'] for digits.
    save_filename : str, optional
        Name of the file to save the plot (default: 'confusion_matrix.png').

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    _ensure_output_dir()

    # Convert torch tensor to numpy if needed
    if hasattr(confusion_mat, "cpu"):
        confusion_mat = confusion_mat.cpu().numpy()

    # Default class names for digit recognition (0-9)
    if class_names is None:
        num_classes = confusion_mat.shape[0]
        class_names = [str(i) for i in range(num_classes)]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap using seaborn
    sns.heatmap(
        confusion_mat,
        annot=True,           # Show numbers in each cell
        fmt="d",              # Format as integers
        cmap="Blues",         # Blue color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,          # Make cells square
        linewidths=0.5,       # Add grid lines between cells
        cbar_kws={"shrink": 0.8},  # Slightly smaller color bar
        ax=ax
    )

    # Add labels and title
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Rotate tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix plot saved to: {save_path}")
    return save_path


# =============================================================================
# Example usage (uncomment to test):
# =============================================================================
# if __name__ == "__main__":
#     # Simulated training data for demonstration
#     example_losses = [2.3, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08]
#     example_accuracies = [20.0, 45.0, 60.0, 75.0, 85.0, 90.0, 93.0, 95.0, 97.0, 98.0]
#
#     # Plot individual curves
#     plot_training_loss(example_losses)
#     plot_training_accuracy(example_accuracies)
#
#     # Plot both on same figure
#     plot_loss_and_accuracy(example_losses, example_accuracies)
#
#     # Simulated confusion matrix (10x10 for digits 0-9)
#     example_cm = np.random.randint(0, 50, size=(10, 10))
#     np.fill_diagonal(example_cm, np.random.randint(200, 500, size=10))
#     plot_confusion_matrix(example_cm)
#
#     # Simulated predictions visualization
#     example_images = np.random.rand(16, 1, 28, 28)
#     example_preds = np.random.randint(0, 10, size=16)
#     example_actuals = np.random.randint(0, 10, size=16)
#     visualize_predictions(example_images, example_preds, example_actuals)
