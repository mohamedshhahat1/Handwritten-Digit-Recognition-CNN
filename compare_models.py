"""
Model Comparison Script: CNN vs MLP for Handwritten Digit Recognition

This script trains both a CNN (Convolutional Neural Network) and an MLP
(Multi-Layer Perceptron) on the MNIST dataset, then compares their performance.

The comparison highlights why CNNs are preferred for image tasks:
- CNNs exploit spatial structure (nearby pixels are related)
- CNNs use weight sharing (same filter applied everywhere)
- CNNs typically achieve higher accuracy with fewer parameters on images

Usage:
    python compare_models.py

Outputs:
    - outputs/comparison_accuracy.png  : Training accuracy curves for both models
    - outputs/comparison_loss.png      : Training loss curves for both models
    - outputs/comparison_summary.png   : Bar chart comparing final metrics
    - Printed comparison table in the terminal
"""

import sys
import os
import time

# Add the project root to the Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Import our models
from model.cnn_model import CNN
from model.mlp_model import MLP

# Import data loading utility
from data.data_loader import get_data_loaders

# Import configuration
import config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.

    This helps compare model complexity. More parameters generally means:
    - More memory usage
    - Longer training times
    - Higher risk of overfitting (without proper regularization)

    Args:
        model (nn.Module): A PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one complete pass through the training data.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing training batches.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer (e.g., Adam).
        device: Device to run computations on (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy) for this epoch.
    """
    model.train()  # Set model to training mode (enables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the appropriate device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass: compute predictions
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients for all parameters

        # Update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader providing test batches.
        criterion: Loss function for computing test loss.
        device: Device to run computations on.

    Returns:
        tuple: (average_loss, accuracy) on the test set.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradient computation needed during evaluation (saves memory and time)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_model(model, model_name, train_loader, test_loader, num_epochs,
                learning_rate, device):
    """
    Train a model for a specified number of epochs and track metrics.

    Args:
        model (nn.Module): The model to train.
        model_name (str): Name for display purposes (e.g., "CNN" or "MLP").
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device: Device to run on.

    Returns:
        dict: Dictionary containing training history and final metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Move model to the appropriate device
    model = model.to(device)

    # Define loss function and optimizer
    # CrossEntropyLoss combines LogSoftmax + NLLLoss (standard for classification)
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: adaptive learning rate algorithm that works well in practice
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    # Measure total training time
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on the test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print progress
        print(f"  Epoch [{epoch:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Calculate total training time
    training_time = time.time() - start_time

    # Final evaluation
    final_test_loss, final_test_acc = evaluate(
        model, test_loader, criterion, device
    )

    # Store summary metrics
    history['final_test_acc'] = final_test_acc
    history['final_test_loss'] = final_test_loss
    history['training_time'] = training_time
    history['num_parameters'] = count_parameters(model)
    history['model_name'] = model_name

    print(f"\n  {model_name} Training Complete!")
    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Total Parameters: {count_parameters(model):,}")

    return history


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_accuracy_comparison(cnn_history, mlp_history, save_path):
    """
    Plot training accuracy curves for both models on the same graph.

    This allows visual comparison of how quickly each model learns and
    what accuracy level each model converges to.

    Args:
        cnn_history (dict): Training history for the CNN model.
        mlp_history (dict): Training history for the MLP model.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(cnn_history['train_acc']) + 1)

    # Plot training accuracy for both models
    plt.plot(epochs, cnn_history['train_acc'], 'b-o', label='CNN - Train Acc',
             markersize=4, linewidth=2)
    plt.plot(epochs, cnn_history['test_acc'], 'b--s', label='CNN - Test Acc',
             markersize=4, linewidth=2, alpha=0.7)
    plt.plot(epochs, mlp_history['train_acc'], 'r-o', label='MLP - Train Acc',
             markersize=4, linewidth=2)
    plt.plot(epochs, mlp_history['test_acc'], 'r--s', label='MLP - Test Acc',
             markersize=4, linewidth=2, alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Comparison: Training & Test Accuracy\n(CNN vs MLP on MNIST)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Accuracy comparison plot saved to: {save_path}")


def plot_loss_comparison(cnn_history, mlp_history, save_path):
    """
    Plot training loss curves for both models on the same graph.

    Lower loss indicates better fit to the data. Comparing loss curves
    shows which model converges faster and to a lower minimum.

    Args:
        cnn_history (dict): Training history for the CNN model.
        mlp_history (dict): Training history for the MLP model.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(cnn_history['train_loss']) + 1)

    # Plot training loss for both models
    plt.plot(epochs, cnn_history['train_loss'], 'b-o', label='CNN - Train Loss',
             markersize=4, linewidth=2)
    plt.plot(epochs, cnn_history['test_loss'], 'b--s', label='CNN - Test Loss',
             markersize=4, linewidth=2, alpha=0.7)
    plt.plot(epochs, mlp_history['train_loss'], 'r-o', label='MLP - Train Loss',
             markersize=4, linewidth=2)
    plt.plot(epochs, mlp_history['test_loss'], 'r--s', label='MLP - Test Loss',
             markersize=4, linewidth=2, alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Comparison: Training & Test Loss\n(CNN vs MLP on MNIST)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss comparison plot saved to: {save_path}")


def plot_summary_comparison(cnn_history, mlp_history, save_path):
    """
    Create a bar chart comparing final metrics between the two models.

    Shows side-by-side bars for:
    - Test Accuracy (%)
    - Parameters (in thousands)
    - Training Time (seconds)
    - Final Test Loss (scaled for visibility)

    Args:
        cnn_history (dict): Training history for the CNN model.
        mlp_history (dict): Training history for the MLP model.
        save_path (str): File path to save the plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    bar_width = 0.35
    colors_cnn = '#2196F3'  # Blue
    colors_mlp = '#F44336'  # Red

    # --- Subplot 1: Test Accuracy ---
    ax = axes[0]
    values = [cnn_history['final_test_acc'], mlp_history['final_test_acc']]
    bars = ax.bar(['CNN', 'MLP'], values, color=[colors_cnn, colors_mlp],
                  width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(min(values) - 2, 100)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # --- Subplot 2: Number of Parameters ---
    ax = axes[1]
    values = [cnn_history['num_parameters'] / 1000,
              mlp_history['num_parameters'] / 1000]
    bars = ax.bar(['CNN', 'MLP'], values, color=[colors_cnn, colors_mlp],
                  width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Parameters (thousands)', fontsize=11)
    ax.set_title('Model Size', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{val:.1f}K', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # --- Subplot 3: Training Time ---
    ax = axes[2]
    values = [cnn_history['training_time'], mlp_history['training_time']]
    bars = ax.bar(['CNN', 'MLP'], values, color=[colors_cnn, colors_mlp],
                  width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # --- Subplot 4: Final Test Loss ---
    ax = axes[3]
    values = [cnn_history['final_test_loss'], mlp_history['final_test_loss']]
    bars = ax.bar(['CNN', 'MLP'], values, color=[colors_cnn, colors_mlp],
                  width=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Final Test Loss', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    plt.suptitle('CNN vs MLP: Performance Comparison on MNIST',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary comparison plot saved to: {save_path}")


def print_comparison_table(cnn_history, mlp_history):
    """
    Print a formatted comparison table to the terminal.

    Displays key metrics side-by-side for easy comparison between
    the CNN and MLP models.

    Args:
        cnn_history (dict): Training history for the CNN model.
        mlp_history (dict): Training history for the MLP model.
    """
    # Format values as strings for alignment
    cnn_acc = f"{cnn_history['final_test_acc']:.2f}%"
    mlp_acc = f"{mlp_history['final_test_acc']:.2f}%"
    cnn_params = f"{cnn_history['num_parameters']:,}"
    mlp_params = f"{mlp_history['num_parameters']:,}"
    cnn_time = f"{cnn_history['training_time']:.1f}s"
    mlp_time = f"{mlp_history['training_time']:.1f}s"
    cnn_loss = f"{cnn_history['final_test_loss']:.4f}"
    mlp_loss = f"{mlp_history['final_test_loss']:.4f}"

    print("\n")
    print("=" * 60)
    print("         MODEL COMPARISON RESULTS: CNN vs MLP")
    print("=" * 60)
    print()
    print(f"| {'Metric':<17}| {'CNN':<12}| {'MLP':<12}|")
    print(f"|{'-'*18}|{'-'*13}|{'-'*13}|")
    print(f"| {'Test Accuracy':<17}| {cnn_acc:<12}| {mlp_acc:<12}|")
    print(f"| {'Parameters':<17}| {cnn_params:<12}| {mlp_params:<12}|")
    print(f"| {'Training Time':<17}| {cnn_time:<12}| {mlp_time:<12}|")
    print(f"| {'Final Loss':<17}| {cnn_loss:<12}| {mlp_loss:<12}|")
    print(f"|{'-'*18}|{'-'*13}|{'-'*13}|")
    print()

    # Determine winner
    if cnn_history['final_test_acc'] > mlp_history['final_test_acc']:
        print("  Winner (Accuracy): CNN")
    elif mlp_history['final_test_acc'] > cnn_history['final_test_acc']:
        print("  Winner (Accuracy): MLP")
    else:
        print("  Winner (Accuracy): TIE")

    if cnn_history['num_parameters'] < mlp_history['num_parameters']:
        print("  More Efficient (Params): CNN")
    else:
        print("  More Efficient (Params): MLP")

    if cnn_history['training_time'] < mlp_history['training_time']:
        print("  Faster Training: CNN")
    else:
        print("  Faster Training: MLP")

    print()
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*   MNIST Model Comparison: CNN vs MLP                    *")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Both models use the SAME hyperparameters for a fair comparison
    NUM_EPOCHS = 10          # Number of training epochs
    LEARNING_RATE = 0.001    # Learning rate for Adam optimizer
    BATCH_SIZE = 64          # Batch size for training and evaluation

    # Determine the computation device (GPU if available, else CPU)
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n--- Loading MNIST Dataset ---")
    # Use standard transforms (no augmentation) for fair comparison
    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        augment=False  # Disable augmentation for fair comparison
    )

    # =========================================================================
    # CREATE MODELS
    # =========================================================================
    print("\n--- Creating Models ---")
    cnn_model = CNN()
    mlp_model = MLP()

    print(f"  CNN Parameters: {count_parameters(cnn_model):,}")
    print(f"  MLP Parameters: {count_parameters(mlp_model):,}")

    # =========================================================================
    # TRAIN BOTH MODELS
    # =========================================================================

    # Train the CNN
    cnn_history = train_model(
        model=cnn_model,
        model_name="CNN",
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # Train the MLP
    mlp_history = train_model(
        model=mlp_model,
        model_name="MLP",
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # =========================================================================
    # GENERATE COMPARISON OUTPUTS
    # =========================================================================
    print("\n\n--- Generating Comparison Plots ---")

    # Ensure the output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Plot 1: Accuracy curves
    plot_accuracy_comparison(
        cnn_history, mlp_history,
        save_path=os.path.join(config.OUTPUT_DIR, 'comparison_accuracy.png')
    )

    # Plot 2: Loss curves
    plot_loss_comparison(
        cnn_history, mlp_history,
        save_path=os.path.join(config.OUTPUT_DIR, 'comparison_loss.png')
    )

    # Plot 3: Summary bar chart
    plot_summary_comparison(
        cnn_history, mlp_history,
        save_path=os.path.join(config.OUTPUT_DIR, 'comparison_summary.png')
    )

    # =========================================================================
    # PRINT COMPARISON TABLE
    # =========================================================================
    print_comparison_table(cnn_history, mlp_history)

    print("\nAll comparison outputs saved to the 'outputs/' directory.")
    print("Done!")
