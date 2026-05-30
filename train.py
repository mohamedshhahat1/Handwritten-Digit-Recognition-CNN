"""
Training Pipeline for Handwritten Digit Recognition using CNN

This script trains a Convolutional Neural Network (CNN) on the MNIST dataset
to recognize handwritten digits (0-9). It handles:
- Loading data using custom data loaders
- Training the model with GPU support (if available)
- Tracking training/validation metrics (loss and accuracy)
- Saving the trained model weights for later use

Author: Handwritten Digit Recognition Project
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import our custom CNN model and data loaders
from model.cnn_model import CNN
from data.data_loader import get_data_loaders


def train_model(num_epochs=10, learning_rate=0.001, batch_size=64):
    """
    Main training function that handles the complete training pipeline.

    Args:
        num_epochs (int): Number of complete passes through the training dataset.
                          Default is 10 epochs.
        learning_rate (float): Step size for the optimizer. Controls how much
                               the model weights are updated each step. Default: 0.001.
        batch_size (int): Number of samples processed before updating model weights.
                          Default: 64.

    Returns:
        dict: Training history containing loss and accuracy per epoch.
    """

    # =========================================================================
    # STEP 1: Set up the device (GPU if available, otherwise CPU)
    # =========================================================================
    # Using a GPU significantly speeds up training for neural networks.
    # torch.device allows us to move our model and data to the appropriate hardware.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)

    # =========================================================================
    # STEP 2: Load the MNIST dataset
    # =========================================================================
    # The data loader handles downloading, preprocessing, and batching the data.
    # It returns separate loaders for training and testing/validation.
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("=" * 60)

    # =========================================================================
    # STEP 3: Initialize the CNN model
    # =========================================================================
    # Create an instance of our CNN and move it to the selected device (GPU/CPU).
    model = CNN().to(device)
    print("Model architecture:")
    print(model)
    print("=" * 60)

    # =========================================================================
    # STEP 4: Define the loss function and optimizer
    # =========================================================================
    # CrossEntropyLoss is the standard loss function for multi-class classification.
    # It combines LogSoftmax and NLLLoss in one single class.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer adapts the learning rate for each parameter individually.
    # It generally works well out-of-the-box for most deep learning tasks.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # =========================================================================
    # STEP 5: Training loop
    # =========================================================================
    # Dictionary to store training history for later plotting/analysis
    history = {
        'train_loss': [],       # Average training loss per epoch
        'train_accuracy': [],   # Training accuracy per epoch (percentage)
        'val_accuracy': []      # Validation/test accuracy per epoch (percentage)
    }

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ----- Training Phase -----
        # Set the model to training mode (enables dropout, batch norm updates, etc.)
        model.train()

        # Variables to track metrics within this epoch
        running_loss = 0.0      # Accumulates loss over all batches
        correct = 0             # Number of correct predictions
        total = 0               # Total number of samples processed

        # Progress bar for visual feedback during training
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            leave=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to the device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: compute model predictions
            outputs = model(images)

            # Calculate the loss between predictions and true labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            optimizer.zero_grad()   # Clear previous gradients
            loss.backward()         # Compute gradients via backpropagation

            # Update model weights based on gradients
            optimizer.step()

            # Track metrics
            running_loss += loss.item()

            # Get predicted class (highest probability)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })

        # Calculate epoch-level metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        # ----- Validation/Test Phase -----
        # Evaluate the model on the test set after each epoch
        val_accuracy = evaluate_model(model, test_loader, device)

        # Store metrics in history
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['val_accuracy'].append(val_accuracy)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary: "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}%")
        print("-" * 60)

    print("\nTraining complete!")

    # =========================================================================
    # STEP 6: Save the trained model
    # =========================================================================
    save_model(model)

    # =========================================================================
    # STEP 7: Save training history for plotting
    # =========================================================================
    save_history(history)

    return history


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a dataset (validation/test set).

    Args:
        model: The trained CNN model.
        data_loader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on (CPU/GPU).

    Returns:
        float: Accuracy as a percentage (0-100).
    """
    # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for efficiency during evaluation
    # (we don't need gradients when we're not training)
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only (no backward pass needed)
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy percentage
    accuracy = 100.0 * correct / total
    return accuracy


def save_model(model, path='saved_models/mnist_cnn.pth'):
    """
    Save the trained model weights to disk.

    Args:
        model: The trained CNN model.
        path (str): File path where the model weights will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save only the model's state dictionary (weights and biases)
    # This is the recommended approach as it's more flexible than saving the entire model
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to '{path}'")


def save_history(history, path='saved_models/training_history.json'):
    """
    Save training history to a JSON file for later plotting/analysis.

    Args:
        history (dict): Dictionary containing training metrics per epoch.
        path (str): File path where the history will be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save as JSON for easy loading later
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to '{path}'")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# This block runs only when the script is executed directly (not imported)
if __name__ == '__main__':
    # You can modify these hyperparameters to experiment with training
    NUM_EPOCHS = 10         # Number of training epochs
    LEARNING_RATE = 0.001   # Adam optimizer learning rate
    BATCH_SIZE = 64         # Number of samples per training batch

    # Start the training pipeline
    history = train_model(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )

    # Print final results summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(history['val_accuracy']):.2f}% "
          f"(Epoch {history['val_accuracy'].index(max(history['val_accuracy'])) + 1})")
    print("=" * 60)
