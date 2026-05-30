"""
Evaluation Module for Handwritten Digit Recognition (MNIST) using a CNN in PyTorch.

This script:
  1. Loads a trained CNN model from 'saved_models/mnist_cnn.pth'
  2. Evaluates it on the MNIST test set (10,000 images)
  3. Prints overall accuracy
  4. Generates a per-digit classification report (precision, recall, f1-score)
  5. Generates and displays a confusion matrix
  6. Shows how many digits were correctly vs incorrectly predicted

Requirements:
  pip install torch torchvision scikit-learn numpy matplotlib seaborn
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

# Import the CNN model from our model package
from model.cnn_model import CNN


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def evaluate_model():
    """
    Main evaluation function that loads the model, runs inference on the
    MNIST test set, and prints/saves evaluation metrics.
    """

    # --- Device configuration ---
    # Use GPU if available for faster inference, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("-" * 60)

    # --- Path to the saved model ---
    model_path = os.path.join('saved_models', 'mnist_cnn.pth')

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at '{model_path}'.")
        print("Please train the model first before running evaluation.")
        print("Expected path: saved_models/mnist_cnn.pth")
        return

    # --- Data preprocessing ---
    # The same transforms used during training must be applied during evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),                          # Convert PIL image to tensor [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))     # Normalize with MNIST mean and std
    ])

    # --- Load the MNIST test dataset ---
    print("Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root='./data/mnist',    # Directory to store/download the dataset
        train=False,            # Use the test split (10,000 images)
        download=True,          # Download if not already present
        transform=transform     # Apply preprocessing
    )

    # Create a DataLoader for batched evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,        # Process 1000 images at a time
        shuffle=False           # Keep order consistent for analysis
    )
    print(f"Test set size: {len(test_dataset)} images")
    print("-" * 60)

    # --- Load the trained model ---
    print(f"Loading trained model from '{model_path}'...")
    model = CNN().to(device)

    # Load the saved weights (map to the current device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode (disables dropout, batchnorm behaves differently)
    model.eval()
    print("Model loaded successfully!")
    print("-" * 60)

    # --- Run inference on the test set ---
    print("Evaluating model on test set...")

    all_predictions = []   # Store all predicted labels
    all_targets = []       # Store all true labels
    correct = 0            # Count of correct predictions
    total = 0              # Total number of samples

    # Disable gradient computation for faster inference (no backprop needed)
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the device (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: get model predictions
            outputs = model(images)

            # Get the predicted class (digit with highest score)
            _, predicted = torch.max(outputs, 1)

            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and targets for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Convert to numpy arrays for sklearn
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # --- Calculate and display overall accuracy ---
    accuracy = 100.0 * correct / total
    incorrect = total - correct

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total test samples:    {total}")
    print(f"Correctly predicted:   {correct}")
    print(f"Incorrectly predicted: {incorrect}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # --- Per-digit accuracy breakdown ---
    print("Per-digit prediction summary:")
    print("-" * 40)
    digit_names = [str(i) for i in range(10)]
    for digit in range(10):
        digit_mask = (all_targets == digit)
        digit_total = digit_mask.sum()
        digit_correct = ((all_predictions == digit) & digit_mask).sum()
        digit_incorrect = digit_total - digit_correct
        digit_acc = 100.0 * digit_correct / digit_total if digit_total > 0 else 0
        print(f"  Digit {digit}: {digit_correct:4d} correct, {digit_incorrect:4d} incorrect "
              f"(out of {digit_total:4d}) -> {digit_acc:.2f}% accuracy")
    print()

    # --- Classification report (precision, recall, f1-score per class) ---
    print("Classification Report (per-digit precision, recall, f1-score):")
    print("-" * 60)
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=[f"Digit {i}" for i in range(10)],
        digits=4  # Show 4 decimal places for precision
    )
    print(report)

    # --- Confusion Matrix ---
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(all_targets, all_predictions)

    # Print the confusion matrix as text
    print("         Predicted")
    print("          ", end="")
    for i in range(10):
        print(f"{i:5d}", end="")
    print()
    print("Actual")
    for i in range(10):
        print(f"  {i}:   ", end="")
        for j in range(10):
            print(f"{cm[i][j]:5d}", end="")
        print()
    print()

    # --- Save confusion matrix as a heatmap image ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in cells
        fmt='d',              # Integer format
        cmap='Blues',          # Color scheme
        xticklabels=digit_names,
        yticklabels=digit_names
    )
    plt.title('Confusion Matrix - MNIST Digit Classification')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()

    # Save the figure
    os.makedirs('outputs', exist_ok=True)
    confusion_matrix_path = os.path.join('outputs', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {confusion_matrix_path}")
    print("-" * 60)

    print("\nEvaluation complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_model()
