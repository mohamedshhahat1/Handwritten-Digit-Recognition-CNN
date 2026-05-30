"""
Evaluation Module for Handwritten Digit Recognition (MNIST) using a CNN in PyTorch.
===================================================================================

This script:
  1. Loads a trained CNN model (best version via ModelManager)
  2. Evaluates it on the MNIST test set (10,000 images)
  3. Prints overall accuracy
  4. Generates a per-digit classification report (precision, recall, f1-score)
  5. Generates and displays a confusion matrix
  6. Shows how many digits were correctly vs incorrectly predicted

Usage:
    python evaluate.py
    python evaluate.py --model-path saved_models/mnist_cnn_best.pth
"""

import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
import config
from model.cnn_model import CNN
from utils.model_manager import ModelManager


def evaluate_model(model_path=None):
    """
    Main evaluation function that loads the model, runs inference on the
    MNIST test set, and prints/saves evaluation metrics.

    Args:
        model_path (str, optional): Path to a specific model file to evaluate.
            If None, loads the best model via ModelManager.
    """

    # --- Device configuration ---
    device = config.DEVICE
    print(f"Using device: {device}")
    print("-" * 60)

    # --- Load the model ---
    model = CNN().to(device)

    if model_path and os.path.exists(model_path):
        # Load from specific path
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {model_path}")
    else:
        # Use ModelManager to find best or latest model
        manager = ModelManager()
        try:
            epoch, metrics = manager.load_model(model, version="best")
        except FileNotFoundError:
            try:
                epoch, metrics = manager.load_model(model, version="latest")
            except FileNotFoundError:
                # Fall back to legacy path
                legacy_path = os.path.join(config.MODEL_DIR, 'mnist_cnn.pth')
                if os.path.exists(legacy_path):
                    model.load_state_dict(torch.load(legacy_path, map_location=device))
                    print(f"Model loaded from legacy path: {legacy_path}")
                else:
                    print("ERROR: No trained model found!")
                    print("Please train the model first by running: python train.py")
                    return

    model.eval()
    print("-" * 60)

    # --- Data preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --- Load the MNIST test dataset ---
    print("Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    print(f"Test set size: {len(test_dataset)} images")
    print("-" * 60)

    # --- Run inference on the test set ---
    print("Evaluating model on test set...")

    all_predictions = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

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

    # --- Classification report ---
    print("Classification Report (per-digit precision, recall, f1-score):")
    print("-" * 60)
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=[f"Digit {i}" for i in range(10)],
        digits=4
    )
    print(report)

    # --- Confusion Matrix ---
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(all_targets, all_predictions)

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

    # --- Save confusion matrix as a heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=digit_names,
        yticklabels=digit_names
    )
    plt.title('Confusion Matrix - MNIST Digit Classification')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    confusion_matrix_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {confusion_matrix_path}")
    print("-" * 60)
    print("\nEvaluation complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN model")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a specific model file to evaluate')
    args = parser.parse_args()
    evaluate_model(model_path=args.model_path)
