"""
predict.py - Prediction Module for Handwritten Digit Recognition

This module provides functions to predict handwritten digits using a trained
CNN model. It supports:
  - Predicting from an image file (PNG/JPG)
  - Predicting from a PyTorch tensor (programmatic use)
  - A demo mode that predicts on random MNIST test set images

Usage:
    # From command line (runs demo):
    python predict.py

    # From another script:
    from predict import predict_image, predict_from_tensor
    digit, confidence = predict_image("my_digit.png")
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Import the CNN model architecture from our model package
from model.cnn_model import CNN


# =============================================================================
# Configuration
# =============================================================================

# Path to the saved trained model weights
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "saved_models", "mnist_cnn.pth")

# MNIST normalization values (mean and standard deviation)
# These are the standard values used during training
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Image transformation pipeline for inference
# Converts any input image to the format expected by the model
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),                   # Resize to 28x28 pixels
    transforms.ToTensor(),                         # Convert to tensor [0, 1]
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  # Normalize
])


# =============================================================================
# Device Setup
# =============================================================================

def get_device():
    """
    Determine the best available device (GPU or CPU).

    Returns:
        torch.device: The device to use for inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU for inference")
    return device


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path=MODEL_PATH, device=None):
    """
    Load the trained CNN model from disk.

    Args:
        model_path (str): Path to the saved model weights (.pth file).
        device (torch.device, optional): Device to load the model onto.
            If None, automatically selects GPU if available.

    Returns:
        tuple: (model, device) - The loaded model and the device it's on.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if device is None:
        device = get_device()

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please train the model first by running: python train.py"
        )

    # Create model instance and load trained weights
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    print(f"[INFO] Model loaded successfully from: {model_path}")
    return model, device


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_image(image_path, model=None, device=None):
    """
    Predict the digit in a single image file.

    This function loads an image from disk, preprocesses it (convert to
    grayscale, resize to 28x28, normalize), and runs it through the model.

    Args:
        image_path (str): Path to the image file (PNG, JPG, etc.).
        model (nn.Module, optional): Pre-loaded model. If None, loads from disk.
        device (torch.device, optional): Device for inference.

    Returns:
        tuple: (predicted_digit, confidence)
            - predicted_digit (int): The predicted digit (0-9).
            - confidence (float): Confidence score (0.0 to 1.0).

    Raises:
        FileNotFoundError: If the image file does not exist.

    Example:
        >>> digit, confidence = predict_image("test_digit.png")
        >>> print(f"Predicted: {digit} (confidence: {confidence:.2%})")
    """
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load model if not provided
    if model is None:
        model, device = load_model(device=device)

    if device is None:
        device = next(model.parameters()).device

    # Load and preprocess the image
    image = Image.open(image_path)
    tensor = inference_transform(image)

    # Add batch dimension: (1, 28, 28) -> (1, 1, 28, 28)
    tensor = tensor.unsqueeze(0).to(device)

    # Run prediction (no gradient computation needed for inference)
    with torch.no_grad():
        output = model(tensor)
        # Apply softmax to get probability distribution
        probabilities = F.softmax(output, dim=1)
        # Get the highest probability and its index (the predicted digit)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_digit = predicted.item()
    confidence_score = confidence.item()

    return predicted_digit, confidence_score


def predict_from_tensor(tensor, model=None, device=None):
    """
    Predict the digit from a preprocessed PyTorch tensor.

    This function is useful for programmatic use when you already have
    the image data as a tensor (e.g., from a data loader or camera feed).

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, 28, 28) or
            (batch_size, 1, 28, 28). Should be normalized with MNIST stats.
        model (nn.Module, optional): Pre-loaded model. If None, loads from disk.
        device (torch.device, optional): Device for inference.

    Returns:
        tuple: (predicted_digit, confidence)
            - predicted_digit (int): The predicted digit (0-9).
              If batch input, returns the prediction for the first image.
            - confidence (float): Confidence score (0.0 to 1.0).

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize((0.1307,), (0.3081,))
        ... ])
        >>> # Assume 'img' is a 28x28 grayscale PIL image
        >>> tensor = transform(img)
        >>> digit, confidence = predict_from_tensor(tensor)
    """
    # Load model if not provided
    if model is None:
        model, device = load_model(device=device)

    if device is None:
        device = next(model.parameters()).device

    # Ensure tensor has correct dimensions
    if tensor.dim() == 3:
        # Single image: (1, 28, 28) -> (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        # Raw 2D image: (28, 28) -> (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Move tensor to the correct device
    tensor = tensor.to(device)

    # Run prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_digit = predicted.item()
    confidence_score = confidence.item()

    return predicted_digit, confidence_score


# =============================================================================
# Demo Function
# =============================================================================

def demo_predict_random_samples(num_samples=5):
    """
    Demo: Predict on random images from the MNIST test set.

    Downloads the MNIST test set (if not already cached), selects random
    samples, predicts the digit, and displays each image alongside the
    prediction result.

    Args:
        num_samples (int): Number of random test images to predict on.
    """
    print("=" * 60)
    print("  HANDWRITTEN DIGIT RECOGNITION - DEMO")
    print("=" * 60)
    print()

    # Load the trained model
    try:
        model, device = load_model()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nTo run this demo, you need a trained model.")
        print("Train one by running: python train.py")
        return

    print()

    # Load MNIST test dataset
    print("[INFO] Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root="./data/mnist",
        train=False,           # Use test set
        download=True,         # Download if not present
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])
    )

    # Select random indices from the test set
    total_samples = len(test_dataset)
    random_indices = random.sample(range(total_samples), num_samples)

    print(f"[INFO] Predicting on {num_samples} random test images...\n")

    # Create a figure to display results
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 4))

    # Handle case where num_samples == 1 (axes won't be a list)
    if num_samples == 1:
        axes = [axes]

    correct_count = 0

    for i, idx in enumerate(random_indices):
        # Get the image tensor and true label
        image_tensor, true_label = test_dataset[idx]

        # Predict using the tensor
        predicted_digit, confidence = predict_from_tensor(
            image_tensor, model=model, device=device
        )

        # Check if prediction is correct
        is_correct = predicted_digit == true_label
        if is_correct:
            correct_count += 1

        # Print result to console
        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Sample {i+1}: True={true_label}, "
              f"Predicted={predicted_digit}, "
              f"Confidence={confidence:.2%} [{status}]")

        # Display the image
        # Denormalize for visualization
        img_display = image_tensor.squeeze().numpy()
        img_display = img_display * MNIST_STD + MNIST_MEAN  # Undo normalization
        img_display = np.clip(img_display, 0, 1)

        axes[i].imshow(img_display, cmap="gray")
        axes[i].set_title(
            f"Pred: {predicted_digit}\n"
            f"True: {true_label}\n"
            f"Conf: {confidence:.1%}",
            fontsize=10,
            color="green" if is_correct else "red"
        )
        axes[i].axis("off")

    # Print summary
    print(f"\n  Accuracy on these samples: {correct_count}/{num_samples} "
          f"({correct_count/num_samples:.0%})")
    print()

    # Finalize and save the plot
    plt.suptitle("Handwritten Digit Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save the figure
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "prediction_demo.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Results saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Run the prediction demo when this script is executed directly.

    Usage:
        python predict.py

    This will:
        1. Load the trained model from saved_models/mnist_cnn.pth
        2. Download MNIST test data (if needed)
        3. Predict on 5 random test images
        4. Display results with images and confidence scores
    """
    demo_predict_random_samples(num_samples=5)
