"""
predict.py - Prediction & Inference CLI for Handwritten Digit Recognition
==========================================================================

This module provides functions and a command-line interface to predict
handwritten digits using a trained CNN model. It supports:
  - Predicting from an image file (PNG/JPG) via the CLI
  - Predicting from a PyTorch tensor (programmatic use)
  - A demo mode that predicts on random MNIST test set images
  - Verbose mode showing all class probabilities

Usage (Command Line):
    # Predict a single image:
    python predict.py --image test.png

    # Predict with verbose output (all class probabilities):
    python predict.py --image path/to/digit.png --verbose

    # Run demo mode with random MNIST test samples:
    python predict.py --demo --num-samples 10

    # Use a custom model path:
    python predict.py --image test.png --model-path saved_models/my_model.pth

Usage (Programmatic):
    from predict import predict_image, predict_from_tensor, get_all_probabilities
    digit, confidence = predict_image("my_digit.png")
    probabilities = get_all_probabilities("my_digit.png", model, device)
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Import the CNN model architecture from our model package
from model.cnn_model import CNN

# Import configuration defaults from config.py
import config

# Import ModelManager for smart model loading
from utils.model_manager import ModelManager


# =============================================================================
# Configuration
# =============================================================================

# Path to the saved trained model weights (from config)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          config.MODEL_DIR, f"{config.MODEL_NAME_PREFIX}.pth")

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

def load_model(model_path=None, device=None):
    """
    Load the trained CNN model from disk using ModelManager.

    The loading strategy is:
      1. If a specific model_path is given, load from that exact path.
      2. Otherwise, try to load the "best" model via ModelManager.
      3. If no "best" model exists, fall back to the "latest" version.
      4. If neither exists, raise a clear error.

    Args:
        model_path (str, optional): Path to a specific saved model weights file.
            If None, uses ModelManager to find the best or latest model.
        device (torch.device, optional): Device to load the model onto.
            If None, automatically selects GPU if available.

    Returns:
        tuple: (model, device) - The loaded model and the device it's on.

    Raises:
        FileNotFoundError: If no model file can be found.
    """
    if device is None:
        device = get_device()

    # Create model instance
    model = CNN().to(device)

    # If a specific model path is provided, load directly from it
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please check the path and try again."
            )
        # Try loading as a full checkpoint first (ModelManager format)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fall back to loading as a raw state_dict
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"[INFO] Model loaded successfully from: {model_path}")
        return model, device

    # No specific path given -- use ModelManager to find best or latest model
    manager = ModelManager()

    # Strategy 1: Try to load the "best" model
    try:
        epoch, metrics = manager.load_model(model, version="best")
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        pass  # No best model found, try latest

    # Strategy 2: Try to load the "latest" versioned model
    try:
        epoch, metrics = manager.load_model(model, version="latest")
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        pass  # No latest model found either

    # Strategy 3: Fall back to the legacy MODEL_PATH (mnist_cnn.pth)
    legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "saved_models", "mnist_cnn.pth")
    if os.path.exists(legacy_path):
        state_dict = torch.load(legacy_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.eval()
        print(f"[INFO] Model loaded successfully from: {legacy_path}")
        return model, device

    # No model found anywhere -- provide a clear error message
    raise FileNotFoundError(
        "No trained model found!\n"
        "  Looked for:\n"
        f"    - Best model: {config.MODEL_DIR}/{config.MODEL_NAME_PREFIX}_best.pth\n"
        f"    - Latest versioned model in: {config.MODEL_DIR}/\n"
        f"    - Legacy model: saved_models/mnist_cnn.pth\n\n"
        "  Please train the model first by running:\n"
        "    python train.py"
    )


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
# Get All Probabilities
# =============================================================================

def get_all_probabilities(image_path_or_tensor, model, device):
    """
    Get the probability for each digit class (0-9).

    This is useful for detailed analysis of model predictions, such as
    understanding which digits the model considers likely.

    Args:
        image_path_or_tensor: Either a file path (str) to an image, or a
            PyTorch tensor (already preprocessed and normalized).
        model (torch.nn.Module): The loaded CNN model.
        device (torch.device): The device to run inference on.

    Returns:
        dict: A dictionary mapping each digit (0-9) to its probability (float).
              Probabilities sum to 1.0 across all classes.

    Example:
        >>> probs = get_all_probabilities("digit.png", model, device)
        >>> for digit, prob in sorted(probs.items(), key=lambda x: -x[1]):
        ...     print(f"  Digit {digit}: {prob:.4f}")
    """
    # Determine if input is a file path or a tensor
    if isinstance(image_path_or_tensor, str):
        # It's a file path -- load and preprocess the image
        if not os.path.exists(image_path_or_tensor):
            raise FileNotFoundError(f"Image not found: {image_path_or_tensor}")
        image = Image.open(image_path_or_tensor)
        tensor = inference_transform(image)
    else:
        # It's already a tensor
        tensor = image_path_or_tensor

    # Ensure tensor has correct dimensions for the model
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Move tensor to the correct device
    tensor = tensor.to(device)

    # Run inference and get softmax probabilities
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)

    # Convert to a dictionary mapping digit -> probability
    probs_array = probabilities.squeeze().cpu().numpy()
    prob_dict = {digit: float(probs_array[digit]) for digit in range(10)}

    return prob_dict


# =============================================================================
# Demo Function
# =============================================================================

def demo_predict_random_samples(num_samples=5, no_display=False):
    """
    Demo: Predict on random images from the MNIST test set.

    Downloads the MNIST test set (if not already cached), selects random
    samples, predicts the digit, and displays each image alongside the
    prediction result.

    Args:
        num_samples (int): Number of random test images to predict on.
        no_display (bool): If True, don't try to show the matplotlib window.
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

    # Optionally show the plot
    if not no_display:
        try:
            plt.show()
        except Exception:
            pass  # Silently handle display errors in headless environments

    plt.close()
    print(f"[INFO] Results saved to: {output_path}")


# =============================================================================
# CLI Helper: Save Prediction Result Image
# =============================================================================

def _save_prediction_result(image_path, predicted_digit, confidence, prob_dict=None):
    """
    Save a visualization of the prediction result to outputs/prediction_result.png.

    Creates an image showing:
      - The input image
      - The predicted digit and confidence
      - (Optional) A bar chart of all class probabilities

    Args:
        image_path (str): Path to the original input image.
        predicted_digit (int): The model's predicted digit.
        confidence (float): Confidence score for the prediction.
        prob_dict (dict, optional): All class probabilities (for verbose mode).
    """
    os.makedirs("outputs", exist_ok=True)

    if prob_dict is not None:
        # Create a figure with the image and a probability bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        # Just show the image with the prediction
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))

    # Display the input image
    image = Image.open(image_path).convert("L")
    ax1.imshow(image, cmap="gray")
    ax1.set_title(
        f"Predicted: {predicted_digit}\nConfidence: {confidence:.2%}",
        fontsize=12, fontweight="bold"
    )
    ax1.axis("off")

    # If verbose mode, add a bar chart of all probabilities
    if prob_dict is not None:
        digits = list(range(10))
        probs = [prob_dict[d] for d in digits]

        # Color the bars: green for the predicted digit, blue for others
        colors = ['green' if d == predicted_digit else 'steelblue' for d in digits]

        ax2.bar(digits, probs, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel("Digit Class", fontsize=10)
        ax2.set_ylabel("Probability", fontsize=10)
        ax2.set_title("Class Probabilities", fontsize=12, fontweight="bold")
        ax2.set_xticks(digits)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join("outputs", "prediction_result.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


# =============================================================================
# Argument Parser
# =============================================================================

def build_argument_parser():
    """
    Build and return the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Handwritten Digit Recognition CNN - Prediction & Inference Tool\n"
            "================================================================\n\n"
            "Predict handwritten digits (0-9) from image files using a trained\n"
            "CNN model. Supports single image prediction, batch demo mode,\n"
            "and verbose probability output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Predict a digit from an image file:\n"
            "  python predict.py --image test.png\n\n"
            "  # Predict with detailed probability output:\n"
            "  python predict.py --image digit.png --verbose\n\n"
            "  # Run demo mode with 10 random MNIST test samples:\n"
            "  python predict.py --demo --num-samples 10\n\n"
            "  # Use a specific model file:\n"
            "  python predict.py --image test.png --model-path saved_models/my_model.pth\n\n"
            "  # Run in headless environment (no display window):\n"
            "  python predict.py --demo --no-display\n"
        )
    )

    # --- Prediction mode arguments ---
    mode_group = parser.add_argument_group("Prediction Mode")
    mode_group.add_argument(
        "--image",
        type=str,
        metavar="PATH",
        help="Path to an image file (PNG, JPG, BMP, etc.) containing a handwritten digit."
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo mode: predict on random images from the MNIST test set."
    )

    # --- Options ---
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--num-samples",
        type=int,
        default=5,
        metavar="N",
        help="Number of random samples for demo mode (default: 5)."
    )
    options_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom model weights file (.pth). "
            "If not specified, automatically loads the best available model."
        )
    )
    options_group.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed output including probabilities for all 10 digit classes."
    )
    options_group.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Don't try to show matplotlib window (for headless/server environments)."
    )

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main function: parse CLI arguments and run the appropriate prediction mode.

    This is called when the script is executed directly:
        python predict.py --image test.png
        python predict.py --demo
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    # If no arguments provided, show help and exit
    if not args.image and not args.demo:
        parser.print_help()
        print("\n[INFO] No arguments provided. Use --image or --demo to get started.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Demo Mode: Predict on random MNIST test samples
    # -------------------------------------------------------------------------
    if args.demo:
        demo_predict_random_samples(
            num_samples=args.num_samples,
            no_display=args.no_display
        )
        return

    # -------------------------------------------------------------------------
    # Image Mode: Predict a single image file
    # -------------------------------------------------------------------------
    if args.image:
        # Validate the image file exists
        if not os.path.exists(args.image):
            print(f"[ERROR] Image file not found: {args.image}")
            print("  Please check the file path and try again.")
            sys.exit(1)

        # Load the model
        print("=" * 60)
        print("  HANDWRITTEN DIGIT RECOGNITION - PREDICTION")
        print("=" * 60)
        print()

        try:
            model, device = load_model(model_path=args.model_path)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        print()

        # Run prediction
        print(f"[INFO] Predicting digit from: {args.image}")
        predicted_digit, confidence = predict_image(
            args.image, model=model, device=device
        )

        # Display the main result
        print()
        print("-" * 40)
        print(f"  Predicted Digit:  {predicted_digit}")
        print(f"  Confidence:       {confidence:.2%}")
        print("-" * 40)

        # Verbose mode: show all class probabilities
        prob_dict = None
        if args.verbose:
            prob_dict = get_all_probabilities(args.image, model, device)
            print()
            print("  All Class Probabilities:")
            print("  " + "-" * 36)
            # Sort by probability descending for readability
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            for digit, prob in sorted_probs:
                # Create a simple text bar chart
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                marker = " <-- predicted" if digit == predicted_digit else ""
                print(f"    Digit {digit}: [{bar}] {prob:.4f}{marker}")
            print("  " + "-" * 36)

        # Save result image
        output_path = _save_prediction_result(
            args.image, predicted_digit, confidence, prob_dict
        )
        print(f"\n[INFO] Result image saved to: {output_path}")
        print()


if __name__ == "__main__":
    """
    Run the prediction CLI when this script is executed directly.

    Usage:
        python predict.py --image test.png
        python predict.py --image test.png --verbose
        python predict.py --demo --num-samples 10
        python predict.py --help
    """
    main()
