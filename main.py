"""
main.py - Entry Point for Handwritten Digit Recognition CNN

This is the main command-line interface for the Handwritten Digit Recognition
project. It provides a unified way to train, evaluate, and predict with the
CNN model through simple command-line arguments.

Usage Examples:
    python main.py --mode train                          # Train the model
    python main.py --mode train --epochs 20 --lr 0.0005 # Train with custom settings
    python main.py --mode evaluate                       # Evaluate on test set
    python main.py --mode predict --image digit.png      # Predict a single image
    python main.py --mode demo                           # Demo on random test images

Author: Handwritten Digit Recognition Project
"""

import argparse
import sys

# =============================================================================
# WELCOME BANNER
# =============================================================================

def print_banner():
    """
    Print a welcome banner when the program starts.
    This gives the user a clear indication of what application is running.
    """
    banner = r"""
    ============================================================
    |   Handwritten Digit Recognition using CNN (PyTorch)      |
    |                                                          |
    |   Recognizes digits 0-9 from handwritten images          |
    |   Trained on the MNIST dataset (60,000 training images)  |
    ============================================================
    """
    print(banner)


# =============================================================================
# ARGUMENT PARSER SETUP
# =============================================================================

def create_parser():
    """
    Create and configure the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: Configured parser with all arguments defined.
    """
    parser = argparse.ArgumentParser(
        description="Handwritten Digit Recognition CNN - Train, evaluate, or predict digits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode train                    Train the model with default settings
  %(prog)s --mode train --epochs 20        Train for 20 epochs
  %(prog)s --mode evaluate                 Evaluate model on MNIST test set
  %(prog)s --mode predict --image img.png  Predict digit in an image file
  %(prog)s --mode demo                     Run demo on random test images
        """
    )

    # -------------------------------------------------------------------------
    # Required argument: mode of operation
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'demo'],
        help=(
            "Mode of operation: "
            "'train' to train the model, "
            "'evaluate' to test on the full test set, "
            "'predict' to classify a single image, "
            "'demo' to see predictions on random test images."
        )
    )

    # -------------------------------------------------------------------------
    # Optional argument: image path (required only for 'predict' mode)
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help="Path to an image file for prediction (required when mode is 'predict')."
    )

    # -------------------------------------------------------------------------
    # Optional training hyperparameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of training epochs (default: 10). Only used in 'train' mode."
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="Batch size for training (default: 64). Only used in 'train' mode."
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001). Only used in 'train' mode."
    )

    return parser


# =============================================================================
# MODE HANDLERS
# =============================================================================

def run_training(args):
    """
    Handle the 'train' mode: trains the CNN model on the MNIST dataset.

    Args:
        args: Parsed command-line arguments containing epochs, batch_size, and lr.
    """
    # Import the training function from train.py
    from train import train_model

    print(f"[MODE] Training the model...")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    # Run the training pipeline with the specified hyperparameters
    history = train_model(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    # Print final results summary after training completes
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Final Training Loss:       {history['train_loss'][-1]:.4f}")
    print(f"  Final Training Accuracy:   {history['train_accuracy'][-1]:.2f}%")
    print(f"  Final Validation Accuracy: {history['val_accuracy'][-1]:.2f}%")
    best_val = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val) + 1
    print(f"  Best Validation Accuracy:  {best_val:.2f}% (Epoch {best_epoch})")
    print("=" * 60)


def run_evaluation(args):
    """
    Handle the 'evaluate' mode: evaluates the trained model on the MNIST test set.

    This generates a full classification report with per-digit metrics,
    a confusion matrix, and overall accuracy statistics.

    Args:
        args: Parsed command-line arguments (not used for evaluation, but kept
              for consistent interface).
    """
    # Import the evaluation function from evaluate.py
    from evaluate import evaluate_model

    print("[MODE] Evaluating the trained model on the test set...")
    print()

    # Run the evaluation pipeline
    evaluate_model()


def run_prediction(args):
    """
    Handle the 'predict' mode: predicts the digit in a single image file.

    Args:
        args: Parsed command-line arguments containing the image path.
    """
    # Import the prediction function from predict.py
    from predict import predict_image

    # Validate that an image path was provided
    if args.image is None:
        print("[ERROR] You must provide an image path with --image when using 'predict' mode.")
        print("  Example: python main.py --mode predict --image path/to/digit.png")
        sys.exit(1)

    print(f"[MODE] Predicting digit in image: {args.image}")
    print()

    # Run the prediction
    try:
        predicted_digit, confidence = predict_image(args.image)

        # Display the result
        print("=" * 40)
        print("  PREDICTION RESULT")
        print("=" * 40)
        print(f"  Image:      {args.image}")
        print(f"  Predicted:  {predicted_digit}")
        print(f"  Confidence: {confidence:.2%}")
        print("=" * 40)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def run_demo(args):
    """
    Handle the 'demo' mode: shows predictions on random MNIST test images.

    This is a great way to quickly see how the model performs without
    needing to provide your own images.

    Args:
        args: Parsed command-line arguments (not used for demo, but kept
              for consistent interface).
    """
    # Import the demo function from predict.py
    from predict import demo_predict_random_samples

    print("[MODE] Running demo with random test images...")
    print()

    # Run the demo (predicts on 5 random test images by default)
    demo_predict_random_samples(num_samples=5)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function that orchestrates the program flow:
    1. Prints the welcome banner
    2. Parses command-line arguments
    3. Routes to the appropriate mode handler
    """
    # Print the welcome banner so users know what program they are running
    print_banner()

    # Create the argument parser and parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Route to the appropriate handler based on the selected mode
    # Using a dictionary as a dispatcher for clean, extensible code
    mode_handlers = {
        'train': run_training,
        'evaluate': run_evaluation,
        'predict': run_prediction,
        'demo': run_demo,
    }

    # Get the handler for the selected mode and execute it
    handler = mode_handlers[args.mode]
    handler(args)


# This block ensures main() only runs when the script is executed directly,
# not when it is imported as a module by another script.
if __name__ == '__main__':
    main()
