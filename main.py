"""
main.py - Entry Point for Handwritten Digit Recognition CNN
============================================================

This is the main command-line interface for the Handwritten Digit Recognition
project. It provides a unified way to train, evaluate, and predict with the
CNN model through simple command-line arguments.

Usage Examples:
    python main.py --mode train                          # Train the model
    python main.py --mode train --epochs 20 --lr 0.0005 # Train with custom settings
    python main.py --mode train --no-augment            # Train without augmentation
    python main.py --mode evaluate                       # Evaluate on test set
    python main.py --mode predict --image digit.png      # Predict a single image
    python main.py --mode demo                           # Demo on random test images
"""

import argparse
import sys


# =============================================================================
# WELCOME BANNER
# =============================================================================

def print_banner():
    """Print a welcome banner when the program starts."""
    banner = r"""
    ============================================================
    |   Handwritten Digit Recognition using CNN (PyTorch)      |
    |                                                          |
    |   Recognizes digits 0-9 from handwritten images          |
    |   Trained on the MNIST dataset (60,000 training images)  |
    |                                                          |
    |   Features: TensorBoard | Early Stopping | Augmentation  |
    |             Model Versioning | Config System              |
    |             LR Scheduling | Quantization                 |
    ============================================================
    """
    print(banner)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Handwritten Digit Recognition CNN - Train, evaluate, or predict digits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode train                    Train the model with default settings
  %(prog)s --mode train --epochs 20        Train for 20 epochs
  %(prog)s --mode train --no-augment       Train without data augmentation
  %(prog)s --mode evaluate                 Evaluate model on MNIST test set
  %(prog)s --mode predict --image img.png  Predict digit in an image file
  %(prog)s --mode demo                     Run demo on random test images
  %(prog)s --mode quantize                 Quantize model for edge deployment
  %(prog)s --mode quantize --quantize-mode static  Static quantization
        """
    )

    # Required: mode of operation
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'demo', 'quantize'],
        help="Mode: 'train', 'evaluate', 'predict', 'demo', or 'quantize'."
    )

    # Image path for predict mode
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help="Path to an image file (required for 'predict' mode)."
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Number of training epochs (default: from config.py)."
    )

    parser.add_argument(
        '--batch-size', type=int, default=None,
        help="Batch size for training (default: from config.py)."
    )

    parser.add_argument(
        '--lr', type=float, default=None,
        help="Learning rate (default: from config.py)."
    )

    # Data augmentation toggle
    parser.add_argument(
        '--no-augment', action='store_true',
        help="Disable data augmentation during training."
    )

    # Verbose mode for predict
    parser.add_argument(
        '--verbose', action='store_true',
        help="Show detailed output (all class probabilities in predict mode)."
    )

    # Model path override
    parser.add_argument(
        '--model-path', type=str, default=None,
        help="Path to a specific model file to use."
    )

    # Quantization options
    parser.add_argument(
        '--quantize-mode', type=str, default='dynamic',
        choices=['dynamic', 'static', 'qat'],
        help="Quantization method: 'dynamic', 'static', or 'qat' (default: dynamic)."
    )

    parser.add_argument(
        '--compare', action='store_true',
        help="Compare all quantization methods (for 'quantize' mode)."
    )

    return parser


# =============================================================================
# MODE HANDLERS
# =============================================================================

def run_training(args):
    """Handle the 'train' mode."""
    from train import train_model

    print(f"[MODE] Training the model...")
    if args.epochs:
        print(f"  Epochs:        {args.epochs}")
    if args.lr:
        print(f"  Learning rate: {args.lr}")
    if args.batch_size:
        print(f"  Batch size:    {args.batch_size}")
    if args.no_augment:
        print(f"  Augmentation:  DISABLED")
    print()

    # Determine augmentation setting
    augment = False if args.no_augment else None  # None = use config default

    history = train_model(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        augment=augment
    )

    # Print final results
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
    """Handle the 'evaluate' mode."""
    from evaluate import evaluate_model

    print("[MODE] Evaluating the trained model on the test set...")
    print()
    evaluate_model(model_path=args.model_path)


def run_prediction(args):
    """Handle the 'predict' mode."""
    from predict import predict_image, get_all_probabilities, load_model

    if args.image is None:
        print("[ERROR] You must provide an image path with --image when using 'predict' mode.")
        print("  Example: python main.py --mode predict --image path/to/digit.png")
        sys.exit(1)

    print(f"[MODE] Predicting digit in image: {args.image}")
    print()

    try:
        predicted_digit, confidence = predict_image(args.image, model_path=args.model_path)

        print("=" * 40)
        print("  PREDICTION RESULT")
        print("=" * 40)
        print(f"  Image:      {args.image}")
        print(f"  Predicted:  {predicted_digit}")
        print(f"  Confidence: {confidence:.2%}")
        print("=" * 40)

        # Show all probabilities if verbose
        if args.verbose:
            model, device = load_model(model_path=args.model_path)
            probs = get_all_probabilities(args.image, model, device)
            print("\n  All class probabilities:")
            for digit, prob in sorted(probs.items()):
                bar = "█" * int(prob * 40)
                print(f"    {digit}: {prob:.4f} {bar}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def run_demo(args):
    """Handle the 'demo' mode."""
    from predict import demo_predict_random_samples

    print("[MODE] Running demo with random test images...")
    print()
    demo_predict_random_samples(num_samples=5)


def run_quantize(args):
    """Handle the 'quantize' mode."""
    from quantize import (
        load_trained_model, get_test_loader, get_calibration_loader,
        get_train_loader, quantize_dynamic, quantize_static, quantize_qat,
        evaluate_accuracy, measure_inference_time, get_model_size,
        save_quantized_model, print_comparison_report, run_comparison
    )

    print("[MODE] Quantizing model for edge deployment...")
    print(f"  Method: {args.quantize_mode}")
    print()

    # Load model
    model = load_trained_model(model_type="cnn", model_path=args.model_path)
    test_loader = get_test_loader()

    if args.compare:
        run_comparison(model, "cnn", test_loader, qat_epochs=args.epochs or 3)
    else:
        from quantize import run_single
        run_single(model, "cnn", args.quantize_mode, test_loader,
                   qat_epochs=args.epochs or 3)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function: parse args and route to appropriate handler."""
    print_banner()

    parser = create_parser()
    args = parser.parse_args()

    mode_handlers = {
        'train': run_training,
        'evaluate': run_evaluation,
        'predict': run_prediction,
        'demo': run_demo,
        'quantize': run_quantize,
    }

    handler = mode_handlers[args.mode]
    handler(args)


if __name__ == '__main__':
    main()
