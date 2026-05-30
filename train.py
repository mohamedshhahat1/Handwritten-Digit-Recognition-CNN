"""
Training Pipeline for Handwritten Digit Recognition using CNN
=============================================================

This script trains a Convolutional Neural Network (CNN) on the MNIST dataset
to recognize handwritten digits (0-9). It integrates:
- Centralized configuration from config.py
- Data augmentation for improved robustness
- TensorBoard logging for live training monitoring
- Model versioning with timestamped checkpoints
- Early stopping to prevent overfitting
- GPU support for accelerated training

Usage:
    python train.py
    python train.py --epochs 20 --lr 0.0005 --batch-size 128
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import project modules
import config
from model.cnn_model import CNN
from data.data_loader import get_data_loaders
from utils.logger import TBLogger
from utils.model_manager import ModelManager
from utils.early_stopping import EarlyStopping


def train_model(num_epochs=None, learning_rate=None, batch_size=None, augment=None):
    """
    Main training function that handles the complete training pipeline.

    All parameters default to values from config.py if not explicitly provided.
    This allows both programmatic use with custom settings and CLI use with defaults.

    Args:
        num_epochs (int, optional): Number of training epochs. Default: config.EPOCHS.
        learning_rate (float, optional): Learning rate for optimizer. Default: config.LEARNING_RATE.
        batch_size (int, optional): Samples per batch. Default: config.BATCH_SIZE.
        augment (bool, optional): Whether to use data augmentation. Default: config.AUGMENTATION_ENABLED.

    Returns:
        dict: Training history containing loss and accuracy per epoch.
    """

    # Use config defaults for any parameters not explicitly provided
    num_epochs = num_epochs if num_epochs is not None else config.EPOCHS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    augment = augment if augment is not None else config.AUGMENTATION_ENABLED

    # =========================================================================
    # STEP 1: Display configuration
    # =========================================================================
    print("=" * 60)
    print("  HANDWRITTEN DIGIT RECOGNITION - TRAINING")
    print("=" * 60)
    config.print_config()

    # =========================================================================
    # STEP 2: Set up device (GPU if available, otherwise CPU)
    # =========================================================================
    device = config.DEVICE
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # =========================================================================
    # STEP 3: Load the MNIST dataset (with optional augmentation)
    # =========================================================================
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size, augment=augment)
    print("=" * 60)

    # =========================================================================
    # STEP 4: Initialize the CNN model
    # =========================================================================
    model = CNN().to(device)
    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # =========================================================================
    # STEP 5: Define loss function and optimizer
    # =========================================================================
    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler (cosine annealing with warm restarts)
    scheduler = None
    if config.LR_SCHEDULER_ENABLED:
        if config.LR_SCHEDULER_TYPE == "cosine_warm_restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.LR_T0,
                T_mult=config.LR_T_MULT,
                eta_min=config.LR_ETA_MIN
            )
            print(f"\nLR Scheduler: CosineAnnealingWarmRestarts "
                  f"(T0={config.LR_T0}, T_mult={config.LR_T_MULT}, "
                  f"eta_min={config.LR_ETA_MIN})")
        elif config.LR_SCHEDULER_TYPE == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=config.LR_ETA_MIN
            )
            print(f"\nLR Scheduler: CosineAnnealingLR "
                  f"(T_max={num_epochs}, eta_min={config.LR_ETA_MIN})")
        elif config.LR_SCHEDULER_TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )
            print(f"\nLR Scheduler: StepLR (step_size=5, gamma=0.5)")
        else:
            print(f"\nWARNING: Unknown scheduler type '{config.LR_SCHEDULER_TYPE}', "
                  f"using constant LR")
    else:
        print("\nLR Scheduler: Disabled (constant learning rate)")

    # =========================================================================
    # STEP 6: Initialize utilities
    # =========================================================================
    # TensorBoard logger for live monitoring
    logger = TBLogger(log_dir=config.LOG_DIR)

    # Log the model graph to TensorBoard
    sample_input = torch.randn(1, 1, 28, 28).to(device)
    logger.log_model_graph(model, sample_input)

    # Model manager for versioned saving
    model_manager = ModelManager()

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode='max',  # Monitor validation accuracy (higher is better)
        verbose=True
    )

    # =========================================================================
    # STEP 7: Training loop
    # =========================================================================
    # Dictionary to store training history for plotting
    history = {
        'train_loss': [],       # Average training loss per epoch
        'train_accuracy': [],   # Training accuracy per epoch (percentage)
        'val_accuracy': []      # Validation/test accuracy per epoch (percentage)
    }

    best_val_acc = 0.0  # Track best validation accuracy for model saving

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ----- Training Phase -----
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            leave=True
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log per-batch loss to TensorBoard (at configured interval)
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                logger.log_training_step(
                    loss.item(), batch_idx, epoch, len(train_loader)
                )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })

        # Calculate epoch-level metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        # ----- Validation Phase -----
        val_accuracy = evaluate_model(model, test_loader, device)

        # Store metrics in history
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['val_accuracy'].append(val_accuracy)

        # ----- TensorBoard Logging -----
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch_metrics(epoch, epoch_loss, epoch_accuracy, val_accuracy, current_lr)

        # Log sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Get a batch for prediction logging
            sample_images, sample_labels = next(iter(test_loader))
            sample_images = sample_images.to(device)
            with torch.no_grad():
                sample_outputs = model(sample_images)
                _, sample_preds = torch.max(sample_outputs, 1)
            logger.log_predictions(
                sample_images, sample_preds, sample_labels.to(device), epoch
            )

        # ----- Model Saving -----
        metrics = {
            'train_loss': epoch_loss,
            'train_acc': epoch_accuracy,
            'val_acc': val_accuracy
        }

        # Save versioned checkpoint
        model_manager.save_model(model, optimizer, epoch + 1, metrics)

        # Save best model if validation accuracy improved
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model_manager.save_best_model(model, optimizer, epoch + 1, metrics)
            print(f"  ★ New best model! Val Acc: {val_accuracy:.2f}%")

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary: "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 60)

        # ----- Learning Rate Scheduler Step -----
        if scheduler is not None:
            scheduler.step()

        # ----- Early Stopping Check -----
        if early_stopping(val_accuracy, epoch + 1):
            print(f"\n⚡ Early stopping triggered at epoch {epoch + 1}!")
            print(f"   Best validation accuracy: {early_stopping.best_score:.2f}% "
                  f"(epoch {early_stopping.best_epoch})")
            break

    # =========================================================================
    # STEP 8: Training complete - cleanup and save
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Close TensorBoard logger
    logger.close()

    # Also save as the legacy model path for backward compatibility
    legacy_path = os.path.join(config.MODEL_DIR, 'mnist_cnn.pth')
    torch.save(model.state_dict(), legacy_path)
    print(f"Legacy model saved to: {legacy_path}")

    # Save training history
    save_history(history)

    # Clean up old model versions (keep last 5)
    model_manager.delete_old_versions(keep_n=5)

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
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def save_history(history, path=None):
    """
    Save training history to a JSON file for later plotting/analysis.

    Args:
        history (dict): Dictionary containing training metrics per epoch.
        path (str, optional): File path. Default: saved_models/training_history.json.
    """
    if path is None:
        path = os.path.join(config.MODEL_DIR, 'training_history.json')

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to '{path}'")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train the Handwritten Digit Recognition CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                           # Train with default config settings
  python train.py --epochs 20               # Train for 20 epochs
  python train.py --lr 0.0005 --batch-size 128  # Custom hyperparameters
  python train.py --no-augment              # Disable data augmentation
        """
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {config.EPOCHS})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {config.LEARNING_RATE})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Determine augmentation setting
    augment = not args.no_augment if args.no_augment else None

    # Start the training pipeline
    history = train_model(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        augment=augment
    )

    # Print final results summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Loss:       {history['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy:   {history['train_accuracy'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2f}%")
    print(f"Best Validation Accuracy:  {max(history['val_accuracy']):.2f}% "
          f"(Epoch {history['val_accuracy'].index(max(history['val_accuracy'])) + 1})")
    print("=" * 60)
    print(f"\nTo view training logs in TensorBoard:")
    print(f"  tensorboard --logdir={config.LOG_DIR}")
