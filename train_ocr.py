"""
OCR Training Pipeline
======================

Trains the CRNN model for handwritten text recognition using CTC loss.
Uses synthetic data generation (no external dataset download needed).

Usage:
    python train_ocr.py
    python train_ocr.py --epochs 30 --batch-size 32

The trained model is saved to saved_models/ocr_crnn_best.pth
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.crnn_model import CRNN
from model.ocr_utils import OCRCharset, ctc_decode_batch, ctc_collate_fn, compute_cer
from model.ocr_dataset import SyntheticOCRDataset


def train_ocr(epochs=20, batch_size=32, learning_rate=0.001, num_samples=5000):
    """
    Train the CRNN OCR model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Initial learning rate.
        num_samples (int): Number of synthetic samples per epoch.
    """
    print("=" * 60)
    print("  OCR TRAINING — CRNN (CNN + BiLSTM + CTC)")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Character set
    charset = OCRCharset()
    print(f"Character set: {charset}")
    print(f"Total classes (with blank): {charset.num_classes}")

    # Create model
    model = CRNN(
        num_classes=charset.num_classes,
        hidden_size=256,
        num_layers=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Datasets
    print(f"\nGenerating {num_samples} synthetic training samples...")
    train_dataset = SyntheticOCRDataset(
        charset=charset,
        num_samples=num_samples,
        img_height=32,
        max_text_len=20,
    )

    val_dataset = SyntheticOCRDataset(
        charset=charset,
        num_samples=num_samples // 5,
        img_height=32,
        max_text_len=20,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_cer = float('inf')
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (images, labels, label_lengths, input_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)  # (seq_len, batch, num_classes)

            # CTC loss requires: log_probs, targets, input_lengths, target_lengths
            loss = criterion(outputs, labels, input_lengths, label_lengths)

            # Skip if loss is inf/nan (can happen with CTC on bad samples)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        # --- Validation ---
        model.eval()
        total_cer = 0.0
        num_val_samples = 0
        sample_predictions = []

        with torch.no_grad():
            for images, labels, label_lengths, input_lengths in val_loader:
                images = images.to(device)

                outputs = model(images)
                predicted_texts = ctc_decode_batch(outputs, charset)

                # Decode ground truth labels
                offset = 0
                for i, length in enumerate(label_lengths):
                    gt_indices = labels[offset:offset + length].tolist()
                    gt_text = charset.decode(gt_indices)
                    pred_text = predicted_texts[i]

                    cer = compute_cer(pred_text, gt_text)
                    total_cer += cer
                    num_val_samples += 1

                    # Save a few samples for display
                    if len(sample_predictions) < 3:
                        sample_predictions.append((gt_text, pred_text))

                    offset += length

        avg_cer = total_cer / max(num_val_samples, 1)

        # Learning rate scheduling
        scheduler.step(avg_cer)

        # Print epoch results
        print(f"Epoch [{epoch}/{epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"CER: {avg_cer:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Show sample predictions
        if epoch % 5 == 1 or epoch == epochs:
            print("  Samples:")
            for gt, pred in sample_predictions:
                print(f"    GT:   '{gt}'")
                print(f"    Pred: '{pred}'")
                print()

        # Save best model
        if avg_cer < best_cer:
            best_cer = avg_cer
            save_path = os.path.join(save_dir, "ocr_crnn_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'charset_size': charset.num_classes,
                'hidden_size': 256,
                'num_layers': 2,
                'epoch': epoch,
                'cer': avg_cer,
                'characters': charset.characters,
            }, save_path)
            print(f"  ✓ Best model saved (CER: {avg_cer:.4f})")

        print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best CER: {best_cer:.4f}")
    print(f"  Model saved: {os.path.join(save_dir, 'ocr_crnn_best.pth')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR CRNN model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples", type=int, default=5000, help="Training samples per epoch")
    args = parser.parse_args()

    train_ocr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_samples=args.samples,
    )
