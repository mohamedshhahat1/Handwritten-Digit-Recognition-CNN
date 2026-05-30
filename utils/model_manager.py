"""
Model Versioning and Management System
=======================================

This module provides a complete system for saving, loading, and managing
different versions of trained CNN models. It handles:

- Saving model checkpoints with automatic version naming
- Loading specific versions or the best/latest model
- Listing all saved versions with their performance metrics
- Cleaning up old versions to save disk space

Usage:
    from utils.model_manager import ModelManager

    # Create a manager instance
    manager = ModelManager()

    # Save a model after training
    manager.save_model(model, optimizer, epoch=5, metrics={
        'train_loss': 0.05,
        'train_acc': 98.5,
        'val_acc': 99.1
    })

    # Load the best model later
    epoch, metrics = manager.load_model(model, optimizer, version="best")

    # See all saved versions
    manager.list_versions()

    # Clean up old versions, keeping only the 5 most recent
    manager.delete_old_versions(keep_n=5)
"""

import os
import sys
import json
import glob
import torch
from datetime import datetime

# Add the project root to the Python path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_DIR, MODEL_NAME_PREFIX


class ModelManager:
    """
    Manages saving, loading, and versioning of trained PyTorch models.

    This class provides a structured way to keep track of multiple model
    versions, making it easy to:
    - Go back to a previous version if a new training run performs worse
    - Compare performance across different training sessions
    - Automatically identify and quick-load the best performing model
    - Keep disk usage under control by pruning old versions

    Attributes:
        model_dir (str): Directory where model files are stored.
        model_prefix (str): Prefix used in model filenames for identification.
    """

    def __init__(self, model_dir=MODEL_DIR, model_prefix=MODEL_NAME_PREFIX):
        """
        Initialize the ModelManager.

        Args:
            model_dir (str): Directory to store saved model files.
                             Defaults to MODEL_DIR from config.py ("./saved_models").
            model_prefix (str): Prefix for model filenames.
                                Defaults to MODEL_NAME_PREFIX from config.py ("mnist_cnn").

        Example:
            # Use default settings from config.py
            manager = ModelManager()

            # Or specify custom directory and prefix
            manager = ModelManager(model_dir="./my_models", model_prefix="digit_model")
        """
        self.model_dir = model_dir
        self.model_prefix = model_prefix

        # Create the model directory if it does not exist yet
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, optimizer, epoch, metrics, version=None):
        """
        Save a model checkpoint with versioning information.

        A checkpoint contains everything needed to resume training or run inference:
        - The model weights (state_dict)
        - The optimizer state (for resuming training)
        - The epoch number (to know where training left off)
        - Performance metrics (to compare versions)
        - A version string and timestamp (for identification)

        Args:
            model (torch.nn.Module): The trained model to save.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
            epoch (int): The current epoch number.
            metrics (dict): Dictionary with keys like 'train_loss', 'train_acc', 'val_acc'.
            version (str, optional): A custom version string. If None, one is
                                     auto-generated using the epoch and current timestamp.

        Returns:
            str: The full file path where the model was saved.

        Example:
            metrics = {'train_loss': 0.03, 'train_acc': 99.1, 'val_acc': 99.3}
            path = manager.save_model(model, optimizer, epoch=10, metrics=metrics)
            # Saves as: ./saved_models/mnist_cnn_v10_20260530_143022.pth
        """
        # Auto-generate version string if none provided
        if version is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{epoch}_{timestamp_str}"

        # Build the checkpoint dictionary with all information needed to
        # restore the model and understand its performance
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'version': version,
            'timestamp': datetime.now().isoformat(),
        }

        # Construct the filename and full save path
        filename = f"{self.model_prefix}_{version}.pth"
        save_path = os.path.join(self.model_dir, filename)

        # Save the checkpoint to disk
        torch.save(checkpoint, save_path)

        print(f"[ModelManager] Model saved successfully!")
        print(f"  Version:  {version}")
        print(f"  Epoch:    {epoch}")
        print(f"  Metrics:  {metrics}")
        print(f"  Path:     {save_path}")

        return save_path

    def save_best_model(self, model, optimizer, epoch, metrics):
        """
        Save the current model as the "best" model.

        This overwrites the previous best model file and updates the
        metadata JSON file with information about why this model is the best.

        Use this when validation accuracy improves during training.

        Args:
            model (torch.nn.Module): The best-performing model to save.
            optimizer (torch.optim.Optimizer): The optimizer state.
            epoch (int): The epoch at which the best performance was achieved.
            metrics (dict): Performance metrics (should include 'val_acc').

        Returns:
            str: The full file path where the best model was saved.

        Example:
            if current_val_acc > best_val_acc:
                manager.save_best_model(model, optimizer, epoch=7, metrics={
                    'train_loss': 0.02, 'train_acc': 99.5, 'val_acc': 99.6
                })
        """
        # Build the checkpoint (same structure as save_model)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'version': 'best',
            'timestamp': datetime.now().isoformat(),
        }

        # Save the model checkpoint as the designated "best" file
        best_path = os.path.join(self.model_dir, f"{self.model_prefix}_best.pth")
        torch.save(checkpoint, best_path)

        # Also save a human-readable JSON file with metadata about the best model.
        # This makes it easy to check the best model's stats without loading the
        # full checkpoint (which can be large).
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_file': f"{self.model_prefix}_best.pth",
        }
        metadata_path = os.path.join(self.model_dir, "best_model_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[ModelManager] Best model updated!")
        print(f"  Epoch:    {epoch}")
        print(f"  Val Acc:  {metrics.get('val_acc', 'N/A')}")
        print(f"  Path:     {best_path}")
        print(f"  Metadata: {metadata_path}")

        return best_path

    def load_model(self, model, optimizer=None, version="best"):
        """
        Load a saved model checkpoint.

        Supports three loading modes:
        - "best": Load the best-performing model (most common for inference)
        - "latest": Load the most recently saved version (useful for resuming training)
        - A specific version string: Load that exact version

        Args:
            model (torch.nn.Module): The model architecture to load weights into.
                                     Must have the same structure as when it was saved.
            optimizer (torch.optim.Optimizer, optional): If provided, restores the
                                                         optimizer state for continued training.
            version (str): Which version to load. Options:
                          - "best" (default): The best performing model
                          - "latest": The most recently saved model
                          - A version string like "v10_20260530_143022"

        Returns:
            tuple: (epoch, metrics) from the loaded checkpoint.

        Raises:
            FileNotFoundError: If the requested model file does not exist.

        Example:
            # Load the best model for inference
            epoch, metrics = manager.load_model(model, version="best")

            # Resume training from the latest checkpoint
            epoch, metrics = manager.load_model(model, optimizer, version="latest")
        """
        # Determine which file to load based on the version argument
        if version == "best":
            # Load the designated best model file
            load_path = os.path.join(self.model_dir, f"{self.model_prefix}_best.pth")

        elif version == "latest":
            # Find the most recently modified .pth file (excluding the best model)
            pattern = os.path.join(self.model_dir, f"{self.model_prefix}_v*.pth")
            model_files = glob.glob(pattern)

            if not model_files:
                raise FileNotFoundError(
                    f"[ModelManager] No versioned model files found in '{self.model_dir}'. "
                    f"Train and save a model first!"
                )

            # Sort by modification time, newest first
            model_files.sort(key=os.path.getmtime, reverse=True)
            load_path = model_files[0]

        else:
            # Load a specific version by its version string
            load_path = os.path.join(self.model_dir, f"{self.model_prefix}_{version}.pth")

        # Check that the file actually exists before trying to load
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"[ModelManager] Model file not found: '{load_path}'\n"
                f"  Available versions can be listed with manager.list_versions()"
            )

        # Load the checkpoint from disk
        # map_location ensures it works even if saved on GPU but loaded on CPU
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

        # Restore the model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optionally restore the optimizer state (needed for resuming training)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Extract metadata from the checkpoint
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        print(f"[ModelManager] Model loaded successfully!")
        print(f"  Version:  {checkpoint.get('version', 'unknown')}")
        print(f"  Epoch:    {epoch}")
        print(f"  Metrics:  {metrics}")
        print(f"  From:     {load_path}")

        return epoch, metrics

    def list_versions(self):
        """
        List all saved model versions with their performance metrics.

        Scans the model directory for all checkpoint files and displays a
        formatted table showing key information about each version.

        Returns:
            list: A list of dictionaries, each containing:
                  - 'version': The version string
                  - 'epoch': Training epoch
                  - 'train_acc': Training accuracy
                  - 'val_acc': Validation accuracy
                  - 'file_size': File size in megabytes
                  - 'path': Full file path

        Example:
            versions = manager.list_versions()
            # Prints a nice table and returns the data for programmatic use
        """
        # Find all model files matching our prefix pattern
        pattern = os.path.join(self.model_dir, f"{self.model_prefix}_*.pth")
        model_files = glob.glob(pattern)

        if not model_files:
            print(f"[ModelManager] No saved models found in '{self.model_dir}'.")
            return []

        # Sort files by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)

        # Collect information about each version
        versions_info = []

        for filepath in model_files:
            try:
                # Load only the metadata (not the full model weights into GPU)
                checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

                # Extract relevant information
                version = checkpoint.get('version', 'unknown')
                epoch = checkpoint.get('epoch', 0)
                metrics = checkpoint.get('metrics', {})
                train_acc = metrics.get('train_acc', 0.0)
                val_acc = metrics.get('val_acc', 0.0)

                # Get the file size in megabytes
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

                versions_info.append({
                    'version': version,
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'file_size': round(file_size_mb, 2),
                    'path': filepath,
                })
            except Exception as e:
                # If a file is corrupted or incompatible, skip it gracefully
                print(f"  [Warning] Could not read '{filepath}': {e}")

        # Print a nicely formatted table
        print("\n" + "=" * 78)
        print(f"{'MODEL VERSIONS':^78}")
        print("=" * 78)
        print(f"{'Version':<28} {'Epoch':<7} {'Train Acc':<11} {'Val Acc':<11} {'Size (MB)':<10}")
        print("-" * 78)

        for info in versions_info:
            print(
                f"{info['version']:<28} "
                f"{info['epoch']:<7} "
                f"{info['train_acc']:<11.2f} "
                f"{info['val_acc']:<11.2f} "
                f"{info['file_size']:<10.2f}"
            )

        print("-" * 78)
        print(f"Total: {len(versions_info)} version(s) found")
        print("=" * 78 + "\n")

        return versions_info

    def delete_old_versions(self, keep_n=5):
        """
        Delete old model versions to free up disk space.

        Keeps the N most recent versioned checkpoints plus the best model.
        All other versioned checkpoints are deleted.

        Args:
            keep_n (int): Number of most recent versions to keep. Defaults to 5.

        Example:
            # Keep only the 3 most recent versions (+ best model)
            manager.delete_old_versions(keep_n=3)
        """
        # Find all versioned model files (those with "v" in the name pattern)
        pattern = os.path.join(self.model_dir, f"{self.model_prefix}_v*.pth")
        model_files = glob.glob(pattern)

        if not model_files:
            print(f"[ModelManager] No versioned models to clean up.")
            return

        # Sort by modification time, newest first
        model_files.sort(key=os.path.getmtime, reverse=True)

        # Identify files to keep (the N most recent) and files to delete
        files_to_keep = model_files[:keep_n]
        files_to_delete = model_files[keep_n:]

        if not files_to_delete:
            print(f"[ModelManager] Nothing to delete. Only {len(model_files)} "
                  f"version(s) exist (keep_n={keep_n}).")
            return

        # Delete the old files
        print(f"[ModelManager] Cleaning up old model versions...")
        print(f"  Keeping {keep_n} most recent + best model")
        print(f"  Deleting {len(files_to_delete)} old version(s):\n")

        total_freed = 0
        for filepath in files_to_delete:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            filename = os.path.basename(filepath)
            os.remove(filepath)
            total_freed += file_size_mb
            print(f"    Deleted: {filename} ({file_size_mb:.2f} MB)")

        print(f"\n  Total space freed: {total_freed:.2f} MB")
        print(f"  Remaining versions: {len(files_to_keep)}")


# =============================================================================
# Standalone usage example
# =============================================================================
if __name__ == "__main__":
    # Quick demonstration of how ModelManager works
    print("ModelManager - Model Versioning System")
    print("=" * 40)

    # Create a manager instance with default settings
    manager = ModelManager()

    # List any existing saved models
    print("\nLooking for existing model versions...")
    manager.list_versions()

    print("\nModelManager is ready to use!")
    print("See the module docstring for usage examples.")
