"""
Data Loading and Preprocessing Module for Handwritten Digit Recognition.

This module handles downloading, transforming, and loading the MNIST dataset
using PyTorch's torchvision library. The MNIST dataset contains 70,000
grayscale images of handwritten digits (0-9), each 28x28 pixels.

- Training set: 60,000 images
- Test set: 10,000 images
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_transforms():
    """
    Define the image transformations to apply to the MNIST dataset.

    Transforms applied:
        1. ToTensor(): Converts PIL Image (0-255) to a PyTorch tensor (0.0-1.0)
        2. Normalize((0.1307,), (0.3081,)): Normalizes the tensor using the
           global mean (0.1307) and standard deviation (0.3081) of the MNIST
           dataset. This helps the model train faster and more reliably.

    Returns:
        torchvision.transforms.Compose: A composed transform pipeline.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor with values in [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
    ])
    return transform


def get_data_loaders(batch_size=64):
    """
    Create and return DataLoaders for the MNIST train and test datasets.

    This function downloads the MNIST dataset if it is not already present,
    applies preprocessing transforms, and wraps the datasets in DataLoader
    objects for efficient batching and shuffling during training.

    Args:
        batch_size (int): Number of images per batch. Default is 64.
            - Smaller batch sizes use less memory but may train slower.
            - Larger batch sizes are faster but require more memory.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set
              (60,000 images, shuffled each epoch).
            - test_loader (DataLoader): DataLoader for the test set
              (10,000 images, not shuffled).
    """
    # Get the preprocessing transforms
    transform = get_transforms()

    # Download and load the training dataset
    # download=True will download the dataset if it's not found in the root directory
    train_dataset = datasets.MNIST(
        root='./data/mnist',      # Directory where data will be stored
        train=True,               # Load the training split (60,000 images)
        download=True,            # Download if not already present
        transform=transform       # Apply our preprocessing transforms
    )

    # Download and load the test dataset
    test_dataset = datasets.MNIST(
        root='./data/mnist',      # Same root directory
        train=False,              # Load the test split (10,000 images)
        download=True,            # Download if not already present
        transform=transform       # Apply the same preprocessing transforms
    )

    # Create DataLoader for the training set
    # shuffle=True randomizes the order of samples each epoch, which helps
    # prevent the model from memorizing the order of training examples
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True              # Shuffle training data for better generalization
    )

    # Create DataLoader for the test set
    # shuffle=False keeps the test data in a consistent order for reproducible evaluation
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False             # No need to shuffle test data
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


def visualize_samples(data_loader, num_samples=12):
    """
    Visualize a grid of sample images from the dataset.

    This function is useful for verifying that the data is loaded correctly
    and for getting an intuitive feel for what the model will be learning from.

    Args:
        data_loader (DataLoader): A DataLoader containing MNIST images and labels.
        num_samples (int): Number of sample images to display. Default is 12.
            Will be arranged in a grid with 4 columns.
    """
    # Get one batch of images and labels from the data loader
    images, labels = next(iter(data_loader))

    # Limit to the requested number of samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Calculate grid dimensions (4 columns, as many rows as needed)
    num_cols = 4
    num_rows = (num_samples + num_cols - 1) // num_cols  # Ceiling division

    # Create a figure with subplots arranged in a grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    for i in range(num_samples):
        # Convert tensor back to a displayable format
        # squeeze() removes the channel dimension (1, 28, 28) -> (28, 28)
        # We also undo the normalization for display purposes
        image = images[i].squeeze().numpy()
        image = image * 0.3081 + 0.1307  # Undo normalization for display

        # Display the image in grayscale
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}", fontsize=12)
        axes[i].axis('off')  # Hide axis ticks and labels

    # Hide any unused subplot positions
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Sample MNIST Digits", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./data/sample_digits.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Sample visualization saved to ./data/sample_digits.png")


# Run this module directly to test data loading and visualize samples
if __name__ == "__main__":
    print("=" * 50)
    print("MNIST Data Loading and Preprocessing")
    print("=" * 50)
    print()

    # Load the data with default batch size
    train_loader, test_loader = get_data_loaders(batch_size=64)

    print()
    print("-" * 50)

    # Show the shape of one batch to understand the data dimensions
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"  -> (batch_size, channels, height, width)")
    print(f"Batch label tensor shape: {labels.shape}")
    print(f"  -> (batch_size,)")
    print(f"Image pixel value range: [{images.min():.4f}, {images.max():.4f}]")

    print()
    print("-" * 50)

    # Visualize some sample images
    print("Generating sample visualization...")
    visualize_samples(train_loader, num_samples=12)
