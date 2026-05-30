"""
Data Loading, Preprocessing, and Augmentation Module for Handwritten Digit Recognition.

This module handles downloading, transforming, and loading the MNIST dataset
using PyTorch's torchvision library. The MNIST dataset contains 70,000
grayscale images of handwritten digits (0-9), each 28x28 pixels.

- Training set: 60,000 images
- Test set: 10,000 images

Data Augmentation:
    This module also supports data augmentation, which artificially expands the
    training set by applying random transformations (rotation, shifting, scaling,
    and noise). This helps the model generalize better to new handwriting styles
    it has never seen before.
"""

import sys
import os

# Add the project root to the Python path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Import configuration values from our centralized config file
import config


# =============================================================================
# CUSTOM TRANSFORM CLASSES
# =============================================================================

class AddGaussianNoise:
    """
    Custom transform that adds random Gaussian noise to an image tensor.

    This simulates imperfections in real-world handwriting scans, such as:
    - Scanner noise
    - Paper texture
    - Slight ink variations

    Adding noise during training makes the model more robust to these
    imperfections when it encounters real-world images.

    Args:
        mean (float): Mean of the Gaussian noise distribution. Default is 0.0.
        std (float): Standard deviation of the noise. Default is 0.01.
            Higher values = more noise = more aggressive augmentation.
    """

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Add Gaussian noise to the input tensor.

        Args:
            tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Image tensor with added Gaussian noise.
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def get_transforms():
    """
    Define the STANDARD image transformations (no augmentation).

    These transforms are used for:
    - Test/validation data (we never augment test data)
    - Training data when augmentation is disabled

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


def get_augmented_transforms():
    """
    Define AUGMENTED image transformations for training data.

    Data augmentation applies random transformations to training images each time
    they are loaded. This means the model sees slightly different versions of each
    image every epoch, effectively expanding the training set and reducing overfitting.

    Augmentations applied (in order):
        1. RandomRotation: Rotates the image by a random angle within the
           specified degree range. Simulates tilted handwriting.
        2. RandomAffine: Applies random translation (shifting) and scaling.
           Simulates digits that are off-center or written at different sizes.
        3. ToTensor(): Converts the augmented PIL Image to a tensor.
        4. Normalize(): Standardizes pixel values using MNIST statistics.
        5. AddGaussianNoise: Adds slight random noise for robustness.

    Configuration values are imported from config.py for easy tuning.

    Returns:
        torchvision.transforms.Compose: A composed augmented transform pipeline.
    """
    augmented_transform = transforms.Compose([
        # Randomly rotate the image by up to ROTATION_DEGREES in either direction
        # This simulates handwriting that is slightly tilted
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),

        # Randomly shift and scale the image
        # translate: shift up to TRANSLATE_RANGE fraction of image size
        # scale: resize between SCALE_RANGE[0] and SCALE_RANGE[1] of original
        transforms.RandomAffine(
            degrees=0,  # No additional rotation (already handled above)
            translate=config.TRANSLATE_RANGE,
            scale=config.SCALE_RANGE
        ),

        # Convert the augmented PIL Image to a PyTorch tensor
        transforms.ToTensor(),

        # Normalize using MNIST dataset statistics
        transforms.Normalize((0.1307,), (0.3081,)),

        # Add slight Gaussian noise to simulate real-world imperfections
        AddGaussianNoise(mean=0.0, std=0.01),
    ])
    return augmented_transform


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_data_loaders(batch_size=config.BATCH_SIZE, augment=config.AUGMENTATION_ENABLED):
    """
    Create and return DataLoaders for the MNIST train and test datasets.

    This function downloads the MNIST dataset if it is not already present,
    applies preprocessing transforms, and wraps the datasets in DataLoader
    objects for efficient batching and shuffling during training.

    When augmentation is enabled, training data receives random transformations
    (rotation, shifting, scaling, noise) to improve model robustness. Test data
    NEVER receives augmentation to ensure consistent evaluation.

    Args:
        batch_size (int): Number of images per batch. Default is config.BATCH_SIZE.
            - Smaller batch sizes use less memory but may train slower.
            - Larger batch sizes are faster but require more memory.
        augment (bool): Whether to apply data augmentation to training data.
            Default is config.AUGMENTATION_ENABLED.
            - True: Apply augmented transforms to training data only.
            - False: Use standard transforms for both train and test data.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set
              (60,000 images, shuffled each epoch).
            - test_loader (DataLoader): DataLoader for the test set
              (10,000 images, not shuffled).
    """
    # Choose the appropriate transforms based on augmentation setting
    if augment:
        train_transform = get_augmented_transforms()
        print("Data augmentation: ENABLED")
        print(f"  - Random rotation: up to {config.ROTATION_DEGREES} degrees")
        print(f"  - Random translation: up to {config.TRANSLATE_RANGE}")
        print(f"  - Random scaling: {config.SCALE_RANGE}")
        print(f"  - Gaussian noise: enabled (std=0.01)")
    else:
        train_transform = get_transforms()
        print("Data augmentation: DISABLED")

    # Test data always uses standard transforms (no augmentation)
    # We want consistent, unmodified images for fair evaluation
    test_transform = get_transforms()

    # Download and load the training dataset
    # download=True will download the dataset if it's not found in the root directory
    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,         # Directory where data will be stored
        train=True,                   # Load the training split (60,000 images)
        download=True,                # Download if not already present
        transform=train_transform     # Apply training transforms (possibly augmented)
    )

    # Download and load the test dataset
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,         # Same root directory
        train=False,                  # Load the test split (10,000 images)
        download=True,                # Download if not already present
        transform=test_transform      # Always use standard transforms for testing
    )

    # Create DataLoader for the training set
    # shuffle=True randomizes the order of samples each epoch, which helps
    # prevent the model from memorizing the order of training examples
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,                         # Shuffle training data for better generalization
        num_workers=config.NUM_WORKERS        # Parallel data loading workers
    )

    # Create DataLoader for the test set
    # shuffle=False keeps the test data in a consistent order for reproducible evaluation
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,                        # No need to shuffle test data
        num_workers=config.NUM_WORKERS        # Parallel data loading workers
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

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


def visualize_augmentations(num_versions=8):
    """
    Visualize the same image with different random augmentations applied.

    This function demonstrates how data augmentation works by showing multiple
    augmented versions of the same original image side by side. Each version
    has different random rotations, shifts, and noise applied.

    This is useful for:
    - Understanding what augmentation looks like visually
    - Tuning augmentation parameters (too much? too little?)
    - Verifying that augmented images still look like valid digits

    Args:
        num_versions (int): Number of augmented versions to generate. Default is 8.
    """
    # Load one image from the MNIST dataset WITHOUT any transforms
    # We need the raw PIL image to apply our augmented transforms manually
    raw_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=None  # No transforms - we want the raw PIL image
    )

    # Get the first image and its label
    original_image, label = raw_dataset[0]

    # Get our augmented transform pipeline
    augmented_transform = get_augmented_transforms()

    # Also get standard transform for comparison
    standard_transform = get_transforms()

    # Create a figure: first image is original, rest are augmented versions
    total_images = 1 + num_versions  # Original + augmented versions
    num_cols = 4
    num_rows = (total_images + num_cols - 1) // num_cols  # Ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    axes = axes.flatten()

    # Show the original image (with standard transforms for consistent display)
    original_tensor = standard_transform(original_image)
    original_display = original_tensor.squeeze().numpy()
    original_display = original_display * 0.3081 + 0.1307  # Undo normalization

    axes[0].imshow(original_display, cmap='gray')
    axes[0].set_title(f"Original (Label: {label})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Show augmented versions of the same image
    for i in range(num_versions):
        # Apply augmented transforms - each call produces a different result
        # because the transforms are random
        augmented_tensor = augmented_transform(original_image)
        augmented_display = augmented_tensor.squeeze().numpy()
        augmented_display = augmented_display * 0.3081 + 0.1307  # Undo normalization

        axes[i + 1].imshow(augmented_display, cmap='gray')
        axes[i + 1].set_title(f"Augmented #{i + 1}", fontsize=11)
        axes[i + 1].axis('off')

    # Hide any unused subplot positions
    for i in range(total_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        f"Data Augmentation Examples (Digit: {label})\n"
        f"Rotation: +/-{config.ROTATION_DEGREES} deg | "
        f"Translate: {config.TRANSLATE_RANGE} | "
        f"Scale: {config.SCALE_RANGE} | Noise: std=0.01",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    # Save to the outputs directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, 'augmentation_examples.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Augmentation visualization saved to {save_path}")


# =============================================================================
# MAIN - Run this module directly to test data loading and visualize samples
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("MNIST Data Loading and Preprocessing")
    print("=" * 50)
    print()

    # Load the data with augmentation enabled (default from config)
    train_loader, test_loader = get_data_loaders()

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

    print()
    print("-" * 50)

    # Visualize augmentation effects
    print("Generating augmentation visualization...")
    visualize_augmentations(num_versions=8)
