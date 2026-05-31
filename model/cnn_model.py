"""
CNN Models for Handwritten Digit Recognition (MNIST)

This module defines CNN architectures for classifying handwritten digits (0-9)
from 28x28 grayscale images:

1. CNN — Standard 2-layer CNN with batch normalization (original architecture)
2. ResNetCNN — Deeper architecture with ResNet-style skip connections

ResNet (Residual Networks) Key Insight:
    In very deep networks, gradients can vanish or explode, making training
    difficult. Skip connections (shortcuts) solve this by allowing gradients
    to flow directly through the network via identity mappings:

        output = F.relu(layer(x) + x)   ← skip connection adds input to output

    This means the network only needs to learn the RESIDUAL (the difference
    between input and desired output), which is easier than learning the full
    transformation from scratch. Benefits:
    - Enables training of much deeper networks (100+ layers)
    - Faster convergence due to better gradient flow
    - No extra parameters for identity shortcuts
    - Acts as an implicit ensemble of shallow networks

Batch Normalization:
    BatchNorm normalizes activations between layers, which provides:
    - Faster training convergence (allows higher learning rates)
    - Reduced sensitivity to weight initialization
    - Acts as a mild regularizer (reduces need for dropout)
    - Stabilizes the distribution of layer inputs (reduces internal covariate shift)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import config for model hyperparameters
import config


class CNN(nn.Module):
    """
    Convolutional Neural Network for digit classification.

    This network takes 28x28 grayscale images as input and outputs
    predictions for 10 digit classes (0 through 9).

    Architecture Summary:
        Input (1x28x28)
        -> Conv2d (32 filters) -> BatchNorm2d -> ReLU -> MaxPool2d
        -> Conv2d (64 filters) -> BatchNorm2d -> ReLU -> MaxPool2d
        -> Flatten -> FC (128 units) -> BatchNorm1d -> ReLU -> Dropout
        -> FC (10 units, output)
    """

    def __init__(self):
        """
        Initialize the CNN layers.

        The network consists of:
        - Two convolutional blocks (conv -> batchnorm -> activation -> pooling)
        - Two fully connected (dense) layers with batch normalization and dropout
        """
        super(CNN, self).__init__()

        # ============================================================
        # CONVOLUTIONAL LAYER 1
        # ============================================================
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        # ============================================================
        # CONVOLUTIONAL LAYER 2
        # ============================================================
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # ============================================================
        # MAX POOLING, FC LAYERS, DROPOUT
        # ============================================================
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
        self.fc2 = nn.Linear(128, config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass: Conv1 -> BN -> ReLU -> Pool -> Conv2 -> BN -> ReLU -> Pool -> FC.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):
    """
    A single residual block with skip connection.

    Implements the fundamental building block of ResNet:

        input ──┬── Conv -> BN -> ReLU -> Conv -> BN ──┬── (+) -> ReLU -> output
                │                                       │
                └───────── shortcut (identity or 1x1) ──┘

    When the input and output have the same dimensions, the skip connection
    is a simple identity (no parameters). When dimensions change (e.g.,
    doubling channels or halving spatial size), a 1x1 convolution adapts
    the shortcut to match.

    Args:
        in_channels (int): Number of input feature map channels.
        out_channels (int): Number of output feature map channels.
        stride (int): Stride for the first convolution (use 2 to downsample).
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # --- Main path (two convolutions with batch norm) ---
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Shortcut connection ---
        # If dimensions change (channel count or spatial size), use a 1x1 conv
        # to project the input to the correct dimensions for addition.
        # Otherwise, the shortcut is just the identity (no-op).
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass with skip connection.

        The input is added to the output of the conv layers BEFORE the
        final ReLU activation. This is the "pre-activation" variant that
        allows unimpeded gradient flow through the skip path.

        Args:
            x (torch.Tensor): Input feature maps.

        Returns:
            torch.Tensor: Output feature maps (same spatial size if stride=1,
                         halved if stride=2).
        """
        # Main path: Conv -> BN -> ReLU -> Conv -> BN
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection: add the (possibly projected) input
        out += self.shortcut(x)

        # Final activation after addition
        out = F.relu(out)

        return out


# =============================================================================
# RESNET CNN — DEEPER ARCHITECTURE WITH SKIP CONNECTIONS
# =============================================================================

class ResNetCNN(nn.Module):
    """
    ResNet-style CNN for digit classification.

    A deeper architecture (14 conv layers) that uses residual skip connections
    to enable effective training despite increased depth. Designed for MNIST
    (28x28 grayscale images) with appropriate sizing.

    Architecture Summary:
        Input (1x28x28)
        -> Initial Conv(1→32, 3x3) -> BN -> ReLU              [32x28x28]
        -> ResBlock(32→32) x2                                  [32x28x28]
        -> ResBlock(32→64, stride=2) + ResBlock(64→64)         [64x14x14]
        -> ResBlock(64→128, stride=2) + ResBlock(128→128)      [128x7x7]
        -> Global Average Pooling                              [128]
        -> FC(128→10)                                          [10]

    Key Differences from Standard CNN:
        - 6 residual blocks (12 conv layers) vs 2 plain conv layers
        - Skip connections prevent vanishing gradients in deeper network
        - Global average pooling instead of flatten (fewer parameters)
        - No dropout needed (ResNet + BN provides sufficient regularization)

    Total Parameters: ~295K (actually FEWER than the plain CNN's 422K,
    because global average pooling eliminates the large FC(3136→128) layer)
    """

    def __init__(self):
        """
        Initialize the ResNet CNN layers.

        Structure:
        - Initial convolution to expand from 1 channel to 32
        - 3 stages of 2 residual blocks each (32, 64, 128 channels)
        - Global average pooling to collapse spatial dimensions
        - Single fully connected layer for classification
        """
        super(ResNetCNN, self).__init__()

        # ============================================================
        # INITIAL CONVOLUTION
        # ============================================================
        # Expands the single grayscale channel to 32 feature maps.
        # This is NOT a residual block — just a standard conv to set up
        # the channel dimension for the residual blocks that follow.
        self.conv_initial = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(32)

        # ============================================================
        # STAGE 1: 32 channels, 28x28 spatial (no downsampling)
        # ============================================================
        # Two residual blocks at 32 channels.
        # Input and output dimensions match → identity shortcuts.
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )

        # ============================================================
        # STAGE 2: 64 channels, 14x14 spatial (downsample 2x)
        # ============================================================
        # First block uses stride=2 to halve spatial dimensions.
        # Also doubles channels: 32 → 64 (requires 1x1 shortcut).
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),   # 32→64, 28x28→14x14
            ResidualBlock(64, 64, stride=1),   # 64→64, 14x14→14x14
        )

        # ============================================================
        # STAGE 3: 128 channels, 7x7 spatial (downsample 2x)
        # ============================================================
        # First block uses stride=2 to halve spatial dimensions again.
        # Doubles channels: 64 → 128.
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # 64→128, 14x14→7x7
            ResidualBlock(128, 128, stride=1), # 128→128, 7x7→7x7
        )

        # ============================================================
        # GLOBAL AVERAGE POOLING
        # ============================================================
        # Collapses each 7x7 feature map into a single value by averaging.
        # This is more parameter-efficient than flattening (128 vs 128*7*7=6272).
        # Also provides translation invariance and reduces overfitting.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ============================================================
        # CLASSIFICATION HEAD
        # ============================================================
        # Maps the 128-dimensional pooled features to 10 digit classes.
        # No dropout needed — BatchNorm + skip connections provide
        # sufficient regularization for MNIST.
        self.fc = nn.Linear(128, config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass through the ResNet CNN.

        Data flow:
            (batch, 1, 28, 28) → initial conv → stage1 → stage2 → stage3
            → global avg pool → flatten → FC → (batch, 10)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        # Initial convolution: (batch, 1, 28, 28) → (batch, 32, 28, 28)
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual stages
        x = self.stage1(x)   # (batch, 32, 28, 28) → (batch, 32, 28, 28)
        x = self.stage2(x)   # (batch, 32, 28, 28) → (batch, 64, 14, 14)
        x = self.stage3(x)   # (batch, 64, 14, 14) → (batch, 128, 7, 7)

        # Global average pooling: (batch, 128, 7, 7) → (batch, 128, 1, 1)
        x = self.global_avg_pool(x)

        # Flatten: (batch, 128, 1, 1) → (batch, 128)
        x = x.view(x.size(0), -1)

        # Classification: (batch, 128) → (batch, 10)
        x = self.fc(x)

        return x
