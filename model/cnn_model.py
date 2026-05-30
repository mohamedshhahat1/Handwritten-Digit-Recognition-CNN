"""
CNN Model for Handwritten Digit Recognition (MNIST)

This module defines a Convolutional Neural Network (CNN) designed to classify
handwritten digits (0-9) from 28x28 grayscale images. The architecture uses
two convolutional layers followed by fully connected layers, which is a
well-established pattern for image classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for digit classification.

    This network takes 28x28 grayscale images as input and outputs
    predictions for 10 digit classes (0 through 9).

    Architecture Summary:
        Input (1x28x28)
        -> Conv2d (32 filters) -> ReLU -> MaxPool2d
        -> Conv2d (64 filters) -> ReLU -> MaxPool2d
        -> Flatten -> FC (128 units) -> ReLU -> Dropout
        -> FC (10 units, output)
    """

    def __init__(self):
        """
        Initialize the CNN layers.

        The network consists of:
        - Two convolutional blocks (conv -> activation -> pooling)
        - Two fully connected (dense) layers with dropout for regularization
        """
        super(CNN, self).__init__()

        # ============================================================
        # CONVOLUTIONAL LAYER 1
        # ============================================================
        # Input: 1 channel (grayscale), Output: 32 feature maps
        # Kernel size: 3x3, Padding: 1 (preserves spatial dimensions)
        # After this layer: 32 x 28 x 28 (padding keeps size the same)
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale images have 1 color channel
            out_channels=32,    # Produce 32 different feature maps
            kernel_size=3,      # Each filter is 3x3 pixels
            padding=1           # Add 1 pixel border to keep dimensions unchanged
        )

        # ============================================================
        # CONVOLUTIONAL LAYER 2
        # ============================================================
        # Input: 32 feature maps, Output: 64 feature maps
        # Kernel size: 3x3, Padding: 1 (preserves spatial dimensions)
        # After this layer (before pooling): 64 x 14 x 14
        # After max pooling: 64 x 7 x 7
        self.conv2 = nn.Conv2d(
            in_channels=32,     # Takes the 32 feature maps from conv1
            out_channels=64,    # Produce 64 different feature maps
            kernel_size=3,      # Each filter is 3x3 pixels
            padding=1           # Add 1 pixel border to keep dimensions unchanged
        )

        # ============================================================
        # MAX POOLING LAYER
        # ============================================================
        # Reduces spatial dimensions by half (takes the max in each 2x2 region)
        # This helps reduce computation and provides translation invariance
        # Applied after each convolutional block:
        #   After pool1: 32 x 14 x 14 (from 32 x 28 x 28)
        #   After pool2: 64 x 7 x 7   (from 64 x 14 x 14)
        self.pool = nn.MaxPool2d(
            kernel_size=2,      # Look at 2x2 regions
            stride=2            # Move 2 pixels at a time (no overlap)
        )

        # ============================================================
        # FULLY CONNECTED LAYER 1
        # ============================================================
        # After two rounds of convolution + pooling, our feature maps are 64 x 7 x 7
        # We flatten these into a single vector: 64 * 7 * 7 = 3136 values
        # This layer maps those 3136 features down to 128 neurons
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7,   # Flattened feature map size
            out_features=128           # Compressed representation
        )

        # ============================================================
        # DROPOUT LAYER
        # ============================================================
        # Randomly sets 25% of neurons to zero during training
        # This prevents overfitting by forcing the network to not rely
        # too heavily on any single neuron
        self.dropout = nn.Dropout(p=0.25)

        # ============================================================
        # FULLY CONNECTED LAYER 2 (OUTPUT)
        # ============================================================
        # Maps the 128 features to 10 output classes (digits 0-9)
        # The output values are raw scores (logits) for each digit class
        self.fc2 = nn.Linear(
            in_features=128,    # Input from previous FC layer
            out_features=10     # One output per digit (0-9)
        )

    def forward(self, x):
        """
        Define the forward pass of the network.

        This method specifies how input data flows through the layers
        to produce the final output predictions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                              representing a batch of grayscale digit images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10) containing
                         raw scores (logits) for each of the 10 digit classes.
        """
        # --- Convolutional Block 1 ---
        # Apply first convolution, then ReLU activation, then max pooling
        # Shape: (batch, 1, 28, 28) -> (batch, 32, 28, 28) -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))

        # --- Convolutional Block 2 ---
        # Apply second convolution, then ReLU activation, then max pooling
        # Shape: (batch, 32, 14, 14) -> (batch, 64, 14, 14) -> (batch, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))

        # --- Flatten ---
        # Reshape the 3D feature maps into a 1D vector for the fully connected layers
        # Shape: (batch, 64, 7, 7) -> (batch, 3136)
        x = x.view(-1, 64 * 7 * 7)

        # --- Fully Connected Block ---
        # Apply first FC layer with ReLU activation
        # Shape: (batch, 3136) -> (batch, 128)
        x = F.relu(self.fc1(x))

        # Apply dropout (only active during training, automatically disabled in eval mode)
        x = self.dropout(x)

        # --- Output Layer ---
        # Apply second FC layer to get final class scores
        # Shape: (batch, 128) -> (batch, 10)
        x = self.fc2(x)

        return x
