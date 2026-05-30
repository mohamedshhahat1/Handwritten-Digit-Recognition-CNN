"""
MLP Model for Handwritten Digit Recognition (MNIST)

This module defines a Multi-Layer Perceptron (MLP) for classifying handwritten
digits (0-9) from 28x28 grayscale images.

=============================================================================
KEY DIFFERENCE: MLP vs CNN
=============================================================================

CNN (Convolutional Neural Network):
    - Uses convolutional layers that scan small regions (filters) across the image
    - Preserves spatial structure: knows that nearby pixels are related
    - Shares weights across positions: the same filter detects a feature anywhere
    - Much fewer parameters for image tasks due to weight sharing
    - Excellent for images because it exploits spatial locality

MLP (Multi-Layer Perceptron):
    - Uses only fully connected (dense) layers
    - Flattens the entire image into a 1D vector first (loses spatial structure)
    - Every input pixel connects to every neuron (no weight sharing)
    - More parameters despite being "simpler" in architecture
    - Treats each pixel independently -- doesn't inherently know that
      neighboring pixels are related

In short: CNNs "see" patterns in local regions, while MLPs treat the image
as a flat list of numbers. This is why CNNs typically outperform MLPs on
image tasks, even with fewer parameters.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for digit classification.

    This network takes 28x28 grayscale images as input and outputs
    predictions for 10 digit classes (0 through 9).

    Unlike the CNN, this model does NOT use convolutions. Instead, it
    flattens the image into a vector of 784 pixels and passes it through
    several fully connected layers.

    Architecture Summary:
        Input (1x28x28)
        -> Flatten (784)
        -> Linear(784, 512) -> ReLU -> Dropout(0.2)
        -> Linear(512, 256) -> ReLU -> Dropout(0.2)
        -> Linear(256, 128) -> ReLU -> Dropout(0.2)
        -> Linear(128, 10) -> output
    """

    def __init__(self):
        """
        Initialize the MLP layers.

        The network consists of three hidden fully connected layers with
        ReLU activation and dropout, followed by an output layer.
        Each hidden layer progressively reduces the dimensionality:
        784 -> 512 -> 256 -> 128 -> 10
        """
        super(MLP, self).__init__()

        # ============================================================
        # INPUT SIZE
        # ============================================================
        # MNIST images are 28x28 pixels = 784 values when flattened.
        # Unlike CNN, we cannot work with 2D spatial data directly.
        self.input_size = 28 * 28  # 784

        # ============================================================
        # FULLY CONNECTED LAYER 1
        # ============================================================
        # Maps the 784-dimensional input to 512 neurons.
        # This is the largest layer and does the initial feature extraction.
        # Note: With 784 * 512 = 401,408 weights + 512 biases, this single
        # layer has more parameters than the entire CNN's convolutional layers!
        self.fc1 = nn.Linear(
            in_features=self.input_size,  # 784 (flattened 28x28 image)
            out_features=512              # First hidden layer size
        )

        # Dropout after first hidden layer (20% of neurons randomly zeroed)
        self.dropout1 = nn.Dropout(p=0.2)

        # ============================================================
        # FULLY CONNECTED LAYER 2
        # ============================================================
        # Reduces from 512 to 256 neurons.
        # Learns more abstract representations from the first layer's output.
        self.fc2 = nn.Linear(
            in_features=512,   # Input from first hidden layer
            out_features=256   # Second hidden layer size
        )

        # Dropout after second hidden layer
        self.dropout2 = nn.Dropout(p=0.2)

        # ============================================================
        # FULLY CONNECTED LAYER 3
        # ============================================================
        # Reduces from 256 to 128 neurons.
        # Further compresses the representation before classification.
        self.fc3 = nn.Linear(
            in_features=256,   # Input from second hidden layer
            out_features=128   # Third hidden layer size
        )

        # Dropout after third hidden layer
        self.dropout3 = nn.Dropout(p=0.2)

        # ============================================================
        # OUTPUT LAYER
        # ============================================================
        # Maps the 128 features to 10 output classes (digits 0-9).
        # The output values are raw scores (logits) for each digit class.
        self.fc4 = nn.Linear(
            in_features=128,   # Input from third hidden layer
            out_features=10    # One output per digit (0-9)
        )

    def forward(self, x):
        """
        Define the forward pass of the network.

        This method specifies how input data flows through the layers
        to produce the final output predictions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                              representing a batch of grayscale digit images.
                              (Same interface as the CNN model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10) containing
                         raw scores (logits) for each of the 10 digit classes.
        """
        # --- Flatten ---
        # Reshape the 2D image into a 1D vector.
        # This is where we lose spatial information!
        # Shape: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(-1, self.input_size)

        # --- Hidden Layer 1 ---
        # Apply first FC layer with ReLU activation, then dropout
        # Shape: (batch, 784) -> (batch, 512)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # --- Hidden Layer 2 ---
        # Apply second FC layer with ReLU activation, then dropout
        # Shape: (batch, 512) -> (batch, 256)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # --- Hidden Layer 3 ---
        # Apply third FC layer with ReLU activation, then dropout
        # Shape: (batch, 256) -> (batch, 128)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        # --- Output Layer ---
        # Apply final FC layer to get class scores (no activation here;
        # CrossEntropyLoss applies softmax internally)
        # Shape: (batch, 128) -> (batch, 10)
        x = self.fc4(x)

        return x
