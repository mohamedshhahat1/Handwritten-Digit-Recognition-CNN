"""
CRNN Model (CNN + RNN) for OCR
================================

This module implements a CRNN (Convolutional Recurrent Neural Network)
for Optical Character Recognition (OCR). It can recognize full words
and sentences from handwritten text images.

Architecture:
    Input Image (1, 32, W) — grayscale, height=32, variable width
        │
        ├─ CNN Backbone (7 conv layers) — extracts visual features
        │   Output: (512, 1, W/4) feature maps
        │
        ├─ Map-to-Sequence — reshape features into a sequence
        │   Output: (W/4, batch, 512)
        │
        ├─ BiLSTM (2 layers) — models sequential dependencies
        │   Output: (W/4, batch, num_classes)
        │
        └─ CTC Decoder — converts per-frame predictions to text

Why CRNN?
    - CNN captures spatial/visual features (strokes, curves)
    - RNN captures sequential context (letter order, word patterns)
    - CTC loss handles variable-length alignment without segmentation

Supports: English (A-Z, a-z) + Arabic (28 letters) + Digits (0-9)
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for OCR.

    Takes a grayscale image of fixed height (32px) and variable width,
    and outputs per-timestep character predictions that are decoded
    using CTC (Connectionist Temporal Classification).

    Args:
        num_classes (int): Total number of characters + blank (CTC).
        hidden_size (int): LSTM hidden state size (default 256).
        num_layers (int): Number of BiLSTM layers (default 2).
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # =====================================================================
        # CNN BACKBONE — Feature Extractor
        # Converts (batch, 1, 32, W) image into (batch, 512, 1, W/4) features
        # =====================================================================
        self.cnn = nn.Sequential(
            # Block 1: 1 → 64 channels, output: (64, 16, W/2)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 → 128 channels, output: (128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 → 256 channels, output: (256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 256 → 256, output: (256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5: 256 → 512, output: (512, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: 512 → 512, output: (512, 2, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 7: 512 → 512, output: (512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # =====================================================================
        # SEQUENCE MODEL — BiLSTM
        # Processes the feature sequence and captures temporal dependencies
        # Input: (seq_len, batch, 512)
        # Output: (seq_len, batch, hidden_size * 2)
        # =====================================================================
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # =====================================================================
        # OUTPUT LAYER
        # Maps BiLSTM output to character probabilities
        # Input: (seq_len, batch, hidden_size * 2)
        # Output: (seq_len, batch, num_classes)
        # =====================================================================
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through CRNN.

        Args:
            x (torch.Tensor): Input images of shape (batch, 1, 32, W).
                Height must be 32. Width can vary (determines sequence length).

        Returns:
            torch.Tensor: Log-softmax predictions of shape (seq_len, batch, num_classes).
                seq_len ≈ W/4 (depends on CNN pooling).
        """
        # CNN feature extraction: (batch, 1, 32, W) → (batch, 512, 1, W')
        conv_features = self.cnn(x)

        # Remove height dimension (it's 1 after CNN): (batch, 512, W')
        batch_size, channels, height, width = conv_features.size()
        assert height == 1, f"CNN output height must be 1, got {height}"
        conv_features = conv_features.squeeze(2)

        # Permute to (seq_len, batch, features) for LSTM: (W', batch, 512)
        conv_features = conv_features.permute(2, 0, 1)

        # BiLSTM sequence modeling: (W', batch, 512) → (W', batch, hidden*2)
        rnn_output, _ = self.rnn(conv_features)

        # Linear projection to character classes: (W', batch, num_classes)
        output = self.linear(rnn_output)

        # Log-softmax for CTC loss
        output = nn.functional.log_softmax(output, dim=2)

        return output

    def get_seq_length(self, input_width):
        """
        Calculate the output sequence length for a given input width.

        The CNN reduces width through pooling. This calculates how many
        timesteps the LSTM will see for a given input image width.

        Args:
            input_width (int): Width of input image in pixels.

        Returns:
            int: Number of output timesteps (sequence length).
        """
        # Simulate the width reduction through the CNN
        # Pool layers that reduce width: Block1 (/2), Block2 (/2)
        # Block4 pool is (2,1) — only reduces height
        # Block6 pool is (2,1) — only reduces height
        # Block7 conv with kernel=2, no padding reduces by 1
        w = input_width
        w = w // 2  # Block 1 MaxPool
        w = w // 2  # Block 2 MaxPool
        w = w - 1   # Block 7 Conv2d(kernel=2, padding=0)
        return w
