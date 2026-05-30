"""
OCR Utilities — Character Set, Encoding, Decoding
===================================================

This module defines the character set for the multilingual OCR system
and provides encoding/decoding functions for CTC-based recognition.

Character Set:
    - CTC Blank (index 0) — special token for CTC loss
    - English uppercase: A-Z (26)
    - English lowercase: a-z (26)
    - Digits: 0-9 (10)
    - Arabic letters: 28 basic forms
    - Special: space, punctuation

CTC Decoding:
    The model outputs a prediction for each timestep. CTC decoding:
    1. Collapses consecutive repeated characters: "hh-ee-ll-ll-oo" → "h-e-l-l-o"
    2. Removes blank tokens: "h-e-l-l-o" → "hello"

Usage:
    from model.ocr_utils import OCRCharset, ctc_decode, ctc_collate_fn

    charset = OCRCharset()
    encoded = charset.encode("hello مرحبا")
    text = charset.decode([5, 2, 9, 9, 12])
"""

import torch
import numpy as np


# Arabic isolated letter forms (28 base letters)
ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"

# Full character set
ENGLISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
SPECIAL = " .,-"  # space, period, comma, dash


class OCRCharset:
    """
    Defines the character set for the OCR system and handles encoding/decoding.

    Index 0 is reserved for CTC blank token.
    Characters are indexed starting from 1.

    Attributes:
        characters (str): All supported characters (without blank).
        num_classes (int): Total classes including blank (len(characters) + 1).
        char_to_idx (dict): Character → index mapping.
        idx_to_char (dict): Index → character mapping.
    """

    def __init__(self):
        # Build the full character set
        self.characters = ENGLISH_UPPER + ENGLISH_LOWER + DIGITS + ARABIC_LETTERS + SPECIAL

        # Total classes = characters + 1 (CTC blank at index 0)
        self.num_classes = len(self.characters) + 1

        # Character → index (1-based, 0 is blank)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}

        # Index → character (0 maps to blank/empty)
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.characters)}
        self.idx_to_char[0] = ''  # CTC blank

    def encode(self, text):
        """
        Encode a text string into a list of integer indices.

        Characters not in the charset are skipped.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: List of character indices (1-based).

        Example:
            >>> charset.encode("Hi5")
            [8, 35, 57]  # H=8, i=35, 5=57
        """
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        return encoded

    def decode(self, indices):
        """
        Decode a list of indices back to text.

        Args:
            indices (list[int]): Character indices to decode.

        Returns:
            str: Decoded text string.
        """
        return ''.join(self.idx_to_char.get(idx, '') for idx in indices)

    def __len__(self):
        return self.num_classes

    def __repr__(self):
        return (f"OCRCharset(num_classes={self.num_classes}, "
                f"english={len(ENGLISH_UPPER + ENGLISH_LOWER)}, "
                f"digits={len(DIGITS)}, "
                f"arabic={len(ARABIC_LETTERS)}, "
                f"special={len(SPECIAL)})")


def ctc_decode_greedy(predictions):
    """
    Greedy CTC decoding — takes the most likely character at each timestep.

    Steps:
    1. Take argmax at each timestep
    2. Collapse consecutive duplicates
    3. Remove blank tokens (index 0)

    Args:
        predictions (torch.Tensor): Shape (seq_len, num_classes) — log probabilities.

    Returns:
        list[int]: Decoded character indices (without blanks or repeats).

    Example:
        Input timesteps:  [0, 5, 5, 0, 2, 2, 2, 0, 9]
        After collapse:   [0, 5, 0, 2, 0, 9]
        After blank removal: [5, 2, 9]
    """
    # Get the most likely character at each timestep
    if isinstance(predictions, torch.Tensor):
        indices = predictions.argmax(dim=-1).cpu().numpy()
    else:
        indices = np.argmax(predictions, axis=-1)

    # Collapse consecutive duplicates
    collapsed = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(int(idx))
        prev = idx

    # Remove blanks (index 0)
    decoded = [idx for idx in collapsed if idx != 0]

    return decoded


def ctc_decode_batch(log_probs, charset):
    """
    Decode a batch of CTC outputs to text strings.

    Args:
        log_probs (torch.Tensor): Shape (seq_len, batch, num_classes).
        charset (OCRCharset): Character set for decoding.

    Returns:
        list[str]: Decoded text for each sample in the batch.
    """
    batch_size = log_probs.size(1)
    texts = []

    for b in range(batch_size):
        # Get predictions for this sample: (seq_len, num_classes)
        sample_preds = log_probs[:, b, :]
        # Greedy decode
        indices = ctc_decode_greedy(sample_preds)
        # Convert to text
        text = charset.decode(indices)
        texts.append(text)

    return texts


def ctc_collate_fn(batch):
    """
    Custom collation function for variable-width OCR images.

    Since text images have variable width (longer text = wider image),
    we need to pad them to the same width in a batch.

    Args:
        batch: List of (image_tensor, label_tensor, label_length) tuples.

    Returns:
        tuple: (padded_images, labels, label_lengths, input_lengths)
            - padded_images: (batch, 1, 32, max_width) — zero-padded
            - labels: concatenated label tensor
            - label_lengths: length of each label
            - input_lengths: sequence length for each image after CNN
    """
    # Sort by width (descending) for efficient packing
    batch.sort(key=lambda x: x[0].size(2), reverse=True)

    images, labels, label_lengths = zip(*batch)

    # Find max width in this batch
    max_width = max(img.size(2) for img in images)

    # Pad all images to max_width
    padded_images = torch.zeros(len(images), 1, 32, max_width)
    input_lengths = []

    for i, img in enumerate(images):
        width = img.size(2)
        padded_images[i, :, :, :width] = img
        # Calculate sequence length after CNN processing
        # CNN reduces width: /2 (pool1) /2 (pool2) -1 (final conv)
        seq_len = width // 4 - 1
        input_lengths.append(seq_len)

    # Concatenate all labels into one tensor (CTC format)
    all_labels = torch.cat(labels)
    label_lengths = torch.IntTensor(label_lengths)
    input_lengths = torch.IntTensor(input_lengths)

    return padded_images, all_labels, label_lengths, input_lengths


def compute_cer(predicted, target):
    """
    Compute Character Error Rate (CER) using edit distance.

    CER = (substitutions + insertions + deletions) / len(target)

    Args:
        predicted (str): Predicted text.
        target (str): Ground truth text.

    Returns:
        float: CER between 0.0 (perfect) and 1.0+ (bad).
    """
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    # Levenshtein distance (dynamic programming)
    m, n = len(predicted), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / len(target)


def compute_wer(predicted, target):
    """
    Compute Word Error Rate (WER).

    WER = edit_distance(predicted_words, target_words) / len(target_words)

    Args:
        predicted (str): Predicted text.
        target (str): Ground truth text.

    Returns:
        float: WER between 0.0 (perfect) and 1.0+ (bad).
    """
    pred_words = predicted.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Levenshtein at word level
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / len(target_words)
