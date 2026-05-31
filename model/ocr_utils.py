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


def ctc_decode_beam_search(log_probs, beam_width=10, blank_id=0):
    """
    Beam search CTC decoding — explores multiple hypotheses in parallel.

    Unlike greedy decoding (which only keeps the single best character at each
    timestep), beam search maintains the top-K most probable sequences. This
    finds better overall paths through the probability matrix because it can
    recover from locally suboptimal choices.

    How it works:
        At each timestep, every beam is extended by every possible character.
        The resulting candidates are scored by their cumulative log-probability,
        and only the top beam_width candidates survive to the next step.

    CTC-specific handling:
        - Blank tokens (index 0) are tracked but don't emit characters
        - Repeated characters are collapsed unless separated by a blank
        - Two beams with the same text but different blank/non-blank endings
          are kept separately (they merge differently at the next step)

    Args:
        log_probs (torch.Tensor or np.ndarray): Shape (seq_len, num_classes).
            Log-probabilities at each timestep (output of log_softmax).
        beam_width (int): Number of hypotheses to keep at each step.
            Higher = better accuracy but slower. Default: 10.
            Typical values: 5-25 for good accuracy/speed tradeoff.
        blank_id (int): Index of the CTC blank token (default: 0).

    Returns:
        list[int]: Best decoded character indices (without blanks or repeats).

    Performance:
        - O(T * beam_width * num_classes) time complexity
        - Typically 2-5% CER improvement over greedy on real data
        - ~10-50x slower than greedy (still fast for single samples)

    Example:
        Greedy might pick: [0, 5, 5, 0, 2, 0, 9] → "h e l" (suboptimal)
        Beam search explores alternatives and finds a better global path.
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()

    seq_len, num_classes = log_probs.shape

    # Each beam is: (prefix_tuple, log_prob_blank_end, log_prob_non_blank_end)
    # We track blank/non-blank endings separately for correct CTC merging.
    #
    # Why? In CTC, "aa" and "a-a" (where - is blank) produce different texts:
    # "aa" collapses to "a", but "a-a" gives "aa".
    # So a beam ending in blank can extend differently than one ending in non-blank.

    NEG_INF = float('-inf')

    # Initial state: empty prefix, starts as "blank ending" with probability 1
    # Format: {prefix: (log_prob_blank, log_prob_non_blank)}
    beams = {(): (0.0, NEG_INF)}  # empty prefix, blank prob = log(1) = 0

    for t in range(seq_len):
        new_beams = {}  # Candidates for this timestep

        for prefix, (log_pb, log_pnb) in beams.items():
            # Total log probability of this prefix (log-sum-exp of blank and non-blank)
            log_p_total = _log_add(log_pb, log_pnb)

            for c in range(num_classes):
                log_p_c = log_probs[t, c]

                if c == blank_id:
                    # Extending with blank: doesn't change the prefix text
                    # Both blank-ending and non-blank-ending beams can emit blank
                    new_log_pb = log_p_total + log_p_c
                    _beam_update(new_beams, prefix, new_log_pb, is_blank=True)

                else:
                    # Extending with a character
                    if prefix and prefix[-1] == c:
                        # Same character as the end of prefix:
                        # - From blank-ending: emits a new character (e.g., "a" + blank + "a" = "aa")
                        # - From non-blank-ending: merges (e.g., "a" + "a" = "a", collapsed)
                        new_log_pnb_from_blank = log_pb + log_p_c
                        _beam_update(new_beams, prefix + (c,), new_log_pnb_from_blank, is_blank=False)

                        # Non-blank ending with same char: stays as current prefix (collapse)
                        new_log_pnb_merge = log_pnb + log_p_c
                        _beam_update(new_beams, prefix, new_log_pnb_merge, is_blank=False)
                    else:
                        # Different character: always extends the prefix
                        new_log_pnb = log_p_total + log_p_c
                        _beam_update(new_beams, prefix + (c,), new_log_pnb, is_blank=False)

        # Prune: keep only the top beam_width beams by total probability
        scored_beams = [
            (prefix, log_pb, log_pnb, _log_add(log_pb, log_pnb))
            for prefix, (log_pb, log_pnb) in new_beams.items()
        ]
        scored_beams.sort(key=lambda x: x[3], reverse=True)
        scored_beams = scored_beams[:beam_width]

        beams = {prefix: (log_pb, log_pnb) for prefix, log_pb, log_pnb, _ in scored_beams}

    # Return the best beam's prefix
    if not beams:
        return []

    best_prefix = max(beams.keys(), key=lambda p: _log_add(beams[p][0], beams[p][1]))
    return list(best_prefix)


def _log_add(log_a, log_b):
    """
    Numerically stable log-addition: log(exp(a) + exp(b)).

    Uses the log-sum-exp trick to avoid overflow/underflow:
        log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    """
    if log_a == float('-inf'):
        return log_b
    if log_b == float('-inf'):
        return log_a
    if log_a > log_b:
        return log_a + np.log1p(np.exp(log_b - log_a))
    else:
        return log_b + np.log1p(np.exp(log_a - log_b))


def _beam_update(beams_dict, prefix, log_prob, is_blank):
    """
    Update a beam entry, accumulating probability for the same prefix.

    Args:
        beams_dict (dict): Current beam dictionary.
        prefix (tuple): Character prefix tuple.
        log_prob (float): Log probability to add.
        is_blank (bool): Whether this extension ends in blank.
    """
    if prefix not in beams_dict:
        beams_dict[prefix] = (float('-inf'), float('-inf'))

    log_pb, log_pnb = beams_dict[prefix]

    if is_blank:
        beams_dict[prefix] = (_log_add(log_pb, log_prob), log_pnb)
    else:
        beams_dict[prefix] = (log_pb, _log_add(log_pnb, log_prob))


def ctc_decode_batch(log_probs, charset, method='greedy', beam_width=10):
    """
    Decode a batch of CTC outputs to text strings.

    Supports both greedy and beam search decoding methods.

    Args:
        log_probs (torch.Tensor): Shape (seq_len, batch, num_classes).
        charset (OCRCharset): Character set for decoding.
        method (str): Decoding method — 'greedy' or 'beam_search'.
            - 'greedy': Fast, picks best character at each timestep.
            - 'beam_search': Slower but more accurate, explores multiple paths.
        beam_width (int): Beam width for beam search (default: 10).
            Ignored if method='greedy'.

    Returns:
        list[str]: Decoded text for each sample in the batch.
    """
    batch_size = log_probs.size(1)
    texts = []

    for b in range(batch_size):
        # Get predictions for this sample: (seq_len, num_classes)
        sample_preds = log_probs[:, b, :]

        if method == 'beam_search':
            indices = ctc_decode_beam_search(sample_preds, beam_width=beam_width)
        else:
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
