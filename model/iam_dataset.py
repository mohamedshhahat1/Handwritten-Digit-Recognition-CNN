"""
Real Handwriting Dataset Loaders (IAM, RIMES)
================================================

This module provides PyTorch Dataset classes for real handwriting datasets,
enabling training on genuine handwritten text instead of synthetic data.

Supported Datasets:

1. IAM Handwriting Database (English)
   - 13,353 labeled text line images from 657 writers
   - Source: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
   - Format: grayscale PNG images + text transcriptions
   - Registration required (free for research)

2. RIMES (French)
   - ~12,000 handwritten text lines
   - Source: http://www.a2ialab.com/doku.php?id=rimes_database
   - Format: grayscale images + XML annotations

Dataset Setup:
    Since these datasets require registration/download, this module expects
    the data to be pre-downloaded to a local directory. It provides:
    - Automatic parsing of standard annotation formats
    - Image preprocessing (resize, normalize, augment)
    - CTC-compatible label encoding
    - Train/val/test split management

Usage:
    # Download IAM dataset manually, then:
    python train_ocr.py --dataset iam --data-dir ./data/iam

    # Or use RIMES:
    python train_ocr.py --dataset rimes --data-dir ./data/rimes

    # If no real dataset is available, fall back to synthetic:
    python train_ocr.py --dataset synthetic

Directory Structure Expected:

    IAM:
        data/iam/
        ├── lines/              # Line images (e.g., a01-000u-00.png)
        │   ├── a01/
        │   │   ├── a01-000u/
        │   │   │   ├── a01-000u-00.png
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        └── lines.txt           # Transcription file (from IAM)

    RIMES:
        data/rimes/
        ├── images_lines/       # Line images
        │   ├── img-000001.png
        │   └── ...
        └── groundtruth.txt     # Transcription file
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset

from model.ocr_utils import OCRCharset


# =============================================================================
# IAM HANDWRITING DATASET
# =============================================================================

class IAMDataset(Dataset):
    """
    PyTorch Dataset for the IAM Handwriting Database.

    Loads text line images and their transcriptions from the IAM dataset.
    The IAM dataset contains English handwritten text from 657 writers.

    The standard IAM `lines.txt` format is:
        # comment lines start with #
        id ok/err graylevel #components tag  text
        a01-000u-00 ok 154 408 746 1661 89 A|MOVE|to|stop|...

    Where:
    - Fields are space-separated
    - The last field is the transcription with '|' replacing spaces
    - Lines starting with '#' are comments
    - 'ok' means the segmentation is correct

    Args:
        data_dir (str): Root directory containing IAM data.
        charset (OCRCharset): Character set for encoding.
        split (str): 'train', 'val', or 'test'.
        img_height (int): Target height for images (default 32).
        max_text_len (int): Maximum text length to include (default 100).
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(self, data_dir, charset, split='train', img_height=32,
                 max_text_len=100, augment=False):
        self.data_dir = data_dir
        self.charset = charset
        self.img_height = img_height
        self.max_text_len = max_text_len
        self.augment = augment
        self.split = split

        # Parse annotations
        self.samples = self._load_annotations()

        # Apply train/val/test split
        self.samples = self._apply_split(self.samples, split)

        print(f"  IAM Dataset ({split}): {len(self.samples)} samples loaded")

    def _load_annotations(self):
        """
        Parse IAM lines.txt annotation file.

        Returns:
            list: List of (image_path, transcription) tuples.
        """
        samples = []

        # Try standard IAM annotation file locations
        annotation_paths = [
            os.path.join(self.data_dir, "lines.txt"),
            os.path.join(self.data_dir, "ascii", "lines.txt"),
            os.path.join(self.data_dir, "annotations", "lines.txt"),
        ]

        annotation_file = None
        for path in annotation_paths:
            if os.path.exists(path):
                annotation_file = path
                break

        if annotation_file is None:
            raise FileNotFoundError(
                f"IAM annotation file 'lines.txt' not found in {self.data_dir}\n"
                f"Searched: {annotation_paths}\n\n"
                f"Please download the IAM dataset from:\n"
                f"  https://fki.tic.heia-fr.ch/databases/iam-handwriting-database\n\n"
                f"Expected directory structure:\n"
                f"  {self.data_dir}/\n"
                f"  ├── lines/          (line images)\n"
                f"  └── lines.txt       (annotations)"
            )

        # Parse the annotation file
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                parts = line.split(' ')

                # IAM format: id status graylevel components... transcription
                # Minimum 9 fields before transcription
                if len(parts) < 9:
                    continue

                line_id = parts[0]
                status = parts[1]

                # Only use correctly segmented lines
                if status != 'ok':
                    continue

                # Transcription is the last field, with '|' for spaces
                transcription = parts[-1].replace('|', ' ')

                # Skip if too long
                if len(transcription) > self.max_text_len:
                    continue

                # Skip if contains characters not in our charset
                valid = all(c in self.charset.char_to_idx for c in transcription)
                if not valid:
                    continue

                # Construct image path
                # IAM line IDs look like: a01-000u-00
                # Image path: lines/a01/a01-000u/a01-000u-00.png
                parts_id = line_id.split('-')
                if len(parts_id) >= 3:
                    folder1 = parts_id[0]
                    folder2 = f"{parts_id[0]}-{parts_id[1]}"
                    img_name = f"{line_id}.png"

                    img_path = os.path.join(
                        self.data_dir, "lines", folder1, folder2, img_name
                    )

                    if os.path.exists(img_path):
                        samples.append((img_path, transcription))

        if not samples:
            raise FileNotFoundError(
                f"No valid IAM samples found in {self.data_dir}\n"
                f"Annotation file found at: {annotation_file}\n"
                f"Make sure the 'lines/' directory contains the image files."
            )

        return samples

    def _apply_split(self, samples, split):
        """
        Split samples into train/val/test sets.

        Uses a deterministic split based on the standard IAM partitioning:
        - Train: 80%
        - Validation: 10%
        - Test: 10%
        """
        # Sort for deterministic splitting
        samples = sorted(samples, key=lambda x: x[0])

        n = len(samples)
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)

        if split == 'train':
            return samples[:train_end]
        elif split == 'val':
            return samples[train_end:val_end]
        elif split == 'test':
            return samples[val_end:]
        else:
            return samples  # Return all

    def _preprocess_image(self, img_path):
        """
        Load and preprocess an image for the CRNN model.

        Steps:
        1. Load as grayscale
        2. Resize to fixed height (maintain aspect ratio)
        3. Optionally augment
        4. Normalize to [0, 1]

        Args:
            img_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image of shape (1, img_height, width).
        """
        img = Image.open(img_path).convert('L')

        # Resize to fixed height, maintain aspect ratio
        w, h = img.size
        new_h = self.img_height
        new_w = max(int(w * new_h / h), 32)  # Minimum width of 32
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Invert if light background (OCR expects white text on black)
        arr = np.array(img)
        if arr.mean() > 127:
            arr = 255 - arr
        img = Image.fromarray(arr)

        # Data augmentation
        if self.augment:
            img = self._augment(img)

        # Convert to tensor and normalize to [0, 1]
        tensor = torch.FloatTensor(np.array(img)).unsqueeze(0) / 255.0

        return tensor

    def _augment(self, img):
        """Apply random augmentation to the image."""
        # Random rotation (slight)
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, fillcolor=0)

        # Random blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.0)
            ))

        # Random noise
        if random.random() < 0.2:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Random erosion/dilation
        if random.random() < 0.15:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))
            else:
                img = img.filter(ImageFilter.MaxFilter(3))

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            tuple: (image_tensor, label_tensor, label_length)
                - image_tensor: (1, img_height, width) normalized grayscale
                - label_tensor: encoded text as int tensor
                - label_length: number of characters in label
        """
        img_path, transcription = self.samples[idx]

        # Preprocess image
        img_tensor = self._preprocess_image(img_path)

        # Encode label
        label = self.charset.encode(transcription)
        label_tensor = torch.IntTensor(label)

        return img_tensor, label_tensor, len(label)


# =============================================================================
# RIMES DATASET
# =============================================================================

class RIMESDataset(Dataset):
    """
    PyTorch Dataset for the RIMES Handwriting Database (French).

    Loads text line images and transcriptions from the RIMES dataset.
    RIMES contains French handwritten mail excerpts.

    Expected groundtruth.txt format (tab-separated):
        image_filename\ttranscription

    Args:
        data_dir (str): Root directory containing RIMES data.
        charset (OCRCharset): Character set for encoding.
        split (str): 'train', 'val', or 'test'.
        img_height (int): Target height for images (default 32).
        max_text_len (int): Maximum text length to include (default 100).
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(self, data_dir, charset, split='train', img_height=32,
                 max_text_len=100, augment=False):
        self.data_dir = data_dir
        self.charset = charset
        self.img_height = img_height
        self.max_text_len = max_text_len
        self.augment = augment
        self.split = split

        # Parse annotations
        self.samples = self._load_annotations()

        # Apply split
        self.samples = self._apply_split(self.samples, split)

        print(f"  RIMES Dataset ({split}): {len(self.samples)} samples loaded")

    def _load_annotations(self):
        """
        Parse RIMES groundtruth annotation file.

        Returns:
            list: List of (image_path, transcription) tuples.
        """
        samples = []

        # Try standard annotation file locations
        annotation_paths = [
            os.path.join(self.data_dir, "groundtruth.txt"),
            os.path.join(self.data_dir, "ground_truth.txt"),
            os.path.join(self.data_dir, "annotations.txt"),
            os.path.join(self.data_dir, "labels.txt"),
        ]

        annotation_file = None
        for path in annotation_paths:
            if os.path.exists(path):
                annotation_file = path
                break

        if annotation_file is None:
            raise FileNotFoundError(
                f"RIMES annotation file not found in {self.data_dir}\n"
                f"Searched: {annotation_paths}\n\n"
                f"Please download the RIMES dataset from:\n"
                f"  http://www.a2ialab.com/doku.php?id=rimes_database\n\n"
                f"Expected directory structure:\n"
                f"  {self.data_dir}/\n"
                f"  ├── images_lines/    (line images)\n"
                f"  └── groundtruth.txt  (annotations)"
            )

        # Possible image directories
        img_dirs = [
            os.path.join(self.data_dir, "images_lines"),
            os.path.join(self.data_dir, "lines"),
            os.path.join(self.data_dir, "images"),
            self.data_dir,
        ]

        # Parse annotation file (tab or space separated: filename transcription)
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try tab-separated first, then space-separated
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(' ', 1)

                if len(parts) < 2:
                    continue

                img_filename = parts[0].strip()
                transcription = parts[1].strip()

                # Skip if too long
                if len(transcription) > self.max_text_len:
                    continue

                # Filter to charset-compatible characters
                filtered = ''.join(c for c in transcription
                                   if c in self.charset.char_to_idx)
                if not filtered or len(filtered) < 2:
                    continue

                # Find the image file
                img_path = None
                for img_dir in img_dirs:
                    candidate = os.path.join(img_dir, img_filename)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path:
                    samples.append((img_path, filtered))

        if not samples:
            raise FileNotFoundError(
                f"No valid RIMES samples found in {self.data_dir}\n"
                f"Annotation file: {annotation_file}\n"
                f"Make sure image files exist in one of: {img_dirs}"
            )

        return samples

    def _apply_split(self, samples, split):
        """Split into train/val/test (80/10/10)."""
        samples = sorted(samples, key=lambda x: x[0])
        n = len(samples)
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)

        if split == 'train':
            return samples[:train_end]
        elif split == 'val':
            return samples[train_end:val_end]
        elif split == 'test':
            return samples[val_end:]
        return samples

    def _preprocess_image(self, img_path):
        """Load, resize, and normalize an image."""
        img = Image.open(img_path).convert('L')

        w, h = img.size
        new_h = self.img_height
        new_w = max(int(w * new_h / h), 32)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        arr = np.array(img)
        if arr.mean() > 127:
            arr = 255 - arr
        img = Image.fromarray(arr)

        if self.augment:
            img = self._augment(img)

        tensor = torch.FloatTensor(np.array(img)).unsqueeze(0) / 255.0
        return tensor

    def _augment(self, img):
        """Apply random augmentation."""
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-3, 3), fillcolor=0)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.0)))
        if random.random() < 0.2:
            arr = np.array(img).astype(np.float32)
            arr = np.clip(arr + np.random.normal(0, 8, arr.shape), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample: (image_tensor, label_tensor, label_length)."""
        img_path, transcription = self.samples[idx]
        img_tensor = self._preprocess_image(img_path)
        label = self.charset.encode(transcription)
        label_tensor = torch.IntTensor(label)
        return img_tensor, label_tensor, len(label)


# =============================================================================
# DATASET FACTORY
# =============================================================================

def get_ocr_dataset(dataset_name, data_dir, charset, split='train',
                    img_height=32, max_text_len=100, augment=False,
                    num_samples=5000):
    """
    Factory function to create the appropriate OCR dataset.

    This provides a unified interface for selecting between synthetic
    and real handwriting datasets for OCR training.

    Args:
        dataset_name (str): One of 'synthetic', 'iam', 'rimes'.
        data_dir (str): Root directory for the dataset.
        charset (OCRCharset): Character set for encoding.
        split (str): 'train', 'val', or 'test'.
        img_height (int): Target image height.
        max_text_len (int): Maximum text length.
        augment (bool): Whether to augment training data.
        num_samples (int): Number of samples for synthetic dataset.

    Returns:
        Dataset: The appropriate PyTorch Dataset instance.

    Raises:
        ValueError: If dataset_name is not recognized.
        FileNotFoundError: If real dataset files are not found.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'synthetic':
        from model.ocr_dataset import SyntheticOCRDataset
        return SyntheticOCRDataset(
            charset=charset,
            num_samples=num_samples,
            img_height=img_height,
            max_text_len=max_text_len,
        )

    elif dataset_name == 'iam':
        if data_dir is None:
            data_dir = './data/iam'
        return IAMDataset(
            data_dir=data_dir,
            charset=charset,
            split=split,
            img_height=img_height,
            max_text_len=max_text_len,
            augment=augment and (split == 'train'),
        )

    elif dataset_name == 'rimes':
        if data_dir is None:
            data_dir = './data/rimes'
        return RIMESDataset(
            data_dir=data_dir,
            charset=charset,
            split=split,
            img_height=img_height,
            max_text_len=max_text_len,
            augment=augment and (split == 'train'),
        )

    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Choose from: 'synthetic', 'iam', 'rimes'"
        )


def check_dataset_available(dataset_name, data_dir=None):
    """
    Check if a real handwriting dataset is available locally.

    Args:
        dataset_name (str): 'iam' or 'rimes'.
        data_dir (str, optional): Override data directory.

    Returns:
        tuple: (is_available: bool, message: str)
    """
    if dataset_name == 'synthetic':
        return True, "Synthetic data (always available, no download needed)"

    if dataset_name == 'iam':
        data_dir = data_dir or './data/iam'
        lines_txt = os.path.join(data_dir, "lines.txt")
        lines_dir = os.path.join(data_dir, "lines")

        if os.path.exists(lines_txt) and os.path.isdir(lines_dir):
            # Count images
            img_count = sum(1 for _ in _iter_files(lines_dir, '.png'))
            return True, f"IAM dataset found: {img_count} images in {data_dir}"
        else:
            return False, (
                f"IAM dataset not found at {data_dir}\n"
                f"  Download from: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database\n"
                f"  Place files as:\n"
                f"    {data_dir}/lines.txt\n"
                f"    {data_dir}/lines/<images>"
            )

    elif dataset_name == 'rimes':
        data_dir = data_dir or './data/rimes'
        gt_names = ["groundtruth.txt", "ground_truth.txt", "annotations.txt", "labels.txt"]

        gt_found = any(os.path.exists(os.path.join(data_dir, name)) for name in gt_names)

        if gt_found:
            return True, f"RIMES dataset found at {data_dir}"
        else:
            return False, (
                f"RIMES dataset not found at {data_dir}\n"
                f"  Download from: http://www.a2ialab.com/doku.php?id=rimes_database\n"
                f"  Place files as:\n"
                f"    {data_dir}/groundtruth.txt\n"
                f"    {data_dir}/images_lines/<images>"
            )

    return False, f"Unknown dataset: {dataset_name}"


def _iter_files(directory, extension):
    """Recursively iterate over files with a given extension."""
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extension):
                yield os.path.join(root, f)
