"""
Centralized Configuration for Handwritten Digit Recognition CNN
================================================================

This file contains ALL configurable parameters for the project in one place.
Modify values here to tune training, model architecture, data loading, and more.

Usage:
    from config import *
    # or
    import config
    print(config.BATCH_SIZE)
"""

import os
import torch

# =============================================================================
# 1. TRAINING HYPERPARAMETERS
# =============================================================================
# These control how the model learns during training.

# BATCH_SIZE: Number of samples processed before the model updates its weights.
# Larger batches = faster training but more memory usage.
# Common values: 32, 64, 128, 256
BATCH_SIZE = 64

# LEARNING_RATE: How much the model adjusts its weights each update step.
# Too high = unstable training, too low = very slow convergence.
# Common values: 0.01, 0.001, 0.0001
LEARNING_RATE = 0.001

# EPOCHS: Number of complete passes through the entire training dataset.
# More epochs = more training time but potentially better accuracy (up to a point).
EPOCHS = 15

# WEIGHT_DECAY: L2 regularization factor to prevent overfitting.
# Adds a penalty for large weights, encouraging simpler models.
# Set to 0 to disable. Common values: 1e-4, 1e-5
WEIGHT_DECAY = 1e-4

# =============================================================================
# 2. MODEL CONFIGURATION
# =============================================================================
# These define the structure of the neural network.

# NUM_CLASSES: Number of output categories the model can predict.
# For digit recognition (0-9), this is always 10.
NUM_CLASSES = 10

# DROPOUT_RATE: Fraction of neurons randomly disabled during training.
# Helps prevent overfitting by forcing the network to be redundant.
# 0.0 = no dropout, 1.0 = drop everything. Common values: 0.2, 0.25, 0.5
DROPOUT_RATE = 0.25

# =============================================================================
# 3. DATA CONFIGURATION
# =============================================================================
# These control how data is loaded and preprocessed.

# DATA_DIR: Directory where the MNIST dataset will be downloaded/stored.
DATA_DIR = "./data/mnist"

# NUM_WORKERS: Number of parallel processes for loading data.
# Higher values speed up data loading but use more CPU/memory.
# Set to 0 for debugging (single-process loading).
NUM_WORKERS = 2

# =============================================================================
# 4. DEVICE CONFIGURATION
# =============================================================================
# Automatically selects GPU (CUDA) if available, otherwise uses CPU.
# GPU training is significantly faster for neural networks.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 5. EARLY STOPPING
# =============================================================================
# Stops training automatically when the model stops improving.
# This prevents overfitting and saves time.

# EARLY_STOPPING_PATIENCE: Number of epochs to wait for improvement before stopping.
# Higher values give the model more chances to recover from plateaus.
EARLY_STOPPING_PATIENCE = 5

# EARLY_STOPPING_MIN_DELTA: Minimum change in monitored metric to count as improvement.
# Changes smaller than this are considered noise, not real progress.
EARLY_STOPPING_MIN_DELTA = 0.001

# =============================================================================
# 6. MODEL SAVING
# =============================================================================
# Controls how and where trained models are saved to disk.

# MODEL_DIR: Directory where model checkpoint files are stored.
MODEL_DIR = "./saved_models"

# MODEL_NAME_PREFIX: Prefix for saved model filenames.
# Final filename example: "mnist_cnn_epoch10_acc99.5.pth"
MODEL_NAME_PREFIX = "mnist_cnn"

# SAVE_BEST_ONLY: If True, only saves the model when it achieves a new best score.
# If False, saves the model after every epoch (uses more disk space).
SAVE_BEST_ONLY = True

# =============================================================================
# 7. DATA AUGMENTATION
# =============================================================================
# Artificially expands the training set by applying random transformations.
# This helps the model generalize better to new handwriting styles.

# AUGMENTATION_ENABLED: Master switch to enable/disable all augmentation.
AUGMENTATION_ENABLED = True

# ROTATION_DEGREES: Maximum random rotation applied to images (in degrees).
# Simulates slightly tilted handwriting.
ROTATION_DEGREES = 10

# TRANSLATE_RANGE: Maximum horizontal and vertical shift as a fraction of image size.
# Simulates digits not perfectly centered.
# (0.1, 0.1) means up to 10% shift in each direction.
TRANSLATE_RANGE = (0.1, 0.1)

# SCALE_RANGE: Range of random scaling factors applied to images.
# Simulates digits written at different sizes.
# (0.9, 1.1) means between 90% and 110% of original size.
SCALE_RANGE = (0.9, 1.1)

# =============================================================================
# 8. TENSORBOARD LOGGING
# =============================================================================
# TensorBoard provides real-time visualization of training progress.
# Launch with: tensorboard --logdir=./runs

# LOG_DIR: Directory where TensorBoard event files are written.
LOG_DIR = "./runs"

# LOG_INTERVAL: How often (in batches) to log training metrics.
# Lower values = more detailed logs but slightly slower training.
LOG_INTERVAL = 100

# =============================================================================
# 9. PATHS AND DIRECTORIES
# =============================================================================
# General output directory for any generated files (plots, reports, etc.).

OUTPUT_DIR = "./outputs"

# --- Create all necessary directories automatically ---
# This ensures the project works out of the box without manual folder creation.
_DIRECTORIES = [DATA_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR]

for _dir in _DIRECTORIES:
    os.makedirs(_dir, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_config():
    """
    Display all configuration settings in a readable format.
    Useful for logging at the start of a training run to record exact settings.
    """
    config_str = """
╔══════════════════════════════════════════════════════════════════╗
║           HANDWRITTEN DIGIT RECOGNITION CNN - CONFIG            ║
╚══════════════════════════════════════════════════════════════════╝

┌─── Training Hyperparameters ────────────────────────────────────┐
│  Batch Size:          {batch_size}
│  Learning Rate:       {lr}
│  Epochs:              {epochs}
│  Weight Decay:        {wd}
└─────────────────────────────────────────────────────────────────┘

┌─── Model Configuration ─────────────────────────────────────────┐
│  Number of Classes:   {num_classes}
│  Dropout Rate:        {dropout}
└─────────────────────────────────────────────────────────────────┘

┌─── Data Configuration ──────────────────────────────────────────┐
│  Data Directory:      {data_dir}
│  Number of Workers:   {num_workers}
└─────────────────────────────────────────────────────────────────┘

┌─── Device Configuration ────────────────────────────────────────┐
│  Device:              {device}
│  CUDA Available:      {cuda}
└─────────────────────────────────────────────────────────────────┘

┌─── Early Stopping ──────────────────────────────────────────────┐
│  Patience:            {patience}
│  Min Delta:           {min_delta}
└─────────────────────────────────────────────────────────────────┘

┌─── Model Saving ────────────────────────────────────────────────┐
│  Model Directory:     {model_dir}
│  Model Name Prefix:   {model_prefix}
│  Save Best Only:      {save_best}
└─────────────────────────────────────────────────────────────────┘

┌─── Data Augmentation ───────────────────────────────────────────┐
│  Enabled:             {aug_enabled}
│  Rotation Degrees:    {rotation}
│  Translate Range:     {translate}
│  Scale Range:         {scale}
└─────────────────────────────────────────────────────────────────┘

┌─── TensorBoard Logging ─────────────────────────────────────────┐
│  Log Directory:       {log_dir}
│  Log Interval:        every {log_interval} batches
└─────────────────────────────────────────────────────────────────┘

┌─── Paths ───────────────────────────────────────────────────────┐
│  Output Directory:    {output_dir}
└─────────────────────────────────────────────────────────────────┘
""".format(
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        wd=WEIGHT_DECAY,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        data_dir=DATA_DIR,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        cuda=torch.cuda.is_available(),
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        model_dir=MODEL_DIR,
        model_prefix=MODEL_NAME_PREFIX,
        save_best=SAVE_BEST_ONLY,
        aug_enabled=AUGMENTATION_ENABLED,
        rotation=ROTATION_DEGREES,
        translate=TRANSLATE_RANGE,
        scale=SCALE_RANGE,
        log_dir=LOG_DIR,
        log_interval=LOG_INTERVAL,
        output_dir=OUTPUT_DIR,
    )
    print(config_str)


# =============================================================================
# Run print_config() when this file is executed directly
# =============================================================================
if __name__ == "__main__":
    print_config()
