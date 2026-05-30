"""
Model Quantization for Edge Deployment
========================================

This script quantizes trained PyTorch models (CNN or ResNetCNN) for efficient
deployment on edge devices (mobile phones, Raspberry Pi, microcontrollers, etc.).

Quantization reduces model size and inference latency by converting floating-point
weights and activations (32-bit) to lower precision integers (8-bit), achieving:
    - 2-4x smaller model size (ideal for bandwidth-constrained deployments)
    - 2-4x faster inference on CPU (uses integer arithmetic)
    - Minimal accuracy loss (typically <0.5% on MNIST)

Three quantization modes are supported:

1. DYNAMIC QUANTIZATION (default, simplest)
   - Quantizes weights to INT8 at save time
   - Activations quantized dynamically at runtime
   - No calibration data needed
   - Best for: quick deployment, LSTM/FC-heavy models

2. STATIC QUANTIZATION (post-training)
   - Quantizes both weights AND activations to INT8
   - Requires a calibration dataset to determine activation ranges
   - Smaller and faster than dynamic quantization
   - Best for: CNN models, maximum size reduction

3. QUANTIZATION-AWARE TRAINING (QAT)
   - Simulates quantization during training (fake quantize)
   - Fine-tunes the model to recover accuracy lost from quantization
   - Highest accuracy of all methods
   - Best for: when accuracy is critical, willing to retrain

Usage:
    # Dynamic quantization (quick, no calibration needed)
    python quantize.py

    # Static quantization (smaller model, needs calibration)
    python quantize.py --mode static

    # Quantization-aware training (best accuracy)
    python quantize.py --mode qat --epochs 3

    # Quantize the ResNetCNN model
    python quantize.py --model resnet

    # Specify a custom model path
    python quantize.py --model-path saved_models/mnist_cnn_best.pth

    # Compare all quantization methods
    python quantize.py --compare

Output:
    saved_models/mnist_cnn_quantized_dynamic.pth
    saved_models/mnist_cnn_quantized_static.pth
    saved_models/mnist_cnn_quantized_qat.pth
"""

import os
import sys
import time
import argparse
import copy

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from model.cnn_model import CNN, ResNetCNN
from utils.model_manager import ModelManager


# =============================================================================
# CONSTANTS
# =============================================================================

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
SAVE_DIR = config.MODEL_DIR


# =============================================================================
# DATA LOADING (for calibration and evaluation)
# =============================================================================

def get_calibration_loader(num_samples=1000, batch_size=64):
    """
    Get a small subset of training data for static quantization calibration.

    The calibration dataset is used to determine the range of activations
    in the model, which is needed to choose optimal quantization parameters.
    A few hundred to a few thousand samples is typically sufficient.

    Args:
        num_samples (int): Number of calibration samples.
        batch_size (int): Batch size for calibration.

    Returns:
        DataLoader: Calibration data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    full_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )

    # Take a subset for calibration
    calibration_dataset = torch.utils.data.Subset(
        full_dataset, range(num_samples)
    )

    return DataLoader(
        calibration_dataset, batch_size=batch_size, shuffle=False
    )


def get_test_loader(batch_size=256):
    """Get the MNIST test set for accuracy evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    test_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform
    )

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_train_loader(batch_size=64):
    """Get the full MNIST training set for QAT fine-tuning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_trained_model(model_type="cnn", model_path=None):
    """
    Load a trained model for quantization.

    Args:
        model_type (str): "cnn" or "resnet".
        model_path (str, optional): Explicit path to model weights.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    if model_type == "resnet":
        model = ResNetCNN()
    else:
        model = CNN()

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from: {model_path}")
    else:
        # Use ModelManager to find best model
        manager = ModelManager()
        try:
            epoch, metrics = manager.load_model(model, version="best")
        except FileNotFoundError:
            try:
                epoch, metrics = manager.load_model(model, version="latest")
            except FileNotFoundError:
                legacy_path = os.path.join(SAVE_DIR, "mnist_cnn.pth")
                if os.path.exists(legacy_path):
                    state_dict = torch.load(legacy_path, map_location='cpu',
                                            weights_only=False)
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                else:
                    raise FileNotFoundError(
                        "No trained model found! Train first with: python train.py"
                    )

    model.eval()
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_accuracy(model, test_loader):
    """
    Evaluate model accuracy on the test set.

    Args:
        model (nn.Module): Model to evaluate.
        test_loader (DataLoader): Test data loader.

    Returns:
        float: Accuracy as a percentage (0-100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def measure_inference_time(model, input_shape=(1, 1, 28, 28), num_runs=100):
    """
    Measure average inference time per sample.

    Args:
        model (nn.Module): Model to benchmark.
        input_shape (tuple): Input tensor shape.
        num_runs (int): Number of inference runs to average over.

    Returns:
        float: Average inference time in milliseconds.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    # Measure
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(dummy_input)
    elapsed = (time.time() - start) / num_runs * 1000  # ms

    return elapsed


def get_model_size(model, save_path=None):
    """
    Get model size in bytes (serialized).

    Args:
        model (nn.Module): Model to measure.
        save_path (str, optional): If provided, save and measure file size.

    Returns:
        int: Model size in bytes.
    """
    if save_path:
        torch.save(model.state_dict(), save_path)
        size = os.path.getsize(save_path)
    else:
        # Use a temp file to measure
        tmp_path = os.path.join(SAVE_DIR, "_tmp_size_check.pth")
        torch.save(model.state_dict(), tmp_path)
        size = os.path.getsize(tmp_path)
        os.remove(tmp_path)

    return size


# =============================================================================
# 1. DYNAMIC QUANTIZATION
# =============================================================================

def quantize_dynamic(model):
    """
    Apply dynamic quantization to the model.

    Dynamic quantization converts weights to INT8 at save time, and
    quantizes activations on-the-fly at inference time. It requires
    no calibration data and is the simplest quantization method.

    Best for models with large Linear layers (where compute is dominated
    by matrix multiplication). Less effective for Conv layers.

    Args:
        model (nn.Module): Trained model in eval mode.

    Returns:
        nn.Module: Dynamically quantized model.
    """
    print("\n" + "=" * 60)
    print("  DYNAMIC QUANTIZATION")
    print("=" * 60)
    print("  Method: Weights → INT8 at save time")
    print("          Activations → quantized dynamically at runtime")
    print("  Pros: No calibration needed, simple to apply")
    print("  Cons: Less speedup for Conv layers vs Linear layers")
    print("-" * 60)

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )

    print("  ✓ Dynamic quantization applied successfully")
    return quantized_model


# =============================================================================
# 2. STATIC QUANTIZATION
# =============================================================================

def quantize_static(model, calibration_loader):
    """
    Apply static (post-training) quantization to the model.

    Static quantization pre-determines the quantization parameters for
    activations by running calibration data through the model. Both weights
    and activations are quantized to INT8, providing maximum compression.

    Args:
        model (nn.Module): Trained model in eval mode.
        calibration_loader (DataLoader): Data for calibration (determines
            activation ranges).

    Returns:
        nn.Module: Statically quantized model.
    """
    print("\n" + "=" * 60)
    print("  STATIC QUANTIZATION (Post-Training)")
    print("=" * 60)
    print("  Method: Weights → INT8, Activations → INT8 (pre-calibrated)")
    print("  Pros: Maximum compression, fastest inference")
    print("  Cons: Requires calibration data, may lose more accuracy")
    print("-" * 60)

    # Deep copy to avoid modifying the original
    model_prepared = copy.deepcopy(model)
    model_prepared.eval()

    # Set quantization configuration
    # Use fbgemm for x86 CPUs, qnnpack for ARM (mobile/edge)
    backend = "fbgemm"  # Change to "qnnpack" for ARM devices
    model_prepared.qconfig = quant.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    # Fuse Conv+BN+ReLU layers for efficiency
    # This combines sequential operations into single fused operations
    print("  Fusing Conv+BN+ReLU layers...")
    model_fused = _fuse_model(model_prepared)

    # Insert observers to track activation ranges during calibration
    print("  Inserting quantization observers...")
    quant.prepare(model_fused, inplace=True)

    # Run calibration data through the model
    print(f"  Running calibration ({len(calibration_loader.dataset)} samples)...")
    model_fused.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            model_fused(images)

    # Convert to quantized model
    print("  Converting to quantized model...")
    quantized_model = quant.convert(model_fused, inplace=True)

    print("  ✓ Static quantization applied successfully")
    return quantized_model


def _fuse_model(model):
    """
    Fuse Conv+BN+ReLU layers for quantization efficiency.

    Fused operations run faster because they avoid materializing
    intermediate results between BN and ReLU.

    Args:
        model (nn.Module): Model to fuse.

    Returns:
        nn.Module: Model with fused layers.
    """
    if isinstance(model, CNN):
        # Fuse conv1+bn1, conv2+bn2 (ReLU is applied via F.relu, not a module)
        # For static quantization, we need module-based ReLU
        # Since our model uses F.relu, we'll skip fusion for now
        # and rely on the quantization backend to handle it
        pass
    elif isinstance(model, ResNetCNN):
        # ResNet has more complex fusion opportunities
        pass

    return model


# =============================================================================
# 3. QUANTIZATION-AWARE TRAINING (QAT)
# =============================================================================

def quantize_qat(model, train_loader, test_loader, num_epochs=3, lr=0.0001):
    """
    Apply Quantization-Aware Training (QAT).

    QAT inserts "fake quantization" nodes during training that simulate
    the effects of quantization. The model learns to compensate for
    quantization noise, resulting in higher accuracy after final conversion.

    Args:
        model (nn.Module): Trained model to fine-tune.
        train_loader (DataLoader): Training data.
        test_loader (DataLoader): Test data for validation.
        num_epochs (int): Number of QAT fine-tuning epochs.
        lr (float): Learning rate for fine-tuning (should be low).

    Returns:
        nn.Module: Quantized model after QAT.
    """
    print("\n" + "=" * 60)
    print("  QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 60)
    print(f"  Method: Simulate quantization during training")
    print(f"  Epochs: {num_epochs}, LR: {lr}")
    print("  Pros: Best accuracy preservation")
    print("  Cons: Requires retraining, more compute")
    print("-" * 60)

    # Deep copy
    model_qat = copy.deepcopy(model)
    model_qat.train()

    # Set QAT config
    backend = "fbgemm"
    model_qat.qconfig = quant.get_default_qat_qconfig(backend)
    torch.backends.quantized.engine = backend

    # Prepare for QAT (inserts fake quantize modules)
    print("  Preparing model for QAT...")
    quant.prepare_qat(model_qat, inplace=True)

    # Fine-tune with fake quantization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=lr)

    print(f"  Fine-tuning with fake quantization...")
    for epoch in range(1, num_epochs + 1):
        model_qat.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model_qat(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        val_acc = evaluate_accuracy(model_qat, test_loader)
        avg_loss = running_loss / len(train_loader)

        print(f"  Epoch [{epoch}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    # Convert to final quantized model
    print("  Converting to quantized model...")
    model_qat.eval()
    quantized_model = quant.convert(model_qat, inplace=True)

    print("  ✓ QAT quantization applied successfully")
    return quantized_model


# =============================================================================
# SAVE & REPORT
# =============================================================================

def save_quantized_model(model, mode, model_type="cnn"):
    """
    Save the quantized model to disk.

    Args:
        model (nn.Module): Quantized model.
        mode (str): Quantization mode (dynamic, static, qat).
        model_type (str): Model type (cnn, resnet).

    Returns:
        str: Path to saved model.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"mnist_{model_type}_quantized_{mode}.pth"
    save_path = os.path.join(SAVE_DIR, filename)
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to: {save_path}")
    return save_path


def print_comparison_report(results):
    """
    Print a formatted comparison table of quantization results.

    Args:
        results (list): List of dicts with keys: name, accuracy, size, time.
    """
    print("\n")
    print("=" * 70)
    print("  QUANTIZATION COMPARISON REPORT")
    print("=" * 70)
    print()
    print(f"  {'Model':<28} {'Accuracy':<12} {'Size':<14} {'Inference':<12}")
    print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*12}")

    baseline = results[0]  # Original model is first

    for r in results:
        acc_str = f"{r['accuracy']:.2f}%"
        size_mb = r['size'] / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB"
        time_str = f"{r['time']:.2f} ms"

        # Show compression ratio vs baseline
        if r != baseline:
            compression = baseline['size'] / r['size']
            speedup = baseline['time'] / r['time'] if r['time'] > 0 else 0
            size_str += f" ({compression:.1f}x)"
            time_str += f" ({speedup:.1f}x)"

        print(f"  {r['name']:<28} {acc_str:<12} {size_str:<14} {time_str:<12}")

    print()
    print("=" * 70)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantize trained CNN/ResNet models for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize.py                        # Dynamic quantization (default)
  python quantize.py --mode static          # Static quantization
  python quantize.py --mode qat --epochs 3  # Quantization-aware training
  python quantize.py --model resnet         # Quantize ResNetCNN
  python quantize.py --compare              # Compare all methods
        """
    )

    parser.add_argument(
        '--mode', type=str, default='dynamic',
        choices=['dynamic', 'static', 'qat'],
        help="Quantization method (default: dynamic)"
    )
    parser.add_argument(
        '--model', type=str, default='cnn',
        choices=['cnn', 'resnet'],
        help="Model architecture to quantize (default: cnn)"
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help="Path to specific model weights file"
    )
    parser.add_argument(
        '--epochs', type=int, default=3,
        help="Number of QAT fine-tuning epochs (default: 3)"
    )
    parser.add_argument(
        '--compare', action='store_true',
        help="Run all quantization methods and compare results"
    )

    args = parser.parse_args()

    print("\n")
    print("*" * 60)
    print("*   MODEL QUANTIZATION FOR EDGE DEPLOYMENT                *")
    print("*" * 60)
    print(f"\n  Model type: {args.model.upper()}")
    print(f"  Mode: {args.mode}")

    # Load the trained model
    print("\n  Loading trained model...")
    model = load_trained_model(model_type=args.model, model_path=args.model_path)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    test_loader = get_test_loader()

    if args.compare:
        # Run all methods and compare
        run_comparison(model, args.model, test_loader, qat_epochs=args.epochs)
    else:
        # Run single quantization method
        run_single(model, args.model, args.mode, test_loader, qat_epochs=args.epochs)


def run_single(model, model_type, mode, test_loader, qat_epochs=3):
    """Run a single quantization method."""

    # Baseline metrics
    print("\n  Evaluating original model...")
    orig_acc = evaluate_accuracy(model, test_loader)
    orig_time = measure_inference_time(model)
    orig_size = get_model_size(model)
    print(f"  Original: Acc={orig_acc:.2f}%, "
          f"Size={orig_size/(1024*1024):.2f}MB, "
          f"Time={orig_time:.2f}ms")

    # Quantize
    if mode == 'dynamic':
        quantized = quantize_dynamic(model)
    elif mode == 'static':
        calibration_loader = get_calibration_loader()
        quantized = quantize_static(model, calibration_loader)
    elif mode == 'qat':
        train_loader = get_train_loader()
        quantized = quantize_qat(model, train_loader, test_loader,
                                 num_epochs=qat_epochs)

    # Evaluate quantized model
    print("\n  Evaluating quantized model...")
    q_acc = evaluate_accuracy(quantized, test_loader)
    q_time = measure_inference_time(quantized)

    # Save
    save_path = save_quantized_model(quantized, mode, model_type)
    q_size = os.path.getsize(save_path)

    # Report
    results = [
        {"name": f"Original ({model_type.upper()})",
         "accuracy": orig_acc, "size": orig_size, "time": orig_time},
        {"name": f"Quantized ({mode})",
         "accuracy": q_acc, "size": q_size, "time": q_time},
    ]
    print_comparison_report(results)

    acc_drop = orig_acc - q_acc
    compression = orig_size / q_size
    print(f"  Summary:")
    print(f"    Accuracy drop:    {acc_drop:.2f}%")
    print(f"    Size reduction:   {compression:.1f}x smaller")
    print(f"    Speedup:          {orig_time/q_time:.1f}x faster")
    print()


def run_comparison(model, model_type, test_loader, qat_epochs=3):
    """Run all quantization methods and compare."""

    print("\n  Running full comparison of all quantization methods...")

    # Baseline
    print("\n  [1/4] Evaluating original model...")
    orig_acc = evaluate_accuracy(model, test_loader)
    orig_time = measure_inference_time(model)
    orig_size = get_model_size(model)

    results = [
        {"name": f"Original ({model_type.upper()})",
         "accuracy": orig_acc, "size": orig_size, "time": orig_time}
    ]

    # Dynamic
    print("\n  [2/4] Dynamic quantization...")
    q_dynamic = quantize_dynamic(model)
    d_acc = evaluate_accuracy(q_dynamic, test_loader)
    d_time = measure_inference_time(q_dynamic)
    d_path = save_quantized_model(q_dynamic, "dynamic", model_type)
    d_size = os.path.getsize(d_path)
    results.append({"name": "Dynamic (INT8 weights)",
                    "accuracy": d_acc, "size": d_size, "time": d_time})

    # Static
    print("\n  [3/4] Static quantization...")
    calibration_loader = get_calibration_loader()
    try:
        q_static = quantize_static(model, calibration_loader)
        s_acc = evaluate_accuracy(q_static, test_loader)
        s_time = measure_inference_time(q_static)
        s_path = save_quantized_model(q_static, "static", model_type)
        s_size = os.path.getsize(s_path)
        results.append({"name": "Static (INT8 weights+acts)",
                        "accuracy": s_acc, "size": s_size, "time": s_time})
    except Exception as e:
        print(f"  ⚠ Static quantization failed: {e}")
        print("    (This can happen with F.relu — module-based ReLU needed)")

    # QAT
    print("\n  [4/4] Quantization-Aware Training...")
    train_loader = get_train_loader()
    try:
        q_qat = quantize_qat(model, train_loader, test_loader,
                             num_epochs=qat_epochs)
        qat_acc = evaluate_accuracy(q_qat, test_loader)
        qat_time = measure_inference_time(q_qat)
        qat_path = save_quantized_model(q_qat, "qat", model_type)
        qat_size = os.path.getsize(qat_path)
        results.append({"name": "QAT (fine-tuned INT8)",
                        "accuracy": qat_acc, "size": qat_size, "time": qat_time})
    except Exception as e:
        print(f"  ⚠ QAT quantization failed: {e}")

    # Print comparison
    print_comparison_report(results)


if __name__ == "__main__":
    main()
