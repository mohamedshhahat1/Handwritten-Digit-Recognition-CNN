"""
ONNX Export for Cross-Platform Inference
==========================================

This script exports trained PyTorch models (CNN, ResNetCNN) to the ONNX format
for deployment across different platforms and runtimes:

    - ONNX Runtime (C++, Python, C#, Java, JavaScript)
    - TensorRT (NVIDIA GPU inference)
    - OpenVINO (Intel hardware)
    - CoreML (Apple devices via onnx-coreml)
    - TensorFlow/TFLite (via onnx-tf converter)
    - Web browsers (ONNX.js / ort-web)
    - Mobile (ONNX Runtime Mobile for Android/iOS)

ONNX (Open Neural Network Exchange) is an open standard format that enables
model interoperability between different ML frameworks and hardware platforms.

Usage:
    # Export the default CNN model
    python export_onnx.py

    # Export the ResNet model
    python export_onnx.py --model resnet

    # Export with a custom model path
    python export_onnx.py --model-path saved_models/mnist_cnn_best.pth

    # Export and validate with test inference
    python export_onnx.py --validate

    # Export with dynamic batch size (for variable batch inference)
    python export_onnx.py --dynamic-batch

    # Specify output path
    python export_onnx.py --output exports/my_model.onnx

Output:
    saved_models/mnist_cnn.onnx
    saved_models/mnist_resnet.onnx
"""

import os
import sys
import argparse
import time

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from model.cnn_model import CNN, ResNetCNN
from utils.model_manager import ModelManager


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_trained_model(model_type="cnn", model_path=None):
    """
    Load a trained model for ONNX export.

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
        print(f"  Model loaded from: {model_path}")
    else:
        manager = ModelManager()
        try:
            epoch, metrics = manager.load_model(model, version="best")
        except FileNotFoundError:
            try:
                epoch, metrics = manager.load_model(model, version="latest")
            except FileNotFoundError:
                legacy_path = os.path.join(config.MODEL_DIR, "mnist_cnn.pth")
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
# ONNX EXPORT
# =============================================================================

def export_to_onnx(model, output_path, dynamic_batch=False, opset_version=18):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (nn.Module): Trained model in eval mode.
        output_path (str): Path to save the .onnx file.
        dynamic_batch (bool): If True, allow variable batch sizes at inference.
        opset_version (int): ONNX opset version (default: 18, widely supported).

    Returns:
        str: Path to the saved ONNX file.
    """
    print(f"\n  Exporting to ONNX (opset {opset_version})...")

    # Create dummy input matching MNIST dimensions
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Configure dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print("  Dynamic batch size: enabled")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Export using the legacy API for broader compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,          # Store trained weights in the model
        opset_version=opset_version, # ONNX version to export to
        do_constant_folding=True,    # Optimize constant folding for inference
        input_names=['input'],       # Name the input tensor
        output_names=['output'],     # Name the output tensor
        dynamic_axes=dynamic_axes,   # Dynamic batch if requested
    )

    # Ensure all weights are embedded in a single file (no external data)
    # The new PyTorch exporter may split weights into .onnx.data
    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        import onnx
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx.save_model(onnx_model, output_path,
                        save_as_external_data=False)
        # Remove the now-unnecessary external data file
        if os.path.exists(external_data_path):
            os.remove(external_data_path)

    file_size = os.path.getsize(output_path)
    print(f"  ✓ ONNX model saved: {output_path}")
    print(f"    File size: {file_size / (1024*1024):.2f} MB")

    return output_path


# =============================================================================
# ONNX VALIDATION
# =============================================================================

def validate_onnx(onnx_path, model_pytorch):
    """
    Validate the ONNX model by comparing outputs with PyTorch.

    This ensures the export was correct by running the same input through
    both the original PyTorch model and the exported ONNX model, then
    comparing the outputs.

    Args:
        onnx_path (str): Path to the ONNX model file.
        model_pytorch (nn.Module): Original PyTorch model for comparison.

    Returns:
        bool: True if validation passes.
    """
    import onnx
    import onnxruntime as ort

    print("\n  Validating ONNX model...")

    # --- Step 1: Check ONNX model structure ---
    print("  [1/4] Checking ONNX model structure...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("        ✓ Model structure is valid")

    # Print model metadata
    graph = onnx_model.graph
    print(f"        Inputs:  {[i.name for i in graph.input]}")
    print(f"        Outputs: {[o.name for o in graph.output]}")
    print(f"        Nodes:   {len(graph.node)} operations")

    # --- Step 2: Create ONNX Runtime session ---
    print("  [2/4] Creating ONNX Runtime inference session...")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    print("        ✓ Session created successfully")

    # --- Step 3: Compare outputs ---
    print("  [3/4] Comparing PyTorch vs ONNX outputs...")

    # Generate random test input (batch_size=1 to work with/without dynamic batch)
    test_input = torch.randn(1, 1, 28, 28)

    # PyTorch inference
    model_pytorch.eval()
    with torch.no_grad():
        pytorch_output = model_pytorch(test_input).numpy()

    # ONNX Runtime inference
    ort_inputs = {'input': test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare (allow small numerical differences from float precision)
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_diff = np.mean(np.abs(pytorch_output - ort_output))
    outputs_match = np.allclose(pytorch_output, ort_output, rtol=1e-3, atol=1e-5)

    print(f"        Max difference:  {max_diff:.2e}")
    print(f"        Mean difference: {mean_diff:.2e}")

    if outputs_match:
        print("        ✓ Outputs match (within tolerance)")
    else:
        print("        ✗ WARNING: Outputs differ beyond tolerance!")
        print("          This may indicate an export issue.")

    # --- Step 4: Benchmark ONNX Runtime speed ---
    print("  [4/4] Benchmarking ONNX Runtime inference speed...")

    single_input = {'input': np.random.randn(1, 1, 28, 28).astype(np.float32)}

    # Warmup
    for _ in range(10):
        ort_session.run(None, single_input)

    # Benchmark
    num_runs = 200
    start = time.time()
    for _ in range(num_runs):
        ort_session.run(None, single_input)
    elapsed = (time.time() - start) / num_runs * 1000

    # PyTorch benchmark for comparison
    torch_input = torch.randn(1, 1, 28, 28)
    for _ in range(10):
        with torch.no_grad():
            model_pytorch(torch_input)
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model_pytorch(torch_input)
    pytorch_elapsed = (time.time() - start) / num_runs * 1000

    print(f"        ONNX Runtime: {elapsed:.3f} ms/sample")
    print(f"        PyTorch CPU:  {pytorch_elapsed:.3f} ms/sample")
    speedup = pytorch_elapsed / elapsed if elapsed > 0 else 0
    print(f"        Speedup:      {speedup:.2f}x")

    print("\n  ✓ Validation complete!")
    return outputs_match


# =============================================================================
# ACCURACY TEST ON MNIST
# =============================================================================

def test_onnx_accuracy(onnx_path, num_samples=10000):
    """
    Test ONNX model accuracy on the MNIST test set.

    Args:
        onnx_path (str): Path to the ONNX model.
        num_samples (int): Number of test samples to evaluate.

    Returns:
        float: Accuracy percentage.
    """
    import onnxruntime as ort
    from torchvision import datasets, transforms

    print("\n  Testing ONNX model accuracy on MNIST test set...")

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform
    )

    # Create ONNX Runtime session
    session = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider']
    )

    correct = 0
    total = min(num_samples, len(test_dataset))

    for i in range(total):
        image, label = test_dataset[i]
        # Add batch dimension: (1, 1, 28, 28)
        input_data = image.unsqueeze(0).numpy()
        output = session.run(None, {'input': input_data})[0]
        predicted = np.argmax(output, axis=1)[0]
        if predicted == label:
            correct += 1

    accuracy = 100.0 * correct / total
    print(f"  ONNX Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export trained models to ONNX format for cross-platform inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_onnx.py                      # Export CNN to ONNX
  python export_onnx.py --model resnet       # Export ResNetCNN
  python export_onnx.py --validate           # Export and validate
  python export_onnx.py --dynamic-batch      # Enable variable batch size
  python export_onnx.py --output model.onnx  # Custom output path

Supported deployment targets:
  - ONNX Runtime (C++, Python, C#, Java, JavaScript)
  - TensorRT (NVIDIA GPUs)
  - OpenVINO (Intel CPUs/GPUs/VPUs)
  - CoreML (Apple devices)
  - TFLite (Android/embedded via onnx-tf)
  - Web browsers (ONNX.js / ort-web)
        """
    )

    parser.add_argument(
        '--model', type=str, default='cnn',
        choices=['cnn', 'resnet'],
        help="Model architecture to export (default: cnn)"
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help="Path to specific model weights file"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output .onnx file path (default: saved_models/mnist_{model}.onnx)"
    )
    parser.add_argument(
        '--validate', action='store_true',
        help="Validate exported model (compare with PyTorch, benchmark speed)"
    )
    parser.add_argument(
        '--dynamic-batch', action='store_true',
        help="Enable dynamic batch size in the exported model"
    )
    parser.add_argument(
        '--opset', type=int, default=18,
        help="ONNX opset version (default: 18)"
    )
    parser.add_argument(
        '--test-accuracy', action='store_true',
        help="Test ONNX model accuracy on MNIST test set"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ONNX EXPORT — Cross-Platform Model Deployment")
    print("=" * 60)
    print(f"\n  Model:        {args.model.upper()}")
    print(f"  Opset:        {args.opset}")
    print(f"  Dynamic batch: {'Yes' if args.dynamic_batch else 'No'}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        output_path = os.path.join(config.MODEL_DIR, f"mnist_{args.model}.onnx")

    print(f"  Output:       {output_path}")

    # Load model
    print("\n  Loading trained model...")
    model = load_trained_model(model_type=args.model, model_path=args.model_path)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {num_params:,}")

    # Export
    export_to_onnx(
        model, output_path,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset
    )

    # Validate
    if args.validate:
        validate_onnx(output_path, model)

    # Test accuracy
    if args.test_accuracy:
        test_onnx_accuracy(output_path)

    # Summary
    pytorch_size = sum(p.numel() * p.element_size() for p in model.parameters())
    onnx_size = os.path.getsize(output_path)

    print("\n" + "=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Model:          {args.model.upper()}")
    print(f"  Parameters:     {num_params:,}")
    print(f"  PyTorch size:   {pytorch_size / (1024*1024):.2f} MB (weights only)")
    print(f"  ONNX size:      {onnx_size / (1024*1024):.2f} MB")
    print(f"  Output:         {output_path}")
    print(f"\n  Deploy with ONNX Runtime:")
    print(f"    import onnxruntime as ort")
    print(f"    session = ort.InferenceSession('{output_path}')")
    print(f"    result = session.run(None, {{'input': image_array}})")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
