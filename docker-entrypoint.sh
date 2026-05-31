#!/bin/bash
# Docker entrypoint script
# Trains the model on first run if no saved model exists, then starts the server.

set -e

MODEL_DIR="/app/saved_models"
CNN_MODEL="$MODEL_DIR/mnist_cnn_best.pth"
OCR_MODEL="$MODEL_DIR/ocr_crnn_best.pth"

echo "============================================================"
echo "  Handwritten Digit & Text Recognition — Docker Container"
echo "============================================================"
echo ""

# --- Train CNN if no model exists ---
if [ ! -f "$CNN_MODEL" ] && [ ! -f "$MODEL_DIR/mnist_cnn.pth" ]; then
    echo "🏋️  No trained CNN model found. Training now (first run only)..."
    echo "    This takes ~3-5 minutes on CPU."
    echo ""
    python train.py --epochs 10
    echo ""
    echo "✅ CNN training complete!"
    echo ""
else
    echo "✓ CNN model found: $(ls $MODEL_DIR/mnist_cnn*.pth 2>/dev/null | head -1)"
fi

# --- Train OCR if no model exists ---
if [ ! -f "$OCR_MODEL" ]; then
    echo ""
    echo "🔤 No trained OCR model found. Training now..."
    echo "    This takes ~2-3 minutes on CPU."
    echo ""
    python train_ocr.py --epochs 10 --samples 3000
    echo ""
    echo "✅ OCR training complete!"
    echo ""
else
    echo "✓ OCR model found: $OCR_MODEL"
fi

echo ""
echo "🚀 Starting web server on port 8000..."
echo "   Open http://localhost:8000 in your browser"
echo "============================================================"
echo ""

# --- Start the FastAPI server ---
exec uvicorn api.app:app --host 0.0.0.0 --port 8000
