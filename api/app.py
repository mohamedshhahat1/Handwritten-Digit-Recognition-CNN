"""
FastAPI Prediction API for Handwritten Digit Recognition CNN

This module provides a REST API for digit recognition inference.
Endpoints allow image upload (multipart or base64) and return
predicted digits with confidence scores.
"""

import base64
import io
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path so we can import model and config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.cnn_model import CNN

# =============================================================================
# App Initialization
# =============================================================================

app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="API for recognizing handwritten digits (0-9) using a CNN model",
    version="1.0.0",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Loading
# =============================================================================

# Global model reference
model = None
model_loaded = False


def load_model():
    """
    Load the CNN model weights at startup.
    Tries best model first, then falls back to standard checkpoint.
    """
    global model, model_loaded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model.to(device)

    # Try loading best model first, then fall back
    model_paths = [
        os.path.join(PROJECT_ROOT, "saved_models", "mnist_cnn_best.pth"),
        os.path.join(PROJECT_ROOT, "saved_models", "mnist_cnn.pth"),
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device)
                # Handle both ModelManager format and raw state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                model_loaded = True
                print(f"Model loaded successfully from: {path}")
                return
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
                continue

    print("WARNING: No model weights found. Predictions will use an untrained model.")
    model.eval()
    model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    load_model()


# =============================================================================
# Image Preprocessing
# =============================================================================

# MNIST normalization parameters
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Preprocessing pipeline matching MNIST training transforms
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.

    Args:
        image: PIL Image in any mode/size.

    Returns:
        Tensor of shape (1, 1, 28, 28) ready for the model.
    """
    tensor = preprocess(image)
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    device = next(model.parameters()).device
    return tensor.to(device)


def run_inference(tensor: torch.Tensor) -> dict:
    """
    Run inference on a preprocessed tensor and return results.

    Args:
        tensor: Preprocessed image tensor of shape (1, 1, 28, 28).

    Returns:
        Dictionary with digit, confidence, and probabilities.
    """
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Convert probabilities to a dictionary
    prob_dict = {
        str(i): round(probabilities[0][i].item(), 4)
        for i in range(10)
    }

    return {
        "digit": predicted.item(),
        "confidence": round(confidence.item(), 4),
        "probabilities": prob_dict,
    }


# =============================================================================
# Request/Response Models
# =============================================================================

class Base64ImageRequest(BaseModel):
    """Request body for base64 image prediction."""
    image: str


class PredictionResponse(BaseModel):
    """Response body for prediction endpoints."""
    digit: int
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    model_loaded: bool


# =============================================================================
# Endpoints
# =============================================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict digit from an uploaded image file.

    Accepts multipart/form-data with an image file.
    The image is converted to grayscale, resized to 28x28,
    and normalized using MNIST statistics before inference.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        tensor = preprocess_image(image)
        result = run_inference(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return result


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    """
    Predict digit from a base64-encoded image.

    Accepts JSON body with a base64-encoded image string.
    Useful for sending canvas data from a web UI.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle data URL prefix (e.g., "data:image/png;base64,...")
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        tensor = preprocess_image(image)
        result = run_inference(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return result


@app.get("/")
async def serve_root():
    """Serve the web UI from the web/ directory."""
    web_dir = os.path.join(PROJECT_ROOT, "web")
    index_path = os.path.join(web_dir, "index.html")

    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {"message": "Handwritten Digit Recognition API", "docs": "/docs"}


@app.get("/ocr")
async def serve_ocr():
    """Serve the OCR web UI."""
    web_dir = os.path.join(PROJECT_ROOT, "web")
    ocr_path = os.path.join(web_dir, "ocr.html")

    if os.path.exists(ocr_path):
        return FileResponse(ocr_path)

    return {"message": "OCR page not found. Place ocr.html in web/ directory."}


# =============================================================================
# OCR Mode — CRNN Text Recognition
# =============================================================================

ocr_model = None
ocr_charset = None
ocr_loaded = False


def load_ocr_model():
    """Load the CRNN OCR model if available."""
    global ocr_model, ocr_charset, ocr_loaded

    try:
        from model.crnn_model import CRNN
        from model.ocr_utils import OCRCharset

        ocr_charset = OCRCharset()

        model_path = os.path.join(PROJECT_ROOT, "saved_models", "ocr_crnn_best.pth")
        if not os.path.exists(model_path):
            print("WARNING: No OCR model found. Train with: python train_ocr.py")
            return

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        ocr_model = CRNN(
            num_classes=checkpoint.get('charset_size', ocr_charset.num_classes),
            hidden_size=checkpoint.get('hidden_size', 256),
            num_layers=checkpoint.get('num_layers', 2),
        )
        ocr_model.load_state_dict(checkpoint['model_state_dict'])
        ocr_model.eval()
        ocr_loaded = True
        print(f"OCR model loaded successfully (CER: {checkpoint.get('cer', 'N/A')})")

    except Exception as e:
        print(f"WARNING: Failed to load OCR model: {e}")


# Load OCR model on startup
load_ocr_model()


def preprocess_ocr_image(image):
    """
    Preprocess an image for OCR inference.

    Resizes to height=32, maintains aspect ratio, converts to grayscale,
    and normalizes to [0, 1].

    Args:
        image (PIL.Image): Input image.

    Returns:
        torch.Tensor: Shape (1, 1, 32, W) ready for CRNN.
    """
    # Convert to grayscale
    image = image.convert('L')

    # Resize to height=32, maintain aspect ratio
    w, h = image.size
    new_h = 32
    new_w = max(int(w * new_h / h), 32)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Invert if light background (OCR expects white text on black)
    import numpy as np
    arr = np.array(image)
    if arr.mean() > 127:
        arr = 255 - arr

    # Convert to tensor and normalize
    tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0) / 255.0

    return tensor


class OCRResponse(BaseModel):
    """Response body for OCR prediction."""
    text: str
    confidence: float
    model_loaded: bool


class OCRBase64Request(BaseModel):
    """Request body for base64 OCR prediction."""
    image: str


@app.post("/predict/ocr", response_model=OCRResponse)
async def predict_ocr(file: UploadFile = File(...)):
    """
    Recognize text from an uploaded image using OCR (CRNN + CTC).

    Returns the recognized text string.
    """
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="OCR model not loaded. Train with: python train_ocr.py")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        tensor = preprocess_ocr_image(image)
        result = run_ocr_inference(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR inference error: {str(e)}")

    return result


@app.post("/predict/ocr/base64", response_model=OCRResponse)
async def predict_ocr_base64(request: OCRBase64Request):
    """
    Recognize text from a base64-encoded image using OCR.

    Accepts JSON body with a base64-encoded image (or data URL).
    Returns recognized text.
    """
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="OCR model not loaded. Train with: python train_ocr.py")

    try:
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        tensor = preprocess_ocr_image(image)
        result = run_ocr_inference(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR inference error: {str(e)}")

    return result


def run_ocr_inference(tensor):
    """
    Run OCR inference on a preprocessed image tensor.

    Args:
        tensor (torch.Tensor): Shape (1, 1, 32, W).

    Returns:
        dict: {"text": str, "confidence": float, "model_loaded": True}
    """
    from model.ocr_utils import ctc_decode_batch

    with torch.no_grad():
        output = ocr_model(tensor)  # (seq_len, 1, num_classes)

        # Get confidence (average max probability across timesteps)
        probs = torch.exp(output)  # Convert log-probs to probs
        max_probs = probs.max(dim=2)[0]  # Max prob at each timestep
        confidence = float(max_probs.mean())

        # Decode with CTC
        texts = ctc_decode_batch(output, ocr_charset)
        text = texts[0] if texts else ""

    return {
        "text": text,
        "confidence": round(confidence, 4),
        "model_loaded": True,
    }


@app.get("/health")
async def health_check_extended():
    """Extended health check with OCR status."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "ocr_loaded": ocr_loaded,
    }
