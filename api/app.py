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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy", "model_loaded": model_loaded}


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
