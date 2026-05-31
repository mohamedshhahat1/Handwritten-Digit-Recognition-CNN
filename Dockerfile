# Dockerfile for Handwritten Digit Recognition CNN
# Multi-stage optimized build for FastAPI serving

FROM python:3.11-slim

# Labels
LABEL maintainer="digit-recognition-team"
LABEL description="Handwritten Digit Recognition CNN with FastAPI API and Web UI"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyTorch and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
