# Dockerfile for Handwritten Digit Recognition CNN
# Auto-trains on first run if no model exists, then serves the API.

FROM python:3.11-slim

# Labels
LABEL maintainer="mohamedshhahat1"
LABEL description="Handwritten Digit & Text Recognition with FastAPI, Web UI, and auto-training"
LABEL version="2.0"

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

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create model directory
RUN mkdir -p saved_models

# Expose the API port
EXPOSE 8000

# Use entrypoint script (auto-trains if needed, then starts server)
ENTRYPOINT ["./docker-entrypoint.sh"]
