# Use Python 3.10 slim for smaller image size
FROM python:3.10-slim-bullseye

# Install system dependencies for OpenCV, Pygame, and display
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    pulseaudio-utils \
    libpulse0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
# Using specific versions known to work well together
RUN pip install --no-cache-dir \
    mediapipe==0.10.9 \
    opencv-python-headless==4.8.1.78 \
    pygame==2.5.2 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    joblib==1.3.2

# Install PyTorch CPU version for MobileNet segmentation
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Set environment variables for display
ENV PYGAME_HIDE_SUPPORT_PROMPT=1
ENV SDL_VIDEODRIVER=x11

# Copy application code
COPY . /app

# Default command
CMD ["python", "main.py"]
