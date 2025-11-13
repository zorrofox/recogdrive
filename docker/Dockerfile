FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
# We explicitly install torch CPU first to avoid pulling in large CUDA dependencies,
# as we are targeting TPU (via JAX/Torchax).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Set environment variables for TPU if needed (optional, but good practice)
ENV PJRT_DEVICE=TPU
