# Build arguments for different configurations
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.12-slim

# ==============================================================================
# STAGE 0: CUDA Build Stage
# Builds Python dependencies using the CUDA development image.
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-builder

ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python 3.12 from the deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    software-properties-common \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 && ln -sf python3.12 /usr/bin/python

WORKDIR /app

# Copy the CUDA-specific requirements and install them.
# PyTorch is installed first from its specific index for CUDA compatibility.
COPY requirements_cuda.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# ==============================================================================
# STAGE 1: Full Build Stage
# Builds Python dependencies for the non-GPU full version.
# ==============================================================================
FROM python:${PYTHON_VERSION} AS full-builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install the standard requirements.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# ==============================================================================
# STAGE 2: Small Build Stage
# Builds Python dependencies for the non-GPU small version.
# ==============================================================================
FROM python:${PYTHON_VERSION} AS small-builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the small-build requirements and install them.
COPY requirements_small.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# ==============================================================================
# STAGE 3: CUDA Final Stage
# Creates the final, runnable CUDA image.
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS cuda-final

ENV DEBIAN_FRONTEND=noninteractive

# CRITICAL: Install the same Python version as the builder to ensure compatibility.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 \
    && ln -sf python3.12 /usr/bin/python && ln -sf python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install all runtime dependencies for the CUDA build in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu ghostscript poppler-utils libreoffice \
    texlive-xetex texlive-latex-recommended texlive-fonts-recommended \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick inkscape \
    resvg potrace pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor libcudnn8 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed Python packages and binaries from the builder stage.
COPY --from=cuda-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=cuda-builder /usr/local/bin /usr/local/bin

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]

# ==============================================================================
# STAGE 4: Full Final Stage
# Creates the final, runnable full image (non-GPU).
# ==============================================================================
FROM python:${PYTHON_VERSION} AS full-final

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install all runtime dependencies in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu ghostscript poppler-utils libreoffice \
    pandoc lmodern texlive-xetex texlive-latex-recommended texlive-fonts-recommended \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick inkscape \
    resvg pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the full builder stage.
COPY --from=full-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=full-builder /usr/local/bin /usr/local/bin

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]

# ==============================================================================
# STAGE 5: Small Final Stage
# Creates the final, runnable small image (non-GPU).
# ==============================================================================
FROM python:${PYTHON_VERSION} AS small-final

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install the reduced set of runtime dependencies in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu ghostscript poppler-utils libreoffice pandoc \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick \
    pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the small builder stage.
COPY --from=small-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=small-builder /usr/local/bin /usr/local/bin

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]
