# STAGE 1: BUILDER
# This stage installs build tools and Python dependencies
FROM python:3.12.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only the build-time dependencies needed for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# STAGE 2: FINAL
# This is the lean, final image for running the application
FROM python:3.12.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only the essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR dependencies
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu ghostscript poppler-utils \
    libreoffice \
    pandoc texlive-xetex \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    calibre \
    ffmpeg \
    libvips-tools \
    libxml2-dev \
    graphicsmagick \
    inkscape \
    resvg \
    potrace \
    pngquant \
    sox \
    jpegoptim \
    libsox-fmt-mp3 \
    lame \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    # Runtime libraries for Python packages
    libxml2 \
    # Process manager
    supervisor \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy supervisor config and application code
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

# Expose port and set executable permissions
EXPOSE 8000
RUN chmod +x run.sh

# Start the application
CMD ["/usr/bin/supervisord", "-n"]
