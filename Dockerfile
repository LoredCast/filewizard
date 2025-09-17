# Dockerfile
FROM python:3.13.7-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    ghostscript \
    poppler-utils \
    libreoffice \
    imagemagick \
    graphicsmagick \
    libvips-tools \
    ffmpeg \
    libheif-examples \
    inkscape \
    calibre \
    pngquant \
    pandoc \
    resvg \
    libvips-tools \
    build-essential \
    pkg-config \
    git \
    potrace \
    curl \
    texlive \
    jpegoptim \
    texlive-latex-extra \
    texlive-fonts-recommended \
    libxml2 \
    texlive-xetex \
    && rm -rf /var/lib/apt/lists/*



# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the app port
EXPOSE 8000
RUN chmod +x run.sh
# Command to run when container starts
CMD ["./run.sh"]
