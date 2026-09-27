# syntax=docker/dockerfile:1.7
#
# FileWizard container image. Pick the variant with the VARIANT build argument:
#
#   full   (default)  all tools incl. TeX, Inkscape, Docling       linux/amd64, linux/arm64
#   small             without TeX, Inkscape and Docling            linux/amd64, linux/arm64
#   cuda              full + CUDA runtime for GPU transcription     linux/amd64
#
#   docker buildx build --build-arg VARIANT=small -t filewizard:small .
#
# Builds are reproducible: base images are pinned by digest, Ubuntu packages come from a
# dated snapshot of the archive (amd64; the arm64 ports archive has no public snapshots),
# Python packages from the hash-pinned lock files in locks/. See docs/docker.md.
#
# Behind a TLS-inspecting proxy, pass its CA certificate as a build secret (it is not
# stored in the image):  docker buildx build --secret id=build_ca,src=/path/to/ca.crt .

ARG VARIANT=full
ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
# Only supplies a CA bundle for the first HTTPS apt update (the Ubuntu base image has none).
# It is replaced by Ubuntu's own ca-certificates package in the same step.
ARG BOOTSTRAP_CA_IMAGE=alpine:3.22@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8
# Ubuntu archive snapshot (https://snapshot.ubuntu.com) for amd64; empty = current archive.
ARG UBUNTU_SNAPSHOT=20260920T000000Z
# Tesseract OCR language packs (Tesseract codes, e.g. "eng deu spa"), or "all".
ARG TESSERACT_LANGS="eng deu fra spa ita por nld pol"
# PyTorch (needed by Docling) comes from this index as a CPU-only build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Fixes timestamps written during the build (TeX formats, .pyc files, /etc/shadow).
ARG SOURCE_DATE_EPOCH=0
ARG VERSION=dev

FROM ${BOOTSTRAP_CA_IMAGE} AS bootstrap-ca

# ==============================================================================
# System packages (converters, OCR, TeX, ...) and the unprivileged app user
# ==============================================================================
FROM ${UBUNTU_IMAGE} AS system-base
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
COPY --from=bootstrap-ca /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

FROM system-base AS system-full
ARG TARGETARCH UBUNTU_SNAPSHOT TESSERACT_LANGS SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh full

FROM system-base AS system-small
ARG TARGETARCH UBUNTU_SNAPSHOT TESSERACT_LANGS SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh small

# The CUDA image has the same system packages as the full one (and shares its layers).
FROM system-full AS system-cuda

FROM system-${VARIANT} AS system

# ==============================================================================
# Python dependencies in a virtualenv, installed from hash-pinned lock files
# ==============================================================================
FROM system AS python-deps
ARG VARIANT TORCH_INDEX_URL SOURCE_DATE_EPOCH
COPY locks/ /tmp/locks/
RUN --mount=type=secret,id=build_ca,required=false,mode=0444 set -eux; \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    python3 -m venv /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check --no-deps --only-binary :all: \
        --require-hashes -r "/tmp/locks/${VARIANT}.txt"; \
    if [ "${VARIANT}" != "small" ]; then \
        /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check --no-deps --only-binary :all: \
            --index-url "${TORCH_INDEX_URL}" -r /tmp/locks/torch-cpu.txt; \
    fi; \
    /opt/venv/bin/pip check

# ==============================================================================
# Runtime image
# ==============================================================================
FROM system AS runtime
ARG VARIANT VERSION

LABEL org.opencontainers.image.title="FileWizard" \
      org.opencontainers.image.description="Self-hosted file conversion, OCR, transcription and text-to-speech" \
      org.opencontainers.image.source="https://github.com/LoredCast/filewizard" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

COPY --from=python-deps /opt/venv /opt/venv
COPY docker/entrypoint.sh /usr/local/bin/filewizard-entrypoint
COPY docker/supervisord.conf /etc/filewizard/supervisord.conf
WORKDIR /app
COPY main.py settings.default.yml ./
COPY templates/ templates/
COPY static/ static/
RUN set -eux; \
    mkdir -p /app/config /app/data /app/models /app/uploads /app/processed; \
    chown filewizard:filewizard /app/config /app/data /app/models /app/uploads /app/processed

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/filewizard \
    PUID=1000 \
    PGID=1000 \
    UPLOADS_DIR=/app/uploads \
    PROCESSED_DIR=/app/processed \
    DATA_DIR=/app/data \
    HF_HOME=/app/models/huggingface \
    WEB_CONCURRENCY=4 \
    HUEY_WORKERS=4 \
    FORWARDED_ALLOW_IPS=* \
    QT_QPA_PLATFORM=offscreen \
    QTWEBENGINE_DISABLE_SANDBOX=1 \
    QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox \
    FILEWIZARD_VARIANT=${VARIANT} \
    FILEWIZARD_VERSION=${VERSION}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["filewizard-entrypoint"]
CMD ["supervisord", "-c", "/etc/filewizard/supervisord.conf"]

# Variant-specific settings; the last stage below is the image that gets built.
FROM runtime AS variant-full
FROM runtime AS variant-small
FROM runtime AS variant-cuda
# cuBLAS for CTranslate2 comes from the nvidia-cublas-cu12 wheel; the driver from the NVIDIA runtime.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TRANSCRIPTION_DEVICE=auto

FROM variant-${VARIANT}
