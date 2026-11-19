# syntax=docker/dockerfile:1.7
#
# FileWizard container image. Pick the variant with the VARIANT build argument:
#
#   full   (default)  all tools incl. TeX, Inkscape, Docling       linux/amd64, linux/arm64
#   small             without TeX, Inkscape, Docling and Kokoro TTS linux/amd64, linux/arm64
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
ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
# Only supplies a CA bundle for the first HTTPS apt update (the Ubuntu base image has none).
# It is replaced by Ubuntu's own ca-certificates package in the same step.
ARG BOOTSTRAP_CA_IMAGE=alpine:3.22@sha256:bfe7ceb9f09ec0566c48fb93356e45a47a00be5a1dd14fa5b909e7709c66b034
# Ubuntu archive snapshot (https://snapshot.ubuntu.com) for amd64; empty = current archive.
ARG UBUNTU_SNAPSHOT=20261107T000000Z
# Tesseract OCR language packs (Tesseract codes, e.g. "eng deu spa"), or "all".
ARG TESSERACT_LANGS="eng deu fra spa ita por nld pol"
# PyTorch (needed by Docling) comes from this index as a CPU-only build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Fixes timestamps written during the build (TeX formats, .pyc files, /etc/shadow).
ARG SOURCE_DATE_EPOCH=0
ARG VERSION=dev

FROM ${BOOTSTRAP_CA_IMAGE} AS bootstrap-ca

# ==============================================================================
# System packages (shared base -> variant extras -> Calibre layer)
# ==============================================================================
FROM ${UBUNTU_IMAGE} AS system-base
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
COPY --from=bootstrap-ca /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

# Base converters shared by small, full, and cuda.
FROM system-base AS system-small-apt
ARG TARGETARCH UBUNTU_SNAPSHOT TESSERACT_LANGS SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=cache,id=apt-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh base

# Full-only converters layered on top of system-small-apt so base layers are shared.
FROM system-small-apt AS system-full-apt
ARG TARGETARCH SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=cache,id=apt-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh full-extra

# The CUDA image has the same system packages as the full one (and shares its layers).
FROM system-full-apt AS system-cuda-apt

# Calibre is installed in its own layer so version bumps in docker/versions.env do not
# invalidate the apt/TeX layers above.
FROM system-${VARIANT}-apt AS system
ARG TARGETARCH SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=bind,source=docker/versions.env,target=/tmp/versions.env \
    --mount=type=cache,id=apt-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=calibre-${TARGETARCH},target=/var/cache/calibre,sharing=locked \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh calibre

# ==============================================================================
# Python dependencies in a virtualenv (built in parallel with system packages)
# ==============================================================================
FROM system-base AS python-builder
ARG TARGETARCH UBUNTU_SNAPSHOT SOURCE_DATE_EPOCH
RUN --mount=type=bind,source=docker/install-system-packages.sh,target=/tmp/install-system-packages.sh \
    --mount=type=cache,id=apt-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-system-packages.sh python-builder

FROM python-builder AS python-deps-small
ARG TARGETARCH SOURCE_DATE_EPOCH
COPY locks/small.txt /tmp/locks/requirements.txt
RUN --mount=type=bind,source=docker/install-python-deps.sh,target=/tmp/install-python-deps.sh \
    --mount=type=cache,id=pip-${TARGETARCH},target=/root/.cache/pip \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-python-deps.sh small

FROM python-builder AS python-deps-full
ARG TARGETARCH TORCH_INDEX_URL SOURCE_DATE_EPOCH
COPY locks/full.txt /tmp/locks/requirements.txt
COPY locks/torch-cpu.txt /tmp/locks/torch-cpu.txt
RUN --mount=type=bind,source=docker/install-python-deps.sh,target=/tmp/install-python-deps.sh \
    --mount=type=cache,id=pip-${TARGETARCH},target=/root/.cache/pip \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-python-deps.sh full

FROM python-builder AS python-deps-cuda
ARG TARGETARCH TORCH_INDEX_URL SOURCE_DATE_EPOCH
COPY locks/cuda.txt /tmp/locks/requirements.txt
COPY locks/torch-cpu.txt /tmp/locks/torch-cpu.txt
RUN --mount=type=bind,source=docker/install-python-deps.sh,target=/tmp/install-python-deps.sh \
    --mount=type=cache,id=pip-${TARGETARCH},target=/root/.cache/pip \
    --mount=type=secret,id=build_ca,required=false,mode=0444 \
    sh /tmp/install-python-deps.sh cuda

FROM python-deps-${VARIANT} AS python-deps

# ==============================================================================
# Runtime image
# ==============================================================================
FROM system AS runtime
ARG VARIANT VERSION

LABEL org.opencontainers.image.title="FileWizard" \
      org.opencontainers.image.description="Self-hosted file conversion, OCR, transcription and text-to-speech" \
      org.opencontainers.image.source="https://github.com/LoredCast/filewizard" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}"

COPY --from=python-deps /opt/venv /opt/venv
COPY docker/entrypoint.sh /usr/local/bin/filewizard-entrypoint
COPY docker/supervisord.conf /etc/filewizard/supervisord.conf
WORKDIR /app
COPY main.py settings.default.yml ./
COPY filewizard/ filewizard/
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
