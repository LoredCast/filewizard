#!/bin/sh
# Installs FileWizard's system packages and creates the app user. Runs during the image
# build (see Dockerfile); PACKAGE_SET is "full" or "small".
#
# Build arguments used: TARGETARCH, UBUNTU_SNAPSHOT, TESSERACT_LANGS, SOURCE_DATE_EPOCH.
set -eux

package_set="$1"

# Package sources: a dated snapshot of the Ubuntu archive makes rebuilds install the same
# versions. Canonical publishes no public snapshots of the ports archive (arm64).
if [ "${TARGETARCH}" = "amd64" ]; then
    uri="https://archive.ubuntu.com/ubuntu"
    if [ -n "${UBUNTU_SNAPSHOT}" ]; then
        uri="https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}"
    fi
else
    uri="https://ports.ubuntu.com/ubuntu-ports"
fi
rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
cat > /etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: ${uri}
Suites: noble noble-updates noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
Check-Valid-Until: no
EOF

if [ "${TESSERACT_LANGS}" = "all" ]; then
    ocr_langs="tesseract-ocr-all"
else
    ocr_langs=""
    for lang in ${TESSERACT_LANGS}; do
        ocr_langs="${ocr_langs} tesseract-ocr-${lang}"
    done
fi

# Calibre: Ubuntu's package, or the official build pinned in docker/versions.env. The official
# build bundles its own Qt; these are the system libraries it loads (Ubuntu's package pulls in
# the equivalent ones as dependencies).
. /tmp/versions.env
calibre_packages="calibre"
if [ -n "${CALIBRE_VERSION:-}" ]; then
    calibre_packages="xz-utils libegl1 libopengl0 libglx0 libxcb-cursor0 libxkbcommon0 libxkbcommon-x11-0 \
        libfontconfig1 libfreetype6 libnss3 libxcomposite1 libxdamage1 libxrandr2 libxtst6 libxkbfile1 \
        libasound2t64 libdbus-1-3"
fi
extra=""
if [ "${package_set}" = "full" ]; then
    extra="texlive-xetex texlive-latex-recommended texlive-fonts-recommended lmodern inkscape potrace libjxl-tools librsvg2-bin"
fi

# Optional CA certificate for building behind a TLS-inspecting proxy (build secret "build_ca").
# apt downloads as the unprivileged _apt user, hence the secret is mounted world-readable.
set --
if [ -f /run/secrets/build_ca ]; then
    set -- -o Acquire::https::CAInfo=/run/secrets/build_ca
fi

apt-get "$@" update
# Security updates for the packages already in the base image (from the same snapshot on amd64).
FORCE_SOURCE_DATE=1 apt-get "$@" upgrade -y --no-install-recommends
# shellcheck disable=SC2086  # the package lists are meant to be split into words
FORCE_SOURCE_DATE=1 apt-get "$@" install -y --no-install-recommends \
    ca-certificates python3 python3-venv supervisor \
    tesseract-ocr ${ocr_langs} \
    ghostscript poppler-utils unpaper \
    libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui libreoffice-draw-nogui \
    pandoc ${calibre_packages} ffmpeg libvips-tools graphicsmagick \
    sox libsox-fmt-mp3 lame pngquant jpegoptim libjpeg-turbo-progs libportaudio2 \
    fonts-dejavu-core fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea \
    ${extra}

# The base image ships a user "ubuntu" with uid 1000; the app user takes its place.
if [ -n "${CALIBRE_VERSION:-}" ]; then
    case "${TARGETARCH}" in
        amd64) calibre_arch=x86_64; calibre_sha256="${CALIBRE_SHA256_X86_64}" ;;
        arm64) calibre_arch=arm64; calibre_sha256="${CALIBRE_SHA256_ARM64}" ;;
        *) echo "No official Calibre build for ${TARGETARCH}" >&2; exit 1 ;;
    esac
    if [ -z "${calibre_sha256}" ]; then
        echo "docker/versions.env has no checksum for Calibre ${CALIBRE_VERSION} (${calibre_arch}); run scripts/update-deps.sh calibre" >&2
        exit 1
    fi
    calibre_url="https://github.com/kovidgoyal/calibre/releases/download/v${CALIBRE_VERSION}/calibre-${CALIBRE_VERSION}-${calibre_arch}.txz"
    python3 - "${calibre_url}" /tmp/calibre.txz <<'PY'
import os, ssl, sys, urllib.request
cafile = "/run/secrets/build_ca" if os.path.exists("/run/secrets/build_ca") else None
context = ssl.create_default_context(cafile=cafile)
with urllib.request.urlopen(sys.argv[1], context=context, timeout=600) as response, open(sys.argv[2], "wb") as out:
    while chunk := response.read(1 << 20):
        out.write(chunk)
PY
    if ! echo "${calibre_sha256}  /tmp/calibre.txz" | sha256sum -c -; then
        echo "Calibre ${CALIBRE_VERSION} (${calibre_arch}) does not match the checksum in docker/versions.env" >&2
        exit 1
    fi
    mkdir -p /opt/calibre
    tar -xJf /tmp/calibre.txz -C /opt/calibre
    rm /tmp/calibre.txz
    for tool in ebook-convert ebook-meta ebook-polish calibre-debug; do
        ln -sf "/opt/calibre/${tool}" "/usr/local/bin/${tool}"
    done
    ebook-convert --version
fi
userdel --remove ubuntu 2>/dev/null || true
groupadd --gid 1000 filewizard
useradd --uid 1000 --gid 1000 --create-home --home-dir /home/filewizard --shell /usr/sbin/nologin filewizard

# Drop package lists, logs and caches: they are not needed at runtime and contain timestamps.
rm -rf /var/lib/apt/lists/* /var/log/apt /var/log/dpkg.log /var/log/alternatives.log \
    /var/cache/ldconfig/aux-cache /var/cache/fontconfig /var/cache/debconf/*-old
