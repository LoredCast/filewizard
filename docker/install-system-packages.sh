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
# shellcheck disable=SC2086  # the package lists are meant to be split into words
FORCE_SOURCE_DATE=1 apt-get "$@" install -y --no-install-recommends \
    ca-certificates python3 python3-venv supervisor \
    tesseract-ocr ${ocr_langs} \
    ghostscript poppler-utils unpaper \
    libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui libreoffice-draw-nogui \
    pandoc calibre ffmpeg libvips-tools graphicsmagick \
    sox libsox-fmt-mp3 lame pngquant jpegoptim libjpeg-turbo-progs libportaudio2 \
    fonts-dejavu-core fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea \
    ${extra}

# The base image ships a user "ubuntu" with uid 1000; the app user takes its place.
userdel --remove ubuntu 2>/dev/null || true
groupadd --gid 1000 filewizard
useradd --uid 1000 --gid 1000 --create-home --home-dir /home/filewizard --shell /usr/sbin/nologin filewizard

# Drop package lists, logs and caches: they are not needed at runtime and contain timestamps.
rm -rf /var/lib/apt/lists/* /var/log/apt /var/log/dpkg.log /var/log/alternatives.log \
    /var/cache/ldconfig/aux-cache /var/cache/fontconfig /var/cache/debconf/*-old
