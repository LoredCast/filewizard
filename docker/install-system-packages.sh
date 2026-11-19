#!/bin/sh
# Installs FileWizard's system packages, Calibre, and creates the app user.
# Runs during the image build (see Dockerfile) in one of four modes:
#
#   base            core converters, Calibre runtime libs, and the filewizard user
#   full-extra      TeX Live, Inkscape, Potrace, libjxl, librsvg (layered on base)
#   calibre         official Calibre tarball (cached + SHA-256 verified) or Ubuntu package
#   python-builder  minimal python3 + venv stage for building /opt/venv in parallel
#
# Build arguments used: TARGETARCH, UBUNTU_SNAPSHOT, TESSERACT_LANGS, SOURCE_DATE_EPOCH.
set -eux

mode="$1"

configure_apt() {
    # Skip docs, manpages, and non-English locales (~150-300 MB smaller and faster dpkg unpack).
    cat > /etc/dpkg/dpkg.cfg.d/01_nodoc <<'EOF'
path-exclude=/usr/share/doc/*
path-include=/usr/share/doc/*/copyright
path-exclude=/usr/share/man/*
path-exclude=/usr/share/info/*
path-exclude=/usr/share/groff/*
path-exclude=/usr/share/lintian/*
path-exclude=/usr/share/linda/*
path-exclude=/usr/share/locale/*
path-include=/usr/share/locale/locale.alias
path-include=/usr/share/locale/en*
EOF

    # Keep downloaded .deb files inside the BuildKit /var/cache/apt cache mount across builds
    # (the cache mount is not committed into the image layer).
    rm -f /etc/apt/apt.conf.d/docker-clean
    printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";\n' > /etc/apt/apt.conf.d/99-keep-cache

    # Package sources: a dated snapshot of the Ubuntu archive makes rebuilds install the same
    # versions. Canonical publishes no public snapshots of the ports archive (arm64).
    if [ "${TARGETARCH}" = "amd64" ]; then
        uri="https://archive.ubuntu.com/ubuntu"
        if [ -n "${UBUNTU_SNAPSHOT:-}" ]; then
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
}

cleanup_apt_metadata() {
    # Drop logs and caches that are not needed at runtime and contain timestamps.
    # (/var/lib/apt/lists and /var/cache/apt are BuildKit cache mounts and never enter the layer.)
    rm -rf /var/log/apt /var/log/dpkg.log /var/log/alternatives.log \
        /var/cache/ldconfig/aux-cache /var/cache/fontconfig /var/cache/debconf/*-old
}

run_apt() {
    # Optional CA certificate for building behind a TLS-inspecting proxy (build secret "build_ca").
    # apt downloads as the unprivileged _apt user, hence the secret is mounted world-readable.
    if [ -f /run/secrets/build_ca ]; then
        apt-get -o Acquire::https::CAInfo=/run/secrets/build_ca "$@"
    else
        apt-get "$@"
    fi
}

case "${mode}" in
    python-builder)
        configure_apt
        run_apt update
        FORCE_SOURCE_DATE=1 run_apt install -y --no-install-recommends \
            ca-certificates python3 python3-venv
        cleanup_apt_metadata
        ;;

    base)
        configure_apt
        if [ "${TESSERACT_LANGS}" = "all" ]; then
            ocr_langs="tesseract-ocr-all"
        else
            ocr_langs=""
            for lang in ${TESSERACT_LANGS}; do
                ocr_langs="${ocr_langs} tesseract-ocr-${lang}"
            done
        fi

        # Runtime libraries loaded by the official Calibre build's bundled Qt.
        calibre_libs="xz-utils libegl1 libopengl0 libglx0 libxcb-cursor0 libxkbcommon0 libxkbcommon-x11-0 \
            libfontconfig1 libfreetype6 libnss3 libxcomposite1 libxdamage1 libxrandr2 libxtst6 libxkbfile1 \
            libasound2t64 libdbus-1-3"

        run_apt update
        # Security updates for the packages already in the base image (from the same snapshot on amd64).
        FORCE_SOURCE_DATE=1 run_apt upgrade -y --no-install-recommends
        # shellcheck disable=SC2086  # the package lists are meant to be split into words
        FORCE_SOURCE_DATE=1 run_apt install -y --no-install-recommends \
            ca-certificates python3 python3-venv supervisor \
            tesseract-ocr ${ocr_langs} \
            ghostscript poppler-utils unpaper \
            libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui libreoffice-draw-nogui \
            pandoc ${calibre_libs} ffmpeg libvips-tools graphicsmagick \
            sox libsox-fmt-mp3 lame pngquant jpegoptim libjpeg-turbo-progs libportaudio2 \
            fonts-dejavu-core fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea

        # The base image ships a user "ubuntu" with uid 1000; the app user takes its place.
        userdel --remove ubuntu 2>/dev/null || true
        groupadd --gid 1000 filewizard
        useradd --uid 1000 --gid 1000 --create-home --home-dir /home/filewizard --shell /usr/sbin/nologin filewizard
        cleanup_apt_metadata
        ;;

    full-extra)
        run_apt update
        FORCE_SOURCE_DATE=1 run_apt install -y --no-install-recommends \
            texlive-xetex texlive-latex-recommended texlive-fonts-recommended lmodern \
            inkscape potrace libjxl-tools librsvg2-bin
        cleanup_apt_metadata
        ;;

    calibre)
        # Calibre: Ubuntu's package when CALIBRE_VERSION is empty, or the official build pinned
        # in docker/versions.env (installed in its own layer so Calibre updates do not re-run apt).
        . /tmp/versions.env
        if [ -z "${CALIBRE_VERSION:-}" ]; then
            run_apt update
            FORCE_SOURCE_DATE=1 run_apt install -y --no-install-recommends calibre
            cleanup_apt_metadata
            exit 0
        fi

        case "${TARGETARCH}" in
            amd64) calibre_arch=x86_64; calibre_sha256="${CALIBRE_SHA256_X86_64}" ;;
            arm64) calibre_arch=arm64; calibre_sha256="${CALIBRE_SHA256_ARM64}" ;;
            *) echo "No official Calibre build for ${TARGETARCH}" >&2; exit 1 ;;
        esac
        if [ -z "${calibre_sha256}" ]; then
            echo "docker/versions.env has no checksum for Calibre ${CALIBRE_VERSION} (${calibre_arch}); run scripts/update-deps.sh calibre" >&2
            exit 1
        fi

        mkdir -p /var/cache/calibre
        archive="/var/cache/calibre/calibre-${CALIBRE_VERSION}-${calibre_arch}.txz"
        calibre_url="https://github.com/kovidgoyal/calibre/releases/download/v${CALIBRE_VERSION}/calibre-${CALIBRE_VERSION}-${calibre_arch}.txz"

        python3 - "${calibre_url}" "${archive}" "${calibre_sha256}" <<'PY'
import hashlib, http.client, os, ssl, sys, time, urllib.error, urllib.request

url, dest, expected_sha = sys.argv[1:4]
cafile = "/run/secrets/build_ca" if os.path.exists("/run/secrets/build_ca") else None
context = ssl.create_default_context(cafile=cafile)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

if os.path.exists(dest) and sha256_file(dest) == expected_sha:
    print(f"Using cached {dest}", file=sys.stderr)
    sys.exit(0)

for attempt in range(1, 6):
    try:
        offset = os.path.getsize(dest) if os.path.exists(dest) else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=context, timeout=60) as response:
            mode = "ab" if (offset and response.status == 206) else "wb"
            expected_len = int(response.headers.get("Content-Length") or 0)
            received = 0
            with open(dest, mode) as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    received += len(chunk)
            if expected_len and received < expected_len:
                raise http.client.IncompleteRead(b"", expected_len - received)
        got = sha256_file(dest)
        if got == expected_sha:
            sys.exit(0)
        print(f"Calibre download attempt {attempt}: checksum mismatch ({got}); retrying", file=sys.stderr)
        os.unlink(dest)
    except (OSError, ssl.SSLError, http.client.HTTPException, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
            if os.path.exists(dest) and sha256_file(dest) == expected_sha:
                sys.exit(0)
            if os.path.exists(dest):
                os.unlink(dest)
        print(f"Calibre download attempt {attempt} failed: {exc}", file=sys.stderr)
    if attempt < 5:
        time.sleep(min(2 ** attempt, 20))

sys.exit("Failed to download Calibre after 5 attempts")
PY
        if ! echo "${calibre_sha256}  ${archive}" | sha256sum -c -; then
            echo "Calibre ${CALIBRE_VERSION} (${calibre_arch}) does not match the checksum in docker/versions.env" >&2
            exit 1
        fi
        mkdir -p /opt/calibre
        tar -xJf "${archive}" -C /opt/calibre
        for tool in ebook-convert ebook-meta ebook-polish calibre-debug; do
            ln -sf "/opt/calibre/${tool}" "/usr/local/bin/${tool}"
        done
        ebook-convert --version
        ;;

    *)
        echo "Unknown mode '${mode}' (expected base, full-extra, calibre, or python-builder)" >&2
        exit 1
        ;;
esac
