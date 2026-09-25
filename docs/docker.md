# Docker images

## Variants and tags

| Variant | Tags | Platforms | Contents |
|---|---|---|---|
| full | `latest`, `X.Y.Z`, `X.Y`, `X.Y-latest` | linux/amd64, linux/arm64 | All converters incl. TeX (Pandoc → PDF), Inkscape, Docling, Whisper, OCR, TTS |
| small | `small`, `X.Y.Z-small`, `X.Y-small` | linux/amd64, linux/arm64 | Without TeX, Inkscape and Docling |
| cuda | `cuda`, `latest-cuda`, `X.Y.Z-cuda`, `X.Y-cuda` | linux/amd64 | Full image that transcribes on an NVIDIA GPU |

`edge`, `edge-small` and `edge-cuda` are test builds, published only when the workflow is run manually with *push*.

## Running

Use [`docker-compose.yml`](../docker-compose.yml) as a starting point:

| Mount | Purpose |
|---|---|
| `/app/config` | `settings.yml` (created from the defaults on first start) and the generated session key |
| `/app/data` | Job history database |
| `/app/models` | Downloaded Whisper and TTS models; mounting it avoids re-downloading them after updates |
| `/app/uploads` | Uploaded files while they are processed |
| `/app/processed` | Results |

The app runs as an unprivileged user. `PUID`/`PGID` (default `1000`/`1000`, Unraid: `99`/`100`) choose its user and group; on start the container gives that user ownership of the mounted folders. You can also start the container with `--user UID:GID`, in which case the folders must already be writable for that user.

### NVIDIA GPU (`cuda` image)

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and give the container the GPU (see the `deploy` section in `docker-compose.yml`, or `--gpus all`). The image uses the GPU for transcription when one is available and falls back to the CPU otherwise (`TRANSCRIPTION_DEVICE=auto`). It ships the CUDA 12 libraries it needs, so any driver with CUDA 12 support (version 525 or newer) works. Check the GPU is visible:

```bash
docker compose exec web python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

### Unraid

Templates are in [`unraid/`](../unraid): `filewizard.xml` and `filewizard-cuda.xml` (needs the Nvidia Driver plugin).

## Upgrading from 0.4

Your existing `docker-compose.yml`, `settings.yml` and mounted folders keep working; `docker compose pull && docker compose up -d` is enough. Things to know:

- **Folder ownership:** 0.4 ran as root. On the first start, 0.5 hands the mounted folders to `PUID`/`PGID` (default `1000:1000`). Set them to the user that should own the files on the host (Unraid: `99`/`100`). On storage that refuses ownership changes (some NFS/SMB shares) the log shows a warning; the folders then have to be writable for that user, otherwise the container stops with an error naming the folder.
- **Job history:** 0.4 kept it inside the container, so every update lost it. Add the `/app/data` volume (and `/app/models`, so downloaded models survive updates too). To carry the current history over, copy it out of the old container *before* updating:
  ```bash
  mkdir -p data && docker compose cp web:/app/jobs.db ./data/jobs.db   # "web": the service name in docker-compose.yml
  ```
  Then add `- ./data:/app/data` and `- ./models:/app/models` under `volumes:` and update. Result files in the processed folder are kept either way.
- **Settings:** `settings.yml` is used as is; new options take their defaults. Customised conversion commands keep working but are read-only on the settings page (edit the file, or set `ALLOW_COMMAND_EDITS=true`).
- **Reverse proxy:** 0.5 rejects state-changing requests from other origins. If uploads fail with *403* behind a proxy, forward the original `Host` header, or set `app_public_url` in the settings or `CSRF_TRUSTED_ORIGINS`.
- **Login (OIDC):** `SECRET_KEY` is optional now (generated and stored in `config/.secret_key`). `allowed_users`/`allowed_domains` in `auth_settings` restrict who may log in; an account whose e-mail the provider marks as unverified (`email_verified: false`) is not treated as an admin.
- **Image tags:** `latest` (full) and `small` keep their meaning; if you pinned `0.4-latest`, switch to `0.5-latest` or `0.5`. The GPU image is `cuda` (was `0.3-cuda`). The old build targets (`full-final`, `small-final`) are gone: remove the `build:` section, or build with the `VARIANT` argument (see *Building locally*).
- **Converter versions:** the image is based on Ubuntu 24.04 LTS instead of Debian 13, so some converters are older releases (e.g. LibreOffice 24.2, Pandoc 3.1.3, Tesseract 5.3).

Going back: set the image to `loredcast/filewizard:0.4-latest`. 0.4 can still read the files and settings written by 0.5.

## Building locally

```bash
docker buildx build -t filewizard:local .                                   # full
docker buildx build --build-arg VARIANT=small -t filewizard:small .
docker buildx build --build-arg VARIANT=cuda --platform linux/amd64 -t filewizard:cuda .
docker/smoke-test.sh filewizard:local                                       # quick check
```

Or `docker compose build` with `VARIANT` set in `docker-compose.yml`.

Behind a proxy that inspects TLS, pass its CA certificate as a build secret; it is only used during the build and not stored in the image:

```bash
docker buildx build --secret id=build_ca,src=/path/to/proxy-ca.crt -t filewizard:local .
```

| Build argument | Default | Purpose |
|---|---|---|
| `VARIANT` | `full` | `full`, `small` or `cuda` |
| `TESSERACT_LANGS` | `eng deu fra spa ita por nld pol` | OCR language packs ([Tesseract codes](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)), or `all` (several hundred MB) |
| `UBUNTU_SNAPSHOT` | a fixed date | Ubuntu archive snapshot for amd64 packages; empty uses the current archive |
| `TORCH_INDEX_URL` | PyTorch CPU index | Where the CPU build of PyTorch (for Docling) comes from |
| `VERSION` | `dev` | Shown in `/health` and the image label |
| `SOURCE_DATE_EPOCH` | `0` | Timestamp for files created during the build; CI uses the commit time |

## Publishing to Docker Hub

### Automatically (recommended)

The workflow [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml) builds every variant on native amd64 and arm64 GitHub runners and publishes the multi-arch tags.

One-time setup:

1. On Docker Hub, create an access token with *Read & Write* scope (Account settings → Personal access tokens).
2. In the GitHub repository, add the secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (Settings → Secrets and variables → Actions). To publish to a repository other than `loredcast/filewizard`, add the variable `DOCKERHUB_IMAGE`.

Releasing a version:

```bash
git tag v0.5.0
git push origin v0.5.0
```

The workflow publishes `0.5.0`, `0.5`, `0.5-latest`, `latest`, the `-small` and `-cuda` equivalents, `small`, `cuda` and `latest-cuda`. A pre-release tag such as `v0.5.0-rc1` only gets its version tags. Progress is visible in the *Actions* tab; afterwards check the result with `docker buildx imagetools inspect loredcast/filewizard:0.5.0`. *Run workflow* on the Actions page builds (and optionally publishes as `edge`) without a tag. Pull requests that change the image build and smoke-test the small image.

### Manually

Test first: build for your own platform with `--load` and run `docker/smoke-test.sh` on it. Multi-arch builds need a `docker-container` builder, and QEMU when building for the other architecture (slow: expect an hour or more for arm64 on an amd64 machine):

```bash
docker login
docker buildx create --name filewizard --driver docker-container --use
docker run --privileged --rm tonistiigi/binfmt --install arm64   # on amd64 hosts

export VERSION=0.5.0 SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
for variant in full small cuda; do
    case $variant in
        full)  suffix="";       extra="latest";            platforms=linux/amd64,linux/arm64 ;;
        small) suffix="-small"; extra="small";             platforms=linux/amd64,linux/arm64 ;;
        cuda)  suffix="-cuda";  extra="cuda latest-cuda";  platforms=linux/amd64 ;;
    esac
    tags="-t loredcast/filewizard:${VERSION}${suffix} -t loredcast/filewizard:${VERSION%.*}${suffix}"
    for t in $extra; do tags="$tags -t loredcast/filewizard:$t"; done
    docker buildx build --platform "$platforms" \
        --build-arg VARIANT=$variant --build-arg VERSION=$VERSION --build-arg SOURCE_DATE_EPOCH \
        --output type=image,push=true,rewrite-timestamp=true $tags .
done
```

## Reproducible builds

Rebuilding the same commit produces the same software:

- Base images are pinned by digest (`UBUNTU_IMAGE`, `BOOTSTRAP_CA_IMAGE` in the Dockerfile).
- Ubuntu packages come from a dated snapshot of the archive (`UBUNTU_SNAPSHOT`, [snapshot.ubuntu.com](https://snapshot.ubuntu.com)). This applies to amd64; Canonical offers no public snapshots of the arm64 ports archive, so arm64 builds install the package versions current at build time.
- Python packages are installed from lock files in [`locks/`](../locks) with pinned versions and SHA-256 hashes, wheels only. PyTorch (CPU build, from the PyTorch index) is pinned by version in `locks/torch-cpu.txt`.
- Timestamps written during the build follow `SOURCE_DATE_EPOCH`, and `rewrite-timestamp=true` normalises file times in the layers.

To check, build twice without cache and compare the image digests:

```bash
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
for i in 1 2; do
    docker buildx build --no-cache --platform linux/amd64 --build-arg VARIANT=small --build-arg SOURCE_DATE_EPOCH \
        --output type=oci,dest=build-$i.tar,rewrite-timestamp=true .
done
tar -xOf build-1.tar index.json; echo; tar -xOf build-2.tar index.json
```

### Updating dependencies

- **Python packages:** edit `requirements*.txt`, then run `pip install uv && scripts/lock-requirements.sh` and commit `locks/`. The script resolves against PyPI for Python 3.12 and checks that every package installs from wheels on amd64 and arm64.
- **Ubuntu packages:** set `UBUNTU_SNAPSHOT` in the Dockerfile to a recent timestamp (`YYYYMMDDTHHMMSSZ`, UTC).
- **Base images:** update the digests, e.g. from `docker buildx imagetools inspect ubuntu:24.04`.

## Troubleshooting

- **The container exits with "… is not writable for user …"** (or permission errors in the log): set `PUID`/`PGID` to the owner of the folder on the host, or fix the ownership.
- **A converter runs out of memory:** converter processes are limited to 4 GB of memory each; raise `CHILD_MEMORY_LIMIT_MB` (`0` = no limit).
- **Behind a reverse proxy, uploads fail with 403:** forward the original `Host` header, or set `app_public_url` in the settings or `CSRF_TRUSTED_ORIGINS`.
