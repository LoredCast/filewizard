# Changelog

## Unreleased

### Faster

- The job history only fetches jobs that changed, polls less often while nothing changes, and pauses in background tabs.
- Jobs start within a second of being submitted (the background worker could take up to 10 s after a quiet spell).
- Image OCR runs Tesseract once per page instead of twice, and multi-page TIFFs no longer have to fit in memory. PDF OCR no longer reads the whole result to build its preview.
- Output formats for the chosen files appear instantly; with several files, only formats that work for all of them are offered.
- Up to 3 files upload at a time and the rest wait in a queue, so the first ones are processed sooner.
- Pages, scripts and job lists are compressed, and scripts and styles are cached by the browser until the next update.
- The TTS voice list no longer stalls the page (offline it could take ~90 s), is cached for all workers, and also lists voices that are already downloaded.
- "Download Selected as ZIP" streams to disk and skips re-compressing media, PDF and Office files.

### More reliable

- Jobs interrupted by a restart of the container or worker are marked as failed instead of staying "processing" forever.
- Upload chunks are retried after network errors; files that are too large, not allowed, or not valid for OCR are rejected before uploading. Uploads can be cancelled, and closing the tab mid-upload asks first.
- Webhook callbacks are retried after connection errors and 5xx responses.
- ZIP batches skip files the chosen converter does not accept instead of failing them, and big batches no longer push their own entry out of the history.
- The File Size column shows the output size as soon as a job finishes; cancelling shows immediately; "Clear History" stops running jobs.
- Leftover temporary files of interrupted uploads and conversions are cleaned up.

## 0.5.0

FileWizard 0.5 is about security, reliability and a rebuilt Docker image. The image now also runs on arm64 (Raspberry Pi 4/5, Apple Silicon, ARM servers), no longer runs as root, and the GPU image works again. Many conversions that failed in 0.4 now work. It also adds the most requested features from the issue tracker.

Upgrading is recommended for everyone, especially for instances reachable from the internet (see *Security*).

### Before you upgrade

`docker compose pull && docker compose up -d` works with your existing compose file and settings. The [upgrade notes](https://github.com/LoredCast/filewizard/blob/main/docs/docker.md#upgrading-from-04) explain the details. In short:

- **The app no longer runs as root.** On the first start, the container hands the mounted folders to `PUID`/`PGID` (default `1000`/`1000`, Unraid `99`/`100`). Set these to the user that should own the files.
- **Keep your job history:** 0.4 stored it inside the container. To take it along, copy it out *before* updating, then mount `/app/data` (and `/app/models`, so downloaded models survive updates):
  ```bash
  mkdir -p data && docker compose cp web:/app/jobs.db ./data/jobs.db
  ```
- **Behind a reverse proxy:** requests from other origins are now rejected. If uploads fail with *403*, forward the original `Host` header, or set `app_public_url` in the settings or the `CSRF_TRUSTED_ORIGINS` variable.
- **Settings page:** with OIDC login it is only available to admins (`admin_users`). Conversion commands are read-only there; edit `settings.yml` or set `ALLOW_COMMAND_EDITS=true`.
- **Using the API from a web page on another domain** now needs that origin in `ALLOWED_ORIGINS`. Scripts and tools such as curl or n8n are not affected.
- **Image tags:** `latest` and `small` keep their meaning, and `0.5-latest` follows the old `0.4-latest` naming. The GPU image is now `cuda` (was `0.3-cuda`).
- The image is based on Ubuntu 24.04 LTS instead of Debian, so some converters are older releases (e.g. LibreOffice 24.2, Pandoc 3.1.3, Tesseract 5.3) and output can differ slightly.

### New

- **arm64 images** for `latest` and `small` ([#14](https://github.com/LoredCast/filewizard/issues/14), [#18](https://github.com/LoredCast/filewizard/issues/18)).
- **Working NVIDIA GPU image** (`cuda`): transcribes on the GPU when the container has one and falls back to the CPU otherwise. It ships its own CUDA 12 libraries, so any driver from version 525 works.
- **Unraid templates** for the CPU and GPU images, see [`unraid/`](https://github.com/LoredCast/filewizard/tree/main/unraid) ([#5](https://github.com/LoredCast/filewizard/issues/5)).
- **OCR language selection** ([#26](https://github.com/LoredCast/filewizard/issues/26)): choose the language(s) per job next to *Start OCR*, and set a default in the settings. The image includes English, German, French, Spanish, Italian, Portuguese, Dutch and Polish; the `TESSERACT_LANGS` build argument adds more. The API accepts `ocr_language`.
- **Delete selected history items** ([#25](https://github.com/LoredCast/filewizard/issues/25)), including their files and, for ZIP batches, all sub-jobs. Running jobs are cancelled first.
- **Automatic cleanup:** `retention_days` deletes finished jobs and their files after the given number of days (off by default).
- **Conversion timeouts** can be changed per tool on the settings page ([#21](https://github.com/LoredCast/filewizard/issues/21)).
- **Restrict who may log in** with OIDC: `allowed_users` and `allowed_domains`.
- **SVG conversion** with rsvg-convert (to PNG, PDF, PS, EPS). cjxl (JPEG XL), cjpeg and potrace, which the settings listed but the image lacked, are now installed.
- Converters that are not installed are hidden in the UI and listed on the settings page, instead of failing when used.
- Whisper compute type `auto` (new default): picks the fastest type the CPU or GPU supports.
- Failed uploads say why (e.g. file too large) when you hover over the status.
- The web interface no longer loads anything from CDNs, so it also works without internet access.
- The container has a health check, and `/health` reports version and variant.
- New environment variables: `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `ALLOWED_ORIGINS`, `SESSION_COOKIE_SECURE`, `FRAME_ANCESTORS`, `ALLOW_COMMAND_EDITS`, `CHILD_CPU_LIMIT_SECONDS`, `CHILD_MEMORY_LIMIT_MB`, `STALE_UPLOAD_HOURS` (see the [README](https://github.com/LoredCast/filewizard/blob/main/README.md#security)) and `PUID`, `PGID` (see [docs/docker.md](https://github.com/LoredCast/filewizard/blob/main/docs/docker.md)).

### Security

- **With OIDC login, the settings page was reachable without logging in** and contained the OIDC client secret and the webhook token. It now requires an admin, and secrets are never sent to the browser. **If your instance was reachable by others, change the OIDC client secret at your identity provider and the webhook token.**
- Other websites could send requests to your instance through your browser (for example to change settings). Such cross-site requests are now rejected, and cross-origin API access is off unless allowed with `ALLOWED_ORIGINS`.
- File names shown in the job history could inject HTML and scripts into the page; all values are now escaped.
- Changing the programs that conversions run now needs `ALLOW_COMMAND_EDITS=true`; other settings only accept the fields the settings form has.
- Uploads: the size limit now also applies to chunked uploads (the way the web interface uploads), uploads of different users are kept apart, and ZIP files are limited in number of files and unpacked size.
- Whisper models and TTS voices are checked against the allowed ones.
- The container runs as an unprivileged user; converter processes have CPU and memory limits, and TeX shell escape is disabled.
- Security headers (Content Security Policy, frame protection, `nosniff`) and optional host name checking (`ALLOWED_HOSTS`).
- Admin rights and login restrictions ignore e-mail addresses the identity provider marks as unverified.

### Fixed

- LibreOffice conversions failed ([#16](https://github.com/LoredCast/filewizard/issues/16)).
- Conversions with a lot of console output (Calibre, ffmpeg, Docling) stalled and failed with "Conversion timed out" ([#21](https://github.com/LoredCast/filewizard/issues/21)).
- Calibre could not create PDFs in the container ([#9](https://github.com/LoredCast/filewizard/issues/9)).
- Logging out with Keycloak and other OIDC providers ([#20](https://github.com/LoredCast/filewizard/issues/20)).
- Without `SECRET_KEY`, OIDC login failed at random because every worker process made up its own session key. The key is now generated once and stored in `config/.secret_key`.
- *Cancel* now stops the running converter instead of letting it run until its timeout.
- Ghostscript PDF presets (screen, ebook, printer, prepress, PDF/A) were ignored.
- Pandoc conversions to plain text, Markdown and LaTeX failed.
- SoX "44k" presets produced 44000 Hz instead of 44100 Hz.
- Files with non-Latin names (e.g. `文件.pdf`) failed.
- ZIP batches could get stuck on "processing".
- Dropping a single file showed an empty format list; it now shows the formats available for that file type ([#23](https://github.com/LoredCast/filewizard/issues/23)).
- *Delete Processed Files* only covered the newest 100 jobs, and *Clear History* left the files behind.
- Settings changes now take effect in all worker processes right away.
- API: download links and status polling work with API tokens, webhooks fire for chunked API uploads, and tools with `_` in their id (e.g. `ghostscript_pdf`) are accepted.
- Whisper models stayed in memory for good; they are now unloaded when idle.
- `run.sh` no longer requires a `.env` file ([#13](https://github.com/LoredCast/filewizard/issues/13)).
- The app works with current Starlette releases; fresh installs failed to render pages.

### Docker image

| Image | Tags | Platforms |
|---|---|---|
| All tools | `latest`, `0.5.0`, `0.5`, `0.5-latest` | amd64, arm64 |
| Without TeX, Inkscape and Docling | `small`, `0.5.0-small`, `0.5-small` | amd64, arm64 |
| NVIDIA GPU | `cuda`, `latest-cuda`, `0.5.0-cuda`, `0.5-cuda` | amd64 |

- One Dockerfile for all variants. Builds are reproducible: base images pinned by digest, Ubuntu packages from a dated archive snapshot, Python packages from hash-pinned lock files.
- Job history, task queue and log live in `/app/data`, models in `/app/models`; both can be kept on volumes across updates.
- Building and publishing the images, from your own machine or with GitHub Actions, is described in [docs/docker.md](https://github.com/LoredCast/filewizard/blob/main/docs/docker.md).

Older releases: see the [GitHub releases](https://github.com/LoredCast/filewizard/releases).
