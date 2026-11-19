"""Path, upload, archive, command, and HTTP security helpers."""

import html
import logging
import mimetypes
import os
import re
import shlex
import zipfile
from pathlib import Path
from string import Formatter
from typing import Dict, List, Optional
from urllib.parse import urlparse

from starlette.datastructures import Headers
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

_SAFE_SUFFIX_RE = re.compile(r"^\.[a-z0-9]{1,10}$")
UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
MAX_UPLOAD_CHUNKS = 100_000
ZIP_JUNK_RE = re.compile(r"(^|/)(__MACOSX/|\._|\.DS_Store$|Thumbs\.db$)")
_TTS_NAME_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OCR_LANG_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")

ALLOWED_COMMAND_VARS = frozenset({
    "input", "output", "output_dir", "output_ext", "quality", "speed", "preset",
    "device", "dpi", "samplerate", "bitdepth", "filter", "model_name",
    "model_path", "voices_path", "lang",
})

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
CSP_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json")


def env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("true", "1", "t", "yes", "y")


def env_list(name: str) -> List[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def ensure_path_is_safe(p: Path, allowed_bases: List[Path]) -> Path:
    """Enhanced path safety check with traversal prevention."""
    try:
        resolved_p = p.resolve()
        if not any(resolved_p.is_relative_to(base.resolve()) for base in allowed_bases):
            raise ValueError(f"Path {resolved_p} is outside of allowed directories.")
        return resolved_p
    except Exception as e:
        logger.error(f"Path safety check failed for {p}: {e}")
        raise ValueError("Invalid or unsafe path specified.")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and XSS."""
    safe_name = secure_filename(filename or "")
    return html.escape(safe_name)


def safe_basename(filename: Optional[str]) -> str:
    """
    Filesystem-safe version of a user supplied file name that keeps a usable extension.
    secure_filename() alone drops non-ASCII characters, so "文件.pdf" became "pdf" and
    lost its extension; here the stem falls back to "file" instead.
    """
    name = re.split(r"[\\/]", filename or "")[-1]
    suffix = Path(name).suffix.lower()
    if not _SAFE_SUFFIX_RE.fullmatch(suffix):
        suffix = ""
    stem = secure_filename(name[: len(name) - len(suffix)] if suffix else name)[:100].strip("._") or "file"
    return f"{stem}{suffix}"


def safe_extract_zip(zip_path: Path, dest: Path, app_settings: dict) -> List[Path]:
    """
    Extracts regular files from an uploaded ZIP with limits on member count and total
    uncompressed size (zip bombs). Directory, symlink and OS metadata entries are skipped.
    zipfile.extract() already strips absolute paths and '..' components.
    """
    max_members = int(app_settings.get("max_zip_members") or 500)
    max_file_mb = float(app_settings.get("max_file_size_bytes", 100 * 1024 * 1024)) / (1024 * 1024)
    max_total_mb = float(app_settings.get("max_zip_uncompressed_mb") or max(4 * max_file_mb, 1024))
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info for info in zf.infolist()
            if not info.is_dir()
            and not ZIP_JUNK_RE.search(info.filename)
            and (info.external_attr >> 16) & 0o170000 != 0o120000  # symlink entries
        ]
        if len(members) > max_members:
            raise ValueError(
                f"ZIP archive has {len(members)} files; the limit is {max_members} (app_settings.max_zip_members)."
            )
        total_mb = sum(info.file_size for info in members) / (1024 * 1024)
        if total_mb > max_total_mb:
            raise ValueError(
                f"ZIP archive would extract to {total_mb:.0f} MB; the limit is {max_total_mb:.0f} MB "
                "(app_settings.max_zip_uncompressed_mb)."
            )
        dest.mkdir(parents=True, exist_ok=True)
        extracted = []
        for info in members:
            target = Path(zf.extract(info, dest))
            ensure_path_is_safe(target, [dest])
            extracted.append(target)
    return extracted


def resolve_whisper_model(model_size: Optional[str], whisper_settings: dict) -> str:
    """
    Only models listed in transcription_settings.whisper.allowed_models may be used.
    faster-whisper also accepts local paths and arbitrary Hugging Face repo ids, which
    must not be reachable from a request.
    """
    allowed = [str(m) for m in (whisper_settings.get("allowed_models") or [])]
    if not model_size:
        if "base" in allowed:
            return "base"
        if allowed:
            return allowed[0]
        raise ValueError("No Whisper models are allowed (transcription_settings.whisper.allowed_models).")
    if model_size not in allowed:
        raise ValueError(f"Invalid model size '{model_size}'. Allowed: {', '.join(allowed)}")
    return model_size


def validate_tts_model_name(model_name: Optional[str]) -> str:
    """Accepts 'piper/<voice>', '<voice>' (Piper) or 'kokoro/<lang>/<voice>'."""
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("A TTS voice (model_name) is required.")
    parts = model_name.split("/")
    engine, names = (parts[0], parts[1:]) if len(parts) > 1 else ("piper", parts)
    expected = {"piper": 1, "kokoro": 2}.get(engine)
    if expected != len(names) or not all(_TTS_NAME_PART_RE.fullmatch(n) and ".." not in n for n in names):
        raise ValueError(f"Invalid TTS voice '{model_name}'.")
    return model_name


def sanitize_output(output: str) -> str:
    """Sanitize output to prevent XSS."""
    if not output:
        return ""
    return html.escape(output[:2000])


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """Validate file type by extension."""
    if not allowed_extensions:
        return True
    return Path(filename).suffix.lower() in allowed_extensions


def get_file_mime_type(filename: str) -> str:
    """Get MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return Path(filename).suffix.lower()


def validate_and_build_command(template_str: str, mapping: Dict[str, str]) -> List[str]:
    fmt = Formatter()
    used = {fname for _, fname, _, _ in fmt.parse(template_str) if fname}
    bad = used - ALLOWED_COMMAND_VARS
    if bad:
        raise ValueError(f"Command template contains disallowed placeholders: {bad}")

    safe_mapping = dict(mapping)
    for name in used:
        if name not in safe_mapping:
            safe_mapping[name] = safe_mapping.get("output_ext", "") if name == "filter" else ""

    # Securely build the command by splitting the template BEFORE formatting.
    # This prevents argument injection if a value in the mapping (e.g. a filename)
    # contains spaces or other shell-special characters.
    command_parts = shlex.split(template_str)
    formatted_command = [part.format(**safe_mapping) for part in command_parts]
    return [part for part in formatted_command if part]


def host_key(netloc: str) -> str:
    """Normalizes host[:port] for comparison, dropping default ports."""
    netloc = (netloc or "").strip().lower()
    for default_port in (":80", ":443"):
        if netloc.endswith(default_port):
            return netloc[: -len(default_port)]
    return netloc


def csrf_allowed_hosts(headers: Headers, public_url: Optional[str], trusted_origins: List[str]) -> set:
    hosts = {host_key(headers.get("host", ""))}
    forwarded_host = headers.get("x-forwarded-host")
    if forwarded_host:
        hosts.add(host_key(forwarded_host.split(",")[0]))
    if public_url:
        hosts.add(host_key(urlparse(public_url).netloc))
    for origin in trusted_origins:
        hosts.add(host_key(urlparse(origin).netloc or origin))
    hosts.discard("")
    return hosts


def is_csrf_safe(method: str, headers: Headers, public_url: Optional[str], trusted_origins: List[str]) -> bool:
    """
    Rejects state-changing requests that a browser sent on behalf of another site.
    Browsers always attach Origin to cross-site POSTs; clients that send neither
    Origin nor Referer (curl, n8n, ...) are not browsers and are let through.
    """
    if method in SAFE_METHODS:
        return True
    source = headers.get("origin") or headers.get("referer")
    if not source:
        return True
    if source == "null":
        return False
    try:
        netloc = urlparse(source).netloc
    except ValueError:
        return False
    return host_key(netloc) in csrf_allowed_hosts(headers, public_url, trusted_origins)


def build_csp(frame_ancestors: str) -> str:
    return (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; "
        f"base-uri 'self'; form-action 'self'; frame-ancestors {frame_ancestors}"
    )
