# main.py (merged)
import html
import threading
import logging
import shutil
import subprocess
import traceback
import uuid
import shlex
import yaml
import os
import httpx
import glob
import secrets
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import resource
import fcntl
import signal
import tempfile
import gc
from threading import Semaphore
from logging.handlers import RotatingFileHandler
from urllib.parse import urljoin, urlparse, urlencode, quote
from io import BytesIO
import zipfile
import sys
import re
import importlib
import collections.abc
import copy
import time
import ocrmypdf
import pypdf
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
from pytesseract import TesseractNotFoundError
from PIL import Image, UnidentifiedImageError
from faster_whisper import WhisperModel
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status, Body, WebSocket, Query, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from huey import SqliteHuey, crontab
from pydantic import BaseModel, ConfigDict, field_serializer
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine, delete, event, text)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
from string import Formatter
from werkzeug.utils import secure_filename
from typing import List as TypingList
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import Headers, MutableHeaders
import warnings
import authlib.deprecate  # installs Authlib's "always show" filter, which ours has to come after
with warnings.catch_warnings():
    # Authlib 1.8 deprecates its httpx integration, which its own Starlette client still imports.
    warnings.filterwarnings("ignore", message="The httpx module is deprecated")
    from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from piper import PiperVoice
import wave
import io
import mimetypes

ENABLE_WEBSOCKETS = False


load_dotenv()

# Converters process untrusted documents. kpathsea reads texmf.cnf overrides from the
# environment, so this disables TeX's \write18 shell escape for every child process.
os.environ.setdefault("shell_escape", "f")

# --- Optional Dependency Handling for Piper TTS ---
try:
    from piper.synthesis import SynthesisConfig
    # download helpers: some piper versions export download_voice, others expose ensure_voice_exists/find_voice
    try:
        # prefer the more explicit helpers if present
        from piper.download import get_voices, ensure_voice_exists, find_voice, VoiceNotFoundError
    except Exception:
        # fall back to older API if available
        try:
            from piper.download import get_voices, download_voice, VoiceNotFoundError
            ensure_voice_exists = None
            find_voice = None
        except Exception:
            # partial import failed -> treat as piper-not-installed for download helpers
            get_voices = None
            download_voice = None
            ensure_voice_exists = None
            find_voice = None
            VoiceNotFoundError = None
except ImportError:
    SynthesisConfig = None
    get_voices = None
    download_voice = None
    ensure_voice_exists = None
    find_voice = None
    VoiceNotFoundError = None


try:
    from PyPDF2 import PdfMerger
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

# Instantiate OAuth object (was referenced in code)
oauth = OAuth()


# --------------------------------------------------------------------------------
# --- 1. CONFIGURATION & SECURITY HELPERS
# --------------------------------------------------------------------------------
# --- Path Safety ---
UPLOADS_BASE = Path(os.environ.get("UPLOADS_DIR", "/app/uploads")).resolve()
PROCESSED_BASE = Path(os.environ.get("PROCESSED_DIR", "/app/processed")).resolve()
CHUNK_TMP_BASE = Path(os.environ.get("CHUNK_TMP_DIR", str(UPLOADS_BASE / "tmp"))).resolve()

def ensure_path_is_safe(p: Path, allowed_bases: List[Path]):
    """Enhanced path safety check with traversal prevention"""
    try:
        # Resolve the path first to get the absolute path
        resolved_p = p.resolve()

        # Check if resolved path is within allowed base directories
        if not any(resolved_p.is_relative_to(base) for base in allowed_bases):
            raise ValueError(f"Path {resolved_p} is outside of allowed directories.")

        return resolved_p
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Path safety check failed for {p}: {e}")
        raise ValueError("Invalid or unsafe path specified.")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and XSS"""
    from werkzeug.utils import secure_filename
    # Use secure_filename and additional sanitization
    safe_name = secure_filename(filename or "")
    # Sanitize for HTML output
    return html.escape(safe_name)

_SAFE_SUFFIX_RE = re.compile(r"^\.[a-z0-9]{1,10}$")


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


# --- Upload sessions ---
UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
MAX_UPLOAD_CHUNKS = 100_000


def chunk_dir_for(user: dict, upload_id: str) -> Path:
    """
    Temp directory for a chunked upload, namespaced per user so one user can neither
    write into nor finalize another user's upload. Upload ids are validated strictly:
    an id that sanitized to "" used to resolve to the shared chunk root itself.
    """
    if not isinstance(upload_id, str) or not UPLOAD_ID_RE.fullmatch(upload_id):
        raise HTTPException(status_code=400, detail="Invalid upload_id (8-128 letters, digits, '-' or '_').")
    user_key = hashlib.sha256(str(user.get("sub", "")).encode()).hexdigest()[:16]
    path = PATHS.CHUNK_TMP_DIR / f"{user_key}_{upload_id}"
    ensure_path_is_safe(path, [PATHS.CHUNK_TMP_DIR])
    return path


# --- ZIP archives ---
ZIP_JUNK_RE = re.compile(r"(^|/)(__MACOSX/|\._|\.DS_Store$|Thumbs\.db$)")


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
            raise ValueError(f"ZIP archive has {len(members)} files; the limit is {max_members} (app_settings.max_zip_members).")
        total_mb = sum(info.file_size for info in members) / (1024 * 1024)
        if total_mb > max_total_mb:
            raise ValueError(f"ZIP archive would extract to {total_mb:.0f} MB; the limit is {max_total_mb:.0f} MB "
                             "(app_settings.max_zip_uncompressed_mb).")
        dest.mkdir(parents=True, exist_ok=True)
        extracted = []
        for info in members:
            target = Path(zf.extract(info, dest))
            ensure_path_is_safe(target, [dest])
            extracted.append(target)
    return extracted


# --- Model / voice names coming from requests ---
_TTS_NAME_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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


# --- OCR languages ---
_OCR_LANG_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")
_ocr_languages_cache: Optional[List[str]] = None


def installed_ocr_languages() -> List[str]:
    """Tesseract languages installed on this server (cached; empty if tesseract is unavailable)."""
    global _ocr_languages_cache
    if _ocr_languages_cache is None:
        try:
            languages = pytesseract.get_languages(config="")
        except Exception as e:
            logger.warning(f"Could not list Tesseract languages: {e}")
            return []
        _ocr_languages_cache = sorted(l for l in languages if l != "osd" and _OCR_LANG_RE.fullmatch(l))
    return _ocr_languages_cache


def resolve_ocr_language(requested: Optional[str], ocr_settings: dict) -> str:
    """
    Returns a validated Tesseract language spec such as 'eng' or 'eng+spa'. Falls back to
    ocr_settings.language (default 'eng') and rejects languages that are not installed.
    """
    spec = requested or ocr_settings.get("language") or "eng"
    if isinstance(spec, (list, tuple)):
        spec = "+".join(str(part) for part in spec)
    parts = [part.strip() for part in str(spec).split("+") if part.strip()]
    if not parts or len(parts) > 8 or not all(_OCR_LANG_RE.fullmatch(part) for part in parts):
        raise ValueError(f"Invalid OCR language '{spec}'.")
    installed = installed_ocr_languages()
    missing = [part for part in parts if installed and part not in installed]
    if missing:
        raise ValueError(f"OCR language(s) not installed: {', '.join(missing)}. Installed: {', '.join(installed)}")
    return "+".join(parts)


def sanitize_output(output: str) -> str:
    """Sanitize output to prevent XSS"""
    if not output:
        return ""
    # Limit length and escape HTML
    output = output[:2000]  # Limit length
    return html.escape(output)

def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """Validate file type by extension"""
    if not allowed_extensions:  # If set is empty, allow all
        return True
    return Path(filename).suffix.lower() in allowed_extensions

def get_file_mime_type(filename: str) -> str:
    """Get MIME type from file extension"""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"  # Default to binary if unknown

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()

def get_supported_output_formats_for_file(filename: str, conversion_tools_config: dict) -> list:
    """
    Get all supported output formats for a given input file based on its extension
    and the supported_input specifications in the tools configuration.
    """
    file_ext = get_file_extension(filename)
    supported_formats = []
    
    for tool_name, tool_config in conversion_tools_config.items():
        supported_inputs = tool_config.get("supported_input", [])
        # Convert supported inputs to lowercase for comparison
        supported_inputs_lower = [ext.lower() for ext in supported_inputs]
        
        if file_ext in supported_inputs_lower:
            # Add all available formats for this tool
            for format_key, format_label in tool_config.get("formats", {}).items():
                full_format_key = f"{tool_name}_{format_key}"
                supported_formats.append({
                    "value": full_format_key,
                    "label": f"{tool_config['name']} - {format_label}",
                    "tool": tool_name,
                    "format": format_key
                })
    
    return supported_formats

# --- Resource Limiting ---
def _compute_child_rlimits() -> list:
    """
    Limits for converter processes, computed once in the parent and clamped to the
    current hard limits (raising a hard limit fails without privileges).
    CHILD_CPU_LIMIT_SECONDS (default 6000) and CHILD_MEMORY_LIMIT_MB (default 4096, 0 = no limit).
    The memory limit is RLIMIT_DATA (memory a process can actually write to), not RLIMIT_AS:
    Chromium-based tools such as Calibre's PDF output reserve far more address space than
    they use and fail under an address-space limit.
    """
    wanted = [(resource.RLIMIT_CPU, int(os.environ.get("CHILD_CPU_LIMIT_SECONDS", "6000")))]
    memory_mb = int(os.environ.get("CHILD_MEMORY_LIMIT_MB", "4096"))
    if memory_mb > 0:
        wanted.append((resource.RLIMIT_DATA, memory_mb * 1024 * 1024))
    limits = []
    for res, value in wanted:
        if value <= 0:
            continue
        try:
            _soft, hard = resource.getrlimit(res)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            limits.append((res, value))
        except (ValueError, OSError):
            pass
    return limits


_CHILD_RLIMITS = _compute_child_rlimits()


def _limit_resources_preexec():
    """
    Runs in the forked child before exec. The parent is multi-threaded (gunicorn/huey),
    so this must not log or take locks, which could deadlock the child.
    """
    for res, value in _CHILD_RLIMITS:
        try:
            resource.setrlimit(res, (value, value))
        except (ValueError, OSError):
            pass


def _kill_process_tree(proc: subprocess.Popen):
    """Kills a child started with start_new_session=True together with everything it spawned."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=10)  # reap, so no zombies are left behind
    except subprocess.TimeoutExpired:
        logger.warning("Process %s did not exit after SIGKILL.", proc.pid)


def _read_tail(f, limit: int = 64 * 1024) -> str:
    """Returns the last `limit` bytes written to a temp file as text."""
    try:
        f.flush()
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - limit))
        return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

# --- Model concurrency semaphore (lazily initialized) ---
_model_semaphore: Optional[Semaphore] = None

def get_model_semaphore() -> Semaphore:
    """Lazily initializes and returns the global model semaphore."""
    global _model_semaphore
    if _model_semaphore is None:
        # Read from app config, fall back to env var, then to a hardcoded default of 1
        model_concurrency_from_env = int(os.environ.get("MODEL_CONCURRENCY", "1"))
        model_concurrency = APP_CONFIG.get("app_settings", {}).get("model_concurrency", model_concurrency_from_env)
        _model_semaphore = Semaphore(model_concurrency)
        logger.info(f"Model concurrency semaphore initialized with limit: {model_concurrency}")
    return _model_semaphore


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Mode ---
LOCAL_ONLY_MODE = os.getenv('LOCAL_ONLY', 'True').lower() in ('true', '1', 't')
if LOCAL_ONLY_MODE:
    logger.warning("Authentication is DISABLED. Running in LOCAL_ONLY mode.")

class AppPaths(BaseModel):
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = UPLOADS_BASE
    PROCESSED_DIR: Path = PROCESSED_BASE
    CHUNK_TMP_DIR: Path = CHUNK_TMP_BASE
    # Job database, task queue and log. Defaults to the app directory; the Docker image uses a volume.
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR") or BASE_DIR).resolve()
    TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts"
    KOKORO_TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts" / "kokoro"
    KOKORO_MODEL_FILE: Path = KOKORO_TTS_MODELS_DIR / "kokoro-v1.0.onnx"
    KOKORO_VOICES_FILE: Path = KOKORO_TTS_MODELS_DIR / "voices-v1.0.bin"
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'jobs.db'}"
    HUEY_DB_PATH: str = str(DATA_DIR / "huey.db")
    CONFIG_DIR: Path = BASE_DIR / "config"
    SETTINGS_FILE: Path = CONFIG_DIR / "settings.yml"
    DEFAULT_SETTINGS_FILE: Path = BASE_DIR / "settings.default.yml"

PATHS = AppPaths()
APP_CONFIG: Dict[str, Any] = {}
PATHS.UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CHUNK_TMP_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CONFIG_DIR.mkdir(exist_ok=True, parents=True)
PATHS.DATA_DIR.mkdir(exist_ok=True, parents=True)
PATHS.TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.KOKORO_TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)

try:
    _log_handler = RotatingFileHandler(PATHS.DATA_DIR / "app.log", maxBytes=10*1024*1024, backupCount=1)
    _log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(_log_handler)
except OSError as e:
    logger.warning(f"File logging disabled: {e}")

# --- WebSocket Connection Manager ---
import json
import asyncio
import threading
from typing import Dict, List
from collections import defaultdict
import time

class ConnectionManager:
    def __init__(self):
        # Maps user_id to list of WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        # Maps WebSocket to user_id
        self.connection_to_user: Dict[WebSocket, str] = {}
        # Maps WebSocket to connection metadata
        self.connection_metadata: Dict[WebSocket, dict] = {}
        # Logger
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str = None):
        await websocket.accept()
        self.connection_to_user[websocket] = user_id
        self.connection_metadata[websocket] = {"connection_id": connection_id or str(uuid.uuid4())}
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket):
        user_id = self.connection_to_user.pop(websocket, None)
        if user_id and websocket in self.active_connections[user_id]:
            self.active_connections[user_id].remove(websocket)
        self.connection_metadata.pop(websocket, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_user_jobs(self, user_id: str, message: str):
        """Send message to all connections for a specific user"""
        self.logger.debug(f"Broadcasting message to user {user_id}: {message}")
        if user_id in self.active_connections:
            disconnected = []
            sent_count = 0
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(message)
                    sent_count += 1
                except WebSocketDisconnect:
                    disconnected.append(websocket)
            
            # Remove disconnected connections
            for websocket in disconnected:
                self.disconnect(websocket)
            
            if sent_count > 0:
                self.logger.info(f"Sent WebSocket message to {sent_count} connections for user {user_id}")
        else:
            self.logger.info(f"No active connections for user {user_id}")

    async def broadcast_job_status_update(self, user_id: str, job_data: dict):
        """Send job status update to user's connections"""
        logger = logging.getLogger(__name__)
        logger.info(f"Broadcasting job update to user {user_id}: job_id={job_data.get('id')}, status={job_data.get('status')}")
        
        message = json.dumps({
            "type": "job_update",
            "job": job_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        await self.broadcast_user_jobs(user_id, message)
        logger.info(f"Finished broadcasting job update to user {user_id}")

    async def broadcast_multiple_jobs_update(self, user_id: str, jobs_data: List):
        """Send multiple job updates to user's connections"""
        message = json.dumps({
            "type": "batch_job_update",
            "jobs": jobs_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        await self.broadcast_user_jobs(user_id, message)

    def sync_broadcast_job_status_update(self, user_id: str, job_data: dict):
        """Synchronously broadcast job status update - for use from sync contexts like Huey tasks"""
        logger = logging.getLogger(__name__)
        job_id = job_data.get('id')
        status = job_data.get('status')
        progress = job_data.get('progress')
        logger.info(f"Queueing WebSocket notification for user {user_id}: job_id={job_id}, status={status}, progress={progress}")
        
        try:
            db = SessionLocal()
            notification = Notification(
                user_id=user_id,
                job_data=json.dumps(job_data)
            )
            db.add(notification)
            db.commit()
            logger.info(f"Queued WebSocket notification for user {user_id}, job {job_id}")
        except Exception as e:
            logger.warning(f"Could not queue WebSocket notification for user {user_id}, job {job_id}: {e}")
        finally:
            if db:
                db.close()

    async def process_notification_queue(self):
        """Process queued notifications and send them to WebSocket clients"""
        db = SessionLocal()
        claimed_notification_data = None
        try:
            # Find a notification to process
            notification_to_process = db.query(Notification).order_by(Notification.created_at).first()
            if not notification_to_process:
                return

            # Try to "claim" it by deleting it.
            notification_id = notification_to_process.id
            
            # We need to copy the data before deleting.
            claimed_notification_data = {
                "id": notification_id,
                "user_id": notification_to_process.user_id,
                "job_data": notification_to_process.job_data
            }

            deleted_count = db.query(Notification).filter_by(id=notification_id).delete(synchronize_session=False)
            db.commit()

            if deleted_count == 0:
                # Another worker got it first.
                self.logger.debug(f"Notification {notification_id} was already claimed by another worker.")
                claimed_notification_data = None # Do not process
        
        except Exception as e:
            self.logger.error(f"Error claiming notification from DB: {e}")
            db.rollback()
            claimed_notification_data = None # Do not process
        finally:
            db.close()

        # --- Process the claimed notification outside the DB transaction ---
        if claimed_notification_data:
            try:
                user_id = claimed_notification_data["user_id"]
                notification_id = claimed_notification_data["id"]
                self.logger.debug(f"Processing claimed notification {notification_id} for user {user_id}")

                if user_id in self.active_connections:
                    job_data = json.loads(claimed_notification_data["job_data"])
                    message = json.dumps({
                        "type": "job_update",
                        "job": job_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    await self.broadcast_user_jobs(user_id, message)
                    self.logger.info(f"Sent claimed notification {notification_id} to user {user_id}")
            except Exception as e:
                self.logger.warning(f"Error sending claimed notification {claimed_notification_data['id']}: {e}")

# Initialize manager
manager = ConnectionManager()


def deep_merge(source: dict, dest: dict) -> dict:
    """
    Recursively merges source dict into dest dict. Modifies dest in place.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
            deep_merge(value, dest[key])
        else:
            dest[key] = value
    return dest

def initialize_settings_file():
    """
    Ensures that config/settings.yml exists. If not, it copies it from
    settings.default.yml.
    """
    if not PATHS.SETTINGS_FILE.exists():
        logger.info(f"'{PATHS.SETTINGS_FILE}' not found. Copying from '{PATHS.DEFAULT_SETTINGS_FILE}'.")
        try:
            shutil.copy(PATHS.DEFAULT_SETTINGS_FILE, PATHS.SETTINGS_FILE)
        except FileNotFoundError:
            logger.error(f"CRITICAL: Default settings file '{PATHS.DEFAULT_SETTINGS_FILE}' not found. Cannot initialize settings.")
            PATHS.SETTINGS_FILE.touch()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to copy default settings file: {e}")
            PATHS.SETTINGS_FILE.touch()

_config_mtimes: Optional[tuple] = None
# tool id -> executable that is missing on this server; such tools are hidden and rejected.
UNAVAILABLE_TOOLS: Dict[str, str] = {}


def _compute_unavailable_tools(conversion_tools: dict) -> Dict[str, str]:
    missing = {}
    for tool_id, tool_cfg in conversion_tools.items():
        if not isinstance(tool_cfg, dict):
            continue
        try:
            parts = shlex.split(str(tool_cfg.get("command_template") or ""))
        except ValueError:
            parts = []
        executable = parts[0] if parts else ""
        if not executable or shutil.which(executable) is None:
            missing[tool_id] = executable or "(no command)"
        elif tool_id == "mozjpeg" and shutil.which("vips") is None:
            missing[tool_id] = "vips"
    return missing


def _settings_mtimes() -> tuple:
    result = []
    for settings_path in (PATHS.SETTINGS_FILE, PATHS.DEFAULT_SETTINGS_FILE):
        try:
            result.append(settings_path.stat().st_mtime_ns)
        except OSError:
            result.append(None)
    return tuple(result)


def reload_app_config_if_changed():
    """
    Each gunicorn worker (and the huey consumer) holds its own APP_CONFIG. Settings saved
    through one worker, or edited on disk, are picked up by the others on their next request.
    """
    if _settings_mtimes() == _config_mtimes:
        return
    try:
        load_app_config()
    except Exception:
        logger.exception("Could not reload settings; keeping the previous configuration.")


def load_app_config():
    """
    Loads configuration by deeply merging settings from hardcoded defaults,
    settings.default.yml, and settings.yml, then applies environment variable
    overrides.
    """
    global APP_CONFIG, UNAVAILABLE_TOOLS, _config_mtimes
    # Recorded before reading, so a write that races with this load triggers another reload.
    _config_mtimes = _settings_mtimes()

    # --- 1. Hardcoded Defaults ---
    hardcoded_defaults = {
        "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": [], "app_public_url": ""},
        "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
        "tts_settings": {
            "piper": {"model_dir": str(PATHS.TTS_MODELS_DIR), "use_cuda": False, "synthesis_config": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}},
            "kokoro": {"model_dir": str(PATHS.KOKORO_TTS_MODELS_DIR), "command_template": "kokoro-tts {input} {output} --model {model_path} --voices {voices_path} --lang {lang} --voice {model_name}"}
        },
        "conversion_tools": {},
        "ocr_settings": {"ocrmypdf": {"language": "eng"}},
        "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []},
        "webhook_settings": {"enabled": False, "allow_chunked_api_uploads": False, "allowed_callback_urls": [], "callback_bearer_token": ""}
    }

    config = hardcoded_defaults

    # --- 2. Merge settings.default.yml ---
    try:
        with open(PATHS.DEFAULT_SETTINGS_FILE, 'r', encoding='utf8') as f:
            default_cfg = yaml.safe_load(f) or {}
        config = deep_merge(default_cfg, config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Could not load or parse settings.default.yml: {e}. Using hardcoded defaults.")

    # --- 3. Merge settings.yml ---
    try:
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            user_cfg = yaml.safe_load(f) or {}
        config = deep_merge(user_cfg, config)
    except (FileNotFoundError, yaml.YAMLError):
        # This is not an error, just means user is using defaults
        pass

    # --- 4. Environment Variable Overrides for Transcription ---
    # Safely access nested keys
    trans_settings = config.get("transcription_settings", {}).get("whisper", {})
    transcription_device = os.environ.get("TRANSCRIPTION_DEVICE", trans_settings.get("device", "cpu"))
    default_compute_type = "float16" if transcription_device == "cuda" else "int8"
    transcription_compute_type = os.environ.get("TRANSCRIPTION_COMPUTE_TYPE", trans_settings.get("compute_type", default_compute_type))
    transcription_device_index_str = os.environ.get("TRANSCRIPTION_DEVICE_INDEX", "0")

    try:
        if ',' in transcription_device_index_str:
            transcription_device_index = [int(i.strip()) for i in transcription_device_index_str.split(',')]
        else:
            transcription_device_index = int(transcription_device_index_str)
    except ValueError:
        logger.warning(f"Invalid TRANSCRIPTION_DEVICE_INDEX value: '{transcription_device_index_str}'. Defaulting to 0.")
        transcription_device_index = 0

    config.setdefault("transcription_settings", {}).setdefault("whisper", {})
    config["transcription_settings"]["whisper"]["device"] = transcription_device
    config["transcription_settings"]["whisper"]["compute_type"] = transcription_compute_type
    config["transcription_settings"]["whisper"]["device_index"] = transcription_device_index

    # --- 5. Final Processing & Assignment ---
    app_settings = config.get("app_settings", {})
    try:
        max_mb = float(app_settings.get("max_file_size_mb", 100))
    except (TypeError, ValueError):
        logger.warning(f"Invalid max_file_size_mb {app_settings.get('max_file_size_mb')!r}; using 100.")
        max_mb = 100
    app_settings["max_file_size_bytes"] = int(max_mb * 1024 * 1024)
    allowed = app_settings.get("allowed_all_extensions", [])
    if not isinstance(allowed, (list, set)):
        allowed = []
    app_settings["allowed_all_extensions"] = set(allowed)
    config["app_settings"] = app_settings

    UNAVAILABLE_TOOLS = _compute_unavailable_tools(config.get("conversion_tools", {}) or {})
    if UNAVAILABLE_TOOLS:
        logger.info("Conversion tools hidden because their program is not installed: "
                    + ", ".join(f"{t} ({exe})" for t, exe in sorted(UNAVAILABLE_TOOLS.items())))

    APP_CONFIG = config
    logger.info("Application configuration loaded.")


# --------------------------------------------------------------------------------
# --- 2. DATABASE & Schemas
# --------------------------------------------------------------------------------
engine = create_engine(
    PATHS.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    c = dbapi_connection.cursor()
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
    finally:
        c.close()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    parent_job_id = Column(String, index=True, nullable=True)
    task_type = Column(String, index=True)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    original_filename = Column(String)
    input_filepath = Column(String)
    input_filesize = Column(Integer, nullable=True)
    processed_filepath = Column(String, nullable=True)
    output_filesize = Column(Integer, nullable=True)
    result_preview = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    callback_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class JobCreate(BaseModel):
    id: str
    user_id: str | None = None
    parent_job_id: str | None = None
    task_type: str
    original_filename: str
    input_filepath: str
    input_filesize: int | None = None
    callback_url: str | None = None
    processed_filepath: str | None = None

class JobSchema(BaseModel):
    id: str
    parent_job_id: str | None = None
    task_type: str
    status: str
    progress: int
    original_filename: str
    input_filesize: int | None = None
    output_filesize: int | None = None
    processed_filepath: str | None = None
    result_preview: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)
    @field_serializer('created_at', 'updated_at')
    def serialize_dt(self, dt: datetime, _info):
        return dt.isoformat() + "Z"

class FinalizeUploadPayload(BaseModel):
    upload_id: str
    original_filename: str
    total_chunks: int
    task_type: str
    model_size: str = ""
    model_name: str = ""
    output_format: str = ""
    generate_timestamps: bool = False
    ocr_language: str = ""  # e.g. "eng" or "eng+spa"; defaults to ocr_settings.ocrmypdf.language
    callback_url: Optional[str] = None # For API chunked uploads

class JobSelection(BaseModel):
    job_ids: List[str]

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    job_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# --------------------------------------------------------------------------------
# --- 3. CRUD OPERATIONS & WEBHOOKS
# --------------------------------------------------------------------------------





def get_job(db: Session, job_id: str):
    # return db.query(Job).filter(Job.id == job_id).first()
    return  db.query(Job).filter(Job.id == job_id).first()

def get_jobs(db: Session, user_id: str | None = None, skip: int = 0, limit: int | None = 100):
    query = db.query(Job)
    if user_id:
        query = query.filter(Job.user_id == user_id)
    query = query.order_by(Job.created_at.desc()).offset(skip)
    return (query.limit(limit) if limit else query).all()


def create_job(db: Session, job: JobCreate):
    db_job = Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    # Broadcast the new job to UI clients via Huey task
    job_schema = JobSchema.model_validate(db_job)
    return db_job

def update_job_status(db: Session, job_id: str, status: str, progress: int = None, error: str = None):
    # End the current transaction so the committed row is read (e.g. a cancellation by the web process).
    try:
        db.commit()
    except Exception:
        db.rollback()
    db_job = get_job(db, job_id)
    if db_job:
        if db_job.status == "cancelled" and status != "cancelled":
            # A cancelled job stays cancelled even if its task later finishes or fails.
            return db_job
        old_status = db_job.status
        old_progress = db_job.progress

        db_job.status = status
        if progress is not None:
            db_job.progress = progress
        if error:
            db_job.error_message = error

        status_changed = old_status != status
        progress_changed = progress is not None and old_progress != progress

        db.commit()

        job_schema = JobSchema.model_validate(db_job)

        # Notifications are only consumed by the WebSocket processor; without it they piled up forever.
        if ENABLE_WEBSOCKETS and (status_changed or progress_changed) and db_job.user_id:
            manager.sync_broadcast_job_status_update(db_job.user_id, job_schema.model_dump())
    return db_job

def mark_job_as_completed(db: Session, job_id: str, output_filepath_str: str | None = None, preview: str | None = None):
    db_job = get_job(db, job_id)
    if db_job and db_job.status != 'cancelled':
        if preview:
            db_job.result_preview = preview.strip()[:2000]
        if output_filepath_str:
            try:
                output_path = Path(output_filepath_str)
                if output_path.exists():
                    db_job.output_filesize = output_path.stat().st_size
            except Exception:
                logger.exception(f"Could not stat output file {output_filepath_str} for job {job_id}")

        update_job_status(db, job_id, "completed", progress=100)
    return db_job

def send_webhook_notification(job_id: str, app_config: Dict[str, Any], base_url: str):
    """Sends a notification to the callback URL if one is configured for the job."""
    webhook_config = app_config.get("webhook_settings", {})
    if not webhook_config.get("enabled", False):
        return

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or not job.callback_url:
            return

        download_url = None
        if job.status == "completed" and job.processed_filepath:
            filename = Path(job.processed_filepath).name
            public_url = app_config.get("app_settings", {}).get("app_public_url", base_url)
            if not public_url:
                logger.warning(f"app_public_url is not set. Cannot generate a full download URL for job {job_id}.")
                download_url = f"/download/{filename}" # Relative URL as fallback
            else:
                download_url = urljoin(public_url, f"/download/{filename}")

        payload = {
            "job_id": job.id,
            "status": job.status,
            "original_filename": job.original_filename,
            "download_url": download_url,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() + "Z",
            "updated_at": job.updated_at.isoformat() + "Z",
        }

        headers = {"Content-Type": "application/json", "User-Agent": "FileProcessor-Webhook/1.0"}
        token = webhook_config.get("callback_bearer_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with httpx.Client() as client:
                response = client.post(job.callback_url, json=payload, headers=headers, timeout=15)
                response.raise_for_status()
            logger.info(f"Sent webhook notification for job {job_id} to {job.callback_url} (Status: {response.status_code})")
        except httpx.RequestError as e:
            logger.error(f"Failed to send webhook for job {job_id} to {job.callback_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Webhook for job {job_id} received non-2xx response {e.response.status_code} from {job.callback_url}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred in send_webhook_notification for job {job_id}: {e}")
    finally:
        db.close()


# --------------------------------------------------------------------------------
# --- 4. BACKGROUND TASK SETUP
# --------------------------------------------------------------------------------
huey = SqliteHuey(filename=PATHS.HUEY_DB_PATH)
WHISPER_MODELS_CACHE: Dict[str, WhisperModel] = {}
PIPER_VOICES_CACHE: Dict[str, "PiperVoice"] = {}
AVAILABLE_TTS_VOICES_CACHE: Dict[str, Any] | None = None
WHISPER_MODELS_LAST_USED: Dict[str, float] = {}

# --- Cache Eviction Settings ---
_cache_cleanup_thread: Optional[threading.Thread] = None
_cache_lock = threading.Lock() # Global lock for modifying cache dictionaries
_model_locks: Dict[str, threading.Lock] = {}
_global_lock = threading.Lock() # Lock for initializing model-specific locks

def _whisper_cache_cleanup_worker():
    """
    Periodically checks for and unloads Whisper models that have been inactive.
    The timeout and check interval are configured in the application settings.
    """
    while True:
        # Read settings within the loop to allow for live changes
        app_settings = APP_CONFIG.get("app_settings", {})
        check_interval = app_settings.get("cache_check_interval", 300)
        inactivity_timeout = app_settings.get("model_inactivity_timeout", 1800)

        time.sleep(check_interval)

        with _cache_lock:
            # Create a copy of items to avoid issues with modifying dict while iterating
            expired_models = []
            for model_size, last_used in WHISPER_MODELS_LAST_USED.items():
                if time.time() - last_used > inactivity_timeout:
                    expired_models.append(model_size)

            if not expired_models:
                continue

            logger.info(f"Found {len(expired_models)} inactive Whisper models to unload: {expired_models}")

            for model_size in expired_models:
                # Acquire the specific model lock before removing to prevent race conditions
                model_lock = _get_or_create_model_lock(model_size)
                with model_lock:
                    # Check if the model is still in the cache (it should be)
                    if model_size in WHISPER_MODELS_CACHE:
                        logger.info(f"Unloading inactive Whisper model: {model_size}")
                        # Remove from caches
                        model_to_unload = WHISPER_MODELS_CACHE.pop(model_size, None)
                        WHISPER_MODELS_LAST_USED.pop(model_size, None)

                        # Explicitly delete the object to encourage garbage collection
                        if model_to_unload:
                            del model_to_unload

        # Explicitly run garbage collection outside the main lock
        import gc
        gc.collect()

def get_whisper_model(model_size: str, whisper_settings: dict) -> Any:
    # Fast path: check cache. If hit, update timestamp and return.
    with _cache_lock:
        if model_size in WHISPER_MODELS_CACHE:
            logger.debug(f"Cache hit for model '{model_size}'")
            WHISPER_MODELS_LAST_USED[model_size] = time.time()
            return WHISPER_MODELS_CACHE[model_size]

    # Model not in cache, prepare for loading.
    model_lock = _get_or_create_model_lock(model_size)

    with model_lock:
        # Re-check cache inside lock in case another thread loaded it
        with _cache_lock:
            if model_size in WHISPER_MODELS_CACHE:
                WHISPER_MODELS_LAST_USED[model_size] = time.time()
                return WHISPER_MODELS_CACHE[model_size]

        logger.info(f"Loading Whisper model '{model_size}'...")
        try:
            device = whisper_settings.get("device", "cpu")
            compute_type = whisper_settings.get("compute_type", "int8")
            device_index = whisper_settings.get("device_index", 0)

            model = WhisperModel(
                model_size,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=max(1, os.cpu_count() // 2),
                num_workers=1
            )

            # Add the new model to the cache under lock
            with _cache_lock:
                WHISPER_MODELS_CACHE[model_size] = model
                WHISPER_MODELS_LAST_USED[model_size] = time.time()

            logger.info(f"Model '{model_size}' loaded (device={device}, compute={compute_type})")
            return model

        except Exception as e:
            logger.error(f"Model '{model_size}' failed to load: {str(e)}", exc_info=True)
            raise RuntimeError(f"Whisper model initialization failed: {e}") from e

def _get_or_create_model_lock(model_size: str) -> threading.Lock:
    """Thread-safe lock acquisition with minimal global contention"""
    # Fast path: lock already exists
    if model_size in _model_locks:
        return _model_locks[model_size]

    # Slow path: create lock under global lock
    with _global_lock:
        return _model_locks.setdefault(model_size, threading.Lock())

def get_piper_voice(model_name: str, tts_settings: dict | None) -> "PiperVoice":
    """
    Load (or download + load) a Piper voice in a robust way:
      - Try Python API helpers (get_voices, ensure_voice_exists/find_voice, download_voice)
      - On any failure, try CLI fallback (download_voice_cli)
      - Attempt to locate model files after download (search subdirs)
      - Try re-importing piper if bindings were previously unavailable
    """
    # ----- Defensive normalization -----
    if tts_settings is None or not isinstance(tts_settings, dict):
        logger.debug("get_piper_voice: normalizing tts_settings (was %r)", tts_settings)
        tts_settings = {}

    model_dir_val = tts_settings.get("model_dir", None)
    if model_dir_val is None:
        model_dir = Path(str(PATHS.TTS_MODELS_DIR))
    else:
        try:
            model_dir = Path(model_dir_val)
        except Exception:
            logger.warning("Could not coerce tts_settings['model_dir']=%r to Path; using default.", model_dir_val)
            model_dir = Path(str(PATHS.TTS_MODELS_DIR))
    model_dir.mkdir(parents=True, exist_ok=True)

    # If PiperVoice already cached, reuse
    if model_name in PIPER_VOICES_CACHE:
        logger.info("Reusing cached Piper voice '%s'.", model_name)
        return PIPER_VOICES_CACHE[model_name]

    with get_model_semaphore():
        if model_name in PIPER_VOICES_CACHE:
            return PIPER_VOICES_CACHE[model_name]

        # If Python bindings are missing, attempt CLI download first (and try re-import)
        if PiperVoice is None:
            logger.info("Piper Python bindings missing; attempting CLI download fallback for '%s' before failing import.", model_name)
            cli_ok = False
            try:
                cli_ok = download_voice_cli(model_name, model_dir)
            except Exception as e:
                logger.warning("CLI download attempt raised: %s", e)
                cli_ok = False

            if cli_ok:
                # attempt to re-import piper package (maybe import issue was transient)
                try:
                    importlib.invalidate_caches()
                    piper_mod = importlib.import_module("piper")
                    from piper import PiperVoice as _PiperVoice  # noqa: F401
                    from piper.synthesis import SynthesisConfig as _SynthesisConfig  # noqa: F401
                    globals().update({"PiperVoice": _PiperVoice, "SynthesisConfig": _SynthesisConfig})
                    logger.info("Successfully re-imported piper after CLI download.")
                except Exception:
                    logger.warning("Could not import piper after CLI download; bindings still unavailable.")
            # If bindings still absent, we cannot load models; raise helpful error
            if PiperVoice is None:
                raise RuntimeError(
                    "Piper Python bindings are not installed or failed to import. "
                    "Tried CLI download fallback but python bindings are still unavailable. "
                    "Please install 'piper-tts' in the runtime used by this process."
                )

        # Now we have Piper bindings (or they were present to begin with). Attempt Python helpers.
        onnx_path = None
        config_path = None

        # Prefer using get_voices to update the index if available
        voices_info = None
        try:
            if get_voices:
                try:
                    voices_info = get_voices(str(model_dir), update_voices=True)
                except TypeError:
                    # some versions may not support update_voices kwarg
                    voices_info = get_voices(str(model_dir))
        except Exception as e:
            logger.debug("get_voices failed or unavailable: %s", e)
            voices_info = None

        try:
            # Preferred modern helpers
            if ensure_voice_exists and find_voice:
                try:
                    ensure_voice_exists(model_name, [model_dir], model_dir, voices_info)
                    onnx_path, config_path = find_voice(model_name, [model_dir])
                except Exception as e:
                    # Could be VoiceNotFoundError or other download error
                    logger.warning("ensure/find voice failed for %s: %s", model_name, e)
                    raise
            elif download_voice:
                # older API: call download helper directly
                try:
                    download_voice(model_name, model_dir)
                    # attempt to locate files
                    onnx_path = model_dir / f"{model_name}.onnx"
                    config_path = model_dir / f"{model_name}.onnx.json"
                except Exception:
                    logger.warning("download_voice failed for %s", model_name)
                    raise
            else:
                # No python download helper available
                raise RuntimeError("No Python download helper available in installed piper package.")
        except Exception as py_exc:
            # Python helper route failed; try CLI fallback BEFORE giving up
            logger.info("Python download route failed for '%s' (%s). Trying CLI fallback...", model_name, py_exc)
            try:
                cli_ok = download_voice_cli(model_name, model_dir)
            except Exception as e:
                logger.warning("CLI fallback attempt raised: %s", e)
                cli_ok = False

            if not cli_ok:
                # If CLI also failed, re-raise the original python exception to preserve context
                logger.error("Both Python download helpers and CLI fallback failed for '%s'.", model_name)
                raise

            # CLI succeeded (or at least returned success) — try to find files on disk
            onnx_path, config_path = _find_model_files(model_name, model_dir)
            if not (onnx_path and config_path):
                # maybe CLI wrote into a nested dir or different name; try to search broadly
                logger.info("Could not find model files after CLI download in %s; attempting broader search...", model_dir)
                onnx_path, config_path = _find_model_files(model_name, model_dir)
                if not (onnx_path and config_path):
                    logger.error("Model files still missing after CLI fallback for '%s'.", model_name)
                    raise RuntimeError(f"Piper voice files for '{model_name}' missing after CLI fallback.")
            # continue to loading below

        # Final safety check and last-resort search
        if not (onnx_path and config_path):
            onnx_path, config_path = _find_model_files(model_name, model_dir)

        if not (onnx_path and config_path):
            raise RuntimeError(f"Piper voice files for '{model_name}' are missing after attempts to download.")

        # Load the PiperVoice
        try:
            use_cuda = bool(tts_settings.get("use_cuda", False))
            voice = PiperVoice.load(str(onnx_path), config_path=str(config_path), use_cuda=use_cuda)
            PIPER_VOICES_CACHE[model_name] = voice
            logger.info("Loaded Piper voice '%s' from %s", model_name, onnx_path)
            return voice
        except Exception as e:
            logger.exception("Failed to load Piper voice '%s' from files (%s, %s): %s", model_name, onnx_path, config_path, e)
            raise


def _find_model_files(model_name: str, model_dir: Path):
    """
    Try multiple strategies to find onnx and config files for a given model_name under model_dir.
    Returns (onnx_path, config_path) or (None, None).
    """
    # direct files in model_dir
    onnx = model_dir / f"{model_name}.onnx"
    cfg = model_dir / f"{model_name}.onnx.json"
    if onnx.exists() and cfg.exists():
        return onnx, cfg

    # possible alternative names or nested directories: search recursively
    matches_onnx = list(model_dir.rglob(f"{model_name}*.onnx"))
    matches_cfg = list(model_dir.rglob(f"{model_name}*.onnx.json"))
    if matches_onnx and matches_cfg:
        # prefer same directory match
        for o in matches_onnx:
            for c in matches_cfg:
                if o.parent == c.parent:
                    return o, c
        # otherwise return first matches
        return matches_onnx[0], matches_cfg[0]

    # last-resort: any onnx + any json in same subdir that contain model name token
    for o in model_dir.rglob("*.onnx"):
        if model_name in o.name:
            # try find any matching json in same dir
            cands = list(o.parent.glob("*.onnx.json"))
            if cands:
                return o, cands[0]

    return None, None


# ---------------------------
# CLI: list available voices
# ---------------------------
def list_voices_cli(timeout: int = 30, python_executables: Optional[List[str]] = None) -> List[str]:
    """
    Run `python -m piper.download_voices` (no args) and parse output into a list of voice IDs.
    Returns [] on failure.
    """
    if python_executables is None:
        python_executables = [sys.executable, "python3", "python"]

    # Regex: voice ids look like en_US-lessac-medium (letters/digits/._-)
    voice_regex = re.compile(r'^([A-Za-z0-9_\-\.]+)')

    for py in python_executables:
        cmd = [py, "-m", "piper.download_voices"]
        try:
            logger.debug("Trying Piper CLI list: %s", shlex.join(cmd))
            cp = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout,
            )
            out = cp.stdout.strip()
            # If stdout empty, sometimes the script writes to stderr
            if not out:
                out = cp.stderr.strip()

            if not out:
                logger.debug("Piper CLI listed nothing (empty output) for %s", py)
                continue

            voices = []
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Try to extract first token that matches voice id pattern
                m = voice_regex.match(line)
                if m:
                    v = m.group(1)
                    # basic sanity: avoid capturing words like 'Available' or headings
                    if re.search(r'\d', v) or '-' in v or '_' in v or '.' in v:
                        voices.append(v)
                    else:
                        # allow alphabetic tokens too (defensive)
                        voices.append(v)
                else:
                    # Also handle lines like " - en_US-lessac-medium: description"
                    parts = re.split(r'[:\s]+', line)
                    if parts:
                        candidate = parts[0].lstrip('-').strip()
                        if candidate:
                            voices.append(candidate)
            # Dedupe while preserving order
            seen = set()
            dedup = []
            for v in voices:
                if v not in seen:
                    seen.add(v)
                    dedup.append(v)
            logger.info("Piper CLI list returned %d voices via %s", len(dedup), py)
            return dedup
        except subprocess.CalledProcessError as e:
            logger.debug("Piper CLI list (%s) non-zero exit. stdout=%s stderr=%s", py, e.stdout, e.stderr)
        except FileNotFoundError:
            logger.debug("Python executable not found: %s", py)
        except subprocess.TimeoutExpired:
            logger.warning("Piper CLI list timed out for %s", py)
        except Exception as e:
            logger.exception("Unexpected error running Piper CLI list with %s: %s", py, e)

    logger.error("All Piper CLI list attempts failed.")
    return []

# ---------------------------
# CLI: download a voice
# ---------------------------
def download_voice_cli(model_name: str, model_dir: Path, python_executables: Optional[List[str]] = None, timeout: int = 300) -> bool:
    """
    Try to download a Piper voice using CLI:
      python -m piper.download_voices <model_name> --data-dir <model_dir>
    Returns True if the CLI ran and expected files exist afterwards (best effort).
    """
    if python_executables is None:
        python_executables = [sys.executable, "python3", "python"]

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for py in python_executables:
        cmd = [py, "-m", "piper.download_voices", model_name, "--data-dir", str(model_dir)]
        try:
            logger.info("Trying Piper CLI download: %s", shlex.join(cmd))
            cp = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout,
            )
            logger.debug("Piper CLI download stdout: %s", cp.stdout)
            logger.debug("Piper CLI download stderr: %s", cp.stderr)
            # Heuristic success check
            onnx = model_dir / f"{model_name}.onnx"
            cfg = model_dir / f"{model_name}.onnx.json"
            if onnx.exists() and cfg.exists():
                logger.info("Piper CLI created expected files for %s", model_name)
                return True
            # Some versions might create nested dirs; treat non-error CLI execution as success (caller will re-check)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("Piper CLI (%s) returned non-zero exit. stdout: %s; stderr: %s", py, e.stdout, e.stderr)
        except FileNotFoundError:
            logger.debug("Python executable %s not found.", py)
        except subprocess.TimeoutExpired:
            logger.warning("Piper CLI call timed out for python %s", py)
        except Exception as e:
            logger.exception("Unexpected error running Piper CLI download with %s: %s", py, e)

    logger.error("All Piper CLI attempts failed for model %s", model_name)
    return False

# ---------------------------
# Safe get_voices wrapper
# ---------------------------
def safe_get_voices(model_dir: Path) -> List[Dict]:
    """
    Try to call the in-Python get_voices(..., update_voices=True) and return a list of dicts.
    If that fails, fall back to list_voices_cli() and return a list of simple dicts:
      [{"id": "en_US-lessac-medium", "name": "...", "local": False}, ...]
    Keeps the shape flexible so your existing endpoint can use it with minimal changes.
    """
    # Prefer Python API if available
    try:
        if get_voices:  # get_voices imported earlier in your file
            # Ensure up-to-date index (like CLI)
            raw = get_voices(str(model_dir), update_voices=True)
            # get_voices may already return the desired structure; normalise to a list of dicts
            if isinstance(raw, dict):
                # some versions return mapping id->meta
                items = []
                for vid, meta in raw.items():
                    d = {"id": vid}
                    if isinstance(meta, dict):
                        d.update(meta)
                    items.append(d)
                return items
            elif isinstance(raw, list):
                return raw
            else:
                # unknown format -> fall back to CLI
                logger.debug("get_voices returned unexpected type; falling back to CLI list.")
    except Exception as e:
        logger.warning("In-Python get_voices failed: %s. Falling back to CLI listing.", e)

    # CLI fallback: parse voice ids and create simple dicts
    cli_list = list_voices_cli()
    results = [{"id": vid, "name": vid, "local": False} for vid in cli_list]
    return results

def list_kokoro_voices_cli(timeout: int = 60) -> List[str]:
    """
    Run `kokoro-tts --help-voices` and parse the output for available models.
    Returns [] on failure.
    """
    model_path = PATHS.KOKORO_MODEL_FILE
    voices_path = PATHS.KOKORO_VOICES_FILE
    if not (model_path.exists() and voices_path.exists()):
        logger.warning("Cannot list Kokoro TTS voices because model/voices files are missing.")
        return []

    cmd = ["kokoro-tts", "--help-voices", "--model", str(model_path), "--voices", str(voices_path)]
    try:
        logger.info("Trying Kokoro TTS CLI list: %s", shlex.join(cmd))
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        out = cp.stdout.strip()
        if not out:
            out = cp.stderr.strip()

        if not out:
            logger.warning("Kokoro TTS CLI list returned no output.")
            return []

        voices = []
        voice_pattern = re.compile(r'^\s*\d+\.\s+([a-z]{2,3}_[a-zA-Z0-9]+)$')
        for line in out.splitlines():
            line = line.strip()
            match = voice_pattern.match(line)
            if match:
                voices.append(match.group(1))

        logger.info("Kokoro TTS CLI list returned %d voices.", len(voices))
        return sorted(list(set(voices)))
    except FileNotFoundError:
        logger.info("Kokoro TTS ('kokoro-tts' command) not found in PATH. Kokoro TTS support disabled.")
        return []
    except subprocess.CalledProcessError as e:
        logger.error("Kokoro TTS CLI list command failed. stderr: %s", e.stderr[:1000])
        logger.error("Kokoro TTS CLI list command failed. stdout: %s", e.stdout[:1000])
        return []
    except subprocess.TimeoutExpired:
        logger.warning("Kokoro TTS CLI list command timed out.")
        return []
    except Exception as e:
        logger.exception("Unexpected error running Kokoro TTS CLI list: %s", e)
        return []

def list_kokoro_languages_cli(timeout: int = 60) -> List[str]:
    """
    Run `kokoro-tts --help-languages` and parse the output for available languages.
    Returns [] on failure.
    """
    model_path = PATHS.KOKORO_MODEL_FILE
    voices_path = PATHS.KOKORO_VOICES_FILE
    if not (model_path.exists() and voices_path.exists()):
        logger.warning("Cannot list Kokoro TTS languages because model/voices files are missing.")
        return []

    cmd = ["kokoro-tts", "--help-languages", "--model", str(model_path), "--voices", str(voices_path)]
    try:
        logger.debug("Trying Kokoro TTS language list: %s", shlex.join(cmd))
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout)
        out = cp.stdout.strip()
        if not out:
            out = cp.stderr.strip()

        if not out:
            logger.warning("Kokoro TTS language list returned no output.")
            return []

        languages = []
        lang_pattern = re.compile(r'^\s*([a-z]{2,3}(?:-[a-z]{2,3})?)$')
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("supported languages"):
                continue
            match = lang_pattern.match(line)
            if match:
                languages.append(match.group(1))

        logger.info("Kokoro TTS language list returned %d languages.", len(languages))
        return sorted(list(set(languages)))
    except FileNotFoundError:
        logger.info("Kokoro TTS ('kokoro-tts' command) not found in PATH. Kokoro TTS support disabled.")
        return []
    except subprocess.CalledProcessError as e:
        logger.error("Kokoro TTS language list command failed. stderr: %s", e.stderr[:1000])
        logger.error("Kokoro TTS language list command failed. stdout: %s", e.stdout[:1000])
        return []
    except subprocess.TimeoutExpired:
        logger.warning("Kokoro TTS language list command timed out.")
        return []
    except Exception as e:
        logger.exception("Unexpected error running Kokoro TTS language list: %s", e)
        return []


def run_command(
    argv: List[str],
    timeout: int = 300,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """
    Executes a command with resource limits and a timeout. Output goes to temp files, not
    pipes (a chatty tool can't block on a full pipe), and on timeout the whole process
    group is killed. Returns the tail of stdout/stderr; raises Exception on failure.
    """
    printable = " ".join(shlex.quote(s) for s in argv)
    logger.debug("Executing command: %s with timeout=%ss", printable, timeout)
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f, cwd=cwd,
                preexec_fn=_limit_resources_preexec, start_new_session=True,
            )
        except FileNotFoundError:
            msg = f"Command not found: {argv[0]}"
            logger.error(msg)
            raise Exception(msg) from None
        except Exception as e:
            msg = f"Unexpected error launching command: {e}"
            logger.exception(msg)
            raise Exception(msg) from e
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            msg = f"Command timed out after {timeout}s: {printable}"
            logger.error(msg)
            raise Exception(msg) from None
        stdout, stderr = _read_tail(out_f), _read_tail(err_f)
    if returncode != 0:
        msg = f"Command failed with exit code {returncode}. Stderr: {stderr[-1000:]}"
        logger.error(msg)
        raise Exception(msg)
    logger.debug("Command completed successfully: %s", printable)
    return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr=stderr)

def validate_and_build_command(template_str: str, mapping: Dict[str, str]) -> TypingList[str]:
    fmt = Formatter()
    used = {fname for _, fname, _, _ in fmt.parse(template_str) if fname}
    ALLOWED_VARS = {
        "input", "output", "output_dir", "output_ext", "quality", "speed", "preset",
        "device", "dpi", "samplerate", "bitdepth", "filter", "model_name",
        "model_path", "voices_path", "lang"
    }
    bad = used - ALLOWED_VARS
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

    # Filter out any empty strings that result from empty optional placeholders
    return [part for part in formatted_command if part]

class SrtFormatter:
    def __init__(self):
        self.segment_count = 0

    def _format_time(self, seconds):
        delta = timedelta(seconds=seconds)
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((delta.total_seconds() % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def format_segment(self, segment):
        self.segment_count += 1
        start_time = self._format_time(segment.start)
        end_time = self._format_time(segment.end)
        return f"{self.segment_count}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"

# --- TASK RUNNERS ---
@huey.task()
def run_transcription_task(job_id: str, input_path_str: str, output_path_str: str, model_size: str, whisper_settings: dict, app_config: dict, base_url: str, generate_timestamps: bool = False):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # --- Constants ---
    DB_POLL_INTERVAL_SECONDS = 1

    try:
        job = get_job(db, job_id)
        if not job:
            logger.warning(f"Job {job_id} not found. Aborting task.")
            return
        if job.status == 'cancelled':
            logger.info(f"Job {job_id} was already cancelled before starting. Aborting.")
            return
        allowed_models = whisper_settings.get("allowed_models")
        if allowed_models is not None and model_size not in allowed_models:
            raise ValueError(f"Whisper model '{model_size}' is not in allowed_models.")

        update_job_status(db, job_id, "processing", progress=0)

        model = get_whisper_model(model_size, whisper_settings)
        logger.info(f"Starting transcription for job {job_id} with model '{model_size}'")
        
        transcribe_args = {"beam_size": 5}
        # For SRT format, we need segment timestamps which are generated by default
        # The 'word_timestamps' parameter is for even more granular word-level timestamps
        # which are not needed for basic SRT format

        segments_generator, info = model.transcribe(str(input_path), **transcribe_args)
        
        logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f} for a duration of {info.duration:.2f}s")

        last_update_time = time.time()
        last_db_refresh_time = time.time()
        DB_REFRESH_INTERVAL = 30

        update_job_status(db, job_id, "processing", progress=5)
        logger.info(f"Transcription job {job_id} started, progress at 5%")

        tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")

        preview_segments = []
        PREVIEW_MAX_LENGTH = 1000
        current_preview_length = 0

        with tmp_output_path.open("w", encoding="utf-8") as f:
            if generate_timestamps:
                srt_formatter = SrtFormatter()
                for segment in segments_generator:
                    formatted_srt = srt_formatter.format_segment(segment)
                    f.write(formatted_srt)
                    
                    segment_text = segment.text.strip()
                    if current_preview_length < PREVIEW_MAX_LENGTH:
                        preview_segments.append(segment_text)
                        current_preview_length += len(segment_text)
                    
                    current_time = time.time()
                    if current_time - last_db_refresh_time > DB_REFRESH_INTERVAL:
                        db.close()
                        db = SessionLocal()
                        last_db_refresh_time = current_time

                    if current_time - last_update_time > DB_POLL_INTERVAL_SECONDS:
                        last_update_time = current_time
                        if _current_job_status(db, job_id) in (None, 'cancelled'):
                            logger.info(f"Job {job_id} cancelled or deleted during transcription. Stopping.")
                            return
                        if info.duration > 0:
                            progress = int((segment.end / info.duration) * 100)
                            update_job_status(db, job_id, "processing", progress=progress)
            else:
                for segment in segments_generator:
                    segment_text = segment.text.strip()
                    f.write(segment_text + "\n")

                    if current_preview_length < PREVIEW_MAX_LENGTH:
                        preview_segments.append(segment_text)
                        current_preview_length += len(segment_text)

                    current_time = time.time()
                    if current_time - last_db_refresh_time > DB_REFRESH_INTERVAL:
                        db.close()
                        db = SessionLocal()
                        last_db_refresh_time = current_time

                    if current_time - last_update_time > DB_POLL_INTERVAL_SECONDS:
                        last_update_time = current_time
                        if _current_job_status(db, job_id) in (None, 'cancelled'):
                            logger.info(f"Job {job_id} cancelled or deleted during transcription. Stopping.")
                            return
                        if info.duration > 0:
                            progress = int((segment.end / info.duration) * 100)
                            update_job_status(db, job_id, "processing", progress=progress)

        tmp_output_path.replace(output_path)

        transcript_preview = " ".join(preview_segments)
        if len(transcript_preview) > PREVIEW_MAX_LENGTH:
            transcript_preview = transcript_preview[:PREVIEW_MAX_LENGTH] + "..."

        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=transcript_preview)
        logger.info(f"Transcription for job {job_id} completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred during transcription for job {job_id}")
        update_job_status(db, job_id, "failed", error=str(e))

    finally:
        # This block executes whether the task succeeded, failed, or was cancelled and returned.
        logger.debug(f"Performing cleanup for job {job_id}")

        # Clean up the temporary file if it still exists (e.g., due to cancellation)
        if 'tmp_output_path' in locals() and tmp_output_path.exists():
            try:
                tmp_output_path.unlink()
                logger.debug(f"Removed temporary file: {tmp_output_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {tmp_output_path}: {e}")

        # Clean up the original input file
        try:
            # First, ensure we are not deleting from an unexpected directory
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
            logger.debug(f"Removed input file: {input_path}")
        except Exception as e:
            logger.exception(f"Failed to cleanup input file {input_path} for job {job_id}: {e}")

        if db:
            db.close()

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        # Send notification last, after all state has been finalized.
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def run_tts_task(job_id: str, input_path_str: str, output_path_str: str, model_name: str, tts_settings: dict, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return

        validate_tts_model_name(model_name)
        update_job_status(db, job_id, "processing")

        engine, actual_model_name = "piper", model_name
        if '/' in model_name:
            parts = model_name.split('/', 1)
            engine = parts[0]
            actual_model_name = parts[1]


        logger.info(f"Starting TTS for job {job_id} using engine '{engine}' with model '{actual_model_name}'")
        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")

        if engine == "piper":
            piper_settings = tts_settings.get("piper", {})
            voice = get_piper_voice(actual_model_name, piper_settings)

            with open(input_path, 'r', encoding='utf-8') as f:
                text_to_speak = f.read()

            synthesis_params = piper_settings.get("synthesis_config", {})
            synthesis_config = SynthesisConfig(**synthesis_params) if SynthesisConfig else None

            with wave.open(str(tmp_out), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                voice.synthesize_wav(text_to_speak, wav_file, synthesis_config)

        elif engine == "kokoro":
            kokoro_settings = tts_settings.get("kokoro", {})
            command_template_str = kokoro_settings.get("command_template")
            if not command_template_str:
                raise ValueError("Kokoro TTS command_template is not defined in settings.")

            try:
                lang, voice_name = actual_model_name.split('/', 1)
            except ValueError:
                raise ValueError(f"Invalid Kokoro model format. Expected 'lang/voice', but got '{actual_model_name}'.")

            mapping = {
                "input": str(input_path),
                "output": str(tmp_out),
                "lang": lang,
                "model_name": voice_name,
                "model_path": str(PATHS.KOKORO_MODEL_FILE),
                "voices_path": str(PATHS.KOKORO_VOICES_FILE),
            }

            command = validate_and_build_command(command_template_str, mapping)
            logger.info(f"Executing Kokoro TTS command: {' '.join(command)}")
            run_command(command, timeout=kokoro_settings.get("timeout", 300))

            if not tmp_out.exists():
                raise FileNotFoundError("Kokoro TTS command did not produce an output file.")

        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")

        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview="Successfully generated audio.")
        logger.info(f"TTS for job {job_id} completed.")

    except Exception as e:
        logger.exception(f"ERROR during TTS for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"TTS failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup input file after TTS.")
        db.close()

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        send_webhook_notification(job_id, app_config, base_url)

@huey.task()
def run_pdf_ocr_task(job_id: str, input_path_str: str, output_path_str: str, ocr_settings: dict, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return

        # Re-check for cancellation right before starting the heavy work
        if get_job(db, job_id).status == 'cancelled':
            logger.info(f"PDF OCR job {job_id} cancelled before starting. Aborting.")
            return
        update_job_status(db, job_id, "processing")
        logger.info(f"Starting PDF OCR for job {job_id}")
        ocrmypdf.ocr(str(input_path), str(output_path_str),
                     language=resolve_ocr_language(None, ocr_settings).split('+'),
                     deskew=ocr_settings.get('deskew', True),
                     force_ocr=ocr_settings.get('force_ocr', True),
                     clean=ocr_settings.get('clean', True),
                     optimize=ocr_settings.get('optimize', 1),
                     progress_bar=False)
        with open(output_path_str, "rb") as f:
            reader = pypdf.PdfReader(f)
            preview = "\n".join(page.extract_text() or "" for page in reader.pages)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=preview)
        logger.info(f"PDF OCR for job {job_id} completed.")
    except Exception as e:
        logger.exception(f"ERROR during PDF OCR for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"PDF OCR failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup input file after PDF OCR.")
        db.close()

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        send_webhook_notification(job_id, app_config, base_url)

@huey.task()
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str, app_config: dict, base_url: str, ocr_language: Optional[str] = None):
    db = SessionLocal()
    input_path = Path(input_path_str)
    out_path = Path(output_path_str)

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return

        # Re-check for cancellation right before starting the heavy work
        if get_job(db, job_id).status == 'cancelled':
            logger.info(f"Image OCR job {job_id} cancelled before starting. Aborting.")
            return

        update_job_status(db, job_id, "processing", progress=10)
        lang = resolve_ocr_language(ocr_language, app_config.get("ocr_settings", {}).get("ocrmypdf", {}))
        logger.info(f"Starting Image OCR for job {job_id} ({lang}) - {input_path}")

        # open image and gather frames (support multi-frame TIFF)
        try:
            pil_img = Image.open(str(input_path))
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Cannot identify/open input image: {e}")

        frames = []
        try:
            # some images support n_frames (multi-page TIFF); iterate safely
            n_frames = getattr(pil_img, "n_frames", 1)
            for i in range(n_frames):
                pil_img.seek(i)
                # copy the frame to avoid problems when the original image object is closed
                frames.append(pil_img.convert("RGB").copy())
        except Exception:
            # fallback: single frame
            frames = [pil_img.convert("RGB")]

        update_job_status(db, job_id, "processing", progress=30)

        pdf_bytes_list = []
        text_parts = []
        for idx, frame in enumerate(frames):
            # produce searchable PDF bytes for the frame and plain text as well
            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(frame, extension="pdf", lang=lang)
            except TesseractNotFoundError as e:
                raise RuntimeError("Tesseract not found. Ensure Tesseract OCR is installed and in PATH.") from e
            except Exception as e:
                raise RuntimeError(f"Failed to run Tesseract on frame {idx}: {e}") from e

            pdf_bytes_list.append(pdf_bytes)

            # also extract plain text for preview and possible fallback
            try:
                page_text = pytesseract.image_to_string(frame, lang=lang)
            except Exception:
                page_text = ""
            text_parts.append(page_text)

            # update progress incrementally
            prog = 30 + int((idx + 1) / max(1, len(frames)) * 50)
            update_job_status(db, job_id, "processing", progress=min(prog, 80))

        # merge per-page pdfs if multiple frames
        final_pdf_bytes = None
        if len(pdf_bytes_list) == 1:
            final_pdf_bytes = pdf_bytes_list[0]
        else:
            if _HAS_PYPDF2:
                merger = PdfMerger()
                for b in pdf_bytes_list:
                    merger.append(io.BytesIO(b))
                out_buffer = io.BytesIO()
                merger.write(out_buffer)
                merger.close()
                final_pdf_bytes = out_buffer.getvalue()
            else:
                # PyPDF2 not installed — try a simple concatenation (not valid PDF merge),
                # better to fail loudly so user can install PyPDF2; but as a fallback
                # write the first page only and include a warning in job preview.
                logger.warning("PyPDF2 not available; only the first frame will be written to output PDF.")
                final_pdf_bytes = pdf_bytes_list[0]
                text_parts.insert(0, "[WARNING] Multiple frames detected but PyPDF2 not available; only first page saved.\n")

        # write out atomically
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix or '.pdf'}")
        try:
            tmp_out.parent.mkdir(parents=True, exist_ok=True)
            with tmp_out.open("wb") as f:
                f.write(final_pdf_bytes)
            tmp_out.replace(out_path)
        except Exception as e:
            raise RuntimeError(f"Failed writing output PDF to {out_path}: {e}") from e

        # create a preview from the recognized text (limit length)
        full_text = "\n\n".join(text_parts).strip()
        preview = full_text[:1000] + ("…" if len(full_text) > 1000 else "")

        mark_job_as_completed(db, job_id, output_filepath_str=str(out_path), preview=preview)
        update_job_status(db, job_id, "completed", progress=100)
        logger.info(f"Image OCR for job {job_id} completed. Output: {out_path}")

    except TesseractNotFoundError:
        logger.exception(f"Tesseract not found for job {job_id}")
        update_job_status(db, job_id, "failed", error="Image OCR failed: Tesseract not found on server.")
    except Exception as e:
        logger.exception(f"ERROR during Image OCR for job {job_id}: {e}")
        update_job_status(db, job_id, "failed", error=f"Image OCR failed: {e}")
    finally:
        # cleanup input file (but only if it lives in allowed uploads dir)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup input file after Image OCR.")
        try:
            db.close()
        except Exception:
            logger.exception("Failed to close DB session after Image OCR.")

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        # send webhook regardless of success/failure (keeps original behavior)
        try:
            send_webhook_notification(job_id, app_config, base_url)
        except Exception:
            logger.exception("Failed to send webhook notification after Image OCR.")


# Executables of LibreOffice; each conversion gets a private profile (see run_conversion_task).
LIBREOFFICE_BINARIES = {"libreoffice", "soffice", "lowriter", "localc", "loimpress", "lodraw"}
# Output keys in the default config whose pandoc writer has a different name.
PANDOC_WRITERS = {"txt": "plain", "md": "markdown", "tex": "latex"}
GHOSTSCRIPT_PDF_PRESETS = {"screen", "ebook", "printer", "prepress", "default"}
SOX_SAMPLE_RATES = {"11k": "11025", "22k": "22050", "44k": "44100", "88k": "88200", "176k": "176400"}


def _find_tool_output(out_dir: Path, input_stem: str, output_ext: str) -> Optional[Path]:
    """Finds the result of a tool that picks its own output file name inside out_dir."""
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    exact = [p for p in files if p.name == f"{input_stem}.{output_ext}"]
    if exact:
        return exact[0]
    same_ext = [p for p in files if p.suffix.lower() == f".{output_ext}".lower()]
    if len(same_ext) == 1:
        return same_ext[0]
    if len(files) == 1:
        return files[0]
    return None


def _current_job_status(db: Session, job_id: str) -> Optional[str]:
    """
    Reads the job's status as currently committed. Ending the session's transaction first
    matters: inside one SQLite read transaction (and SQLAlchemy's identity map) a
    cancellation written by the web process would never become visible.
    """
    db.rollback()
    job = get_job(db, job_id)
    return job.status if job else None


@huey.task()
def run_conversion_task(job_id: str,
                        input_path_str: str,
                        output_path_str: str,
                        tool: str,
                        task_key: str,
                        conversion_tools_config: dict,
                        app_config: dict,
                        base_url: str):
    """
    Runs a configured conversion tool for a job.
    - The tool runs in its own process group with resource limits; cancel/timeout kill the whole group.
    - Output is written to temp files instead of pipes, so verbose tools can't block on a full pipe.
    - Tools that ignore {output} and write into {output_dir} (LibreOffice, docling) get a private
      directory whose result is moved into place.
    """
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    temp_input_file: Optional[Path] = None
    temp_output_file: Optional[Path] = None
    job_output_dir: Optional[Path] = None
    lo_profile_dir: Optional[Path] = None

    POLL_INTERVAL = 1.0
    STDERR_SNIPPET = 4000

    def _parse_task_key(tool_name: str, tk: str, tool_cfg: dict, mapping: dict):
        try:
            if tool_name == "ghostscript_pdf":
                preset = ""
                if tk in GHOSTSCRIPT_PDF_PRESETS:
                    preset = f"-dPDFSETTINGS=/{tk}"
                elif tk == "pdfa":
                    preset = "-dPDFA=2"
                mapping.update({"preset": preset})
            elif tool_name.startswith("ghostscript"):
                parts = tk.split("_", 1)
                device = parts[0] if parts and parts[0] else ""
                setting = parts[1] if len(parts) > 1 else ""
                mapping.update({"device": device, "dpi": setting, "preset": setting})
            elif tool_name == "pngquant":
                parts = tk.split("_", 1)
                quality_key = parts[1] if len(parts) > 1 else (parts[0] if parts else "mq")
                quality_map = {"hq": "80-95", "mq": "65-80", "fast": "65-80"}
                speed_map = {"hq": "1", "mq": "3", "fast": "11"}
                mapping.update({"quality": quality_map.get(quality_key, "65-80"),
                                "speed": speed_map.get(quality_key, "3")})
            elif tool_name == "sox":
                parts = tk.split("_")
                if len(parts) >= 3:
                    rate_token = parts[-2]
                    depth_token = parts[-1]
                elif len(parts) == 2:
                    rate_token = parts[-1]
                    depth_token = ""
                else:
                    rate_token = ""
                    depth_token = ""
                rate_val = SOX_SAMPLE_RATES.get(rate_token) or (rate_token.replace("k", "000") if rate_token else "")
                if depth_token:
                    depth_val = ('-b' + depth_token.replace('b', '')) if 'b' in depth_token else depth_token
                else:
                    depth_val = ''
                mapping.update({"samplerate": rate_val, "bitdepth": depth_val})
            elif tool_name == "mozjpeg":
                parts = tk.split("_", 1)
                quality_token = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
                quality = quality_token.replace("q", "") if quality_token else ""
                mapping.update({"quality": quality})
            elif tool_name == 'libreoffice':
                target_ext = mapping['output_ext']
                filter_val = tool_cfg.get("filters", {}).get(target_ext)
                if not filter_val:
                    filter_val = target_ext
                mapping["filter"] = filter_val
            elif tool_name == "pandoc":
                mapping["output_ext"] = PANDOC_WRITERS.get(mapping["output_ext"], mapping["output_ext"])
        except Exception:
            logger.exception("Failed to parse task_key for tool %s; continuing with defaults.", tool_name)

    def _run_cancellable_command(command: List[str], timeout: int) -> subprocess.CompletedProcess:
        """Runs the tool, polling the DB for cancellation. Raises Exception on failure/timeout/cancel."""
        logger.debug("Launching conversion subprocess: %s", " ".join(shlex.quote(c) for c in command))
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f,
                                    preexec_fn=_limit_resources_preexec, start_new_session=True)
            start = time.monotonic()
            while True:
                try:
                    ret = proc.wait(timeout=POLL_INTERVAL)
                except subprocess.TimeoutExpired:
                    ret = None
                if ret is not None:
                    err = _read_tail(err_f, STDERR_SNIPPET)
                    if ret != 0:
                        raise Exception(f"Conversion command failed (rc={ret}): {err}")
                    return subprocess.CompletedProcess(args=command, returncode=ret, stdout=_read_tail(out_f, STDERR_SNIPPET), stderr=err)

                job_status = _current_job_status(db, job_id)
                if job_status is None:
                    logger.warning("Job %s disappeared; killing conversion process.", job_id)
                    _kill_process_tree(proc)
                    raise Exception("Job disappeared during conversion")
                if job_status == "cancelled":
                    logger.info("Job %s cancelled; terminating conversion process.", job_id)
                    _kill_process_tree(proc)
                    raise Exception("Conversion cancelled")
                if timeout and time.monotonic() - start > timeout:
                    logger.warning("Conversion command timed out after %ss; terminating.", timeout)
                    _kill_process_tree(proc)
                    raise Exception(f"Conversion command timed out after {timeout}s")

    try:
        job = get_job(db, job_id)
        if not job:
            logger.warning("Job %s not found; aborting conversion.", job_id)
            return
        if job.status == "cancelled":
            logger.info("Job %s already cancelled; aborting conversion.", job_id)
            return

        update_job_status(db, job_id, "processing", progress=25)
        logger.info("Starting conversion for job %s using %s with task %s", job_id, tool, task_key)

        tool_config = conversion_tools_config.get(tool)
        if not tool_config:
            raise ValueError(f"Unknown conversion tool: {tool}")

        current_input_path = input_path

        # Pre-conversion step for mozjpeg uses run_command (resource-limited)
        if tool == "mozjpeg":
            temp_input_file = input_path.with_suffix('.temp.ppm')
            logger.info("Pre-converting for MozJPEG: %s -> %s", input_path, temp_input_file)
            vips_bin = shutil.which("vips") or "vips"
            pre_conv_cmd = [vips_bin, "copy", str(input_path), str(temp_input_file)]
            try:
                run_command(pre_conv_cmd, timeout=int(tool_config.get("timeout", 300)))
            except Exception as ex:
                short_err = (str(ex) or "")[:STDERR_SNIPPET]
                logger.exception("MozJPEG pre-conversion failed: %s", short_err)
                raise Exception(f"MozJPEG pre-conversion to PPM failed: {short_err}")
            current_input_path = temp_input_file

        update_job_status(db, job_id, "processing", progress=50)

        # Prepare atomic temp output on same FS
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_file = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")
        job_output_dir = PATHS.PROCESSED_DIR / f".tmp-{job_id}"
        shutil.rmtree(job_output_dir, ignore_errors=True)
        job_output_dir.mkdir(parents=True)

        mapping = {
            "input": str(current_input_path),
            "output": str(temp_output_file),
            "output_dir": str(job_output_dir),
            "output_ext": output_path.suffix.lstrip('.'),
        }

        _parse_task_key(tool, task_key, tool_config, mapping)

        command_template_str = tool_config.get("command_template")
        if not command_template_str:
            raise ValueError(f"Tool '{tool}' missing 'command_template' in configuration.")
        command = validate_and_build_command(command_template_str, mapping)
        if not isinstance(command, (list, tuple)) or not command:
            raise ValueError("validate_and_build_command must return a non-empty list/tuple command.")
        command = [str(x) for x in command]

        # LibreOffice instances sharing a profile hand documents to whichever instance is already
        # running and exit without converting, so concurrent jobs silently failed.
        if Path(command[0]).name in LIBREOFFICE_BINARIES and not any(a.startswith("-env:UserInstallation=") for a in command):
            lo_profile_dir = Path(tempfile.mkdtemp(prefix="filewizard-lo-"))
            command.insert(1, f"-env:UserInstallation={lo_profile_dir.as_uri()}")

        logger.info("Executing command: %s", " ".join(shlex.quote(c) for c in command))
        _run_cancellable_command(command, timeout=int(tool_config.get("timeout", 300)))

        if not (temp_output_file.exists() and temp_output_file.stat().st_size > 0):
            produced = _find_tool_output(job_output_dir, current_input_path.stem, output_path.suffix.lstrip('.'))
            if produced:
                produced.replace(temp_output_file)
        if not temp_output_file.exists() or temp_output_file.stat().st_size == 0:
            raise Exception("Conversion failed: The tool produced an empty or missing output file.")

        temp_output_file.replace(output_path)
        mark_job_as_completed(db, job_id, output_filepath_str=str(output_path), preview="Successfully converted file.")
        logger.info("Conversion for job %s completed.", job_id)

    except Exception as e:
        logger.exception("ERROR during conversion for job %s: %s", job_id, e)
        try:
            update_job_status(db, job_id, "failed", error=f"Conversion failed: {e}")
        except Exception:
            logger.exception("Failed to update job status to failed after conversion error.")
    finally:
        # clean main input
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup main input file after conversion.")

        for temp_file in (temp_input_file, temp_output_file):
            if temp_file:
                try:
                    ensure_path_is_safe(Path(temp_file), [PATHS.UPLOADS_DIR, PATHS.PROCESSED_DIR])
                    Path(temp_file).unlink(missing_ok=True)
                except Exception:
                    logger.exception("Failed to cleanup temp file %s after conversion.", temp_file)

        for temp_dir in (job_output_dir, lo_profile_dir):
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        try:
            db.close()
        except Exception:
            logger.exception("Failed to close DB session after conversion.")

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        gc.collect()

        try:
            send_webhook_notification(job_id, app_config, base_url)
        except Exception:
            logger.exception("Failed to send webhook notification after conversion.")

def dispatch_single_file_job(original_filename: str, input_filepath: str, task_type: str, user: dict, db: Session, app_config: Dict, base_url: str, job_id: str | None = None, options: Dict = None, parent_job_id: str | None = None) -> Optional[str]:
    """Helper to create and dispatch a job for a single file. Returns the job id, or None if no job was created."""
    if options is None:
        options = {}

    # If no job_id is passed, generate one. This is for sub-tasks from zips.
    if job_id is None:
        job_id = uuid.uuid4().hex

    safe_filename = safe_basename(original_filename)
    final_path = Path(input_filepath)

    # Ensure the input file exists before creating a job
    if not final_path.exists():
        logger.error(f"Input file does not exist, cannot dispatch job: {input_filepath}")
        return None

    job_data = JobCreate(
        id=job_id, user_id=user['sub'], task_type=task_type,
        original_filename=original_filename, input_filepath=str(final_path),
        input_filesize=final_path.stat().st_size,
        parent_job_id=parent_job_id, callback_url=options.get("callback_url")
    )

    if task_type == 'transcription':
        whisper_settings = app_config.get("transcription_settings", {}).get("whisper", {})
        try:
            model_size = resolve_whisper_model(options.get("model_size"), whisper_settings)
        except ValueError as e:
            logger.warning(f"Not dispatching transcription for '{original_filename}': {e}")
            final_path.unlink(missing_ok=True)
            return None
        output_suffix = '.srt' if options.get('generate_timestamps', False) else '.txt'
        processed_path = PATHS.PROCESSED_DIR / f"{Path(safe_filename).stem}_{job_id[:8]}{output_suffix}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_transcription_task(job_data.id, str(final_path), str(processed_path), model_size, whisper_settings, app_config, base_url, generate_timestamps=options.get('generate_timestamps', False))
    elif task_type == "tts":
        try:
            validate_tts_model_name(options.get("model_name"))
        except ValueError as e:
            logger.warning(f"Not dispatching TTS for '{original_filename}': {e}")
            final_path.unlink(missing_ok=True)
            return None
        tts_config = app_config.get("tts_settings", {})
        stem = Path(safe_filename).stem
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.wav"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_tts_task(job_data.id, str(final_path), str(processed_path), options.get("model_name"), tts_config, app_config, base_url)
    elif task_type == "ocr":
        stem, suffix = Path(safe_filename).stem, Path(safe_filename).suffix.lower()
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
        if suffix not in IMAGE_EXTENSIONS and suffix != '.pdf':
            logger.warning(f"Skipping unsupported file type for OCR: {original_filename}")
            # Clean up the orphaned file from the zip extraction
            final_path.unlink(missing_ok=True)
            return None
        ocr_settings = app_config.get("ocr_settings", {}).get("ocrmypdf", {})
        try:
            ocr_language = resolve_ocr_language(options.get("ocr_language"), ocr_settings)
        except ValueError as e:
            logger.warning(f"Not dispatching OCR for '{original_filename}': {e}")
            final_path.unlink(missing_ok=True)
            return None
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.pdf"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        if suffix in IMAGE_EXTENSIONS:
            run_image_ocr_task(job_data.id, str(final_path), str(processed_path), app_config, base_url, ocr_language)
        else:
            run_pdf_ocr_task(job_data.id, str(final_path), str(processed_path), dict(ocr_settings, language=ocr_language), app_config, base_url)
    elif task_type == "conversion":
        try:
            logger.info(f"Preparing to dispatch conversion job for file '{original_filename}' with requested format '{options.get('output_format')}'")
            all_tools = app_config.get("conversion_tools", {}).keys()
            logger.info(f"Available conversion tools: {', '.join(all_tools)}")
            tool, task_key = _parse_tool_and_task_key(options.get("output_format"), all_tools)
            logger.info(f"Dispatching conversion job using tool '{tool}' with task key '{task_key}' for file '{original_filename}'")
        except (AttributeError, ValueError):
            if parent_job_id:
                logger.warning(f"Skipping file '{original_filename}' from batch job '{parent_job_id}' as it is not applicable for the selected conversion format '{options.get('output_format')}'.")
                final_path.unlink(missing_ok=True)
                return None
            else:
                logger.error(f"Invalid or missing output_format for conversion of {original_filename}")
                final_path.unlink(missing_ok=True)
                return None

        original_stem = Path(safe_filename).stem

        if tool == 'pandoc_academic':
            processed_path = PATHS.PROCESSED_DIR / f"{original_stem}_{job_id}.pdf"
            job_data.processed_filepath = str(processed_path)
            job_data.task_type = 'academic_pandoc' # Use a more specific task type for the DB
            create_job(db=db, job=job_data)
            run_academic_pandoc_task(job_data.id, str(final_path), str(processed_path), task_key, APP_CONFIG, base_url)
        else:
            target_ext = task_key.split('_')[0]
            if tool == "ghostscript_pdf": target_ext = "pdf"
            processed_path = PATHS.PROCESSED_DIR / f"{original_stem}_{job_id}.{target_ext}"
            job_data.processed_filepath = str(processed_path)
            create_job(db=db, job=job_data)
            run_conversion_task(job_data.id, str(final_path), str(processed_path), tool, task_key, app_config.get("conversion_tools", {}), app_config, base_url)
    else:
        logger.error(f"Invalid task type '{task_type}' for file {original_filename}")
        final_path.unlink(missing_ok=True)
        return None
    return job_id

@huey.task()
def run_academic_pandoc_task(job_id: str, input_path_str: str, output_path_str: str, task_key: str, app_config: dict, base_url: str):
    """
    Runs a Pandoc conversion for a zipped academic project (e.g., markdown + bibliography).
    """
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    unzip_dir = PATHS.UPLOADS_DIR / f"unzipped_{job_id}"

    def find_first_file_with_ext(directory: Path, extensions: List[str]) -> Optional[Path]:
        for ext in extensions:
            try:
                return next(directory.rglob(f"*{ext}"))
            except StopIteration:
                continue
        return None

    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return

        update_job_status(db, job_id, "processing", progress=10)
        logger.info(f"Starting academic Pandoc task for job {job_id}")

        # 1. Unzip the project
        if not zipfile.is_zipfile(input_path):
            raise ValueError("Input is not a valid ZIP archive.")
        unzip_dir.mkdir()
        safe_extract_zip(input_path, unzip_dir, app_config.get("app_settings", {}))

        update_job_status(db, job_id, "processing", progress=25)

        # 2. Find required files
        main_doc = find_first_file_with_ext(unzip_dir, ['.md', '.tex', '.txt'])
        bib_file = find_first_file_with_ext(unzip_dir, ['.bib'])
        csl_file = find_first_file_with_ext(unzip_dir, ['.csl'])

        if not main_doc:
            raise FileNotFoundError("No main document (.md, .tex, .txt) found in the ZIP archive.")
        if not bib_file:
            raise FileNotFoundError("No bibliography file (.bib) found in the ZIP archive.")

        update_job_status(db, job_id, "processing", progress=40)

        # 3. Build Pandoc command
        command = ['pandoc', str(main_doc), '-o', str(output_path)]
        command.extend(['--bibliography', str(bib_file)])
        command.append('--citeproc') # Use the citation processor

        # Handle CSL style
        style_key = task_key.split('_')[-1] # e.g., 'apa' from 'pdf_apa'
        csl_path_or_url = None

        if csl_file:
            logger.info(f"Using CSL file found in ZIP: {csl_file.name}")
            csl_path_or_url = str(csl_file)
        else:
            # Look up CSL from config
            try:
                csl_path_or_url = app_config['academic_settings']['pandoc']['csl_files'][style_key]
                logger.info(f"Using CSL style '{style_key}' from configuration.")
            except KeyError:
                logger.warning(f"No CSL style found for key '{style_key}'. Pandoc will use its default.")

        if csl_path_or_url:
            command.extend(['--csl', csl_path_or_url])

        command.extend(['--pdf-engine', 'xelatex'])

        update_job_status(db, job_id, "processing", progress=50)
        logger.info(f"Executing Pandoc command for job {job_id}: {' '.join(command)}")

        # 4. Run pandoc in the unzipped directory so relative resources resolve
        try:
            run_command(command, timeout=300, cwd=unzip_dir)
        except Exception as e:
            raise Exception(f"Pandoc compilation failed. Please check your document for errors. {str(e)[:2000]}") from e

        # 5. Verify output
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise Exception("Pandoc conversion failed: The tool produced an empty or missing output file.")

        mark_job_as_completed(db, job_id, output_filepath_str=str(output_path), preview="Successfully created academic PDF.")
        logger.info(f"Academic Pandoc task for job {job_id} completed.")

    except Exception as e:
        logger.exception(f"ERROR during academic Pandoc task for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"Pandoc task failed: {e}")
    finally:
        # 6. Cleanup
        if unzip_dir.exists():
            shutil.rmtree(unzip_dir, ignore_errors=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup input ZIP file after Pandoc task.")

        if db:
            db.close()

        # If this was a sub-job, trigger a progress update for its parent.
        db_for_check = SessionLocal()
        try:
            job = get_job(db_for_check, job_id)
            if job and job.parent_job_id:
                _update_parent_zip_job_progress(job.parent_job_id)
        finally:
            db_for_check.close()

        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def _update_parent_zip_job_progress(parent_job_id: str):
    """Checks and updates the progress of a parent 'unzip' job."""
    db = SessionLocal()
    try:
        parent_job = get_job(db, parent_job_id)
        # The parent stays 'pending' while its sub-jobs are being dispatched; judging completion
        # before all of them exist would finish it early (or get overwritten back to 'processing').
        if not parent_job or parent_job.status != 'processing':
            return # Job is still dispatching, already finalized, or doesn't exist

        child_jobs = db.query(Job).filter(Job.parent_job_id == parent_job.id).all()
        total_children = len(child_jobs)

        if total_children == 0:
            return # Should not happen if dispatched correctly, but safeguard.

        finished_children = 0
        for child in child_jobs:
            if child.status in ['completed', 'failed', 'cancelled']:
                finished_children += 1

        progress = int((finished_children / total_children) * 100) if total_children > 0 else 100

        if finished_children == total_children:
            failed_count = sum(1 for child in child_jobs if child.status == 'failed')
            preview = f"Batch processing complete. {total_children - failed_count}/{total_children} tasks succeeded."
            if failed_count > 0:
                preview += f" ({failed_count} failed)."
            mark_job_as_completed(db, parent_job.id, preview=preview)
            logger.info(f"Batch job {parent_job.id} marked as completed.")
        else:
            if parent_job.progress != progress:
                update_job_status(db, parent_job.id, 'processing', progress=progress)

    except Exception as e:
        logger.exception(f"Error in _update_parent_zip_job_progress for parent {parent_job_id}: {e}")
    finally:
        db.close()


@huey.task()
def unzip_and_dispatch_task(job_id: str, input_path_str: str, sub_task_type: str, sub_task_options: dict, user: dict, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    unzip_dir = PATHS.UPLOADS_DIR / f"unzipped_{job_id}"
    logger.info(f"Starting unzip and dispatch task for job {job_id} into {sub_task_type} jobs. ")

    try:
        if not zipfile.is_zipfile(input_path):
            raise ValueError("Uploaded file is not a valid ZIP archive.")
        unzip_dir.mkdir()
        extracted_files = safe_extract_zip(input_path, unzip_dir, app_config.get("app_settings", {}))

        created_jobs = 0
        for extracted_file_path in extracted_files:
            if dispatch_single_file_job(
                original_filename=extracted_file_path.name,
                input_filepath=str(extracted_file_path),
                task_type=sub_task_type,
                options=sub_task_options,
                user=user,
                db=db,
                app_config=app_config,
                base_url=base_url,
                parent_job_id=job_id
            ):
                created_jobs += 1
        if created_jobs > 0:
            # All sub-jobs exist now; from here on their completion drives the parent's progress.
            update_job_status(db, job_id, "processing", progress=0)
            _update_parent_zip_job_progress(job_id)
        else:
            mark_job_as_completed(db, job_id, preview="No files in the ZIP archive could be processed with the selected task.")

    except Exception as e:
        logger.exception(f"ERROR during ZIP processing for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"Failed to process ZIP file: {e}")
        # If unzipping fails, clean up the directory
        if unzip_dir.exists():
            shutil.rmtree(unzip_dir)
    finally:
        try:
            # CRITICAL FIX: Only delete the original ZIP file.
            # Do NOT delete the unzip_dir here, as the sub-tasks need the files.
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup original ZIP file.")
        db.close()



_huey_startup_lock = threading.Lock()


@huey.on_startup()
def _init_huey_worker():
    """
    The huey consumer is a separate process that never runs the FastAPI lifespan: load the
    settings there and run the Whisper cache eviction where the models are actually loaded.
    """
    global _cache_cleanup_thread
    with _huey_startup_lock:
        if not APP_CONFIG:
            try:
                load_app_config()
            except Exception:
                logger.exception("Huey worker could not load settings.")
        if _cache_cleanup_thread is None:
            _cache_cleanup_thread = threading.Thread(target=_whisper_cache_cleanup_worker, daemon=True)
            _cache_cleanup_thread.start()


@huey.periodic_task(crontab(minute="15"))
def apply_retention_policy():
    """
    Deletes finished jobs (and their files) older than app_settings.retention_days.
    0 or unset keeps everything.
    """
    reload_app_config_if_changed()
    try:
        days = float(APP_CONFIG.get("app_settings", {}).get("retention_days") or 0)
    except (TypeError, ValueError):
        days = 0
    if days <= 0:
        return
    # created_at/updated_at are stored as naive UTC
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
    db = SessionLocal()
    try:
        jobs = db.query(Job).filter(Job.status.in_(FINAL_JOB_STATES), Job.updated_at < cutoff).all()
        if jobs:
            deleted_ids, files_deleted = delete_jobs_and_files(db, jobs)
            logger.info(f"Retention: deleted {len(deleted_ids)} jobs older than {days:g} days and {files_deleted} files.")
    finally:
        db.close()


STALE_UPLOAD_HOURS = float(os.environ.get("STALE_UPLOAD_HOURS", "6"))


@huey.periodic_task(crontab(minute="*/30"))
def cleanup_stale_uploads():
    """Removes abandoned chunked uploads and ZIP extraction directories of finished batches."""
    cutoff = time.time() - STALE_UPLOAD_HOURS * 3600
    removed = 0
    db = SessionLocal()
    try:
        candidates = ([p for p in PATHS.CHUNK_TMP_DIR.iterdir()] + list(PATHS.UPLOADS_DIR.glob("unzipped_*"))
                      + list(PATHS.PROCESSED_DIR.glob(".tmp-*")))
        for entry in candidates:
            try:
                if not entry.is_dir() or entry.is_symlink() or entry.stat().st_mtime >= cutoff:
                    continue
                if entry.name.startswith("unzipped_"):
                    parent = get_job(db, entry.name[len("unzipped_"):])
                    if parent and parent.status in ("pending", "processing"):
                        continue  # sub-jobs may still need their input files
                shutil.rmtree(entry)
                removed += 1
            except OSError:
                logger.exception(f"Could not remove stale upload directory {entry}")
    finally:
        db.close()
    if removed:
        logger.info(f"Removed {removed} stale upload directories.")


# --------------------------------------------------------------------------------
# --- 5. FASTAPI APPLICATION
# --------------------------------------------------------------------------------

# --- SSE Broadcaster for real-time UI updates ---
import asyncio
import json





# --------------------------------------------------------------------------------
# --- 2. DATABASE & Schemas

async def download_kokoro_models_if_missing():
    """Checks for Kokoro TTS model files and downloads them if they don't exist or are empty."""
    files_to_download = {
        "model": {
            "path": PATHS.KOKORO_MODEL_FILE,
            "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx",
            "size": 325532387
        },
        "voices": {
            "path": PATHS.KOKORO_VOICES_FILE,
            "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin",
            "size": 26124436
        }
    }
    # All gunicorn workers run this at startup; only one may download, and files appear atomically.
    lock_file = open(PATHS.KOKORO_TTS_MODELS_DIR / ".download.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        logger.info("Another worker is checking/downloading the Kokoro TTS models.")
        return
    try:
        await _download_kokoro_files(files_to_download)
    finally:
        lock_file.close()


async def _download_kokoro_files(files_to_download: dict):
    async with httpx.AsyncClient() as client:
        for name, details in files_to_download.items():
            path, url, expected_size = details["path"], details["url"], details["size"]
            
            # Check for existence and size
            if not path.exists() or path.stat().st_size != expected_size:
                if path.exists():
                    logger.warning(f"Kokoro TTS {name} file found but has incorrect size. Expected {expected_size}, got {path.stat().st_size}. Re-downloading.")
                else:
                    logger.info(f"Kokoro TTS {name} file missing. Downloading from {url}...")

                partial_path = path.with_name(path.name + ".part")
                for attempt in range(3):
                    try:
                        with partial_path.open("wb") as f:
                            async with client.stream("GET", url, follow_redirects=True, timeout=300) as response:
                                response.raise_for_status()
                                total_downloaded = 0
                                async for chunk in response.aiter_bytes():
                                    f.write(chunk)
                                    total_downloaded += len(chunk)

                        if total_downloaded == expected_size:
                            partial_path.replace(path)
                            logger.info(f"Successfully downloaded Kokoro TTS {name} file to {path}.")
                            break
                        else:
                            logger.warning(f"Kokoro TTS {name} download incomplete. Expected {expected_size}, got {total_downloaded}. Retrying...")
                    except Exception as e:
                        logger.error(f"Failed to download Kokoro TTS {name} file (attempt {attempt + 1}): {e}")
                        partial_path.unlink(missing_ok=True)
                        await asyncio.sleep(5)
                else:
                    logger.critical(f"Failed to download Kokoro TTS {name} file after 3 attempts. TTS will not be available.")

            else:
                logger.info(f"Found existing and valid Kokoro TTS {name} file at {path}.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    global AVAILABLE_TTS_VOICES_CACHE
    AVAILABLE_TTS_VOICES_CACHE = None
    # Base.metadata.create_all(bind=engine)

    create_attempts = 3
    for attempt in range(1, create_attempts + 1):
        try:
            # use engine.begin() to ensure the DDL runs in a connection/transaction context
            with engine.begin() as conn:
                Base.metadata.create_all(bind=conn)
            logger.info("Database tables ensured (create_all succeeded).")
            break
        except OperationalError as oe:
            # Some SQLite drivers raise an OperationalError when two processes try to create the same table at once.
            msg = str(oe).lower()
            # If we see "already exists" we treat this as a race and retry briefly.
            if "already exists" in msg or ("table" in msg and "already exists" in msg):
                logger.warning(
                    "Database table creation race detected (attempt %d/%d): %s. Retrying...",
                    attempt,
                    create_attempts,
                    oe,
                )
                time.sleep(0.5)
                continue
            else:
                logger.exception("Database initialization failed with OperationalError.")
                raise
        except Exception:
            logger.exception("Unexpected error during DB initialization.")
            raise

    initialize_settings_file()
    load_app_config()

    # Start the cache cleanup thread
    global _cache_cleanup_thread
    if _cache_cleanup_thread is None:
        _cache_cleanup_thread = threading.Thread(target=_whisper_cache_cleanup_worker, daemon=True)
        _cache_cleanup_thread.start()
        logger.info("Whisper model cache cleanup thread started.")

    # Download required models on startup
    DOWNLOAD_KOKORO_ON_STARTUP = os.environ.get('DOWNLOAD_KOKORO_ON_STARTUP', 'false').lower() == 'true'
    if shutil.which("kokoro-tts") and DOWNLOAD_KOKORO_ON_STARTUP:
        logger.info("Checking for Kokoro TTS Models (async)")
        app.state.download_kokoro_task = asyncio.create_task(download_kokoro_models_if_missing())

    if PiperVoice is None:
        logger.warning("piper-tts is not installed. Piper TTS features will be disabled. Install with: pip install piper-tts")
    if not shutil.which("kokoro-tts"):
        logger.warning("kokoro-tts command not found in PATH. Kokoro TTS features will be disabled.")

    ENV = os.environ.get('ENV', 'dev').lower()
    ALLOW_LOCAL_ONLY = os.environ.get('ALLOW_LOCAL_ONLY', 'false').lower() == 'true'
    if LOCAL_ONLY_MODE and ENV != 'dev' and not ALLOW_LOCAL_ONLY:
        raise RuntimeError('LOCAL_ONLY_MODE may only be enabled in dev or when ALLOW_LOCAL_ONLY=true is set.')
    if not LOCAL_ONLY_MODE:
        oidc_cfg = APP_CONFIG.get('auth_settings', {})
        if not all(oidc_cfg.get(k) for k in ['oidc_client_id', 'oidc_client_secret', 'oidc_server_metadata_url']):
            logger.warning("OIDC auth settings are incomplete. Auth will be disabled.")
            app.state.oidc_registration_failed = True
        else:
            try:
                oauth.register(
                    name='oidc',
                    client_id=oidc_cfg.get('oidc_client_id'),
                    client_secret=oidc_cfg.get('oidc_client_secret'),
                    server_metadata_url=oidc_cfg.get('oidc_server_metadata_url'),
                    client_kwargs={'scope': 'openid email profile'},
                    userinfo_endpoint=oidc_cfg.get('oidc_userinfo_endpoint'),
                    end_session_endpoint=oidc_cfg.get('oidc_end_session_endpoint')
                )
                logger.info('OAuth registered successfully.')
                app.state.oidc_registration_failed = False
            except Exception as e:
                logger.error(f"Failed to register OIDC OAuth provider: {e}. Authentication will be disabled.")
                app.state.oidc_registration_failed = True
    
    # Background task for processing WebSocket notification queue
    async def process_notifications_periodically():
        """Process WebSocket notification queue periodically"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while True:
            try:
                await manager.process_notification_queue()
                await asyncio.sleep(0.1)  # Process every 100ms
                
                # Reset error counter on success
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error processing WebSocket notifications (consecutive errors: {consecutive_errors}): {e}")
                
                # If we have too many consecutive errors, log a warning
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(f"Too many consecutive errors in WebSocket notification processor. Restarting...")
                    consecutive_errors = 0
                
                await asyncio.sleep(min(1 * consecutive_errors, 10))  # Exponential backoff, max 10s
    
    if ENABLE_WEBSOCKETS:
        # Store the task reference so we can cancel it later
        app.state.notification_processor_task = asyncio.create_task(process_notifications_periodically())
        logger.info("WebSocket notification processor task started")
    
    yield
    
    # Cleanup
    if ENABLE_WEBSOCKETS and hasattr(app.state, 'notification_processor_task'):
        app.state.notification_processor_task.cancel()
        try:
            await app.state.notification_processor_task
        except asyncio.CancelledError:
            pass

    if hasattr(app.state, 'download_kokoro_task'):
        app.state.download_kokoro_task.cancel()
        try:
            await app.state.download_kokoro_task
        except asyncio.CancelledError:
            logger.info("Kokoro download task cancelled.")
            pass

    logger.info('Application shutting down...')

app = FastAPI(lifespan=lifespan)
ENV = os.environ.get('ENV', 'dev').lower()


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ('true', '1', 't', 'yes', 'y')


def _env_list(name: str) -> List[str]:
    return [v.strip() for v in os.environ.get(name, '').split(',') if v.strip()]


def _load_or_create_secret_key() -> str:
    """
    Returns a session signing key shared by all workers. A key generated per process
    would differ between the gunicorn workers, so a login started on one worker would
    fail its OAuth state check on another. The generated key is persisted next to
    settings.yml (mode 0600) so sessions also survive restarts.
    """
    key_file = PATHS.CONFIG_DIR / ".secret_key"
    try:
        existing = key_file.read_text(encoding="utf8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"Could not read {key_file}: {e}")

    new_key = secrets.token_hex(32)
    try:
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write(new_key)
        logger.info(f"SECRET_KEY is not set; generated a persistent session key in {key_file}.")
        return new_key
    except FileExistsError:
        # Another worker is creating the file right now; wait for its content.
        for _ in range(50):
            existing = key_file.read_text(encoding="utf8").strip()
            if existing:
                return existing
            time.sleep(0.1)
    except OSError as e:
        logger.warning(f"Could not persist a session key to {key_file}: {e}")
    logger.warning("Using a temporary per-process session key. Set SECRET_KEY so logins work across workers and restarts.")
    return new_key


SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY and not LOCAL_ONLY_MODE and ENV != 'dev':
    raise RuntimeError('SECRET_KEY must be set in production when authentication is enabled.')
if not SECRET_KEY:
    SECRET_KEY = _load_or_create_secret_key()

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    # Set SESSION_COOKIE_SECURE=true when the app is served over HTTPS (e.g. behind a TLS reverse proxy).
    https_only=_env_flag('SESSION_COOKIE_SECURE', False),
    same_site='lax',
    max_age=14 * 24 * 60 * 60  # 14 days
)

# CORS: the UI is served from the same origin and API clients (curl, n8n, ...) do not need CORS,
# so no cross-origin access is granted unless origins are listed explicitly.
CORS_ALLOWED_ORIGINS = [o for o in _env_list('ALLOWED_ORIGINS') if o != '*']
if '*' in _env_list('ALLOWED_ORIGINS'):
    logger.warning("ALLOWED_ORIGINS='*' is ignored: credentialed CORS must list explicit origins.")
if CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# Optional Host header allowlist (protects LOCAL_ONLY instances against DNS rebinding).
ALLOWED_HOSTS = _env_list('ALLOWED_HOSTS')
if ALLOWED_HOSTS:
    # Loopback names stay allowed for health checks; a rebinding attack arrives with its own host name.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS + ["localhost", "127.0.0.1", "::1"])

# --- CSRF protection & security headers ---
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
CSRF_TRUSTED_ORIGINS = _env_list('CSRF_TRUSTED_ORIGINS') + CORS_ALLOWED_ORIGINS
# Who may embed the UI in a frame (e.g. a dashboard such as Homarr/Organizr): CSP frame-ancestors syntax.
FRAME_ANCESTORS = os.environ.get('FRAME_ANCESTORS', "'self'").strip() or "'self'"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; "
    f"base-uri 'self'; form-action 'self'; frame-ancestors {FRAME_ANCESTORS}"
)
# The interactive API docs load their assets from a CDN; they get the other headers but no CSP.
CSP_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json")


def _host_key(netloc: str) -> str:
    """Normalizes host[:port] for comparison, dropping default ports."""
    netloc = (netloc or "").strip().lower()
    for default_port in (":80", ":443"):
        if netloc.endswith(default_port):
            return netloc[: -len(default_port)]
    return netloc


def _csrf_allowed_hosts(headers: Headers) -> set:
    hosts = {_host_key(headers.get("host", ""))}
    forwarded_host = headers.get("x-forwarded-host")
    if forwarded_host:
        hosts.add(_host_key(forwarded_host.split(",")[0]))
    public_url = APP_CONFIG.get("app_settings", {}).get("app_public_url")
    if public_url:
        hosts.add(_host_key(urlparse(public_url).netloc))
    for origin in CSRF_TRUSTED_ORIGINS:
        hosts.add(_host_key(urlparse(origin).netloc or origin))
    hosts.discard("")
    return hosts


def _is_csrf_safe(method: str, headers: Headers) -> bool:
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
    return _host_key(netloc) in _csrf_allowed_hosts(headers)


class SecurityMiddleware:
    """Pure ASGI middleware: CSRF origin check and security response headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        reload_app_config_if_changed()
        headers = Headers(scope=scope)
        if not _is_csrf_safe(scope["method"], headers):
            logger.warning(
                "Blocked cross-origin %s %s (origin=%r, referer=%r, host=%r).",
                scope["method"], scope.get("path"), headers.get("origin"), headers.get("referer"), headers.get("host"),
            )
            response = JSONResponse(
                {"detail": "Cross-origin request blocked. If FileWizard runs behind a reverse proxy, "
                           "forward the original Host header, or set app_public_url / CSRF_TRUSTED_ORIGINS."},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers.setdefault("X-Content-Type-Options", "nosniff")
                response_headers.setdefault("Referrer-Policy", "same-origin")
                if FRAME_ANCESTORS == "'self'":
                    response_headers.setdefault("X-Frame-Options", "SAMEORIGIN")
                elif FRAME_ANCESTORS == "'none'":
                    response_headers.setdefault("X-Frame-Options", "DENY")
                if not path.startswith(CSP_EXEMPT_PATHS):
                    response_headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityMiddleware)


# Static / templates
app.mount("/static", StaticFiles(directory=str(PATHS.BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PATHS.BASE_DIR / "templates"))

# --- AUTH & USER HELPERS ---
http_bearer = HTTPBearer()

LOCAL_USER = {'sub': 'local_user', 'email': 'local@user.com', 'name': 'Local User'}


def _normalized_email_set(values) -> set:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {v.strip().lower() for v in values if isinstance(v, str) and v.strip()}


def _verified_email(user: dict | None) -> str | None:
    """The user's email, unless the identity provider explicitly marked it as unverified."""
    if not user:
        return None
    email = user.get('email')
    if not isinstance(email, str) or not email.strip() or user.get('email_verified') is False:
        return None
    return email.strip().lower()


def is_user_allowed(user: dict | None) -> bool:
    """
    Authorization for OIDC users. With auth_settings.allowed_users and allowed_domains both
    empty, every account the identity provider authenticates may use the app.
    """
    if LOCAL_ONLY_MODE:
        return True
    if not user:
        return False
    auth_settings = APP_CONFIG.get("auth_settings", {})
    allowed_users = _normalized_email_set(auth_settings.get("allowed_users"))
    allowed_domains = {d.lstrip('@') for d in _normalized_email_set(auth_settings.get("allowed_domains"))}
    if not allowed_users and not allowed_domains:
        return True
    email = _verified_email(user)
    if not email:
        return False
    if email in allowed_users or email in _normalized_email_set(auth_settings.get("admin_users")):
        return True
    return email.rsplit('@', 1)[-1] in allowed_domains


def get_current_user(request: Request):
    if LOCAL_ONLY_MODE:
        return LOCAL_USER
    user = request.session.get('user')
    # Re-checked on every request so removing someone from the allowlist takes effect immediately.
    if user and not is_user_allowed(user):
        return None
    return user


async def _user_from_bearer_token(token: str) -> dict:
    if not check_oidc_availability():
        logger.warning("OIDC not available for API authentication")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication system is not properly configured")
    try:
        user = dict(await oauth.oidc.userinfo(token={'access_token': token}))
    except Exception as e:
        logger.error(f"API token validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    if not user.get('sub') or not is_user_allowed(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is not allowed to use this service.")
    return user


async def require_api_user(request: Request, creds: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Dependency for API routes requiring OIDC bearer token authentication."""
    if LOCAL_ONLY_MODE:
        # Same identity as the UI, so API jobs show up in the history and their downloads work.
        return LOCAL_USER
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return await _user_from_bearer_token(creds.credentials)


async def require_user_or_api(request: Request):
    """Accepts a browser session or, for API clients, an OIDC bearer token."""
    user = get_current_user(request)
    if user:
        return user
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return await _user_from_bearer_token(token.strip())
    raise HTTPException(status_code=401, detail="Not authenticated")


def is_admin(request: Request) -> bool:
    if LOCAL_ONLY_MODE: return True
    email = _verified_email(get_current_user(request))
    if not email: return False
    return email in _normalized_email_set(APP_CONFIG.get("auth_settings", {}).get("admin_users", []))

def require_user(request: Request):
    user = get_current_user(request)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def require_admin(request: Request):
    if not is_admin(request): raise HTTPException(status_code=403, detail="Administrator privileges required.")
    return True

def check_oidc_availability():
    """Check if OIDC is properly configured and registered"""
    if LOCAL_ONLY_MODE:
        return True  # No OIDC needed in local mode
    # In non-local mode, check if registration failed
    return not getattr(app.state, 'oidc_registration_failed', True)

# --- FILE SAVING UTILITY ---
async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    """
    Saves an uploaded file to a destination, handling size limits and validating file type.
    This function is used by both the simple API and the legacy direct-upload routes.
    """
    max_size = APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024)
    
    # Validate file type before processing
    allowed_extensions = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not validate_file_type(upload_file.filename, allowed_extensions):
        raise HTTPException(status_code=400, detail=f"File type '{Path(upload_file.filename).suffix}' not allowed.")
    
    tmp_path = destination.with_name(f"{destination.stem}.tmp-{uuid.uuid4().hex}{destination.suffix}")
    size = 0
    try:
        with tmp_path.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(status_code=413, detail=f"File exceeds {max_size / 1024 / 1024} MB limit")
                buffer.write(chunk)
        tmp_path.replace(destination)
        return size
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.exception(f"Error saving upload file: {e}")
        # Ensure temp file is cleaned up on error
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            logger.exception("Failed to remove temp upload file after error.")
        raise HTTPException(status_code=500, 
                            detail="Failed to save uploaded file due to server error")
    finally:
        try:
            await upload_file.close()
        except Exception:
            pass  # Don't let close failure mask other errors

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    if not allowed_extensions: # If set is empty, allow all
        return True
    return Path(filename).suffix.lower() in allowed_extensions

# --- CHUNKED UPLOADS (for UI) ---
def _max_upload_bytes() -> int:
    return int(APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024))


@app.post("/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    user: dict = Depends(require_user)
):
    temp_dir = chunk_dir_for(user, upload_id)
    if not 0 <= chunk_number < MAX_UPLOAD_CHUNKS:
        raise HTTPException(status_code=400, detail="Invalid chunk_number.")
    chunk_path = temp_dir / f"{chunk_number}.chunk"
    max_size = _max_upload_bytes()

    def save_chunk_sync():
        temp_dir.mkdir(parents=True, exist_ok=True)
        # The size limit applies to the whole file, i.e. all chunks received so far.
        received = sum(p.stat().st_size for p in temp_dir.glob("*.chunk") if p.name != chunk_path.name)
        partial_path = temp_dir / f"{chunk_number}.part"
        try:
            with open(partial_path, "wb") as buffer:
                while True:
                    block = chunk.file.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > max_size:
                        raise HTTPException(status_code=413, detail=f"File exceeds {max_size / 1024 / 1024:.0f} MB limit")
                    buffer.write(block)
            partial_path.replace(chunk_path)  # only complete chunks are ever stitched
        except HTTPException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            chunk.file.close()
            partial_path.unlink(missing_ok=True)

    await run_in_threadpool(save_chunk_sync)

    return JSONResponse({"message": f"Chunk {chunk_number} for {upload_id} uploaded."})


async def _stitch_chunks(temp_dir: Path, final_path: Path, total_chunks: int):
    """Stitches chunks together memory-efficiently and cleans up."""
    ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    ensure_path_is_safe(final_path, [PATHS.UPLOADS_DIR])
    max_size = _max_upload_bytes()

    # This is a blocking function that will be run in a threadpool
    def do_stitch():
        chunk_paths = [temp_dir / f"{i}.chunk" for i in range(total_chunks)]
        for i, chunk_path in enumerate(chunk_paths):
            if not chunk_path.exists():
                # Raise an exception that can be caught and handled
                raise FileNotFoundError(f"Upload failed: missing chunk {i}")
        if sum(p.stat().st_size for p in chunk_paths) > max_size:
            raise HTTPException(status_code=413, detail=f"File exceeds {max_size / 1024 / 1024:.0f} MB limit")
        with open(final_path, "wb") as final_file:
            for chunk_path in chunk_paths:
                with open(chunk_path, "rb") as chunk_file:
                    # Use copyfileobj for memory efficiency
                    shutil.copyfileobj(chunk_file, final_file)

    try:
        await run_in_threadpool(do_stitch)
    except FileNotFoundError as e:
        # If a chunk was missing, clean up and re-raise as HTTPException
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        final_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


UPLOAD_TASK_TYPES = {"conversion", "ocr", "transcription", "tts"}


def _require_tool_available(tool: str):
    if tool in UNAVAILABLE_TOOLS:
        name = APP_CONFIG.get("conversion_tools", {}).get(tool, {}).get("name", tool)
        raise HTTPException(status_code=400, detail=f"{name} is not installed on this server ('{UNAVAILABLE_TOOLS[tool]}' not found).")


def _validate_task_options(task_type: str, options: dict, is_zip: bool) -> Optional[tuple]:
    """
    Validates a processing request before any file is written. Returns (tool, task_key)
    for conversions. Raises HTTPException(400) for invalid input.
    """
    if task_type not in UPLOAD_TASK_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid task_type: '{task_type}'")
    try:
        if task_type == "transcription":
            whisper_settings = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
            options["model_size"] = resolve_whisper_model(options.get("model_size"), whisper_settings)
        elif task_type == "tts":
            validate_tts_model_name(options.get("model_name"))
        elif task_type == "ocr":
            ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
            options["ocr_language"] = resolve_ocr_language(options.get("ocr_language"), ocr_settings)
        elif task_type == "conversion":
            all_tools = APP_CONFIG.get("conversion_tools", {}).keys()
            tool, task_key = _parse_tool_and_task_key(options.get("output_format") or "", all_tools)
            _require_tool_available(tool)
            return tool, task_key
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return None


def _parse_tool_and_task_key(output_format: str, all_tool_keys: list) -> (str, str):
    """Robustly parses an output_format string to find the matching tool and task key."""
    # Sort keys by length descending to match longest prefix first (e.g., 'ghostscript_image' before 'ghostscript')
    for tool_key in sorted(all_tool_keys, key=len, reverse=True):
        if output_format.startswith(tool_key + '_'):
            task_key = output_format[len(tool_key) + 1:]
            return tool_key, task_key
    raise ValueError(f"Could not determine tool from output_format: {output_format}")

@app.post("/upload/finalize", response_model=JobSchema, status_code=status.HTTP_202_ACCEPTED)
async def finalize_upload(request: Request, payload: FinalizeUploadPayload, user: dict = Depends(require_user), db: Session = Depends(get_db)):
    temp_dir = chunk_dir_for(user, payload.upload_id)
    if not temp_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload session not found or already finalized.")

    # Validate everything before stitching, so a rejected request leaves no files behind.
    try:
        if not 1 <= payload.total_chunks <= MAX_UPLOAD_CHUNKS:
            raise HTTPException(status_code=400, detail="Invalid total_chunks.")

        webhook_config = APP_CONFIG.get("webhook_settings", {})
        if payload.callback_url and not is_allowed_callback_url(payload.callback_url, webhook_config.get("allowed_callback_urls", [])):
            raise HTTPException(status_code=400, detail="Provided callback_url is not allowed.")

        # Validate file type before processing
        allowed_extensions = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
        if not validate_file_type(payload.original_filename, allowed_extensions):
            raise HTTPException(status_code=400, detail=f"File type '{Path(payload.original_filename).suffix}' not allowed.")

        safe_filename = safe_basename(payload.original_filename)
        is_zip = Path(safe_filename).suffix == '.zip'
        options = {"model_size": payload.model_size, "model_name": payload.model_name,
                   "output_format": payload.output_format, "generate_timestamps": payload.generate_timestamps,
                   "ocr_language": payload.ocr_language, "callback_url": payload.callback_url}
        tool, task_key = _validate_task_options(payload.task_type, options, is_zip) or (None, None)
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    job_id = uuid.uuid4().hex
    final_path = PATHS.UPLOADS_DIR / f"{Path(safe_filename).stem}_{job_id}{Path(safe_filename).suffix}"
    await _stitch_chunks(temp_dir, final_path, payload.total_chunks)

    base_url = str(request.base_url)

    if tool == 'pandoc_academic':
        # This is a single job that processes a ZIP file as a project.
        dispatch_single_file_job(payload.original_filename, str(final_path), "conversion", user, db, APP_CONFIG, base_url, job_id=job_id, options=options)

    elif is_zip:
        # Batch processing: every file in the archive becomes a sub-job.
        job_data = JobCreate(
            id=job_id, user_id=user['sub'], task_type="unzip",
            original_filename=payload.original_filename, input_filepath=str(final_path),
            input_filesize=final_path.stat().st_size, callback_url=payload.callback_url
        )
        create_job(db=db, job=job_data)
        unzip_and_dispatch_task(job_id, str(final_path), payload.task_type, options, user, APP_CONFIG, base_url)
    else:
        # This is the logic for all other single-file uploads.
        dispatch_single_file_job(payload.original_filename, str(final_path), payload.task_type, user, db, APP_CONFIG, base_url, job_id=job_id, options=options)

    # Return the full job so the frontend can render its row immediately.
    db.expire_all()
    db_job = get_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=400, detail="This file type is not supported for the selected task.")
    return db_job


# --- LEGACY DIRECT-UPLOAD ROUTES (kept for compatibility) ---
@app.post("/transcribe-audio", status_code=status.HTTP_202_ACCEPTED)
async def submit_audio_transcription(
    request: Request, file: UploadFile = File(...), model_size: str = Form("base"),
    generate_timestamps: bool = Form(False),
    db: Session = Depends(get_db), user: dict = Depends(require_user)
):
    allowed_audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
    if not is_allowed_file(file.filename, allowed_audio_exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid audio file type.")

    whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
    if model_size not in whisper_config.get("allowed_models", []):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model size: {model_size}.")

    job_id, safe_name = uuid.uuid4().hex, safe_basename(file.filename)
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    upload_path = PATHS.UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    output_suffix = '.srt' if generate_timestamps else '.txt'
    processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{output_suffix}"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="transcription", original_filename=file.filename,
                         input_filepath=str(upload_path), input_filesize=input_size, processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_transcription_task(new_job.id, str(upload_path), str(processed_path), model_size, whisper_settings=whisper_config, app_config=APP_CONFIG, base_url=base_url, generate_timestamps=generate_timestamps)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/convert-file", status_code=status.HTTP_202_ACCEPTED)
async def submit_file_conversion(request: Request, file: UploadFile = File(...), output_format: str = Form(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    allowed_exts = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail=f"File type '{Path(file.filename).suffix}' not allowed.")
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    try:
        # Tool ids may contain '_' (ghostscript_pdf, ffmpeg_audio, ...), so match against the known ids.
        tool, task_key = _parse_tool_and_task_key(output_format, conversion_tools.keys())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output format selected.")
    _require_tool_available(tool)

    job_id, safe_name = uuid.uuid4().hex, safe_basename(file.filename)
    original_stem = Path(safe_name).stem
    target_ext = task_key.split('_')[0]
    if tool == "ghostscript_pdf": target_ext = "pdf"
    upload_path = PATHS.UPLOADS_DIR / f"{original_stem}_{job_id}{Path(safe_name).suffix}"
    processed_path = PATHS.PROCESSED_DIR / f"{original_stem}_{job_id}.{target_ext}"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="conversion", original_filename=file.filename,
                         input_filepath=str(upload_path), input_filesize=input_size, processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_conversion_task(new_job.id, str(upload_path), str(processed_path), tool, task_key, conversion_tools, APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-pdf", status_code=status.HTTP_202_ACCEPTED)
async def submit_pdf_ocr(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    if not is_allowed_file(file.filename, {".pdf"}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PDF.")
    job_id, safe_name = uuid.uuid4().hex, safe_basename(file.filename)
    unique_filename = f"{Path(safe_name).stem}_{job_id}{Path(safe_name).suffix}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / unique_filename
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="ocr", original_filename=file.filename,
                         input_filepath=str(upload_path), input_filesize=input_size, processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
    run_pdf_ocr_task(new_job.id, str(upload_path), str(processed_path), ocr_settings, APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-image", status_code=status.HTTP_202_ACCEPTED)
async def submit_image_ocr(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    allowed_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PNG, JPG, or TIFF.")
    job_id, safe_name = uuid.uuid4().hex, safe_basename(file.filename)
    file_ext = Path(safe_name).suffix
    unique_filename = f"{Path(safe_name).stem}_{job_id}{file_ext}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / f"{Path(safe_name).stem}_{job_id}.pdf"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="ocr-image", original_filename=file.filename,
                         input_filepath=str(upload_path), input_filesize=input_size, processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_image_ocr_task(new_job.id, str(upload_path), str(processed_path), APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

# --------------------------------------------------------------------------------
# --- API V1 ROUTES (for programmatic access)
# --------------------------------------------------------------------------------
def is_allowed_callback_url(url: str, allowed: List[str]) -> bool:
    if not allowed:
        return False
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        for a in allowed:
            ap = urlparse(a)
            if ap.scheme and ap.netloc:
                if parsed.scheme == ap.scheme and parsed.netloc == ap.netloc:
                    return True
            else:
                # support legacy prefix entries - keep fallback
                if url.startswith(a):
                    return True
        return False
    except Exception:
        return False

@app.get("/api/v1/ocr-languages")
async def get_ocr_languages(user: dict = Depends(require_user_or_api)):
    """Installed Tesseract languages and the configured default."""
    languages = await run_in_threadpool(installed_ocr_languages)
    default = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {}).get("language") or "eng"
    return {"languages": languages, "default": default}


@app.get("/api/v1/tts-voices")
async def get_tts_voices_list(user: dict = Depends(require_user)):
    global AVAILABLE_TTS_VOICES_CACHE

    if AVAILABLE_TTS_VOICES_CACHE is not None:
        return AVAILABLE_TTS_VOICES_CACHE

    kokoro_available = shutil.which("kokoro-tts") is not None
    piper_available = False
    try:
        import piper
        piper_available = True
    except ImportError:
        pass

    if not piper_available and not kokoro_available:
        AVAILABLE_TTS_VOICES_CACHE = []
        return JSONResponse(content={"error": "TTS feature not configured on server (no TTS engines found)."}, status_code=501)

    all_voices = []
    try:
        if piper_available:
            logger.info("Fetching available Piper voices list...")
            piper_voices = safe_get_voices(PATHS.TTS_MODELS_DIR)
            for voice in piper_voices:
                voice['id'] = f"piper/{voice.get('id')}"
                voice['name'] = f"Piper: {voice.get('name', voice.get('id'))}"
            all_voices.extend(piper_voices)

        if kokoro_available:
            logger.info("Fetching available Kokoro TTS voices and languages...")
            kokoro_voices = list_kokoro_voices_cli()
            kokoro_langs = list_kokoro_languages_cli()
            for lang in kokoro_langs:
                for voice in kokoro_voices:
                    all_voices.append({
                        "id": f"kokoro/{lang}/{voice}",
                        "name": f"Kokoro ({lang}): {voice}",
                        "local": False
                    })

        AVAILABLE_TTS_VOICES_CACHE = sorted(all_voices, key=lambda x: x['name'])
        return AVAILABLE_TTS_VOICES_CACHE
    except Exception as e:
        logger.exception("Could not fetch list of TTS voices.")
        AVAILABLE_TTS_VOICES_CACHE = [] # Cache the failure
        raise HTTPException(status_code=500, detail=f"Could not retrieve voices list: {e}")

# --- Standard API endpoint (non-chunked) ---
@app.post("/api/v1/process", status_code=status.HTTP_202_ACCEPTED, tags=["Webhook API"])
async def api_process_file(
    request: Request, file: UploadFile = File(...), task_type: str = Form(...), callback_url: str = Form(...),
    model_size: Optional[str] = Form("base"), model_name: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    generate_timestamps: bool = Form(False),
    ocr_language: Optional[str] = Form(None),
    db: Session = Depends(get_db), user: dict = Depends(require_api_user)
):
    """
    Programmatically submit a file for processing via a single HTTP request.
    This is the recommended endpoint for services like n8n.
    Requires bearer token authentication unless in LOCAL_ONLY_MODE.
    """
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Webhook processing is disabled on the server.")

    if not is_allowed_callback_url(callback_url, webhook_config.get("allowed_callback_urls", [])):
        logger.warning(f"Rejected webhook from user '{user.get('email')}' with disallowed callback URL: {callback_url}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provided callback_url is not in the list of allowed URLs.")

    # Validate the request before anything is written to disk.
    job_id = uuid.uuid4().hex
    safe_name = safe_basename(file.filename)
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    if task_type == "transcription":
        whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
        try:
            model_size = resolve_whisper_model(model_size, whisper_config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{'.srt' if generate_timestamps else '.txt'}"
    elif task_type == "tts":
        if not is_allowed_file(file.filename, {".txt"}):
            raise HTTPException(status_code=400, detail="Invalid file type for TTS, requires .txt")
        try:
            validate_tts_model_name(model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.wav"
    elif task_type == "conversion":
        if not output_format:
            raise HTTPException(status_code=400, detail="output_format is required for conversion task.")
        conversion_tools = APP_CONFIG.get("conversion_tools", {})
        try:
            tool, task_key = _parse_tool_and_task_key(output_format, conversion_tools.keys())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid output_format selected.")
        _require_tool_available(tool)
        target_ext = task_key.split('_')[0]
        if tool == "ghostscript_pdf": target_ext = "pdf"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.{target_ext}"
    elif task_type in ("ocr", "ocr-image"):
        if task_type == "ocr" and not is_allowed_file(file.filename, {".pdf"}):
            raise HTTPException(status_code=400, detail="Invalid file type for ocr, requires .pdf")
        if task_type == "ocr-image" and not is_allowed_file(file.filename, {".png", ".jpg", ".jpeg", ".tiff", ".tif"}):
            raise HTTPException(status_code=400, detail="Invalid file type for ocr-image.")
        ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
        try:
            ocr_language = resolve_ocr_language(ocr_language, ocr_settings)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Both produce a searchable PDF.
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.pdf"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid task_type: '{task_type}'")

    upload_path = PATHS.UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    try:
        input_size = await save_upload_file(file, upload_path)
    except HTTPException as e:
        raise e # Re-raise exceptions from save_upload_file (e.g., file too large)
    except Exception as e:
        logger.exception("Failed to save uploaded file for webhook processing.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save file: {e}")

    base_url = str(request.base_url)
    create_job(db=db, job=JobCreate(
        id=job_id, user_id=user['sub'], original_filename=file.filename,
        input_filepath=str(upload_path), input_filesize=input_size,
        callback_url=callback_url, task_type=task_type, processed_filepath=str(processed_path),
    ))

    # --- API Task Dispatching Logic ---
    if task_type == "transcription":
        run_transcription_task(job_id, str(upload_path), str(processed_path), model_size, whisper_config, APP_CONFIG, base_url, generate_timestamps=generate_timestamps)
    elif task_type == "tts":
        run_tts_task(job_id, str(upload_path), str(processed_path), model_name, APP_CONFIG.get("tts_settings", {}), APP_CONFIG, base_url)
    elif task_type == "conversion":
        run_conversion_task(job_id, str(upload_path), str(processed_path), tool, task_key, conversion_tools, APP_CONFIG, base_url)
    elif task_type == "ocr":
        run_pdf_ocr_task(job_id, str(upload_path), str(processed_path), dict(ocr_settings, language=ocr_language), APP_CONFIG, base_url)
    elif task_type == "ocr-image":
        run_image_ocr_task(job_id, str(upload_path), str(processed_path), APP_CONFIG, base_url, ocr_language)

    return {"job_id": job_id, "status": "pending"}


# --- Chunked API endpoints (optional) ---
@app.post("/api/v1/upload/chunk", tags=["Webhook API"])
async def api_upload_chunk(
    chunk: UploadFile = File(...), upload_id: str = Form(...), chunk_number: int = Form(...),
    user: dict = Depends(require_api_user)
):
    """API endpoint for uploading a single file chunk."""
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False) or not webhook_config.get("allow_chunked_api_uploads", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chunked API uploads are disabled.")

    return await upload_chunk(chunk, upload_id, chunk_number, user)

@app.post("/api/v1/upload/finalize", status_code=status.HTTP_202_ACCEPTED, tags=["Webhook API"])
async def api_finalize_upload(
    request: Request, payload: FinalizeUploadPayload, user: dict = Depends(require_api_user), db: Session = Depends(get_db)
):
    """API endpoint to finalize a chunked upload and start a processing job."""
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False) or not webhook_config.get("allow_chunked_api_uploads", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chunked API uploads are disabled.")

    # Validate callback URL if provided for a webhook job
    if payload.callback_url and not is_allowed_callback_url(payload.callback_url, webhook_config.get("allowed_callback_urls", [])):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provided callback_url is not allowed.")

    # Re-use the main finalization logic, but with API user context
    return await finalize_upload(request, payload, user, db)


# --------------------------------------------------------------------------------
# --- AUTH & PAGE ROUTES
# --------------------------------------------------------------------------------
if not LOCAL_ONLY_MODE:
    @app.get('/login')
    async def login(request: Request):
        if not check_oidc_availability():
            logger.warning("OIDC not available, redirecting to home with error")
            return RedirectResponse(url='/?error=auth_not_configured')
        redirect_uri = request.url_for('auth')
        return await oauth.oidc.authorize_redirect(request, redirect_uri)

    @app.get('/auth')
    async def auth(request: Request):
        if not check_oidc_availability():
            logger.warning("OIDC not available for authentication")
            raise HTTPException(status_code=401, detail="Authentication system is not properly configured")
        
        try:
            token = await oauth.oidc.authorize_access_token(request)
            user = dict(await oauth.oidc.userinfo(token=token))
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
        if not is_user_allowed(user):
            logger.warning(f"Login rejected for '{user.get('email')}': not in auth_settings.allowed_users/allowed_domains.")
            request.session.clear()
            raise HTTPException(status_code=403, detail="This account is not allowed to use this service.")
        request.session['user'] = user
        request.session['id_token'] = token.get('id_token')
        return RedirectResponse(url='/')

    @app.get("/logout")
    async def logout(request: Request):
        if not check_oidc_availability():
            request.session.clear()
            return RedirectResponse(url="/", status_code=302)
            
        try:
            # The metadata is fetched lazily; this worker may not have loaded it yet.
            await oauth.oidc.load_server_metadata()
            logout_endpoint = (oauth.oidc.server_metadata.get("end_session_endpoint")
                               or APP_CONFIG.get("auth_settings", {}).get("oidc_end_session_endpoint"))
            if not logout_endpoint:
                request.session.clear()
                logger.warning("OIDC 'end_session_endpoint' not found. Performing local-only logout.")
                return RedirectResponse(url="/", status_code=302)

            # RP-initiated logout: providers such as Keycloak require id_token_hint (or client_id)
            # alongside post_logout_redirect_uri.
            params = {"post_logout_redirect_uri": str(request.url_for("get_index"))}
            id_token = request.session.get("id_token")
            if id_token:
                params["id_token_hint"] = id_token
            client_id = APP_CONFIG.get("auth_settings", {}).get("oidc_client_id")
            if client_id:
                params["client_id"] = client_id
            separator = "&" if "?" in logout_endpoint else "?"
            logout_url = f"{logout_endpoint}{separator}{urlencode(params)}"
        except Exception as e:
            logger.warning(f"Could not determine OIDC logout endpoint: {e}. Performing local-only logout.")
            request.session.clear()
            return RedirectResponse(url="/", status_code=302)
        
        request.session.clear()
        return RedirectResponse(url=logout_url, status_code=302)


# This is for reverse proxies that use forward auth
@app.get("/api/authz/forward-auth")
async def forward_auth(request: Request):
    if LOCAL_ONLY_MODE:
        raise HTTPException(status_code=404, detail="Not available in LOCAL_ONLY mode.")
    if not check_oidc_availability():
        raise HTTPException(status_code=401, detail="Authentication system is not properly configured")
    redirect_uri = request.url_for('auth')
    return await oauth.oidc.authorize_redirect(request, redirect_uri)

@app.get("/")
async def get_index(request: Request):
    user = get_current_user(request)
    admin_status = is_admin(request)
    whisper_models = APP_CONFIG.get("transcription_settings", {}).get("whisper", {}).get("allowed_models", [])
    # The browser only needs names and output formats; command templates stay server-side.
    conversion_tools = {
        tool_id: {"name": tool.get("name", tool_id), "formats": tool.get("formats") or {}}
        for tool_id, tool in APP_CONFIG.get("conversion_tools", {}).items()
        if isinstance(tool, dict) and tool_id not in UNAVAILABLE_TOOLS
    }
    return templates.TemplateResponse(request, "index.html", {
        "user": user, "is_admin": admin_status,
        "whisper_models": sorted(list(whisper_models)),
        "conversion_tools": conversion_tools, "local_only_mode": LOCAL_ONLY_MODE
    })

# --- Settings page & API ---
# Secrets are never sent to the browser; submitting an empty value keeps the stored one.
SECRET_SETTINGS = (("auth_settings", "oidc_client_secret"), ("webhook_settings", "callback_bearer_token"))
# Editing a command template from the browser decides which programs the server executes, so it
# is off unless explicitly enabled. Templates can always be changed in config/settings.yml.
ALLOW_COMMAND_EDITS = _env_flag("ALLOW_COMMAND_EDITS", False)


def _leaves(*names: str) -> dict:
    return {name: None for name in names}


# Everything the settings form may change. Other keys (TTS command templates, model paths, ...)
# can only be edited in settings.yml.
EDITABLE_SETTINGS = {
    "app_settings": _leaves("app_public_url", "max_file_size_mb", "allowed_all_extensions",
                            "model_concurrency", "model_inactivity_timeout", "cache_check_interval",
                            "retention_days"),
    "ocr_settings": {"ocrmypdf": _leaves("deskew", "clean", "force_ocr", "language")},
    "transcription_settings": {"whisper": _leaves("compute_type")},
    "auth_settings": _leaves("oidc_client_id", "oidc_client_secret", "oidc_server_metadata_url",
                             "admin_users", "allowed_users", "allowed_domains"),
    "webhook_settings": _leaves("enabled", "allow_chunked_api_uploads", "allowed_callback_urls", "callback_bearer_token"),
    "tts_settings": {"piper": {"use_cuda": None, "synthesis_config": _leaves("length_scale", "noise_scale", "noise_w")}},
}
EDITABLE_TOOL_KEYS = {"name", "command_template", "supported_input", "formats", "timeout"}
POSITIVE_NUMBER_SETTINGS = (("app_settings", "max_file_size_mb"), ("app_settings", "model_concurrency"),
                            ("app_settings", "model_inactivity_timeout"), ("app_settings", "cache_check_interval"))


def _collapse_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def _validate_settings_payload(payload: dict) -> List[str]:
    """Returns a list of problems with a settings update coming from the web UI."""
    errors: List[str] = []

    def walk(data: dict, schema: dict, path: tuple):
        for key, value in data.items():
            key_path = ".".join(path + (key,))
            if key not in schema:
                errors.append(f"'{key_path}' cannot be changed from the web UI; edit settings.yml instead.")
                continue
            sub_schema = schema[key]
            if sub_schema is None:
                continue
            if not isinstance(value, dict):
                errors.append(f"'{key_path}' must be an object.")
                continue
            walk(value, sub_schema, path + (key,))

    walk({k: v for k, v in payload.items() if k != "conversion_tools"}, EDITABLE_SETTINGS, ())

    for section, key in POSITIVE_NUMBER_SETTINGS:
        section_data = payload.get(section)
        if isinstance(section_data, dict) and key in section_data:
            try:
                valid = float(section_data[key]) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"'{section}.{key}' must be a positive number.")

    app_section = payload.get("app_settings")
    if isinstance(app_section, dict) and "retention_days" in app_section:
        try:
            valid = app_section["retention_days"] is None or float(app_section["retention_days"]) >= 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append("'app_settings.retention_days' must be 0 (keep forever) or a positive number.")

    ocr_language = (payload.get("ocr_settings") or {}).get("ocrmypdf", {}) if isinstance(payload.get("ocr_settings"), dict) else {}
    if isinstance(ocr_language, dict) and "language" in ocr_language:
        try:
            resolve_ocr_language(str(ocr_language["language"] or ""), {})
        except ValueError as e:
            errors.append(f"'ocr_settings.ocrmypdf.language': {e}")

    tools = payload.get("conversion_tools")
    if tools is None:
        return errors
    if not isinstance(tools, dict):
        errors.append("'conversion_tools' must be an object.")
        return errors
    current_tools = APP_CONFIG.get("conversion_tools", {})
    for tool_id, tool in tools.items():
        if not isinstance(tool, dict):
            errors.append(f"'conversion_tools.{tool_id}' must be an object.")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", str(tool_id)):
            errors.append(f"Invalid tool id '{tool_id}' (letters, digits and underscores only).")
            continue
        for key in sorted(set(tool) - EDITABLE_TOOL_KEYS):
            errors.append(f"'conversion_tools.{tool_id}.{key}' cannot be changed from the web UI; edit settings.yml instead.")
        if "timeout" in tool:
            try:
                valid_timeout = 0 < float(tool["timeout"]) <= 7 * 24 * 3600
            except (TypeError, ValueError):
                valid_timeout = False
            if not valid_timeout:
                errors.append(f"'conversion_tools.{tool_id}.timeout' must be a positive number of seconds.")
        current = current_tools.get(tool_id)
        if current is None and not ALLOW_COMMAND_EDITS:
            errors.append(f"Adding conversion tools ('{tool_id}') from the web UI is disabled. "
                          "Edit config/settings.yml or set ALLOW_COMMAND_EDITS=true.")
            continue
        if "command_template" not in tool:
            if current is None:
                errors.append(f"New tool '{tool_id}' needs a command_template.")
            continue
        template = tool["command_template"]
        unchanged = current is not None and _collapse_whitespace(template) == _collapse_whitespace(current.get("command_template"))
        if not isinstance(template, str) or not template.strip():
            errors.append(f"'conversion_tools.{tool_id}.command_template' must be a non-empty string.")
        elif unchanged:
            continue  # the form always submits every template; only changes are checked
        elif not ALLOW_COMMAND_EDITS:
            errors.append(f"Changing the command of '{tool_id}' from the web UI is disabled. "
                          "Edit config/settings.yml or set ALLOW_COMMAND_EDITS=true.")
        else:
            try:
                validate_and_build_command(template, {})
            except ValueError as e:
                errors.append(f"Invalid command template for '{tool_id}': {e}")
    return errors


def _drop_blank_secrets(payload: dict) -> None:
    for section, key in SECRET_SETTINGS:
        section_data = payload.get(section)
        if isinstance(section_data, dict) and key in section_data:
            value = section_data[key]
            if value is None or (isinstance(value, str) and not value.strip()):
                del section_data[key]


@app.get("/settings")
async def get_settings_page(request: Request):
    """Settings page: configuration for admins, history management for every user."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    admin_status = is_admin(request)
    context = {"config": {}, "config_source": "", "secrets_set": {}, "user": user, "is_admin": admin_status,
               "local_only_mode": LOCAL_ONLY_MODE, "allow_command_edits": ALLOW_COMMAND_EDITS,
               "unavailable_tools": UNAVAILABLE_TOOLS, "ocr_languages": installed_ocr_languages()}
    if not admin_status:
        return templates.TemplateResponse(request, "settings.html", context)

    # Deep copy: the normalization below must not mutate the live configuration.
    current_config = copy.deepcopy(APP_CONFIG)
    secrets_set = {}
    for section, key in SECRET_SETTINGS:
        section_data = current_config.setdefault(section, {})
        secrets_set[key] = bool(section_data.get(key))
        section_data[key] = ""

    # Ensure all required nested dictionaries exist
    current_config.setdefault("app_settings", {})
    current_config.setdefault("transcription_settings", {}).setdefault("whisper", {})
    current_config.setdefault("tts_settings", {}).setdefault("piper", {}).setdefault("synthesis_config", {})
    current_config.setdefault("conversion_tools", {})
    current_config.setdefault("ocr_settings", {}).setdefault("ocrmypdf", {})

    # Fix potential format issues in conversion tools - ensure 'formats' is always a dict
    for tool_id, tool_config in current_config.get("conversion_tools", {}).items():
        if "formats" in tool_config:
            formats_data = tool_config["formats"]
            if isinstance(formats_data, list):
                # Convert list back to dict format - this handles cases where JS processing created a list
                formats_dict = {}
                for item in formats_data:
                    if isinstance(item, str) and ':' in item:
                        parts = item.split(':', 1)  # Split only on first colon to handle values with colons
                        key = parts[0].strip()
                        value = parts[1].strip() if len(parts) > 1 else ""
                        if key:  # Only add if key is not empty
                            formats_dict[key] = value
                tool_config["formats"] = formats_dict
            elif not isinstance(formats_data, dict):
                # If it's neither list nor dict, set to empty dict
                tool_config["formats"] = {}

    # Determine the source file for display purposes
    config_source = "none"
    if PATHS.SETTINGS_FILE.exists():
        config_source = str(PATHS.SETTINGS_FILE.name)
    elif PATHS.DEFAULT_SETTINGS_FILE.exists():
        config_source = str(PATHS.DEFAULT_SETTINGS_FILE.name)

    context.update({"config": current_config, "config_source": config_source, "secrets_set": secrets_set})
    return templates.TemplateResponse(request, "settings.html", context)

def _preprocess_settings_for_saving(config: Dict) -> Dict:
    """
    Pre-processes the settings dictionary before saving it to YAML.
    This function corrects data structures that may be misinterpreted
    when converted from JSON from the frontend.
    """
    if "conversion_tools" in config and isinstance(config["conversion_tools"], dict):
        for tool_name, tool_config in config["conversion_tools"].items():
            if isinstance(tool_config, dict):
                # --- Fix command_template: Convert list back to string ---
                if "command_template" in tool_config:
                    ct = tool_config["command_template"]
                    if isinstance(ct, list) and len(ct) == 1 and isinstance(ct[0], str):
                        tool_config["command_template"] = ct[0]
                    # Also handle the case where it might be a string with incorrect newlines
                    elif isinstance(ct, str):
                        tool_config["command_template"] = ct.replace('\n', '\n')

                # --- Fix formats: Convert list of strings back to a dictionary ---
                if "formats" in tool_config:
                    formats_data = tool_config["formats"]
                    if isinstance(formats_data, list):
                        new_formats = {}
                        for item in formats_data:
                            if isinstance(item, str):
                                # Split the string into lines and then into key-value pairs
                                for line in item.split('\n'):
                                    line = line.strip()
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        new_formats[key.strip()] = value.strip()
                        tool_config["formats"] = new_formats
                    elif isinstance(formats_data, str):
                        # Handle the case where the whole thing is a single string
                        new_formats = {}
                        for line in formats_data.split('\n'):
                            line = line.strip()
                            if ':' in line:
                                key, value = line.split(':', 1)
                                new_formats[key.strip()] = value.strip()
                        tool_config["formats"] = new_formats

    return config
@app.post("/settings/save")
async def save_settings(
    request: Request, new_config_from_ui: Dict = Body(...), admin: bool = Depends(require_admin)
):
    """Safely updates settings.yml by merging UI changes with the existing file."""
    tmp_path = PATHS.SETTINGS_FILE.with_suffix(".tmp")
    user = get_current_user(request)
    try:
        if not new_config_from_ui:
            if PATHS.SETTINGS_FILE.exists():
                PATHS.SETTINGS_FILE.unlink()
                logger.info(f"Admin '{user.get('email')}' reverted to default settings.")
            load_app_config()
            return JSONResponse({"message": "Settings reverted to default."})

        # Pre-process the incoming config to fix formatting issues
        processed_config = _preprocess_settings_for_saving(new_config_from_ui)
        _drop_blank_secrets(processed_config)
        errors = _validate_settings_payload(processed_config)
        if errors:
            logger.warning(f"Rejected settings update from admin '{user.get('email')}': {errors}")
            raise HTTPException(status_code=400, detail=" ".join(errors[:10]))

        try:
            with PATHS.SETTINGS_FILE.open("r", encoding="utf8") as f:
                current_config_on_disk = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current_config_on_disk = {}

        merged_config = deep_merge(processed_config, current_config_on_disk)

        with tmp_path.open("w", encoding="utf8") as f:
            yaml.safe_dump(merged_config, f, default_flow_style=False, sort_keys=False, width=float('inf'))

        tmp_path.replace(PATHS.SETTINGS_FILE)
        logger.info(f"Admin '{user.get('email')}' updated settings.yml.")
        load_app_config()
        return JSONResponse({"message": "Settings saved successfully."})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update settings for admin '{user.get('email')}'")
        if tmp_path.exists(): tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not save settings.yml: {e}")

# WebSocket endpoint for real-time job updates
@app.websocket("/ws/jobs")
async def websocket_job_updates(websocket: WebSocket, 
                               token: str = Query(None),
                               request: Request = None):
    """
    WebSocket endpoint for real-time job status updates.
    Requires authentication via token or session.
    """
    if not ENABLE_WEBSOCKETS:
        await websocket.close(code=1008, reason="WebSockets are disabled")
        return

    # Get user from either token or session
    user = None
    if token:
        # Validate bearer token for API users  
        try:
            from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
            http_bearer = HTTPBearer()
            creds = HTTPAuthorizationCredentials(credentials=token)
            user = await require_api_user(request, creds) if not LOCAL_ONLY_MODE else LOCAL_USER
        except Exception:
            await websocket.close(code=1008, reason="Invalid token")
            return
    elif LOCAL_ONLY_MODE:
        # In local-only mode, always allow with local user
        user = LOCAL_USER
    else:
        # Get from session for UI users
        user = get_current_user(request) if request else None
    
    if not user:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    user_id = user['sub']
    
    # Establish connection
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        logger.info(f"WebSocket connection established for user {user_id}")
        
        # Send initial job states
        db = SessionLocal()
        try:
            # Get recent jobs for user
            recent_jobs = get_jobs(db, user_id=user_id, skip=0, limit=50)  # Configurable limit
            if recent_jobs:
                jobs_data = [JobSchema.model_validate(job).model_dump() for job in recent_jobs]
                await manager.broadcast_multiple_jobs_update(user_id, jobs_data)
                logger.info(f"Sent initial job states to user {user_id}: {len(jobs_data)} jobs")
            else:
                logger.info(f"No initial jobs to send to user {user_id}")
        finally:
            db.close()
        
        # Listen for messages and maintain connection
        # Add server-side heartbeat to prevent timeout
        async def send_keepalive():
            while True:
                await asyncio.sleep(45)  # Send keepalive every 45 seconds
                try:
                    await websocket.send_text(json.dumps({
                        "type": "keepalive", 
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }))
                except Exception:
                    # Connection likely closed, break the loop
                    break
        
        # Start keepalive task
        keepalive_task = asyncio.create_task(send_keepalive())
        
        try:
            while True:
                # WebSocket operations are handled by the manager and job updates
                data = await websocket.receive_text()
                # In the basic implementation, client just sends keep-alive or commands
                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "")
                    if msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}))
                except json.JSONDecodeError:
                    # Ignore malformed messages
                    pass
        finally:
            # Cancel the keepalive task when connection closes
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect as e:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for user {user_id}. Code: {e.code}, Reason: {e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}", exc_info=True)
        manager.disconnect(websocket)

# --------------------------------------------------------------------------------
# --- JOB MANAGEMENT & UTILITY ROUTES
# --------------------------------------------------------------------------------
def _delete_job_files(jobs) -> tuple[int, list]:
    """Deletes the processed files of the given jobs. Returns (deleted_count, failed_names)."""
    deleted_count, errors = 0, []
    for job in jobs:
        if job.processed_filepath:
            try:
                p = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
            except Exception:
                errors.append(Path(job.processed_filepath).name)
                logger.exception(f"Could not delete file {Path(job.processed_filepath).name}")
    return deleted_count, errors


FINAL_JOB_STATES = ("completed", "failed", "cancelled")
MAX_JOBS_PER_DELETE = 1000


def delete_jobs_and_files(db: Session, jobs: list) -> tuple[list, int]:
    """
    Deletes jobs with their processed files (and the inputs of jobs that never started).
    Active jobs are cancelled first so running tools get killed. Returns (deleted_ids, files_deleted).
    """
    active_ids = [job.id for job in jobs if job.status in ("pending", "processing")]
    if active_ids:
        db.query(Job).filter(Job.id.in_(active_ids)).update({"status": "cancelled"}, synchronize_session=False)
        db.commit()
    files_deleted, _errors = _delete_job_files(jobs)
    for job in jobs:
        if job.id in active_ids and job.input_filepath:
            try:
                input_path = ensure_path_is_safe(Path(job.input_filepath), [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
                input_path.unlink(missing_ok=True)
            except Exception:
                logger.debug(f"Could not remove input of deleted job {job.id}", exc_info=True)
    deleted_ids = [job.id for job in jobs]
    for start in range(0, len(deleted_ids), 500):
        db.query(Job).filter(Job.id.in_(deleted_ids[start:start + 500])).delete(synchronize_session=False)
    db.commit()
    return deleted_ids, files_deleted


@app.post("/jobs/delete")
async def delete_selected_jobs(payload: JobSelection, db: Session = Depends(get_db), user: dict = Depends(require_user_or_api)):
    """Deletes the selected jobs of the current user, their files and, for ZIP batches, their sub-jobs."""
    job_ids = list(dict.fromkeys(payload.job_ids))[:MAX_JOBS_PER_DELETE]
    if not job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided.")
    jobs = db.query(Job).filter(Job.id.in_(job_ids), Job.user_id == user['sub']).all()
    batch_ids = [job.id for job in jobs if job.task_type == "unzip"]
    if batch_ids:
        known = {job.id for job in jobs}
        jobs += [child for child in db.query(Job).filter(Job.parent_job_id.in_(batch_ids), Job.user_id == user['sub']).all()
                 if child.id not in known]
    deleted_ids, files_deleted = delete_jobs_and_files(db, jobs)
    logger.info(f"User {user['sub']} deleted {len(deleted_ids)} jobs and {files_deleted} files.")
    return {"deleted": deleted_ids, "files_deleted": files_deleted}


@app.post("/settings/clear-history")
async def clear_job_history(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    """Deletes the user's job records together with their processed files (they'd be unreachable otherwise)."""
    try:
        files_deleted, _errors = _delete_job_files(get_jobs(db, user_id=user['sub'], limit=None))
        num_deleted = db.query(Job).filter(Job.user_id == user['sub']).delete()
        db.commit()
        logger.info(f"Cleared {num_deleted} jobs and {files_deleted} files for user {user['sub']}.")
        return {"deleted_count": num_deleted, "files_deleted": files_deleted}
    except Exception:
        db.rollback()
        logger.exception("Failed to clear job history")
        raise HTTPException(status_code=500, detail="Database error while clearing history.")

@app.post("/settings/delete-files")
async def delete_processed_files(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    # All of the user's jobs, not just the 100 most recent ones.
    deleted_count, errors = _delete_job_files(get_jobs(db, user_id=user['sub'], limit=None))
    if errors:
        raise HTTPException(status_code=500, detail=f"Could not delete some files: {', '.join(errors)}")
    logger.info(f"Deleted {deleted_count} files for user {user['sub']}.")
    return {"deleted_count": deleted_count}

@app.post("/job/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job = get_job(db, job_id)
    if not job or job.user_id != user['sub']:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ["pending", "processing"]:
        update_job_status(db, job_id, status="cancelled")
        return {"message": "Job cancellation requested."}
    raise HTTPException(status_code=400, detail=f"Job is already in a final state ({job.status}).")

@app.get("/jobs", response_model=List[JobSchema])
async def get_all_jobs(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    return get_jobs(db, user_id=user['sub'])

@app.get("/job/{job_id}", response_model=JobSchema)
async def get_job_status(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user_or_api)):
    job = get_job(db, job_id)
    if not job or job.user_id != user['sub']:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job



class JobStatusRequest(BaseModel):
    job_ids: TypingList[str]

@app.post("/api/v1/jobs/status", response_model=TypingList[JobSchema])
async def get_jobs_status(payload: JobStatusRequest, db: Session = Depends(get_db), user: dict = Depends(require_user_or_api)):
    """
    Accepts a list of job IDs and returns their current status.
    This is used by the frontend for polling active jobs.
    """
    if not payload.job_ids:
        return []
    
    # Fetch all requested jobs from the database in a single query
    jobs = db.query(Job).filter(Job.id.in_(payload.job_ids), Job.user_id == user['sub']).all()
    return jobs



@app.get("/download/{filename}")
async def download_file(filename: str, db: Session = Depends(get_db), user: dict = Depends(require_user_or_api)):
    file_path = ensure_path_is_safe(PATHS.PROCESSED_DIR / filename, [PATHS.PROCESSED_DIR])
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    # API users can download files they own via webhook URL. UI users need session.
    job_owner_id = user.get('sub') if user else None
    job = db.query(Job).filter(Job.processed_filepath == str(file_path), Job.user_id == job_owner_id).first()
    if not job:
        raise HTTPException(status_code=403, detail="You do not have permission to download this file.")
    download_filename = Path(job.original_filename).stem + Path(job.processed_filepath).suffix
    return FileResponse(path=file_path, filename=download_filename, media_type="application/octet-stream")

def _content_disposition(filename: str) -> str:
    """Attachment header with an ASCII fallback and an RFC 5987 UTF-8 name."""
    fallback = safe_basename(filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _zip_files_response(entries: List[tuple], download_name: str) -> StreamingResponse:
    """
    Builds a ZIP of (path, archive_name) entries in a spooled temp file (memory up to 32 MB,
    disk beyond) and streams it. Call via run_in_threadpool: zipping blocks.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
    used_names = set()
    with zipfile.ZipFile(spool, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path, arcname in entries:
            stem, suffix = Path(arcname).stem, Path(arcname).suffix
            unique_name, counter = arcname, 1
            while unique_name in used_names:
                counter += 1
                unique_name = f"{stem} ({counter}){suffix}"
            used_names.add(unique_name)
            zip_file.write(file_path, arcname=unique_name)
    spool.seek(0)

    def iter_file():
        try:
            while chunk := spool.read(1024 * 1024):
                yield chunk
        finally:
            spool.close()

    return StreamingResponse(iter_file(), media_type="application/zip",
                             headers={"Content-Disposition": _content_disposition(download_name)})


@app.post("/download/batch", response_class=StreamingResponse)
async def download_batch(payload: JobSelection, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job_ids = payload.job_ids
    if not job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided.")

    entries = []
    for job in db.query(Job).filter(Job.id.in_(job_ids), Job.user_id == user['sub'], Job.status == 'completed').all():
        if job.processed_filepath:
            file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
            if file_path.exists():
                entries.append((file_path, f"{Path(safe_basename(job.original_filename)).stem}_{job.id}{file_path.suffix}"))

    return await run_in_threadpool(_zip_files_response, entries, f"file-wizard-batch-{uuid.uuid4().hex[:8]}.zip")

@app.get("/download/zip-batch/{job_id}", response_class=StreamingResponse)
async def download_zip_batch(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    """Downloads all processed files from a ZIP upload batch as a new ZIP file."""
    parent_job = get_job(db, job_id)
    if not parent_job or parent_job.user_id != user['sub']:
        raise HTTPException(status_code=404, detail="Parent job not found.")
    if parent_job.task_type != 'unzip':
        raise HTTPException(status_code=400, detail="This job is not a batch upload.")

    child_jobs = db.query(Job).filter(Job.parent_job_id == job_id, Job.status == 'completed').all()
    if not child_jobs:
        raise HTTPException(status_code=404, detail="No completed sub-jobs found for this batch.")

    entries = []
    for job in child_jobs:
        if job.processed_filepath:
            file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
            if file_path.exists():
                # Create a more user-friendly name inside the zip
                entries.append((file_path, f"{Path(safe_basename(job.original_filename)).stem}{file_path.suffix}"))

    if not entries:
         raise HTTPException(status_code=404, detail="No processed files found for the completed sub-jobs.")

    batch_filename = f"{Path(parent_job.original_filename).stem}_processed.zip"
    return await run_in_threadpool(_zip_files_response, entries, batch_filename)

@app.get("/api/v1/supported-formats/{file_extension}")
async def get_supported_formats_for_file_type(file_extension: str, user: dict = Depends(require_user)):
    """
    Get supported output formats for a given file extension.
    The file_extension should include the dot (e.g., '.pdf', '.docx').
    """
    # Validate file extension format
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    
    file_extension = file_extension.lower()
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    
    # Find tools that support this input extension
    supported_formats = []
    for tool_name, tool_config in conversion_tools.items():
        if tool_name in UNAVAILABLE_TOOLS:
            continue
        supported_inputs = tool_config.get("supported_input", [])
        # Convert supported inputs to lowercase for comparison
        supported_inputs_lower = [ext.lower() for ext in supported_inputs]
        
        if file_extension in supported_inputs_lower:
            # Add all available formats for this tool
            for format_key, format_label in tool_config.get("formats", {}).items():
                full_format_key = f"{tool_name}_{format_key}"
                supported_formats.append({
                    "value": full_format_key,
                    "label": f"{tool_config['name']} - {format_label}",
                    "tool": tool_name,
                    "format": format_key
                })
    
    return {"formats": supported_formats}


@app.get("/api/formats/count")
async def get_formats_count():
    """
    Returns the number of supported input and output formats of the installed tools.
    """
    input_formats = set()
    output_formats = set()
    for tool, config in APP_CONFIG.get('conversion_tools', {}).items():
        if not isinstance(config, dict) or tool in UNAVAILABLE_TOOLS:
            continue
        input_formats.update(config.get('supported_input') or [])
        output_formats.update((config.get('formats') or {}).keys())
    return {
        "input_formats_count": len(input_formats),
        "output_formats_count": len(output_formats)
    }

@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse({"ok": False}, status_code=500)
    return {"ok": True, "version": os.environ.get("FILEWIZARD_VERSION", "dev"),
            "variant": os.environ.get("FILEWIZARD_VARIANT", "")}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(str(PATHS.BASE_DIR / 'static' / 'favicon.png'))

