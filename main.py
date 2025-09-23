# main.py (merged)
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
import cv2
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import resource
from threading import Semaphore
from logging.handlers import RotatingFileHandler
from urllib.parse import urljoin, urlparse
from io import BytesIO
import zipfile
import sys
import re
import importlib
import collections.abc
import time
import ocrmypdf
import pypdf
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
from pytesseract import TesseractNotFoundError
from PIL import Image, UnidentifiedImageError
from faster_whisper import WhisperModel
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status, Body)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from huey import SqliteHuey, crontab
from pydantic import BaseModel, ConfigDict, field_serializer
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine, delete, event)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
from string import Formatter
from werkzeug.utils import secure_filename
from typing import List as TypingList
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from piper import PiperVoice
import wave
import io


load_dotenv()

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
    """Ensure a path resolves to a location within one of the allowed base directories."""
    try:
        resolved_p = p.resolve()
        if not any(resolved_p.is_relative_to(base) for base in allowed_bases):
            raise ValueError(f"Path {resolved_p} is outside of allowed directories.")
        return resolved_p
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Path safety check failed for {p}: {e}")
        raise ValueError("Invalid or unsafe path specified.")

# --- Resource Limiting ---
def _limit_resources_preexec():
    """Set resource limits for child processes to prevent DoS attacks."""
    try:
        # 6000s CPU, 4GB address space
        resource.setrlimit(resource.RLIMIT_CPU, (6000, 6000))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
    except Exception as e:
        # This may fail in some environments (e.g. Windows, some containers)
        logging.getLogger(__name__).warning(f"Could not set resource limits: {e}")
        pass

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
_log_handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=1)
_log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
_log_handler.setFormatter(_log_formatter)
logging.getLogger().addHandler(_log_handler)
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
    TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts"
    KOKORO_TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts" / "kokoro"
    KOKORO_MODEL_FILE: Path = KOKORO_TTS_MODELS_DIR / "kokoro-v1.0.onnx"
    KOKORO_VOICES_FILE: Path = KOKORO_TTS_MODELS_DIR / "voices-v1.0.bin"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'jobs.db'}"
    HUEY_DB_PATH: str = str(BASE_DIR / "huey.db")
    CONFIG_DIR: Path = BASE_DIR / "config"
    SETTINGS_FILE: Path = CONFIG_DIR / "settings.yml"
    DEFAULT_SETTINGS_FILE: Path = BASE_DIR / "settings.default.yml"

PATHS = AppPaths()
APP_CONFIG: Dict[str, Any] = {}
PATHS.UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CHUNK_TMP_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CONFIG_DIR.mkdir(exist_ok=True, parents=True)
PATHS.TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.KOKORO_TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_app_config():
    """
    Loads configuration from settings.yml, with a fallback to settings.default.yml,
    and finally to hardcoded defaults if both files are missing.
    Reads environment variables for transcription device settings.
    """
    global APP_CONFIG
    try:
        # --- Primary Method: Attempt to load settings.yml ---
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            cfg_raw = yaml.safe_load(f) or {}

        # Read transcription settings from environment variables, providing smart defaults.
        transcription_device = os.environ.get("TRANSCRIPTION_DEVICE", "cpu")
        # Default to float16 for CUDA for better performance, otherwise int8 for CPU.
        default_compute_type = "float16" if transcription_device == "cuda" else "int8"
        transcription_compute_type = os.environ.get("TRANSCRIPTION_COMPUTE_TYPE", default_compute_type)
        transcription_device_index_str = os.environ.get("TRANSCRIPTION_DEVICE_INDEX", "0")

        # Handle multiple device indexes (e.g., "0,1")
        try:
            if ',' in transcription_device_index_str:
                transcription_device_index = [int(i.strip()) for i in transcription_device_index_str.split(',')]
            else:
                transcription_device_index = int(transcription_device_index_str)
        except ValueError:
            logger.warning(f"Invalid TRANSCRIPTION_DEVICE_INDEX value: '{transcription_device_index_str}'. Defaulting to 0.")
            transcription_device_index = 0

        defaults = {
            "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": [], "app_public_url": ""},
            "transcription_settings": {
                "whisper": {
                    "allowed_models": ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                    "compute_type": transcription_compute_type,
                    "device": transcription_device,
                    "device_index": transcription_device_index
                }
            },
            "tts_settings": {
                "piper": {"model_dir": str(PATHS.TTS_MODELS_DIR), "use_cuda": False, "synthesis_config": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}},
                "kokoro": {"model_dir": str(PATHS.KOKORO_TTS_MODELS_DIR), "command_template": "kokoro-tts {input} {output} --model {model_path} --voices {voices_path} --lang {lang} --voice {model_name}"}
            },
            "conversion_tools": {},
            "ocr_settings": {"ocrmypdf": {}},
            "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []},
            "webhook_settings": {"enabled": False, "allow_chunked_api_uploads": False, "allowed_callback_urls": [], "callback_bearer_token": ""}
        }
        cfg = defaults.copy()
        cfg.update(cfg_raw) # Merge loaded settings into defaults
        app_settings = cfg.get("app_settings", {})
        max_mb = app_settings.get("max_file_size_mb", 100)
        app_settings["max_file_size_bytes"] = int(max_mb) * 1024 * 1024
        allowed = app_settings.get("allowed_all_extensions", [])
        if not isinstance(allowed, (list, set)):
            allowed = list(allowed)
        app_settings["allowed_all_extensions"] = set(allowed)
        cfg["app_settings"] = app_settings
        APP_CONFIG = cfg
        logger.info("Successfully loaded settings from settings.yml")

    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Could not load settings.yml: {e}. Falling back to settings.default.yml...")
        try:
            # --- Fallback Method: Attempt to load settings.default.yml ---
            with open(PATHS.DEFAULT_SETTINGS_FILE, 'r', encoding='utf8') as f:
                cfg_raw = yaml.safe_load(f) or {}

            defaults = {
                "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": [], "app_public_url": ""},
                "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
                "tts_settings": {
                    "piper": {"model_dir": str(PATHS.TTS_MODELS_DIR), "use_cuda": False, "synthesis_config": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}},
                    "kokoro": {"model_dir": str(PATHS.KOKORO_TTS_MODELS_DIR), "command_template": "kokoro-tts {input} {output} --model {model_path} --voices {voices_path} --lang {lang} --voice {model_name}"}
                },
                "conversion_tools": {},
                "ocr_settings": {"ocrmypdf": {}},
                "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []},
                "webhook_settings": {"enabled": False, "allow_chunked_api_uploads": False, "allowed_callback_urls": [], "callback_bearer_token": ""}
            }
            cfg = defaults.copy()
            cfg.update(cfg_raw) # Merge loaded settings into defaults
            app_settings = cfg.get("app_settings", {})
            max_mb = app_settings.get("max_file_size_mb", 100)
            app_settings["max_file_size_bytes"] = int(max_mb) * 1024 * 1024
            allowed = app_settings.get("allowed_all_extensions", [])
            if not isinstance(allowed, (list, set)):
                allowed = list(allowed)
            app_settings["allowed_all_extensions"] = set(allowed)
            cfg["app_settings"] = app_settings
            APP_CONFIG = cfg
            logger.info("Successfully loaded settings from settings.default.yml")

        except (FileNotFoundError, yaml.YAMLError) as e_fallback:
            # --- Final Failsafe: Use hardcoded defaults ---
            logger.error(f"CRITICAL: Fallback file settings.default.yml also failed: {e_fallback}. Using hardcoded defaults.")
            APP_CONFIG = {
                "app_settings": {"max_file_size_mb": 100, "max_file_size_bytes": 100 * 1024 * 1024, "allowed_all_extensions": set(), "app_public_url": ""},
                "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
                "tts_settings": {
                    "piper": {"model_dir": str(PATHS.TTS_MODELS_DIR), "use_cuda": False, "synthesis_config": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}},
                    "kokoro": {"model_dir": str(PATHS.KOKORO_TTS_MODELS_DIR), "command_template": "kokoro-tts {input} {output} --model {model_path} --voices {voices_path} --lang {lang} --voice {model_name}"}
                },
                "conversion_tools": {},
                "ocr_settings": {"ocrmypdf": {}},
                "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []},
                "webhook_settings": {"enabled": False, "allow_chunked_api_uploads": False, "allowed_callback_urls": [], "callback_bearer_token": ""}
            }


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
    callback_url: Optional[str] = None # For API chunked uploads

class JobSelection(BaseModel):
    job_ids: List[str]

# --------------------------------------------------------------------------------
# --- 3. CRUD OPERATIONS & WEBHOOKS
# --------------------------------------------------------------------------------





def get_job(db: Session, job_id: str):
    # return db.query(Job).filter(Job.id == job_id).first()
    return  db.query(Job).filter(Job.id == job_id).first()

def get_jobs(db: Session, user_id: str | None = None, skip: int = 0, limit: int = 100):
    query = db.query(Job)
    if user_id:
        query = query.filter(Job.user_id == user_id)
    return query.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()

def create_job(db: Session, job: JobCreate):
    db_job = Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    # Broadcast the new job to UI clients via Huey task
    job_schema = JobSchema.model_validate(db_job)
    return db_job

def update_job_status(db: Session, job_id: str, status: str, progress: int = None, error: str = None):
    db_job = get_job(db, job_id)
    if db_job:
        db_job.status = status
        if progress is not None:
            db_job.progress = progress
        if error:
            db_job.error_message = error
        db.commit()
        db.refresh(db_job)
        # Broadcast the updated job to UI clients via Huey task
        job_schema = JobSchema.model_validate(db_job)
    return db_job

def mark_job_as_completed(db: Session, job_id: str, output_filepath_str: str | None = None, preview: str | None = None):
    db_job = get_job(db, job_id)
    if db_job and db_job.status != 'cancelled':
        db_job.status = "completed"
        db_job.progress = 100
        if preview:
            db_job.result_preview = preview.strip()[:2000]
        if output_filepath_str:
            try:
                output_path = Path(output_filepath_str)
                if output_path.exists():
                    db_job.output_filesize = output_path.stat().st_size
            except Exception:
                logger.exception(f"Could not stat output file {output_filepath_str} for job {job_id}")
        db.commit()
        db.refresh(db_job)
        # Broadcast the final job state to UI clients via Huey task
        job_schema = JobSchema.model_validate(db_job)

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
        logger.debug("Trying Kokoro TTS CLI list: %s", shlex.join(cmd))
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
        return []
    except subprocess.TimeoutExpired:
        logger.warning("Kokoro TTS language list command timed out.")
        return []
    except Exception as e:
        logger.exception("Unexpected error running Kokoro TTS language list: %s", e)
        return []


def run_command(
    argv: List[str],
    timeout: int = 300
) -> subprocess.CompletedProcess:
    """
    Executes a command, captures its output, and handles timeouts and errors.
    Uses resource limits for child processes. This is a simplified, more robust
    implementation using subprocess.run.
    """
    logger.debug("Executing command: %s with timeout=%ss", " ".join(shlex.quote(s) for s in argv), timeout)

    preexec = globals().get("_limit_resources_preexec", None)

    try:
        # subprocess.run handles timeout, output capturing, and error checking.
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,  # Raises CalledProcessError on non-zero exit
            preexec_fn=preexec
        )
        logger.debug("Command completed successfully: %s", " ".join(shlex.quote(s) for s in argv))
        return result
    except FileNotFoundError:
        msg = f"Command not found: {argv[0]}"
        logger.error(msg)
        raise Exception(msg) from None
    except subprocess.TimeoutExpired as e:
        msg = f"Command timed out after {timeout}s: {' '.join(shlex.quote(s) for s in argv)}"
        logger.error(msg)
        raise Exception(msg) from e
    except subprocess.CalledProcessError as e:
        snippet = (e.stderr or "")[:1000]
        msg = f"Command failed with exit code {e.returncode}. Stderr: {snippet}"
        logger.error(msg)
        raise Exception(msg) from e
    except Exception as e:
        msg = f"Unexpected error launching command: {e}"
        logger.exception(msg)
        raise Exception(msg) from e

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

# --- TASK RUNNERS ---
@huey.task()
def run_transcription_task(job_id: str, input_path_str: str, output_path_str: str, model_size: str, whisper_settings: dict, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # --- Constants ---
    # Time in seconds between database checks for progress updates and cancellations.
    # This avoids hammering the database on every single segment.
    DB_POLL_INTERVAL_SECONDS = 5

    try:
        job = get_job(db, job_id)
        if not job:
            logger.warning(f"Job {job_id} not found. Aborting task.")
            return
        if job.status == 'cancelled':
            logger.info(f"Job {job_id} was already cancelled before starting. Aborting.")
            return

        update_job_status(db, job_id, "processing", progress=0)

        model = get_whisper_model(model_size, whisper_settings)
        logger.info(f"Starting transcription for job {job_id} with model '{model_size}'")
        segments_generator, info = model.transcribe(str(input_path), beam_size=5)
        logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f} for a duration of {info.duration:.2f}s")


        last_update_time = time.time()

        # Use a temporary file to ensure atomic writes. The final file will only appear
        # once the transcription is fully and successfully written.
        tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")

        # Store a small preview in memory for the final update without holding the whole transcript.
        preview_segments = []
        PREVIEW_MAX_LENGTH = 1000  # characters
        current_preview_length = 0

        with tmp_output_path.open("w", encoding="utf-8") as f:
            for segment in segments_generator:
                segment_text = segment.text.strip()
                f.write(segment_text + "\n")

                # Build a small preview
                if current_preview_length < PREVIEW_MAX_LENGTH:
                    preview_segments.append(segment_text)
                    current_preview_length += len(segment_text)


                current_time = time.time()
                if current_time - last_update_time > DB_POLL_INTERVAL_SECONDS:
                    last_update_time = current_time

                    # Check for cancellation without overwhelming the DB
                    job_check = get_job(db, job_id)
                    if job_check and job_check.status == 'cancelled':
                        logger.info(f"Job {job_id} cancelled during transcription. Stopping.")
                        # The temporary file will be cleaned up in the finally block
                        return

                    # Update progress
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
        update_job_status(db, job_id, "processing")
        logger.info(f"Starting PDF OCR for job {job_id}")
        ocrmypdf.ocr(str(input_path), str(output_path_str),
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
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    out_path = Path(output_path_str)

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return

        update_job_status(db, job_id, "processing", progress=10)
        logger.info(f"Starting Image OCR for job {job_id} - {input_path}")

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
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(frame, extension="pdf")
            except TesseractNotFoundError as e:
                raise RuntimeError("Tesseract not found. Ensure Tesseract OCR is installed and in PATH.") from e
            except Exception as e:
                raise RuntimeError(f"Failed to run Tesseract on frame {idx}: {e}") from e

            pdf_bytes_list.append(pdf_bytes)

            # also extract plain text for preview and possible fallback
            try:
                page_text = pytesseract.image_to_string(frame)
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
    Drop-in replacement for conversion task.
    - Uses improved run_command for short operations (resource-limited).
    - Uses cancellable Popen runner for long-running conversion to respond to DB cancellations.
    """
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # localize helpers for speed
    _get_job = get_job
    _update_job_status = update_job_status
    _validate_build = validate_and_build_command
    _mark_completed = mark_job_as_completed
    _ensure_safe = ensure_path_is_safe
    _send_webhook = send_webhook_notification

    temp_input_file: Optional[Path] = None
    temp_output_file: Optional[Path] = None

    POLL_INTERVAL = 1.0
    STDERR_SNIPPET = 4000

    def _parse_task_key(tool_name: str, tk: str, tool_cfg: dict, mapping: dict):
        try:
            if tool_name.startswith("ghostscript"):
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
                rate_val = rate_token.replace("k", "000") if rate_token else ""
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
            elif tool_name == "libreoffice":
                target_ext = output_path.suffix.lstrip('.')
                filter_val = tool_cfg.get("filters", {}).get(target_ext, target_ext)
                mapping["filter"] = filter_val
        except Exception:
            logger.exception("Failed to parse task_key for tool %s; continuing with defaults.", tool_name)

    def _run_cancellable_command(command: List[str], timeout: int):
        """
        Run command with Popen and poll the DB for cancellation. Enforce timeout.
        Returns CompletedProcess-like on success. Raises Exception on failure/timeout/cancel.
        """
        preexec = globals().get("_limit_resources_preexec", None)
        logger.debug("Launching conversion subprocess: %s", " ".join(shlex.quote(c) for c in command))
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=preexec)

        start = time.monotonic()
        stderr_accum = []
        stderr_len = 0
        STDERR_LIMIT = STDERR_SNIPPET

        try:
            while True:
                ret = proc.poll()
                # Check job status
                job_check = _get_job(db, job_id)
                if job_check is None:
                    logger.warning("Job %s disappeared; killing conversion process.", job_id)
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise Exception("Job disappeared during conversion")
                if job_check.status == "cancelled":
                    logger.info("Job %s cancelled; terminating conversion process.", job_id)
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise Exception("Conversion cancelled")

                if ret is not None:
                    # process done - read remaining stderr/stdout safely
                    try:
                        out, err = proc.communicate(timeout=2)
                    except Exception:
                        out, err = "", ""
                        try:
                            if proc.stderr:
                                err = proc.stderr.read(STDERR_LIMIT)
                        except Exception:
                            pass
                    if err and len(err) > STDERR_LIMIT:
                        err = err[-STDERR_LIMIT:]
                    if ret != 0:
                        msg = (err or "")[:STDERR_LIMIT]
                        raise Exception(f"Conversion command failed (rc={ret}): {msg}")
                    return subprocess.CompletedProcess(args=command, returncode=ret, stdout=out, stderr=err)

                # timeout check
                elapsed = time.monotonic() - start
                if timeout and elapsed > timeout:
                    logger.warning("Conversion command timed out after %ss; terminating.", timeout)
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise Exception("Conversion command timed out")

                time.sleep(POLL_INTERVAL)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    try:
        job = _get_job(db, job_id)
        if not job:
            logger.warning("Job %s not found; aborting conversion.", job_id)
            return
        if job.status == "cancelled":
            logger.info("Job %s already cancelled; aborting conversion.", job_id)
            return

        _update_job_status(db, job_id, "processing", progress=25)
        logger.info("Starting conversion for job %s using %s with task %s", job_id, tool, task_key)

        tool_config = conversion_tools_config.get(tool)
        if not tool_config:
            raise ValueError(f"Unknown conversion tool: {tool}")

        current_input_path = input_path

        # Pre-conversion step for mozjpeg uses improved run_command (resource-limited)
        if tool == "mozjpeg":
            temp_input_file = input_path.with_suffix('.temp.ppm')
            logger.info("Pre-converting for MozJPEG: %s -> %s", input_path, temp_input_file)
            vips_bin = shutil.which("vips") or "vips"
            pre_conv_cmd = [vips_bin, "copy", str(input_path), str(temp_input_file)]
            try:
                run_command(pre_conv_cmd, timeout=int(tool_config.get("timeout", 300)))
            except Exception as ex:
                err_msg = str(ex)
                short_err = (err_msg or "")[:STDERR_SNIPPET]
                logger.exception("MozJPEG pre-conversion failed: %s", short_err)
                raise Exception(f"MozJPEG pre-conversion to PPM failed: {short_err}")
            current_input_path = temp_input_file

        _update_job_status(db, job_id, "processing", progress=50)

        # Prepare atomic temp output on same FS
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_file = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")

        mapping = {
            "input": str(current_input_path),
            "output": str(temp_output_file),
            "output_dir": str(output_path.parent),
            "output_ext": output_path.suffix.lstrip('.'),
        }

        _parse_task_key(tool, task_key, tool_config, mapping)

        command_template_str = tool_config.get("command_template")
        if not command_template_str:
            raise ValueError(f"Tool '{tool}' missing 'command_template' in configuration.")
        command = _validate_build(command_template_str, mapping)
        if not isinstance(command, (list, tuple)) or not command:
            raise ValueError("validate_and_build_command must return a non-empty list/tuple command.")
        command = [str(x) for x in command]

        logger.info("Executing command: %s", " ".join(shlex.quote(c) for c in command))

        # Run main conversion in cancellable manner
        timeout_val = int(tool_config.get("timeout", 300))

        # call the cancellable runner above
        result = _run_cancellable_command(command, timeout=timeout_val) if False else None
        # the above is replaced with a direct call to the actual function:
        result = _run_cancellable_command(command, timeout=timeout_val)

        # If successful and temp output exists, move it into place atomically
        if temp_output_file and temp_output_file.exists():
            temp_output_file.replace(output_path)

        _mark_completed(db, job_id, output_filepath_str=str(output_path), preview="Successfully converted file.")
        logger.info("Conversion for job %s completed.", job_id)

    except Exception as e:
        logger.exception("ERROR during conversion for job %s: %s", job_id, e)
        try:
            _update_job_status(db, job_id, "failed", error=f"Conversion failed: {e}")
        except Exception:
            logger.exception("Failed to update job status to failed after conversion error.")
    finally:
        # clean main input
        try:
            _ensure_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup main input file after conversion.")

        # cleanup temp input
        if temp_input_file:
            try:
                temp_input_file_path = Path(temp_input_file)
                _ensure_safe(temp_input_file_path, [PATHS.UPLOADS_DIR, PATHS.PROCESSED_DIR])
                temp_input_file_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to cleanup temp input file after conversion.")

        if temp_output_file:
            try:
                temp_output_file_path = Path(temp_output_file)
                _ensure_safe(temp_output_file_path, [PATHS.UPLOADS_DIR, PATHS.PROCESSED_DIR])
                temp_output_file_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to cleanup temp output file after conversion.")

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

        try:
            gc.collect()
        except Exception:
            pass

        try:
            _send_webhook(job_id, app_config, base_url)
        except Exception:
            logger.exception("Failed to send webhook notification after conversion.")

def dispatch_single_file_job(original_filename: str, input_filepath: str, task_type: str, user: dict, db: Session, app_config: Dict, base_url: str, job_id: str | None = None, options: Dict = None, parent_job_id: str | None = None):
    """Helper to create and dispatch a job for a single file."""
    if options is None:
        options = {}

    # If no job_id is passed, generate one. This is for sub-tasks from zips.
    if job_id is None:
        job_id = uuid.uuid4().hex

    safe_filename = secure_filename(original_filename)
    final_path = Path(input_filepath)

    # Ensure the input file exists before creating a job
    if not final_path.exists():
        logger.error(f"Input file does not exist, cannot dispatch job: {input_filepath}")
        return

    job_data = JobCreate(
        id=job_id, user_id=user['sub'], task_type=task_type,
        original_filename=original_filename, input_filepath=str(final_path),
        input_filesize=final_path.stat().st_size,
        parent_job_id=parent_job_id
    )

    if task_type == "transcription":
        stem = Path(safe_filename).stem
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_transcription_task(job_data.id, str(final_path), str(processed_path), options.get("model_size", "base"), app_config.get("transcription_settings", {}).get("whisper", {}), app_config, base_url)
    elif task_type == "tts":
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
            return
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.pdf"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        if suffix in IMAGE_EXTENSIONS:
            run_image_ocr_task(job_data.id, str(final_path), str(processed_path), app_config, base_url)
        else:
            run_pdf_ocr_task(job_data.id, str(final_path), str(processed_path), app_config.get("ocr_settings", {}).get("ocrmypdf", {}), app_config, base_url)
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
                return
            else:
                logger.error(f"Invalid or missing output_format for conversion of {original_filename}")
                final_path.unlink(missing_ok=True)
                return

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
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

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

        # 4. Execute command directly to control working directory and error capture
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,  # Raise CalledProcessError on non-zero exit
                cwd=unzip_dir, # Run pandoc in the unzipped directory
                preexec_fn=globals().get("_limit_resources_preexec", None)
            )
        except subprocess.CalledProcessError as e:
            # Capture the full, detailed error log from pandoc/latex
            error_log = e.stderr or "No stderr output."
            logger.error(f"Pandoc compilation failed. Full log:\n{error_log}")
            # Raise a more informative exception for the user
            raise Exception(f"Pandoc compilation failed. Please check your document for errors. Log: {error_log[:2000]}") from e

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
        if not parent_job or parent_job.status not in ['processing', 'pending']:
            return # Job is already finalized or doesn't exist

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
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        
        file_count = 0
        for extracted_file_path in unzip_dir.rglob('*'):
            if extracted_file_path.is_file():
                file_count += 1
                dispatch_single_file_job(
                    original_filename=extracted_file_path.name,
                    input_filepath=str(extracted_file_path),
                    task_type=sub_task_type,
                    options=sub_task_options,
                    user=user,
                    db=db,
                    app_config=app_config,
                    base_url=base_url,
                    parent_job_id=job_id
                )
        if file_count > 0:
            # Mark parent job as processing, to be completed by the periodic task
            update_job_status(db, job_id, "processing", progress=0)
        else:
            # No files found, mark as completed with a note
            mark_job_as_completed(db, job_id, preview="ZIP archive was empty. No sub-jobs created.")

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



# --------------------------------------------------------------------------------
# --- 5. FASTAPI APPLICATION
# --------------------------------------------------------------------------------

# --- SSE Broadcaster for real-time UI updates ---
import asyncio
import json





# --------------------------------------------------------------------------------
# --- 2. DATABASE & Schemas

async def download_kokoro_models_if_missing():
    """Checks for Kokoro TTS model files and downloads them if they don't exist."""
    files_to_download = {
        "model": {"path": PATHS.KOKORO_MODEL_FILE, "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"},
        "voices": {"path": PATHS.KOKORO_VOICES_FILE, "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"}
    }
    async with httpx.AsyncClient() as client:
        for name, details in files_to_download.items():
            path, url = details["path"], details["url"]
            if not path.exists():
                logger.info(f"Kokoro TTS {name} file missing. Downloading from {url}...")
                try:
                    with path.open("wb") as f:
                        async with client.stream("GET", url, follow_redirects=True, timeout=300) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                    logger.info(f"Successfully downloaded Kokoro TTS {name} file to {path}.")
                except Exception as e:
                    logger.error(f"Failed to download Kokoro TTS {name} file: {e}")
                    if path.exists(): path.unlink(missing_ok=True)
            else:
                logger.info(f"Found existing Kokoro TTS {name} file at {path}.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
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


    load_app_config()

    # Start the cache cleanup thread
    global _cache_cleanup_thread
    if _cache_cleanup_thread is None:
        _cache_cleanup_thread = threading.Thread(target=_whisper_cache_cleanup_worker, daemon=True)
        _cache_cleanup_thread.start()
        logger.info("Whisper model cache cleanup thread started.")

    # Download required models on startup
    if shutil.which("kokoro-tts"):
        await download_kokoro_models_if_missing()

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
            logger.error("OIDC auth settings are incomplete. Auth will be disabled if not in LOCAL_ONLY_MODE.")
        else:
            oauth.register(
                name='oidc',
                client_id=oidc_cfg.get('oidc_client_id'),
                client_secret=oidc_cfg.get('oidc_client_secret'),
                server_metadata_url=oidc_cfg.get('oidc_server_metadata_url'),
                client_kwargs={'scope': 'openid email profile'},
                userinfo_endpoint=oidc_cfg.get('oidc_userinfo_endpoint'),
                end_session_endpoint=oidc_cfg.get('oidc_end_session_endpoint')
            )
            logger.info('OAuth registered.')
    yield
    logger.info('Application shutting down...')

app = FastAPI(lifespan=lifespan)
ENV = os.environ.get('ENV', 'dev').lower()
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY and not LOCAL_ONLY_MODE and ENV != 'dev':
    raise RuntimeError('SECRET_KEY must be set in production when authentication is enabled.')
if not SECRET_KEY:
    logger.warning('SECRET_KEY is not set. Generating a temporary key. Sessions will not persist across restarts.')
    SECRET_KEY = os.urandom(24).hex()

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False, # Set to True if behind HTTPS proxy
    same_site='lax',
    max_age=14 * 24 * 60 * 60  # 14 days
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Static / templates
app.mount("/static", StaticFiles(directory=str(PATHS.BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PATHS.BASE_DIR / "templates"))

# --- AUTH & USER HELPERS ---
http_bearer = HTTPBearer()

def get_current_user(request: Request):
    if LOCAL_ONLY_MODE:
        return {'sub': 'local_user', 'email': 'local@user.com', 'name': 'Local User'}
    return request.session.get('user')

async def require_api_user(request: Request, creds: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Dependency for API routes requiring OIDC bearer token authentication."""
    if LOCAL_ONLY_MODE:
        return {'sub': 'local_api_user', 'email': 'local@api.user.com', 'name': 'Local API User'}

    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = creds.credentials
    try:
        user = await oauth.oidc.userinfo(token={'access_token': token})
        return dict(user)
    except Exception as e:
        logger.error(f"API token validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

def is_admin(request: Request) -> bool:
    if LOCAL_ONLY_MODE: return True
    user = get_current_user(request)
    if not user: return False
    admin_users = APP_CONFIG.get("auth_settings", {}).get("admin_users", [])
    return user.get('email') in admin_users

def require_user(request: Request):
    user = get_current_user(request)
    if not user: raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def require_admin(request: Request):
    if not is_admin(request): raise HTTPException(status_code=403, detail="Administrator privileges required.")
    return True

# --- FILE SAVING UTILITY ---
async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    """
    Saves an uploaded file to a destination, handling size limits.
    This function is used by both the simple API and the legacy direct-upload routes.
    """
    max_size = APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024)
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
    except Exception as e:
        try:
            # Ensure temp file is cleaned up on error
            ensure_path_is_safe(tmp_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove temp upload file after error.")
        # Re-raise the original exception
        raise e
    finally:
        await upload_file.close()

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    if not allowed_extensions: # If set is empty, allow all
        return True
    return Path(filename).suffix.lower() in allowed_extensions

# --- CHUNKED UPLOADS (for UI) ---
@app.post("/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    user: dict = Depends(require_user)
):
    safe_upload_id = secure_filename(upload_id)
    temp_dir = ensure_path_is_safe(PATHS.CHUNK_TMP_DIR / safe_upload_id, [PATHS.CHUNK_TMP_DIR])
    temp_dir.mkdir(exist_ok=True)
    chunk_path = temp_dir / f"{chunk_number}.chunk"

    def save_chunk_sync():
        try:
            with open(chunk_path, "wb") as buffer:
                shutil.copyfileobj(chunk.file, buffer)
        finally:
            chunk.file.close()

    await run_in_threadpool(save_chunk_sync)

    return JSONResponse({"message": f"Chunk {chunk_number} for {safe_upload_id} uploaded."})


async def _stitch_chunks(temp_dir: Path, final_path: Path, total_chunks: int):
    """Stitches chunks together memory-efficiently and cleans up."""
    ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    ensure_path_is_safe(final_path, [PATHS.UPLOADS_DIR])

    # This is a blocking function that will be run in a threadpool
    def do_stitch():
        with open(final_path, "wb") as final_file:
            for i in range(total_chunks):
                chunk_path = temp_dir / f"{i}.chunk"
                if not chunk_path.exists():
                    # Raise an exception that can be caught and handled
                    raise FileNotFoundError(f"Upload failed: missing chunk {i}")
                with open(chunk_path, "rb") as chunk_file:
                    # Use copyfileobj for memory efficiency
                    shutil.copyfileobj(chunk_file, final_file)

    try:
        await run_in_threadpool(do_stitch)
    except FileNotFoundError as e:
        # If a chunk was missing, clean up and re-raise as HTTPException
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # For any other error during stitching, clean up and re-raise
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e # Re-raise the original exception
    else:
        # If successful, clean up the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

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
    safe_upload_id = secure_filename(payload.upload_id)
    temp_dir = ensure_path_is_safe(PATHS.CHUNK_TMP_DIR / safe_upload_id, [PATHS.CHUNK_TMP_DIR])
    if not temp_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload session not found or already finalized.")

    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if payload.callback_url and not is_allowed_callback_url(payload.callback_url, webhook_config.get("allowed_callback_urls", [])):
        raise HTTPException(status_code=400, detail="Provided callback_url is not allowed.")

    job_id = uuid.uuid4().hex
    safe_filename = secure_filename(payload.original_filename)
    final_path = PATHS.UPLOADS_DIR / f"{Path(safe_filename).stem}_{job_id}{Path(safe_filename).suffix}"
    await _stitch_chunks(temp_dir, final_path, payload.total_chunks)

    base_url = str(request.base_url)

    # Check if the selected conversion is the new academic pandoc task
    tool, task_key = None, None
    if payload.task_type == 'conversion':
        try:
            all_tools = APP_CONFIG.get("conversion_tools", {}).keys()
            tool, task_key = _parse_tool_and_task_key(payload.output_format, all_tools)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid or missing output_format for conversion.")

    if tool == 'pandoc_academic':
        # This is a single job that processes a ZIP file as a project.
        options = {"output_format": payload.output_format}
        dispatch_single_file_job(payload.original_filename, str(final_path), "conversion", user, db, APP_CONFIG, base_url, job_id=job_id, options=options)

    elif Path(safe_filename).suffix.lower() == '.zip':
        # This is the original batch processing logic for ZIP files.
        job_data = JobCreate(
            id=job_id, user_id=user['sub'], task_type="unzip",
            original_filename=payload.original_filename, input_filepath=str(final_path),
            input_filesize=final_path.stat().st_size
        )
        create_job(db=db, job=job_data)
        sub_task_options = {
            "model_size": payload.model_size,
            "model_name": payload.model_name,
            "output_format": payload.output_format
        }
        unzip_and_dispatch_task(job_id, str(final_path), payload.task_type, sub_task_options, user, APP_CONFIG, base_url)
    else:
        # This is the logic for all other single-file uploads.
        options = {"model_size": payload.model_size, "model_name": payload.model_name, "output_format": payload.output_format}
        dispatch_single_file_job(payload.original_filename, str(final_path), payload.task_type, user, db, APP_CONFIG, base_url, job_id=job_id, options=options)

    # --- FIX STARTS HERE ---
    # Instead of returning a minimal object, fetch the newly created job
    # from the database and return the full serialized object. This ensures
    # the frontend has all the data it needs to correctly update the UI row.
    db.flush() # Ensure the job is available to be queried
    db_job = get_job(db, job_id)
    if not db_job:
        # This is an unlikely race condition but we handle it just in case.
        # The SSE event will still create the row correctly.
        raise HTTPException(status_code=500, detail="Job was created but could not be retrieved for an immediate response.")
    
    # Also, update the function signature to use the response_model
    # from: @app.post("/upload/finalize", status_code=status.HTTP_202_ACCEPTED)
    # to:   @app.post("/upload/finalize", response_model=JobSchema, status_code=status.HTTP_202_ACCEPTED)
    return db_job
    # --- FIX ENDS HERE ---


# --- LEGACY DIRECT-UPLOAD ROUTES (kept for compatibility) ---
@app.post("/transcribe-audio", status_code=status.HTTP_202_ACCEPTED)
async def submit_audio_transcription(
    request: Request, file: UploadFile = File(...), model_size: str = Form("base"),
    db: Session = Depends(get_db), user: dict = Depends(require_user)
):
    allowed_audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
    if not is_allowed_file(file.filename, allowed_audio_exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid audio file type.")

    whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
    if model_size not in whisper_config.get("allowed_models", []):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model size: {model_size}.")

    job_id, safe_basename = uuid.uuid4().hex, secure_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix
    upload_path = PATHS.UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="transcription", original_filename=file.filename,
                         input_filepath=str(upload_path), input_filesize=input_size, processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_transcription_task(new_job.id, str(upload_path), str(processed_path), model_size, whisper_settings=whisper_config, app_config=APP_CONFIG, base_url=base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/convert-file", status_code=status.HTTP_202_ACCEPTED)
async def submit_file_conversion(request: Request, file: UploadFile = File(...), output_format: str = Form(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    allowed_exts = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail=f"File type '{Path(file.filename).suffix}' not allowed.")
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    try:
        tool, task_key = output_format.split('_', 1)
        if tool not in conversion_tools: raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output format selected.")

    job_id, safe_basename = uuid.uuid4().hex, secure_filename(file.filename)
    original_stem = Path(safe_basename).stem
    target_ext = task_key.split('_')[0]
    if tool == "ghostscript_pdf": target_ext = "pdf"
    upload_path = PATHS.UPLOADS_DIR / f"{original_stem}_{job_id}{Path(safe_basename).suffix}"
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
    job_id, safe_basename = uuid.uuid4().hex, secure_filename(file.filename)
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{Path(safe_basename).suffix}"
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
    job_id, safe_basename = uuid.uuid4().hex, secure_filename(file.filename)
    file_ext = Path(safe_basename).suffix
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{file_ext}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / f"{Path(safe_basename).stem}_{job_id}.pdf"
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

@app.get("/api/v1/tts-voices")
async def get_tts_voices_list(user: dict = Depends(require_user)):
    global AVAILABLE_TTS_VOICES_CACHE

    kokoro_available = shutil.which("kokoro-tts") is not None
    piper_available = PiperVoice is not None

    if not piper_available and not kokoro_available:
        return JSONResponse(content={"error": "TTS feature not configured on server (no TTS engines found)."}, status_code=501)

    if AVAILABLE_TTS_VOICES_CACHE:
        return AVAILABLE_TTS_VOICES_CACHE

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
        raise HTTPException(status_code=500, detail=f"Could not retrieve voices list: {e}")

# --- Standard API endpoint (non-chunked) ---
@app.post("/api/v1/process", status_code=status.HTTP_202_ACCEPTED, tags=["Webhook API"])
async def api_process_file(
    request: Request, file: UploadFile = File(...), task_type: str = Form(...), callback_url: str = Form(...),
    model_size: Optional[str] = Form("base"), model_name: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
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

    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix
    upload_filename = f"{stem}_{job_id}{suffix}"
    upload_path = PATHS.UPLOADS_DIR / upload_filename

    try:
        input_size = await save_upload_file(file, upload_path)
    except HTTPException as e:
        raise e # Re-raise exceptions from save_upload_file (e.g., file too large)
    except Exception as e:
        logger.exception("Failed to save uploaded file for webhook processing.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save file: {e}")

    base_url = str(request.base_url)
    job_data_args = {
        "id": job_id, "user_id": user['sub'], "original_filename": file.filename,
        "input_filepath": str(upload_path), "input_filesize": input_size,
        "callback_url": callback_url, "task_type": task_type,
    }

    # --- API Task Dispatching Logic ---
    if task_type == "transcription":
        whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
        if model_size not in whisper_config.get("allowed_models", []):
            raise HTTPException(status_code=400, detail=f"Invalid model_size '{model_size}'")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_transcription_task(job_id, str(upload_path), str(processed_path), model_size, whisper_config, APP_CONFIG, base_url)

    elif task_type == "tts":
        if not is_allowed_file(file.filename, {".txt"}):
            raise HTTPException(status_code=400, detail="Invalid file type for TTS, requires .txt")
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required for TTS task.")
        tts_config = APP_CONFIG.get("tts_settings", {})
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.wav"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_tts_task(job_id, str(upload_path), str(processed_path), model_name, tts_config, APP_CONFIG, base_url)

    elif task_type == "conversion":
        if not output_format:
            raise HTTPException(status_code=400, detail="output_format is required for conversion task.")
        conversion_tools = APP_CONFIG.get("conversion_tools", {})
        try:
            tool, task_key = output_format.split('_', 1)
            if tool not in conversion_tools: raise ValueError("Invalid tool")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid output_format selected.")
        target_ext = task_key.split('_')[0]
        if tool == "ghostscript_pdf": target_ext = "pdf"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.{target_ext}"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_conversion_task(job_id, str(upload_path), str(processed_path), tool, task_key, conversion_tools, APP_CONFIG, base_url)

    elif task_type == "ocr":
        if not is_allowed_file(file.filename, {".pdf"}):
            raise HTTPException(status_code=400, detail="Invalid file type for ocr, requires .pdf")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{suffix}"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_pdf_ocr_task(job_id, str(upload_path), str(processed_path), APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {}), APP_CONFIG, base_url)

    elif task_type == "ocr-image":
        if not is_allowed_file(file.filename, {".png", ".jpg", ".jpeg", ".tiff", ".tif"}):
             raise HTTPException(status_code=400, detail="Invalid file type for ocr-image.")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_image_ocr_task(job_id, str(upload_path), str(processed_path), APP_CONFIG, base_url)

    else:
        upload_path.unlink(missing_ok=True) # Cleanup orphaned file
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid task_type: '{task_type}'")

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
        redirect_uri = request.url_for('auth')
        return await oauth.oidc.authorize_redirect(request, redirect_uri)

    @app.get('/auth')
    async def auth(request: Request):
        try:
            token = await oauth.oidc.authorize_access_token(request)
            user = await oauth.oidc.userinfo(token=token)
            request.session['user'] = dict(user)
            request.session['id_token'] = token.get('id_token')
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
        return RedirectResponse(url='/')

    @app.get("/logout")
    async def logout(request: Request):
        logout_endpoint = oauth.oidc.server_metadata.get("end_session_endpoint")
        if not logout_endpoint:
            request.session.clear()
            logger.warning("OIDC 'end_session_endpoint' not found. Performing local-only logout.")
            return RedirectResponse(url="/", status_code=302)

        post_logout_redirect_uri = str(request.url_for("get_index"))
        logout_url = f"{logout_endpoint}?post_logout_redirect_uri={post_logout_redirect_uri}"
        request.session.clear()
        return RedirectResponse(url=logout_url, status_code=302)


# This is for reverse proxies that use forward auth
@app.get("/api/authz/forward-auth")
async def forward_auth(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.oidc.authorize_redirect(request, redirect_uri)

@app.get("/")
async def get_index(request: Request):
    user = get_current_user(request)
    admin_status = is_admin(request)
    whisper_models = APP_CONFIG.get("transcription_settings", {}).get("whisper", {}).get("allowed_models", [])
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    return templates.TemplateResponse("index.html", {
        "request": request, "user": user, "is_admin": admin_status,
        "whisper_models": sorted(list(whisper_models)),
        "conversion_tools": conversion_tools, "local_only_mode": LOCAL_ONLY_MODE
    })

@app.get("/settings")
async def get_settings_page(request: Request):
    """Displays the contents of the currently active configuration."""
    user = get_current_user(request)
    admin_status = is_admin(request)

    # Use the globally loaded and merged APP_CONFIG for consistency
    # This ensures all default keys are present before rendering.
    current_config = APP_CONFIG

    # Determine the source file for display purposes
    config_source = "none"
    if PATHS.SETTINGS_FILE.exists():
        config_source = str(PATHS.SETTINGS_FILE.name)
    elif PATHS.DEFAULT_SETTINGS_FILE.exists():
        config_source = str(PATHS.DEFAULT_SETTINGS_FILE.name)

    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "config": current_config, "config_source": config_source,
         "user": user, "is_admin": admin_status, "local_only_mode": LOCAL_ONLY_MODE}
    )

def deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merges dicts."""
    for key, value in source.items():
        if isinstance(value, collections.abc.Mapping):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination

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

        try:
            with PATHS.SETTINGS_FILE.open("r", encoding="utf8") as f:
                current_config_on_disk = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current_config_on_disk = {}

        merged_config = deep_merge(source=new_config_from_ui, destination=current_config_on_disk)

        with tmp_path.open("w", encoding="utf8") as f:
            yaml.safe_dump(merged_config, f, default_flow_style=False, sort_keys=False)

        tmp_path.replace(PATHS.SETTINGS_FILE)
        logger.info(f"Admin '{user.get('email')}' updated settings.yml.")
        load_app_config()
        return JSONResponse({"message": "Settings saved successfully."})

    except Exception as e:
        logger.exception(f"Failed to update settings for admin '{user.get('email')}'")
        if tmp_path.exists(): tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not save settings.yml: {e}")

# --------------------------------------------------------------------------------
# --- JOB MANAGEMENT & UTILITY ROUTES
# --------------------------------------------------------------------------------
@app.post("/settings/clear-history")
async def clear_job_history(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    try:
        num_deleted = db.query(Job).filter(Job.user_id == user['sub']).delete()
        db.commit()
        logger.info(f"Cleared {num_deleted} jobs for user {user['sub']}.")
        return {"deleted_count": num_deleted}
    except Exception:
        db.rollback()
        logger.exception("Failed to clear job history")
        raise HTTPException(status_code=500, detail="Database error while clearing history.")

@app.post("/settings/delete-files")
async def delete_processed_files(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    deleted_count, errors = 0, []
    for job in get_jobs(db, user_id=user['sub']):
        if job.processed_filepath:
            try:
                p = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
            except Exception:
                errors.append(Path(job.processed_filepath).name)
                logger.exception(f"Could not delete file {Path(job.processed_filepath).name}")
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
async def get_job_status(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job = get_job(db, job_id)
    if not job or job.user_id != user['sub']:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job



class JobStatusRequest(BaseModel):
    job_ids: TypingList[str]

@app.post("/api/v1/jobs/status", response_model=TypingList[JobSchema])
async def get_jobs_status(payload: JobStatusRequest, db: Session = Depends(get_db), user: dict = Depends(require_user)):
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
async def download_file(filename: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    safe_filename = secure_filename(filename)
    file_path = ensure_path_is_safe(PATHS.PROCESSED_DIR / safe_filename, [PATHS.PROCESSED_DIR])
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    # API users can download files they own via webhook URL. UI users need session.
    job_owner_id = user.get('sub') if user else None
    job = db.query(Job).filter(Job.processed_filepath == str(file_path), Job.user_id == job_owner_id).first()
    if not job:
        raise HTTPException(status_code=403, detail="You do not have permission to download this file.")
    download_filename = Path(job.original_filename).stem + Path(job.processed_filepath).suffix
    return FileResponse(path=file_path, filename=download_filename, media_type="application/octet-stream")

@app.post("/download/batch", response_class=StreamingResponse)
async def download_batch(payload: JobSelection, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job_ids = payload.job_ids
    if not job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided.")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for job_id in job_ids:
            job = get_job(db, job_id)
            if job and job.user_id == user['sub'] and job.status == 'completed' and job.processed_filepath:
                file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if file_path.exists():
                    download_filename = f"{Path(job.original_filename).stem}_{job_id}{file_path.suffix}"
                    zip_file.write(file_path, arcname=download_filename)

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment; filename="file-wizard-batch-{uuid.uuid4().hex[:8]}.zip"'
    })

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

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        files_added = 0
        for job in child_jobs:
            if job.processed_filepath:
                file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if file_path.exists():
                    # Create a more user-friendly name inside the zip
                    download_filename = f"{Path(job.original_filename).stem}{file_path.suffix}"
                    zip_file.write(file_path, arcname=download_filename)
                    files_added += 1

    if files_added == 0:
         raise HTTPException(status_code=404, detail="No processed files found for the completed sub-jobs.")

    zip_buffer.seek(0)

    # Generate a filename for the download
    batch_filename = f"{Path(parent_job.original_filename).stem}_processed.zip"

    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment; filename="{batch_filename}"'
    })

@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").execute("SELECT 1")
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse({"ok": False}, status_code=500)
    return {"ok": True}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(str(PATHS.BASE_DIR / 'static' / 'favicon.png'))

