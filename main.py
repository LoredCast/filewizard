# main.py (merged)

import logging
import shutil
import subprocess
import traceback
import uuid
import shlex
import yaml
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import resource
from threading import Semaphore
from logging.handlers import RotatingFileHandler
from urllib.parse import urlencode

import ocrmypdf
import pypdf
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status, Body)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huey import SqliteHuey
from pydantic import BaseModel, ConfigDict, field_serializer
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine, delete, event)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from string import Formatter
from werkzeug.utils import secure_filename
from typing import List as TypingList
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv

load_dotenv()
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
        # 6000s CPU, 2GB address space, i dont know if thats too much tbh
        resource.setrlimit(resource.RLIMIT_CPU, (6000, 6000))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
    except Exception as e:
        # This may fail in some environments (e.g. Windows, some containers)
        logging.getLogger(__name__).warning(f"Could not set resource limits: {e}")
        pass

# --- Model concurrency semaphore ---
MODEL_CONCURRENCY = int(os.environ.get("MODEL_CONCURRENCY", "1"))
_model_semaphore = Semaphore(MODEL_CONCURRENCY)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log_handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
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

def load_app_config():
    """
    Loads configuration from settings.yml, with a fallback to settings.default.yml,
    and finally to hardcoded defaults if both files are missing.
    """
    global APP_CONFIG
    try:
        # --- Primary Method: Attempt to load settings.yml ---
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            cfg_raw = yaml.safe_load(f) or {}
        
        # This logic block is intentionally duplicated to maintain compatibility
        defaults = {
            "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": []},
            "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
            "conversion_tools": {},
            "ocr_settings": {"ocrmypdf": {}},
            "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []}
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

            # The same processing logic is applied to the fallback file
            defaults = {
                "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": []},
                "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
                "conversion_tools": {},
                "ocr_settings": {"ocrmypdf": {}},
                "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []}
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
                "app_settings": {"max_file_size_mb": 100, "max_file_size_bytes": 100 * 1024 * 1024, "allowed_all_extensions": set()},
                "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8", "device": "cpu"}},
                "conversion_tools": {},
                "ocr_settings": {"ocrmypdf": {}},
                "auth_settings": {"oidc_client_id": "", "oidc_client_secret": "", "oidc_server_metadata_url": "", "admin_users": []}
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
    task_type: str
    original_filename: str
    input_filepath: str
    input_filesize: int | None = None
    processed_filepath: str | None = None

class JobSchema(BaseModel):
    id: str
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
    output_format: str = ""

# --------------------------------------------------------------------------------
# --- 3. CRUD OPERATIONS
# --------------------------------------------------------------------------------
def get_job(db: Session, job_id: str):
    return db.query(Job).filter(Job.id == job_id).first()

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
    return db_job

# --------------------------------------------------------------------------------
# --- 4. BACKGROUND TASK SETUP
# --------------------------------------------------------------------------------
huey = SqliteHuey(filename=PATHS.HUEY_DB_PATH)
WHISPER_MODELS_CACHE: Dict[str, WhisperModel] = {}

def get_whisper_model(model_size: str, whisper_settings: dict) -> WhisperModel:
    if model_size in WHISPER_MODELS_CACHE:
        logger.info(f"Reusing cached model '{model_size}'.")
        return WHISPER_MODELS_CACHE[model_size]
    with _model_semaphore:
        if model_size in WHISPER_MODELS_CACHE:
            return WHISPER_MODELS_CACHE[model_size]
        logger.info(f"Loading Whisper model '{model_size}'...")
        model = WhisperModel(model_size, device=whisper_settings.get("device", "cpu"), compute_type=whisper_settings.get('compute_type', 'int8'))
        WHISPER_MODELS_CACHE[model_size] = model
        logger.info(f"Model '{model_size}' loaded.")
        return model

def run_command(argv: TypingList[str], timeout: int = 300):
    try:
        res = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, preexec_fn=_limit_resources_preexec)
        if res.returncode != 0:
            raise Exception(f"Command failed with exit code {res.returncode}. Stderr: {res.stderr[:1000]}")
        return res
    except subprocess.TimeoutExpired:
        raise Exception(f"Command timed out after {timeout}s")

def validate_and_build_command(template_str: str, mapping: Dict[str, str]) -> TypingList[str]:
    fmt = Formatter()
    used = {fname for _, fname, _, _ in fmt.parse(template_str) if fname}
    ALLOWED_VARS = {"input", "output", "output_dir", "output_ext", "quality", "speed", "preset", "device", "dpi", "samplerate", "bitdepth", "filter"}
    bad = used - ALLOWED_VARS
    if bad:
        raise ValueError(f"Command template contains disallowed placeholders: {bad}")
    safe_mapping = dict(mapping)
    for name in used:
        if name not in safe_mapping:
            safe_mapping[name] = safe_mapping.get("output_ext", "") if name == "filter" else ""
    formatted = template_str.format(**safe_mapping)
    return shlex.split(formatted)

@huey.task()
def run_transcription_task(job_id: str, input_path_str: str, output_path_str: str, model_size: str, whisper_settings: dict):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled': return
        update_job_status(db, job_id, "processing")
        model = get_whisper_model(model_size, whisper_settings)
        logger.info(f"Starting transcription for job {job_id}")
        segments, info = model.transcribe(str(input_path), beam_size=5)
        full_transcript = []
        for segment in segments:
            job_check = get_job(db, job_id)
            if job_check.status == 'cancelled':
                logger.info(f"Job {job_id} cancelled during transcription.")
                return
            if info.duration > 0:
                progress = int((segment.end / info.duration) * 100)
                update_job_status(db, job_id, "processing", progress=progress)
            full_transcript.append(segment.text.strip())
        transcript_text = "\n".join(full_transcript)
        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")
        with tmp_out.open("w", encoding="utf-8") as f:
            f.write(transcript_text)
        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=transcript_text)
        logger.info(f"Transcription for job {job_id} completed.")
    except Exception as e:
        logger.exception(f"ERROR during transcription for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"Transcription failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            # swallow cleanup errors but log
            logger.exception("Failed to cleanup input file after transcription.")
        db.close()

@huey.task()
def run_pdf_ocr_task(job_id: str, input_path_str: str, output_path_str: str, ocr_settings: dict):
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

@huey.task()
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return
        update_job_status(db, job_id, "processing", progress=50)
        logger.info(f"Starting Image OCR for job {job_id}")
        text = pytesseract.image_to_string(Image.open(str(input_path)))
        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")
        with tmp_out.open("w", encoding="utf-8") as f:
            f.write(text)
        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=text)
        logger.info(f"Image OCR for job {job_id} completed.")
    except Exception as e:
        logger.exception(f"ERROR during Image OCR for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"Image OCR failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup input file after Image OCR.")
        db.close()


@huey.task()
def run_conversion_task(job_id: str, input_path_str: str, output_path_str: str, tool: str, task_key: str, conversion_tools_config: dict):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    temp_input_file = None
    temp_output_file = None
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return
        update_job_status(db, job_id, "processing", progress=25)
        logger.info(f"Starting conversion for job {job_id} using {tool} with task {task_key}")
        tool_config = conversion_tools_config.get(tool)
        if not tool_config:
            raise ValueError(f"Unknown conversion tool: {tool}")

        current_input_path = input_path

        # Pre-processing for specific tools
        if tool == "mozjpeg":
            temp_input_file = input_path.with_suffix('.temp.ppm')
            logger.info(f"Pre-converting for MozJPEG: {input_path} -> {temp_input_file}")
            pre_conv_cmd = ["vips", "copy", str(input_path), str(temp_input_file)]
            pre_conv_result = subprocess.run(pre_conv_cmd, capture_output=True, text=True, check=False, timeout=tool_config.get("timeout", 300))
            if pre_conv_result.returncode != 0:
                err = (pre_conv_result.stderr or "")[:4000]
                raise Exception(f"MozJPEG pre-conversion to PPM failed: {err}")
            current_input_path = temp_input_file

        update_job_status(db, job_id, "processing", progress=50)

        # prepare temporary output and mapping
        temp_output_file = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")
        mapping = {
            "input": str(current_input_path),
            "output": str(temp_output_file),
            "output_dir": str(output_path.parent),
            "output_ext": output_path.suffix.lstrip('.'),
        }

        # tool specific mapping adjustments
        if tool.startswith("ghostscript"):
            # task_key form: "device_setting"
            parts = task_key.split('_', 1)
            device = parts[0] if parts else ""
            setting = parts[1] if len(parts) > 1 else ""
            mapping.update({"device": device, "dpi": setting, "preset": setting})
        elif tool == "pngquant":
            _, quality_key = task_key.split('_')
            quality_map = {"hq": "80-95", "mq": "65-80", "fast": "65-80"}
            speed_map = {"hq": "1", "mq": "3", "fast": "11"}
            mapping.update({"quality": quality_map.get(quality_key, "65-80"), "speed": speed_map.get(quality_key, "3")})
        elif tool == "sox":
            rate, depth = '', ''
            try:
                _, rate, depth = task_key.split('_')
                depth = ('-b' + depth.replace('b', '')) if 'b' in depth else '16b'
            except:
                _, rate = task_key.split('_')
                depth = ''
            
            rate = rate.replace('k', '000') if 'k' in rate else rate
            mapping.update({"samplerate": rate, "bitdepth":  depth})
        elif tool == "mozjpeg":
            _, quality = task_key.split('_')
            quality = quality.replace('q', '')
            mapping.update({"quality": quality})
        elif tool == "libreoffice":
            target_ext = output_path.suffix.lstrip('.')
            filter_val = tool_config.get("filters", {}).get(target_ext, target_ext)
            mapping["filter"] = filter_val

        command_template_str = tool_config["command_template"]
        command = validate_and_build_command(command_template_str, mapping)
        logger.info(f"Executing command: {' '.join(command)}")

        result = run_command(command, timeout=tool_config.get("timeout", 300))

        if temp_output_file and temp_output_file.exists():
            temp_output_file.replace(output_path)

        mark_job_as_completed(db, job_id, output_filepath_str=str(output_path), preview=f"Successfully converted file.")
        logger.info(f"Conversion for job {job_id} completed.")
    except Exception as e:
        logger.exception(f"ERROR during conversion for job {job_id}")
        update_job_status(db, job_id, "failed", error=f"Conversion failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to cleanup main input file after conversion.")
        if temp_input_file:
            try:
                temp_input_file_path = Path(temp_input_file)
                ensure_path_is_safe(temp_input_file_path, [PATHS.UPLOADS_DIR, PATHS.PROCESSED_DIR])
                temp_input_file_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to cleanup temp input file after conversion.")
        if temp_output_file:
            try:
                temp_output_file_path = Path(temp_output_file)
                ensure_path_is_safe(temp_output_file_path, [PATHS.UPLOADS_DIR, PATHS.PROCESSED_DIR])
                temp_output_file_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to cleanup temp output file after conversion.")
        db.close()

# --------------------------------------------------------------------------------
# --- 5. FASTAPI APPLICATION
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    Base.metadata.create_all(bind=engine)
    load_app_config()
    ENV = os.environ.get('ENV', 'dev').lower() # probably reduntant because I load the .env at the start but whatever
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

# Should probably set https_only=True in production behind HTTPS i guess
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False,
    same_site='lax',
    max_age=14 * 24 * 60 * 60  # 14 days in seconds
)


# Static / templates
app.mount("/static", StaticFiles(directory=str(PATHS.BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PATHS.BASE_DIR / "templates"))

# --- AUTH & USER HELPERS ---
def get_current_user(request: Request):
    if LOCAL_ONLY_MODE:
        return {'sub': 'local_user', 'email': 'local@user.com', 'name': 'Local User'}
    return request.session.get('user')

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

# --- CHUNKED UPLOADs ---
@app.post("/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    user: dict = Depends(require_user) # AUTHENTICATION
):
    safe_upload_id = secure_filename(upload_id)
    temp_dir = PATHS.CHUNK_TMP_DIR / safe_upload_id
    temp_dir = ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    temp_dir.mkdir(exist_ok=True)
    chunk_path = temp_dir / f"{chunk_number}.chunk"

    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(chunk.file, buffer)
    finally:
        chunk.file.close()
    return JSONResponse({"message": f"Chunk {chunk_number} for {safe_upload_id} uploaded."})

async def _stitch_chunks(temp_dir: Path, final_path: Path, total_chunks: int):
    """Stitches chunks together and cleans up."""
    ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    ensure_path_is_safe(final_path, [PATHS.UPLOADS_DIR])
    with open(final_path, "wb") as final_file:
        for i in range(total_chunks):
            chunk_path = temp_dir / f"{i}.chunk"
            if not chunk_path.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail=f"Upload failed: missing chunk {i}")
            with open(chunk_path, "rb") as chunk_file:
                final_file.write(chunk_file.read())
    shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/upload/finalize", status_code=status.HTTP_202_ACCEPTED)
async def finalize_upload(payload: FinalizeUploadPayload, user: dict = Depends(require_user), db: Session = Depends(get_db)):
    safe_upload_id = secure_filename(payload.upload_id)
    temp_dir = PATHS.CHUNK_TMP_DIR / safe_upload_id
    temp_dir = ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    if not temp_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload session not found or already finalized.")

    job_id = uuid.uuid4().hex
    safe_filename = secure_filename(payload.original_filename)
    final_path = PATHS.UPLOADS_DIR / f"{Path(safe_filename).stem}_{job_id}{Path(safe_filename).suffix}"
    await _stitch_chunks(temp_dir, final_path, payload.total_chunks)

    job_data = JobCreate(
        id=job_id, user_id=user['sub'], task_type=payload.task_type,
        original_filename=payload.original_filename, input_filepath=str(final_path),
        input_filesize=final_path.stat().st_size
    )

    if payload.task_type == "transcription":
        stem = Path(safe_filename).stem
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_transcription_task(job_id, str(final_path), str(processed_path), payload.model_size, APP_CONFIG.get("transcription_settings", {}).get("whisper", {}))
    elif payload.task_type == "ocr":
        stem, suffix = Path(safe_filename).stem, Path(safe_filename).suffix
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{suffix}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_pdf_ocr_task(job_id, str(final_path), str(processed_path), APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {}))
    elif payload.task_type == "conversion":
        try:
            tool, task_key = payload.output_format.split('_', 1)
        except Exception:
            final_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Invalid output_format for conversion.")
        original_stem = Path(safe_filename).stem
        target_ext = task_key.split('_')[0]
        if tool == "ghostscript_pdf": target_ext = "pdf"
        processed_path = PATHS.PROCESSED_DIR / f"{original_stem}_{job_id}.{target_ext}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_conversion_task(job_id, str(final_path), str(processed_path), tool, task_key, APP_CONFIG.get("conversion_tools", {}))
    else:
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid task type.")

    return {"job_id": job_id, "status": "pending"}

# --- OLD DIRECT-UPLOAD ROUTES (kept for compatibility) ---
# These use the same task functions but accept direct file uploads (no chunking).
async def save_upload_file_chunked(upload_file: UploadFile, destination: Path) -> int:
    """
    Write upload to a tmp file in chunks, then atomically move to final destination.
    Returns the final size of the file in bytes.
    """
    max_size = APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024)
    tmp = destination.with_name(f"{destination.stem}.tmp-{uuid.uuid4().hex}{destination.suffix}")
    size = 0
    try:
        with tmp.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(status_code=413, detail=f"File exceeds {max_size / 1024 / 1024} MB limit")
                buffer.write(chunk)
        tmp.replace(destination)
        return size
    except Exception:
        try:
            ensure_path_is_safe(tmp, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            tmp.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove temp upload file after error.")
        raise

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions

@app.post("/transcribe-audio", status_code=status.HTTP_202_ACCEPTED)
async def submit_audio_transcription(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    db: Session = Depends(get_db),
    user: dict = Depends(require_user)
):
    allowed_audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
    if not is_allowed_file(file.filename, allowed_audio_exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid audio file type.")

    whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
    if model_size not in whisper_config.get("allowed_models", []):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model size: {model_size}.")

    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix

    audio_filename = f"{stem}_{job_id}{suffix}"
    transcript_filename = f"{stem}_{job_id}.txt"
    upload_path = PATHS.UPLOADS_DIR / audio_filename
    processed_path = PATHS.PROCESSED_DIR / transcript_filename

    input_size = await save_upload_file_chunked(file, upload_path)

    job_data = JobCreate(
        id=job_id,
        user_id=user['sub'],
        task_type="transcription",
        original_filename=file.filename,
        input_filepath=str(upload_path),
        input_filesize=input_size,
        processed_filepath=str(processed_path)
    )
    new_job = create_job(db=db, job=job_data)

    run_transcription_task(new_job.id, str(upload_path), str(processed_path), model_size, whisper_settings=whisper_config)

    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/convert-file", status_code=status.HTTP_202_ACCEPTED)
async def submit_file_conversion(file: UploadFile = File(...), output_format: str = Form(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    allowed_exts = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail=f"File type '{Path(file.filename).suffix}' not allowed.")
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    try:
        tool, task_key = output_format.split('_', 1)
        if tool not in conversion_tools or task_key not in conversion_tools[tool].get("formats", {}):
            # fallback: allow tasks that exist but may not be in formats map (some configs only have commands)
            if tool not in conversion_tools:
                raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output format selected.")
    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    original_stem = Path(safe_basename).stem
    target_ext = task_key.split('_')[0]
    if tool == "ghostscript_pdf":
        target_ext = "pdf"
    upload_filename = f"{original_stem}_{job_id}{Path(safe_basename).suffix}"
    processed_filename = f"{original_stem}_{job_id}.{target_ext}"
    upload_path = PATHS.UPLOADS_DIR / upload_filename
    processed_path = PATHS.PROCESSED_DIR / processed_filename
    input_size = await save_upload_file_chunked(file, upload_path)
    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="conversion", original_filename=file.filename,
                         input_filepath=str(upload_path),
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_conversion_task(new_job.id, str(upload_path), str(processed_path), tool, task_key, conversion_tools)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-pdf", status_code=status.HTTP_202_ACCEPTED)
async def submit_pdf_ocr(file: UploadFile = File(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    if not is_allowed_file(file.filename, {".pdf"}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PDF.")
    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{Path(safe_basename).suffix}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / unique_filename
    input_size = await save_upload_file_chunked(file, upload_path)
    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="ocr", original_filename=file.filename,
                         input_filepath=str(upload_path),
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
    run_pdf_ocr_task(new_job.id, str(upload_path), str(processed_path), ocr_settings)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-image", status_code=status.HTTP_202_ACCEPTED)
async def submit_image_ocr(file: UploadFile = File(...), db: Session = Depends(get_db), user: dict = Depends(require_user)):
    allowed_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PNG, JPG, or TIFF.")
    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    file_ext = Path(safe_basename).suffix
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{file_ext}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / f"{Path(safe_basename).stem}_{job_id}.txt"
    input_size = await save_upload_file_chunked(file, upload_path)
    job_data = JobCreate(id=job_id, user_id=user['sub'], task_type="ocr-image", original_filename=file.filename,
                         input_filepath=str(upload_path),
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_image_ocr_task(new_job.id, str(upload_path), str(processed_path))
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

# --- Routes for auth and pages ---
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
            # Store id_token in session for logout
            request.session['id_token'] = token.get('id_token')
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
        return RedirectResponse(url='/')

    @app.get("/logout")
    async def logout(request: Request):
        logout_endpoint = oauth.oidc.server_metadata.get("end_session_endpoint")
        logger.info(f"OIDC end_session_endpoint: {logout_endpoint}")

        # local-only logout if provider doesn't expose end_session_endpoint
        if not logout_endpoint:
            request.session.clear()
            logger.warning("OIDC 'end_session_endpoint' not found. Performing local-only logout.")
            return RedirectResponse(url="/", status_code=302)


        # Prefer a single canonical / registered post-logout redirect URI from config
        post_logout_redirect_uri = str(request.url_for("get_index"))
        logger.info(f"Post logout redirect URI: {post_logout_redirect_uri}")

        logout_url = f"{logout_endpoint}?post_logout_redirect_uri={post_logout_redirect_uri}"

        logger.info(f"Redirecting to provider logout URL: {logout_url}")

        request.session.clear()
        return RedirectResponse(url=logout_url, status_code=302)


#### TODO: Remove this weird forward authz endpoint, its needed if reverse proxy does foward auth

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
        "request": request,
        "user": user,
        "is_admin": admin_status,
        "whisper_models": sorted(list(whisper_models)),
        "conversion_tools": conversion_tools,
        "local_only_mode": LOCAL_ONLY_MODE
    })

@app.get("/settings")
async def get_settings_page(request: Request):
    """
    Displays the contents of the currently active configuration file.
    It prioritizes settings.yml and falls back to settings.default.yml.
    """
    user = get_current_user(request)
    admin_status = is_admin(request)
    current_config = {}
    config_source = "none" # A helper variable to track which file was loaded

    try:
        # 1. Attempt to load the primary, user-provided settings.yml
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            current_config = yaml.safe_load(f) or {}
        config_source = str(PATHS.SETTINGS_FILE.name)
        logger.info(f"Displaying configuration from '{config_source}' on settings page.")

    except FileNotFoundError:
        logger.warning(f"'{PATHS.SETTINGS_FILE.name}' not found. Attempting to display fallback configuration.")
        try:
            # 2. If it's not found, fall back to the default settings file
            with open(PATHS.DEFAULT_SETTINGS_FILE, 'r', encoding='utf8') as f:
                current_config = yaml.safe_load(f) or {}
            config_source = str(PATHS.DEFAULT_SETTINGS_FILE.name)
            logger.info(f"Displaying configuration from fallback '{config_source}' on settings page.")
        
        except Exception as e_fallback:
            # 3. If even the default file fails, log the error and use an empty config
            logger.exception(f"CRITICAL: Could not load fallback '{PATHS.DEFAULT_SETTINGS_FILE.name}' for settings page: {e_fallback}")
            current_config = {} # Failsafe
            config_source = "error"
            
    except Exception as e_primary:
        # Handles other errors with the primary settings.yml (e.g., parsing errors, permissions)
        logger.exception(f"Could not load '{PATHS.SETTINGS_FILE.name}' for settings page: {e_primary}")
        current_config = {} # Failsafe
        config_source = "error"

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "config": current_config,
            "config_source": config_source, # You can use this in the template!
            "user": user,
            "is_admin": admin_status,
            "local_only_mode": LOCAL_ONLY_MODE,
        }
    )


import collections.abc

def deep_merge(source: dict, destination: dict) -> dict:
    """
    Recursively merges the `source` dictionary into the `destination` dictionary.

    Values from `source` will overwrite values in `destination`.
    """
    for key, value in source.items():
        if isinstance(value, collections.abc.Mapping):
            # If the value is a dictionary, recurse
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            # Otherwise, overwrite the value
            destination[key] = value
    return destination

@app.post("/settings/save")
async def save_settings(
    request: Request,
    new_config_from_ui: Dict = Body(...),
    admin: bool = Depends(require_admin)
):
    """
    Safely updates settings.yml by merging UI changes with the existing file,
    preserving any settings not managed by the UI.
    """
    tmp_path = PATHS.SETTINGS_FILE.with_suffix(".tmp")
    user = get_current_user(request)

    try:
        # Handle the special case where the user wants to revert to defaults
        if not new_config_from_ui:
            if PATHS.SETTINGS_FILE.exists():
                PATHS.SETTINGS_FILE.unlink()
                logger.info(f"Admin '{user.get('email')}' reverted to default settings by deleting settings.yml.")
            load_app_config()
            return JSONResponse({"message": "Settings reverted to default."})

        # --- Read-Modify-Write Cycle ---

        # 1. READ: Load the current configuration from settings.yml on disk.
        # If the file doesn't exist, start with an empty dictionary.
        try:
            with PATHS.SETTINGS_FILE.open("r", encoding="utf8") as f:
                current_config_on_disk = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current_config_on_disk = {}

        # 2. MODIFY: Deep merge the changes from the UI into the config from the disk.
        # The UI config (`source`) overwrites keys in the disk config (`destination`).
        merged_config = deep_merge(source=new_config_from_ui, destination=current_config_on_disk)

        # 3. WRITE: Save the fully merged configuration back to the file.
        with tmp_path.open("w", encoding="utf8") as f:
            yaml.safe_dump(merged_config, f, default_flow_style=False, sort_keys=False)
        
        tmp_path.replace(PATHS.SETTINGS_FILE)
        logger.info(f"Admin '{user.get('email')}' successfully updated settings.yml.")

        # Reload the app config to apply changes immediately
        load_app_config()
        
        return JSONResponse({"message": "Settings saved successfully. The new configuration is now active."})

    except Exception as e:
        logger.exception(f"Failed to update settings for admin '{user.get('email')}'")
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not save settings.yml: {e}")

# job management endpoints

@app.post("/settings/clear-history")
async def clear_job_history(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    try:
        num_deleted = db.query(Job).filter(Job.user_id == user['sub']).delete()
        db.commit()
        logger.info(f"Cleared {num_deleted} jobs from history for user {user['sub']}.")
        return {"deleted_count": num_deleted}
    except Exception:
        db.rollback()
        logger.exception("Failed to clear job history")
        raise HTTPException(status_code=500, detail="Database error while clearing history.")

@app.post("/settings/delete-files")
async def delete_processed_files(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    deleted_count = 0
    errors = []
    user_jobs = get_jobs(db, user_id=user['sub'])
    for job in user_jobs:
        if job.processed_filepath:
            try:
                p = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
            except Exception as e:
                errors.append(Path(job.processed_filepath).name)
                logger.exception(f"Could not delete processed file {Path(job.processed_filepath).name}")
    if errors:
        raise HTTPException(status_code=500, detail=f"Could not delete some files: {', '.join(errors)}")
    logger.info(f"Deleted {deleted_count} files from processed directory for user {user['sub']}.")
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

@app.get("/download/{filename}")
async def download_file(filename: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    safe_filename = secure_filename(filename)
    file_path = ensure_path_is_safe(PATHS.PROCESSED_DIR / safe_filename, [PATHS.PROCESSED_DIR])
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    job = db.query(Job).filter(Job.processed_filepath == str(file_path), Job.user_id == user['sub']).first()
    if not job:
        raise HTTPException(status_code=403, detail="You do not have permission to download this file.")
    download_filename = Path(job.original_filename).stem + Path(job.processed_filepath).suffix
    return FileResponse(path=file_path, filename=download_filename, media_type="application/octet-stream")

@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse({"ok": False}, status_code=500)
    return {"ok": True}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(str(PATHS.BASE_DIR / 'static' / 'favicon.png'))