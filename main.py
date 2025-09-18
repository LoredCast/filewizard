import logging
import shutil
import subprocess
import traceback
import uuid
import shlex
import yaml
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import ocrmypdf
import pypdf
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status, Body)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huey import SqliteHuey
from pydantic import BaseModel, ConfigDict, field_serializer # MODIFIED: Import field_serializer
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine, delete, event)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from string import Formatter
from werkzeug.utils import secure_filename
from typing import List as TypingList

# --------------------------------------------------------------------------------
# --- 1. CONFIGURATION
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AppPaths(BaseModel):
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'jobs.db'}"
    HUEY_DB_PATH: str = str(BASE_DIR / "huey.db")
    SETTINGS_FILE: Path = BASE_DIR / "settings.yml"

PATHS = AppPaths()
APP_CONFIG: Dict[str, Any] = {}
PATHS.UPLOADS_DIR.mkdir(exist_ok=True)
PATHS.PROCESSED_DIR.mkdir(exist_ok=True)

def load_app_config():
    global APP_CONFIG
    try:
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            cfg_raw = yaml.safe_load(f) or {}
        # basic defaults
        defaults = {
            "app_settings": {"max_file_size_mb": 100, "allowed_all_extensions": []},
            "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8"}},
            "conversion_tools": {},
            "ocr_settings": {"ocrmypdf": {}}
        }
        # shallow merge (safe for top-level keys)
        cfg = defaults.copy()
        cfg.update(cfg_raw)
        # normalize app settings
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
        logging.getLogger(__name__).exception(f"Could not load settings.yml: {e}. Using defaults.")
        
        APP_CONFIG = {
            "app_settings": {"max_file_size_mb": 100, "max_file_size_bytes": 100 * 1024 * 1024, "allowed_all_extensions": set()},
            "transcription_settings": {"whisper": {"allowed_models": ["tiny", "base", "small"], "compute_type": "int8"}},
            "conversion_tools": {},
            "ocr_settings": {"ocrmypdf": {}}
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
    """
    Enable WAL mode and set sane synchronous for better concurrency
    between the FastAPI process and Huey worker processes.
    """
    c = dbapi_connection.cursor()
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
    finally:
        c.close()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
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

    # NEW: This serializer ensures the datetime string sent to the frontend ALWAYS
    # includes the 'Z' UTC indicator, fixing the timezone bug.
    @field_serializer('created_at', 'updated_at')
    def serialize_dt(self, dt: datetime, _info):
        return dt.isoformat() + "Z"

# --------------------------------------------------------------------------------
# --- 3. CRUD OPERATIONS
# --------------------------------------------------------------------------------
def get_job(db: Session, job_id: str):
    return db.query(Job).filter(Job.id == job_id).first()

def get_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()

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

# ... (The rest of the file is unchanged and remains the same) ...

# --------------------------------------------------------------------------------
# --- 4. BACKGROUND TASK SETUP
# --------------------------------------------------------------------------------
huey = SqliteHuey(filename=PATHS.HUEY_DB_PATH)

# Whisper model cache per worker process
WHISPER_MODELS_CACHE: Dict[str, WhisperModel] = {}

def get_whisper_model(model_size: str, whisper_settings: dict) -> WhisperModel:
    if model_size in WHISPER_MODELS_CACHE:
        logger.info(f"Found model '{model_size}' in cache. Reusing.")
        return WHISPER_MODELS_CACHE[model_size]
    device = whisper_settings.get("device", "cpu")
    compute_type = whisper_settings.get('compute_type', 'int8')
    logger.info(f"Whisper model '{model_size}' not in cache. Loading into memory on device={device}...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        logger.exception("Failed to load whisper model")
        raise
    WHISPER_MODELS_CACHE[model_size] = model
    logger.info(f"Model '{model_size}' loaded successfully.")
    return model

# Helper: safe run_command (trimmed logs + timeout)
def run_command(argv: TypingList[str], timeout: int = 300):
    try:
        res = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Exception(f"Command timed out after {timeout}s")
    if res.returncode != 0:
        stderr = (res.stderr or "")[:4000]
        stdout = (res.stdout or "")[:4000]
        raise Exception(f"Command failed exit {res.returncode}. stderr: {stderr}; stdout: {stdout}")
    return res

# Helper: validate and build command from template with allowlist
ALLOWED_VARS = {"input", "output", "output_dir", "output_ext", "quality", "speed", "preset", "device", "dpi", "samplerate", "bitdepth", "filter"}

def validate_and_build_command(template_str: str, mapping: Dict[str, str]) -> TypingList[str]:
    """
    Validate placeholders against ALLOWED_VARS and build a safe argv list.
    If a template uses allowed placeholders that are missing from `mapping`,
    auto-fill sensible defaults:
      - 'filter' -> mapping.get('output_ext', '')
      - others -> empty string
    This prevents KeyError while preserving the allowlist security check.
    """
    fmt = Formatter()
    used = {fname for _, fname, _, _ in fmt.parse(template_str) if fname}
    bad = used - ALLOWED_VARS
    if bad:
        raise ValueError(f"Command template contains disallowed placeholders: {bad}")

    # auto-fill missing allowed placeholders with safe defaults
    safe_mapping = dict(mapping)  # shallow copy to avoid mutating caller mapping
    for name in used:
        if name not in safe_mapping:
            if name == "filter":
                safe_mapping[name] = safe_mapping.get("output_ext", "")
            else:
                safe_mapping[name] = ""

    formatted = template_str.format(**safe_mapping)
    return shlex.split(formatted)

@huey.task()
def run_transcription_task(job_id: str, input_path_str: str, output_path_str: str, model_size: str, whisper_settings: dict):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return
        update_job_status(db, job_id, "processing")
        model = get_whisper_model(model_size, whisper_settings)
        logger.info(f"Starting transcription for job {job_id}")
        segments, info = model.transcribe(input_path_str, beam_size=5)
        full_transcript = []
        for segment in segments:
            job_check = get_job(db, job_id)  # Check for cancellation during long tasks
            if job_check.status == 'cancelled':
                logger.info(f"Job {job_id} cancelled during transcription.")
                return
            if info.duration > 0:
                progress = int((segment.end / info.duration) * 100)
                update_job_status(db, job_id, "processing", progress=progress)
            full_transcript.append(segment.text.strip())
        transcript_text = "\n".join(full_transcript)
        # atomic write of transcript — keep the real extension and mark tmp in the name
        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")
        with tmp_out.open("w", encoding="utf-8") as f:
            f.write(transcript_text)
        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=transcript_text)
        logger.info(f"Transcription for job {job_id} completed.")
    except Exception:
        logger.exception(f"ERROR during transcription for job {job_id}")
        update_job_status(db, job_id, "failed", error="See server logs for details.")
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()

@huey.task()
def run_pdf_ocr_task(job_id: str, input_path_str: str, output_path_str: str, ocr_settings: dict):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return
        update_job_status(db, job_id, "processing")
        logger.info(f"Starting PDF OCR for job {job_id}")
        ocrmypdf.ocr(input_path_str, output_path_str,
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
    except Exception:
        logger.exception(f"ERROR during PDF OCR for job {job_id}")
        update_job_status(db, job_id, "failed", error="See server logs for details.")
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()

@huey.task()
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            return
        update_job_status(db, job_id, "processing", progress=50)
        logger.info(f"Starting Image OCR for job {job_id}")
        text = pytesseract.image_to_string(Image.open(input_path_str))
        # atomic write of OCR text
        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")        
        with tmp_out.open("w", encoding="utf-8") as f:
            f.write(text)
        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=text)
        logger.info(f"Image OCR for job {job_id} completed.")
    except Exception:
        logger.exception(f"ERROR during Image OCR for job {job_id}")
        update_job_status(db, job_id, "failed", error="See server logs for details.")
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()


@huey.task()
def run_conversion_task(job_id: str, input_path_str: str, output_path_str: str, tool: str, task_key: str, conversion_tools_config: dict):
    db = SessionLocal()
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
        input_path = Path(input_path_str)
        output_path = Path(output_path_str)
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
        # use a temp filename that preserves the real extension, e.g. file.tmp-<uuid>.pdf
        temp_output_file = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")
        mapping = {
            "input": str(current_input_path),
            "output": str(temp_output_file),
            "output_dir": str(output_path.parent),
            "output_ext": output_path.suffix.lstrip('.'),
        }

        # tool specific mapping adjustments
        if tool.startswith("ghostscript"):
            device, setting = task_key.split('_')
            mapping.update({"device": device, "dpi": setting, "preset": setting})
        elif tool == "pngquant":
            _, quality_key = task_key.split('_')
            quality_map = {"hq": "80-95", "mq": "65-80", "fast": "65-80"}
            speed_map = {"hq": "1", "mq": "3", "fast": "11"}
            mapping.update({"quality": quality_map.get(quality_key, "65-80"), "speed": speed_map.get(quality_key, "3")})
        elif tool == "sox":
            _, rate, depth = task_key.split('_')
            rate = rate.replace('k', '000') if 'k' in rate else rate
            depth = depth.replace('b', '') if 'b' in depth else '16'
            mapping.update({"samplerate": rate, "bitdepth": depth})
        elif tool == "mozjpeg":
            _, quality = task_key.split('_')
            quality = quality.replace('q', '')
            mapping.update({"quality": quality})
        elif tool == "libreoffice":
            target_ext = output_path.suffix.lstrip('.')
            # tool_config may include a 'filters' mapping (see settings.yml example)
            filter_val = tool_config.get("filters", {}).get(target_ext, target_ext)
            mapping["filter"] = filter_val

        command_template_str = tool_config["command_template"]
        command = validate_and_build_command(command_template_str, mapping)
        logger.info(f"Executing command: {' '.join(command)}")

        # execute command with timeout and trimmed logs on error
        result = run_command(command, timeout=tool_config.get("timeout", 300))

        # handle LibreOffice special case: sometimes it writes differently
        # Special-case LibreOffice: support per-format export filters via settings.yml


        # move temp output into final location atomically
        if temp_output_file and temp_output_file.exists():
            temp_output_file.replace(output_path)

        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=f"Successfully converted file.")
        logger.info(f"Conversion for job {job_id} completed.")
    except Exception:
        logger.exception(f"ERROR during conversion for job {job_id}")
        update_job_status(db, job_id, "failed", error="See server logs for details.")
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        if temp_input_file:
            temp_input_file.unlink(missing_ok=True)
        if temp_output_file:
            temp_output_file.unlink(missing_ok=True)
        db.close()

# --------------------------------------------------------------------------------
# --- 5. FASTAPI APPLICATION
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    Base.metadata.create_all(bind=engine)
    load_app_config()
    yield
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PATHS.BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=PATHS.BASE_DIR / "templates")

async def save_upload_file_chunked(upload_file: UploadFile, destination: Path) -> int:
    """
    Write upload to a tmp file in chunks, then atomically move to final destination.
    Returns the final size of the file in bytes.
    """
    max_size = APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024)
    # make a temp filename that keeps the real extension, e.g. file.tmp-<uuid>.pdf
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
        tmp.unlink(missing_ok=True)
        raise

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions

# --- Routes (transcription route uses Huey task enqueuing) ---

@app.post("/transcribe-audio", status_code=status.HTTP_202_ACCEPTED)
async def submit_audio_transcription(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    db: Session = Depends(get_db)
):
    if not is_allowed_file(file.filename, {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}):
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
        task_type="transcription", 
        original_filename=file.filename, 
        input_filepath=str(upload_path), 
        input_filesize=input_size,
        processed_filepath=str(processed_path)
    )
    new_job = create_job(db=db, job=job_data)
    
    # enqueue the Huey task (decorated function call enqueues when using huey)
    run_transcription_task(new_job.id, str(upload_path), str(processed_path), model_size=model_size, whisper_settings=whisper_config)
    
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}


@app.get("/")
async def get_index(request: Request):
    whisper_models = APP_CONFIG.get("transcription_settings", {}).get("whisper", {}).get("allowed_models", [])
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    return templates.TemplateResponse("index.html", {
        "request": request,
        "whisper_models": sorted(list(whisper_models)),
        "conversion_tools": conversion_tools
    })

@app.get("/settings")
async def get_settings_page(request: Request):
    try:
        with open(PATHS.SETTINGS_FILE, 'r', encoding='utf8') as f:
            current_config = yaml.safe_load(f) or {}
    except Exception:
        logger.exception("Could not load settings.yml for settings page")
        current_config = {}
    return templates.TemplateResponse("settings.html", {"request": request, "config": current_config})

def deep_merge(base: dict, updates: dict) -> dict:
    """
    Recursively merge `updates` into `base`. Lists and scalars are replaced.
    """
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@app.post("/settings/save")
async def save_settings(new_config: Dict = Body(...)):
    tmp = PATHS.SETTINGS_FILE.with_suffix(".tmp")
    try:
        # load existing config if present
        try:
            with PATHS.SETTINGS_FILE.open("r", encoding="utf8") as f:
                current_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current_config = {}

        # deep merge new values
        merged = deep_merge(current_config, new_config)

        # atomic write back
        with tmp.open("w", encoding="utf8") as f:
            yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False)
        tmp.replace(PATHS.SETTINGS_FILE)

        load_app_config()
        return JSONResponse({"message": "Settings updated successfully."})
    except Exception:
        logger.exception("Failed to update settings")
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not update settings.yml.")


@app.post("/settings/clear-history")
async def clear_job_history(db: Session = Depends(get_db)):
    try:
        num_deleted = db.query(Job).delete()
        db.commit()
        logger.info(f"Cleared {num_deleted} jobs from history.")
        return {"deleted_count": num_deleted}
    except Exception:
        db.rollback()
        logger.exception("Failed to clear job history")
        raise HTTPException(status_code=500, detail="Database error while clearing history.")

@app.post("/settings/delete-files")
async def delete_processed_files():
    deleted_count = 0
    errors = []
    for f in PATHS.PROCESSED_DIR.glob('*'):
        try:
            if f.is_file():
                f.unlink()
                deleted_count += 1
        except Exception:
            errors.append(f.name)
            logger.exception(f"Could not delete processed file {f.name}")
    if errors:
        raise HTTPException(status_code=500, detail=f"Could not delete some files: {', '.join(errors)}")
    logger.info(f"Deleted {deleted_count} files from processed directory.")
    return {"deleted_count": deleted_count}

@app.post("/convert-file", status_code=status.HTTP_202_ACCEPTED)
async def submit_file_conversion(file: UploadFile = File(...), output_format: str = Form(...), db: Session = Depends(get_db)):
    allowed_exts = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail=f"File type '{Path(file.filename).suffix}' not allowed.")
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    try:
        tool, task_key = output_format.split('_', 1)
        if tool not in conversion_tools or task_key not in conversion_tools[tool]["formats"]:
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
    job_data = JobCreate(id=job_id, task_type="conversion", original_filename=file.filename,
                         input_filepath=str(upload_path), 
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_conversion_task(new_job.id, str(upload_path), str(processed_path), tool, task_key, conversion_tools)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-pdf", status_code=status.HTTP_202_ACCEPTED)
async def submit_pdf_ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_allowed_file(file.filename, {".pdf"}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PDF.")
    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{Path(safe_basename).suffix}"
    upload_path = PATHS.UPLOADS_DIR / unique_filename
    processed_path = PATHS.PROCESSED_DIR / unique_filename
    input_size = await save_upload_file_chunked(file, upload_path)
    job_data = JobCreate(id=job_id, task_type="ocr", original_filename=file.filename,
                         input_filepath=str(upload_path), 
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
    run_pdf_ocr_task(new_job.id, str(upload_path), str(processed_path), ocr_settings)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/ocr-image", status_code=status.HTTP_202_ACCEPTED)
async def submit_image_ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
    job_data = JobCreate(id=job_id, task_type="ocr-image", original_filename=file.filename,
                         input_filepath=str(upload_path), 
                         input_filesize=input_size,
                         processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    run_image_ocr_task(new_job.id, str(upload_path), str(processed_path))
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}

@app.post("/job/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ["pending", "processing"]:
        update_job_status(db, job_id, status="cancelled")
        return {"message": "Job cancellation requested."}
    raise HTTPException(status_code=400, detail=f"Job is already in a final state ({job.status}).")

@app.get("/jobs", response_model=List[JobSchema])
async def get_all_jobs(db: Session = Depends(get_db)):
    return get_jobs(db)

@app.get("/job/{job_id}", response_model=JobSchema)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_filename = secure_filename(filename)
    file_path = (PATHS.PROCESSED_DIR / safe_filename).resolve()
    base = PATHS.PROCESSED_DIR.resolve()
    try:
        file_path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=safe_filename, media_type="application/octet-stream")

# Small health endpoint
@app.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse({"ok": False}, status_code=500)
    return {"ok": True}

favicon_path = PATHS.BASE_DIR / 'static' / 'favicon.png'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)