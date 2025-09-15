import logging
import shutil
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Set

import ocrmypdf
import pypdf
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
# MODIFICATION: Added Form for model selection
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huey import SqliteHuey
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings
from sqlalchemy import (Column, DateTime, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------------
# --- 1. CONFIGURATION
# --------------------------------------------------------------------------------
class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'jobs.db'}"
    HUEY_DB_PATH: str = str(BASE_DIR / "huey.db")
    # MODIFICATION: Removed hardcoded model size, added a set of allowed models
    WHISPER_COMPUTE_TYPE: str = "int8"
    ALLOWED_WHISPER_MODELS: Set[str] = {"tiny", "base", "small", "medium", "large-v3", "distil-large-v2"}
    MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_PDF_EXTENSIONS: set = {".pdf"}
    ALLOWED_IMAGE_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    ALLOWED_AUDIO_EXTENSIONS: set = {".mp3", "m4a", ".ogg", ".flac", ".opus"}

settings = Settings()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings.UPLOADS_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------------
# --- 2. DATABASE (for Job Tracking) - NO CHANGES
# --------------------------------------------------------------------------------
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    task_type = Column(String, index=True)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    original_filename = Column(String)
    input_filepath = Column(String)
    processed_filepath = Column(String, nullable=True)
    result_preview = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------------
# --- 3. PYDANTIC SCHEMAS (Data Validation) - NO CHANGES
# --------------------------------------------------------------------------------
class JobCreate(BaseModel):
    id: str
    task_type: str
    original_filename: str
    input_filepath: str
    processed_filepath: str | None = None

class JobSchema(BaseModel):
    id: str
    task_type: str
    status: str
    progress: int
    original_filename: str
    processed_filepath: str | None = None
    result_preview: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------------
# --- 4. CRUD OPERATIONS (Database Interactions) - NO CHANGES
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

def mark_job_as_completed(db: Session, job_id: str, preview: str | None = None):
    db_job = get_job(db, job_id)
    if db_job and db_job.status != 'cancelled':
        db_job.status = "completed"
        db_job.progress = 100
        if preview:
            db_job.result_preview = preview.strip()[:2000]
        db.commit()
    return db_job


# --------------------------------------------------------------------------------
# --- 5. BACKGROUND TASKS (Huey)
# --------------------------------------------------------------------------------
huey = SqliteHuey(filename=settings.HUEY_DB_PATH)

# MODIFICATION: Removed global whisper model and lazy loader.
# The model will now be loaded inside the task itself based on user selection.

@huey.task()
def run_pdf_ocr_task(job_id: str, input_path_str: str, output_path_str: str):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            logger.info(f"Job {job_id} was cancelled before starting.")
            return

        update_job_status(db, job_id, "processing")
        logger.info(f"Starting PDF OCR for job {job_id}")
        
        ocrmypdf.ocr(input_path_str, output_path_str, deskew=True, force_ocr=True, clean=True, optimize=1, progress_bar=False)
        
        with open(output_path_str, "rb") as f:
            reader = pypdf.PdfReader(f)
            preview = "\n".join(page.extract_text() or "" for page in reader.pages)
        mark_job_as_completed(db, job_id, preview=preview)
        logger.info(f"PDF OCR for job {job_id} completed.")
    except Exception as e:
        logger.error(f"ERROR during PDF OCR for job {job_id}: {e}\n{traceback.format_exc()}")
        update_job_status(db, job_id, "failed", error=str(e))
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()

@huey.task()
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            logger.info(f"Job {job_id} was cancelled before starting.")
            return

        update_job_status(db, job_id, "processing", progress=50)
        logger.info(f"Starting Image OCR for job {job_id}")
        text = pytesseract.image_to_string(Image.open(input_path_str))
        with open(output_path_str, "w", encoding="utf-8") as f:
            f.write(text)
        mark_job_as_completed(db, job_id, preview=text)
        logger.info(f"Image OCR for job {job_id} completed.")
    except Exception as e:
        logger.error(f"ERROR during Image OCR for job {job_id}: {e}\n{traceback.format_exc()}")
        update_job_status(db, job_id, "failed", error=str(e))
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()

# MODIFICATION: The task now accepts `model_size` and loads the model dynamically.
@huey.task()
def run_transcription_task(job_id: str, input_path_str: str, output_path_str: str, model_size: str):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or job.status == 'cancelled':
            logger.info(f"Job {job_id} was cancelled before starting.")
            return

        update_job_status(db, job_id, "processing")
        
        # Load the specified model for this task
        logger.info(f"Loading faster-whisper model: {model_size} for job {job_id}...")
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        logger.info(f"Whisper model '{model_size}' loaded successfully.")
        
        logger.info(f"Starting transcription for job {job_id}")
        segments, info = model.transcribe(input_path_str, beam_size=5)
        
        full_transcript = []
        total_duration = info.duration
        for segment in segments:
            job_check = get_job(db, job_id)
            if job_check.status == 'cancelled':
                logger.info(f"Job {job_id} cancelled during transcription.")
                return

            # Update progress based on the segment's end time
            if total_duration > 0:
                progress = int((segment.end / total_duration) * 100)
                update_job_status(db, job_id, "processing", progress=progress)
            full_transcript.append(segment.text.strip())

        transcript_text = "\n".join(full_transcript)
        with open(output_path_str, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        mark_job_as_completed(db, job_id, preview=transcript_text)
        logger.info(f"Transcription for job {job_id} completed.")
    except Exception as e:
        logger.error(f"ERROR during transcription for job {job_id}: {e}\n{traceback.format_exc()}")
        update_job_status(db, job_id, "failed", error=str(e))
    finally:
        Path(input_path_str).unlink(missing_ok=True)
        db.close()


# --------------------------------------------------------------------------------
# --- 6. FASTAPI APPLICATION
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=settings.BASE_DIR / "templates")

# --- Helper Functions ---
async def save_upload_file_chunked(upload_file: UploadFile, destination: Path):
    size = 0
    with open(destination, "wb") as buffer:
        while chunk := await upload_file.read(1024 * 1024):  # 1MB chunks
            if size + len(chunk) > settings.MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds limit of {settings.MAX_FILE_SIZE_BYTES // 1024 // 1024} MB"
                )
            buffer.write(chunk)
            size += len(chunk)

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions

# --- API Endpoints ---
@app.get("/")
async def get_index(request: Request):
    # MODIFICATION: Pass available models to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "whisper_models": sorted(list(settings.ALLOWED_WHISPER_MODELS))
    })

@app.post("/ocr-pdf", status_code=status.HTTP_202_ACCEPTED)
async def submit_pdf_ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_allowed_file(file.filename, settings.ALLOWED_PDF_EXTENSIONS):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PDF.")

    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{Path(safe_basename).suffix}"
    upload_path = settings.UPLOADS_DIR / unique_filename
    processed_path = settings.PROCESSED_DIR / unique_filename

    await save_upload_file_chunked(file, upload_path)
    
    job_data = JobCreate(id=job_id, task_type="ocr", original_filename=file.filename, input_filepath=str(upload_path), processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    
    run_pdf_ocr_task(new_job.id, str(upload_path), str(processed_path))
    return {"job_id": new_job.id, "status": new_job.status}

@app.post("/ocr-image", status_code=status.HTTP_202_ACCEPTED)
async def submit_image_ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_allowed_file(file.filename, settings.ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Please upload a PNG, JPG, or TIFF.")

    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    file_ext = Path(safe_basename).suffix
    unique_filename = f"{Path(safe_basename).stem}_{job_id}{file_ext}"
    upload_path = settings.UPLOADS_DIR / unique_filename
    processed_path = settings.PROCESSED_DIR / f"{Path(safe_basename).stem}_{job_id}.txt"

    await save_upload_file_chunked(file, upload_path)

    job_data = JobCreate(id=job_id, task_type="ocr-image", original_filename=file.filename, input_filepath=str(upload_path), processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)

    run_image_ocr_task(new_job.id, str(upload_path), str(processed_path))
    return {"job_id": new_job.id, "status": new_job.status}

# MODIFICATION: Endpoint now accepts `model_size` as form data.
@app.post("/transcribe-audio", status_code=status.HTTP_202_ACCEPTED)
async def submit_audio_transcription(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    db: Session = Depends(get_db)
):
    if not is_allowed_file(file.filename, settings.ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid audio file type.")

    # Validate the selected model size
    if model_size not in settings.ALLOWED_WHISPER_MODELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model size: {model_size}.")

    job_id = uuid.uuid4().hex
    safe_basename = secure_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix
    
    audio_filename = f"{stem}_{job_id}{suffix}"
    transcript_filename = f"{stem}_{job_id}.txt"
    upload_path = settings.UPLOADS_DIR / audio_filename
    processed_path = settings.PROCESSED_DIR / transcript_filename

    await save_upload_file_chunked(file, upload_path)
    
    job_data = JobCreate(id=job_id, task_type="transcription", original_filename=file.filename, input_filepath=str(upload_path), processed_filepath=str(processed_path))
    new_job = create_job(db=db, job=job_data)
    
    # Pass the selected model size to the background task
    run_transcription_task(new_job.id, str(upload_path), str(processed_path), model_size=model_size)
    return {"job_id": new_job.id, "status": new_job.status}

@app.post("/job/{job_id}/cancel", status_code=status.HTTP_200_OK)
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
    file_path = settings.PROCESSED_DIR / safe_filename
    
    if not file_path.resolve().is_relative_to(settings.PROCESSED_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(path=file_path, filename=safe_filename, media_type="application/octet-stream")