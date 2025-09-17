# File Wizard

File Wizard is a self-hosted, browser-based utility for file conversion, OCR, and audio transcription. It features a modern drag-and-drop interface and a powerful, asynchronous backend to handle long-running tasks without tying up the browser.

## Features

  * **Versatile File Conversion:** Easily convert between various file formats. The system is designed to be extended with any command-line tool (like FFmpeg, ImageMagick, etc.) via a simple `settings.yml` configuration file.
  * **High-Quality OCR:** Perform Optical Character Recognition (OCR) on PDFs and images to extract text.
  * **Accurate Audio Transcription:** Transcribe audio files into text using high-performance Whisper models.
  * **Modern UI & UX:**
      * Clean, responsive, dark-themed interface.
      * Drag-and-drop support for single or multiple files anywhere on the screen.
      * Traditional multi-file selection buttons.
      * A dialog to choose the desired action (Convert, OCR, Transcribe) for dropped files.
  * **Real-time Updates & History:**
      * Jobs are processed in the background, with the UI updating statuses in real-time via polling.
      * A persistent job history table displays file names, tasks, submission times, file sizes (input → output), and final status.
      * Timestamps are automatically displayed in the user's local timezone.
      * Download links are provided for successfully processed files.
      * Pending or processing jobs can be cancelled.
  * **Easy Configuration:**
      * All tools, models, and application settings are managed in a human-readable `settings.yml` file.
      * A dedicated `/settings` page allows for viewing and editing the configuration directly from the UI.

-----

## Tech Stack 

  * **Backend:** FastAPI (Python)
  * **Frontend:** Vanilla HTML, CSS, JavaScript (no framework)
  * **Task Queue:** Huey (with a SQLite backend)
  * **Database:** SQLAlchemy (with a SQLite database)
  * **Configuration:** YAML

-----

## Installation & Setup 

To get File Wizard running, you need to run the web server and the background task processor separately.

### With Docker
Clone the Repo and make sure docker compose is working on your environment
```bash
git clone https://github.com/LoredCast/filewizard.git
cd filewizard
```

Startup Docker, initially the settings.yml file is applied, you can edit it.
For You can edit the .env file for further configuration.

```bash
docker compose up -d
```

### Manually


**1. Clone the Repository**

```bash
git clone https://github.com/LoredCast/filewizard.git
cd filewizard
```

**2. Create a Virtual Environment & Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Make sure you have a requirements.txt file with all dependencies
pip install -r requirements.txt
```

*(**Note:** Dependencies include `fastapi`, `uvicorn`, `sqlalchemy`, `huey`, `faster-whisper`, `ocrmypdf`, `pytesseract`, `python-multipart`, `pyyaml`, etc.)*

**3. Configure the Application**
Copy the default settings file and customize it to your needs. This is where you define your conversion tools and other parameters.


**4. Run the Web Server**
This command starts the Webserver and Huey.

```bash
chmod +x run.sh
./run.sh
```


-----

## Usage 🖱️

1.  Open your browser to `http://127.0.0.1:8000`.
2.  **Drag and drop** any file or multiple files onto the page.
3.  A dialog will appear asking you to choose an action: **Convert**, **OCR**, or **Transcribe**.
4.  Alternatively, use the "Choose File" buttons in any of the three sections.
5.  Your job will appear in the "History" table, and its status will update automatically.