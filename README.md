# File Wizard

File Wizard is a self-hosted, browser-based utility for file conversion, OCR, and audio transcription. It wraps many cli and python converters aswell as fast-whisper and tesseract ocr.

![Screenshot](swappy-20250920_155526.png)

## Features
  *  Convert between various file formats. The system is designed to be extended with any command-line tool (like FFmpeg, ImageMagick, etc.) via a simple `settings.yml` configuration file.
  * **OCR:** Perform Optical Character Recognition on PDFs and images to extract text.
  * **Audio Transcription:** Transcribe audio files into text using Whisper models.
  * **UI:**
      * Clean, responsive, dark-themed interface (hopefully goodlooking idk)
      * Drag-and-drop support for single or multiple files anywhere on the screen.
      * Traditional multi-file selection buttons.
      * A dialog to choose the desired action (Convert, OCR, Transcribe) for dropped files.
  * **Real-time Updates & History:**
      * Jobs are processed in the background, with the UI updating statuses in real-time via polling.
      * A persistent job history table displays file names, tasks, submission times, file sizes (input → output), and final status.
     
  * **Configuration:**
[See the Wiki](https://github.com/LoredCast/filewizard/wiki)

      * A dedicated `/settings` page.
      * OAuth needs to be configured in the `config/settings.yml` file, you can see the default for a reference. By default, it runs without auth in local mode.
      * Currently it only supports cpu operations, but a future image will include the cuda drivers for running whisper on gpu (torch and cuda is large and I didn't want to inflate the image even more)

-----

## NOTE
Run at your own risk! This app is highly vulnerable to arbitrary code executing if left public and without auth. I'm no security expert this app is intended for local use or usage with an OAuth oidc provider!

## Tech Stack:
FastAPI for the Server, Vanilla html, js, css for the frontend (might switch to svelte in the future, kept it light for this), Huey for task queuing and SQlite for any Databse.

-----

## Installation & Setup 

### With Docker (Local Build)
Clone the Repo and make sure docker compose is working on your environment

```bash
git clone https://github.com/LoredCast/filewizard.git
cd filewizard
```

Startup Docker, initially the settings.yml file is applied, you can edit it.
For You can edit the .env file for further configuration.

Note: Building this image will take some time installing all deps ((mostly texlive)).

```bash
docker compose up --build
```


### With Docker (From Dockerhub) [Recommended]
I've published 4 images to dockerhub that you can pull:

- `loredcast/filewizard:0.2`
- `loredcast/filewizard:0.2-small`

and the previous images without tts:
- `loredcast/filewizard:latest`
- `loredcast/filewizard:small`

The smaller one has Tex and many large tools left out.

Copy the `docker-compose.yml` from the repo into a directory on your machine, adjust the file to your needs and startup docker.

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

## Usage 

1.  Open your browser to `http://127.0.0.1:8000`.
2.  **Drag and drop** any file or multiple files onto the page.
3.  A dialog will appear asking you to choose an action: **Convert**, **OCR**, or **Transcribe**.
4.  Alternatively, use the "Choose File" buttons in any of the three sections.
5.  Your job will appear in the "History" table, and its status will update automatically.

