#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
source .env

# Set environment variables for the application
#export LOCAL_ONLY=True
export SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 16)}
export UPLOADS_DIR=${UPLOADS_DIR:-./uploads}
export PROCESSED_DIR=${PROCESSED_DIR:-./processed}
export CHUNK_TMP_DIR=${CHUNK_TMP_DIR:-./uploads/tmp}

echo "Starting application with:"
echo "  UPLOADS_DIR=$UPLOADS_DIR"
echo "  PROCESSED_DIR=$PROCESSED_DIR"
echo "  CHUNK_TMP_DIR=$CHUNK_TMP_DIR"

# Start Gunicorn in the background with increased timeout
gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker --forwarded-allow-ips='*' --timeout 120 main:app -b 0.0.0.0:8000 &
echo "Started Gunicorn..."
# Store the Gunicorn process ID
GUNICORN_PID=$!
echo "Gunicorn PID: $GUNICORN_PID"
# Start the Huey consumer in the foreground with the same env
exec env LOCAL_ONLY=$LOCAL_ONLY SECRET_KEY=$SECRET_KEY UPLOADS_DIR=$UPLOADS_DIR PROCESSED_DIR=$PROCESSED_DIR CHUNK_TMP_DIR=$CHUNK_TMP_DIR huey_consumer.py main.huey -w 4
echo "Started Huey consumer..."
