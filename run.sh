#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
source .env

LOCAL_ONLY=True
SECRET_KEY=
UPLOADS_DIR=./uploads
PROCESSED_DIR=./processed
# Start Gunicorn in the background
gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker --forwarded-allow-ips='*' --error-logfile - --access-logfile - main:app -b 0.0.0.0:8000 &
echo "Started Gunicorn..."
# Store the Gunicorn process ID
GUNICORN_PID=$!
echo "Gunicorn PID: $GUNICORN_PID"
# Start the Huey consumer in the foreground
exec huey_consumer.py main.huey -w 4
echo "Started Huey consumer..."
