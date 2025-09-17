#!/bin/bash
# This script starts the FastAPI application using Gunicorn.

echo "Starting DocProcessor with Gunicorn..."

exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker --forwarded-allow-ips='*' main:app -b 0.0.0.0:8000 &
echo "Done"
echo "Starting huey..."
exec huey_consumer.py main.huey -w 2 &
echo "Done"
