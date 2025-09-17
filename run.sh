#!/bin/bash
# This script starts the FastAPI application using Gunicorn.

echo "Starting DocProcessor with Gunicorn on port 0.0.0.0:8000..."

exec gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker --forwarded-allow-ips='*' main:app -b 0.0.0.0:8000 &
echo "Done"
echo "Starting huey..."
exec huey_consumer.py main.huey -w 4
echo "Done"
