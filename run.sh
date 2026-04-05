#!/usr/bin/env bash

# Exit on error
set -e

echo "Starting MLOps Narrative Bias API..."

# Install dependencies if you want to ensure they exist (uncomment if using venv)
# pip install -r requirements.txt

# Start the uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
