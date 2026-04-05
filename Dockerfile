# Use an official Python runtime as a parent image, slim version for smaller footprint
FROM python:3.11-slim

# Set environment variables:
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache

# Set work directory
WORKDIR /app

# Install system dependencies required for PyTorch and Transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep the image size small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace model cache during the build step.
# This prevents the 1.6GB model from downloading on every container start.
# Note: facebook/bart-large-mnli is hardcoded here to match the config default.
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Copy the application code
COPY app/ ./app/

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' apiapp && \
    chown -R apiapp:apiapp /app
USER apiapp

# Expose port
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
