# Simple Dockerfile for Intelligent Surveillance System
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Create directories and download models
RUN mkdir -p src/data src/models src/assets/files && \
    curl -L -o src/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt || \
    echo "Warning: Could not download YOLOv8 model. It will be downloaded on first use."

# Expose port
EXPOSE 5000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]
