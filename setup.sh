#!/bin/bash

# Intelligent Surveillance System Setup Script
# This script sets up the complete project structure and dependencies

set -e  # Exit on any error

echo "🚀 Setting up Intelligent Surveillance System..."

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/chromadb
mkdir -p data/logs
mkdir -p assets/files
mkdir -p models
mkdir -p tests
mkdir -p scripts

# Create __init__.py files for Python packages
echo "📦 Creating Python package structure..."
touch services/__init__.py
touch models/__init__.py
touch routes/__init__.py
touch controllers/__init__.py
touch helpers/__init__.py

# Set up environment variables
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "ℹ️ .env file already exists"
fi

# Create logging configuration
echo "📋 Creating logging configuration..."
cat > logging.conf << 'EOF'
[loggers]
keys=root,surveillance

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_surveillance]
level=INFO
handlers=consoleHandler,fileHandler
qualname=surveillance
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=detailedFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=detailedFormatter
args=('data/logs/surveillance.log',)

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
EOF

# Create Docker configuration
echo "🐳 Creating Docker configuration..."
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p data/chromadb data/logs assets/files

# Expose port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
EOF

# Create docker-compose for development
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  surveillance-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./assets:/app/assets
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  celery-worker:
    build: .
    command: celery -A services.job_queue worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./assets:/app/assets
    depends_on:
      - redis
    restart: unless-stopped

volumes:
  redis_data:
EOF

# Create development startup script
echo "🔧 Creating development scripts..."
cat > scripts/start_dev.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Intelligent Surveillance System in development mode..."

# Start Redis in background if not running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "📦 Starting Redis server..."
    redis-server --daemonize yes
fi

# Start Celery worker in background
echo "👷 Starting Celery worker..."
celery -A services.job_queue worker --loglevel=info --detach

# Start the FastAPI server
echo "🌐 Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 5000
EOF

chmod +x scripts/start_dev.sh

# Create production startup script
cat > scripts/start_prod.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Intelligent Surveillance System in production mode..."

# Start Celery worker
celery -A services.job_queue worker --loglevel=info --detach

# Start the application with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
EOF

chmod +x scripts/start_prod.sh

# Create database migration script
cat > scripts/migrate_db.py << 'EOF'
#!/usr/bin/env python3
"""
Database migration script for Intelligent Surveillance System
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.database import db_manager

def main():
    print("🗄️ Creating/updating database tables...")
    
    try:
        db_manager.create_tables()
        print("✅ Database tables created/updated successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/migrate_db.py

# Create AI model download script
cat > scripts/download_models.py << 'EOF'
#!/usr/bin/env python3
"""
Download AI models for Intelligent Surveillance System
"""

import os
import sys
from pathlib import Path

def download_yolo_model():
    """Download YOLOv8 model"""
    try:
        from ultralytics import YOLO
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "yolov8n.pt"
        
        if not model_path.exists():
            print("📥 Downloading YOLOv8 model...")
            model = YOLO("yolov8n.pt")  # This will download the model
            
            # Move to models directory
            import shutil
            shutil.move("yolov8n.pt", model_path)
            print(f"✅ YOLOv8 model saved to {model_path}")
        else:
            print("ℹ️ YOLOv8 model already exists")
            
    except ImportError:
        print("⚠️ ultralytics not installed. Install requirements first.")
    except Exception as e:
        print(f"❌ Error downloading YOLOv8 model: {e}")

def download_blip_model():
    """Download BLIP model (will be cached by transformers)"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("📥 Downloading BLIP model (this may take a while)...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ BLIP model downloaded and cached")
        
    except ImportError:
        print("⚠️ transformers not installed. Install requirements first.")
    except Exception as e:
        print(f"❌ Error downloading BLIP model: {e}")

def main():
    print("🤖 Downloading AI models for Intelligent Surveillance System...")
    
    download_yolo_model()
    download_blip_model()
    
    print("🎉 Model download complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/download_models.py

# Create test configuration
echo "🧪 Creating test configuration..."
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
asyncio_mode = auto
EOF

# Create basic test structure
mkdir -p tests/unit tests/integration
cat > tests/__init__.py << 'EOF'
# Test package
EOF

cat > tests/conftest.py << 'EOF'
import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
EOF

# Create a basic API test
cat > tests/integration/test_api.py << 'EOF'
def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"

def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_welcome_endpoint(client):
    """Test the welcome endpoint"""
    response = client.get("/api/welcome")
    assert response.status_code == 200
EOF

# Create gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.pytest_cache/
htmlcov/
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache

# Environment variables
.env
.env.local
.env.*.local

# Data directories
data/
assets/files/
!assets/files/.gitkeep

# Models (large files)
models/*.pt
models/*.pth
models/*.bin

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
EOF

# Create placeholder files
echo "📄 Creating placeholder files..."
touch assets/files/.gitkeep
touch data/.gitkeep

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Set up environment: edit .env file with your settings"
echo "3. Initialize database: python scripts/migrate_db.py"
echo "4. Download AI models: python scripts/download_models.py"
echo "5. Start development: ./scripts/start_dev.sh"
echo ""
echo "🚀 Your Intelligent Surveillance System is ready for development!"
EOF
