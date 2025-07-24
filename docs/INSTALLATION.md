# 🚀 Installation Guide

This guide will walk you through setting up the Intelligent Surveillance System on your local machine or server.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+ with WSL2
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: 10GB+ free space for models and data
- **GPU**: Optional but recommended (CUDA-compatible for AI acceleration)

### Dependencies
- Docker and Docker Compose (recommended)
- Redis server
- Git
- curl

## 🔧 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/amrorabea/intelligent-surveillance
cd intelligent-surveillance

# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh

# Start all services
make fullstack
```

### Option 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/amrorabea/intelligent-surveillance
cd intelligent-surveillance
```

#### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt
```

#### Step 3: Install Redis
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

#### Step 4: Download AI Models
```bash
# Run model setup script
chmod +x scripts/setup_ai_models.sh
./scripts/setup_ai_models.sh
```

#### Step 5: Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration (optional)
nano .env
```

## 🐳 Docker Installation

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+

### Setup
```bash
# Clone repository
git clone https://github.com/amrorabea/intelligent-surveillance
cd intelligent-surveillance

# Build and start services
docker-compose up -d

# Verify installation
docker-compose ps
```

## 🚀 Starting the System

### Development Mode
```bash
# Start all services
make fullstack

# Or start individual components
make api          # FastAPI backend
make worker       # Celery worker
make redis        # Redis server
make frontend     # Streamlit UI
```

### Production Mode
```bash
# Start with production settings
make production

# Or use Docker
docker-compose -f docker-compose.prod.yml up -d
```

## ✅ Verification

### Health Check
```bash
# Check system status
make status-fullstack

# Test API endpoint
curl http://localhost:8000/health

# Test frontend
curl http://localhost:8501
```

### Test Video Processing
```bash
# Run demo
./demo.sh

# Or upload a test video via the web interface
# Navigate to http://localhost:8501
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=8501

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI Models
YOLO_MODEL_PATH=src/models/yolov8n.pt
BLIP_MODEL_NAME=Salesforce/blip-image-captioning-base

# Storage
UPLOAD_DIR=uploads
PROJECT_FILES_DIR=project_files
VECTOR_DB_PATH=vector_db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Model Configuration

The system automatically downloads required models:

- **YOLOv8**: Object detection model (~6MB)
- **BLIP**: Image captioning model (~990MB)
- **ChromaDB**: Vector database for embeddings

## 🛠️ Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find and kill process using port 8000
sudo lsof -t -i:8000 | xargs kill -9

# Or use different ports in .env file
```

#### Permission Errors
```bash
# Fix file permissions
chmod +x fix_permissions.sh
./fix_permissions.sh
```

#### Redis Connection Issues
```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis-server
```

#### GPU/CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
# Follow: https://developer.nvidia.com/cuda-toolkit
```

### Model Download Issues
```bash
# Manually download models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8 downloaded successfully')
"

# Clear model cache if corrupted
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
```

### Memory Issues
```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory usage
htop
```

## 📱 Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev python3-pip redis-server \
    build-essential libssl-dev libffi-dev

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python redis docker
```

### Windows (WSL2)
```bash
# Install WSL2 and Ubuntu
wsl --install

# Inside WSL2, follow Ubuntu instructions
```

## 🔄 Updates

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r src/requirements.txt --upgrade

# Restart services
make restart
```

### Model Updates
```bash
# Update AI models
./scripts/setup_ai_models.sh --force

# Clear cache and restart
make clean && make fullstack
```

## 🚀 Next Steps

After successful installation:

1. **Read the [Quick Start Guide](QUICK_START.md)** for your first video processing workflow
2. **Explore the [API Reference](API_REFERENCE.md)** to understand available endpoints
3. **Check out [Development Guide](DEVELOPMENT.md)** if you plan to contribute
4. **Review [Security Guide](SECURITY.md)** for production deployment

## 📞 Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search existing [GitHub Issues](https://github.com/your-repo/intelligent-surveillance/issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error messages and logs
   - Steps to reproduce

---

**🎉 Welcome to the Intelligent Surveillance System!**
