# 🔍 Intelligent Surveillance System

AI-powered surveillance video processing with semantic search capabilities.

## ⚡ Quick Start

### Option 1: Docker (Recommended)
```bash
# Install Docker
sudo apt install docker.io docker-compose

# Start the system
./run.sh docker

# Access: http://localhost:5000
```

### Option 2: Manual Setup
```bash
# Setup everything
./run.sh full

# Start backend
./run.sh start

# Start frontend (in another terminal)
./run.sh frontend
```

## 🎯 Features

- **Video Upload**: Process surveillance videos
- **AI Detection**: YOLOv8 object detection
- **Semantic Search**: Natural language queries
- **Real-time Processing**: Background job queue

## 📡 API Endpoints

- **Upload**: `POST /api/data/upload/{project_id}`
- **Process**: `POST /api/surveillance/process/{project_id}/{file_id}`
- **Search**: `POST /api/surveillance/query`
- **Docs**: http://localhost:5000/docs

## 🛠️ Commands

```bash
./run.sh help      # Show all commands
./run.sh docker    # Start with Docker
./run.sh start     # Start backend only
./run.sh frontend  # Start Streamlit UI
./run.sh clean     # Clean cache files
```

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM
- Docker (optional but recommended)

## 🏗️ Architecture

```
Frontend (Streamlit) → Backend (FastAPI) → AI Models (YOLOv8, BLIP)
                              ↓
                         Redis Queue → Celery Workers
```

## 🔧 Development

1. Fork the repository
2. Make changes in `src/`
3. Test with `./run.sh start`
4. Submit PR

## 📝 License

MIT License - see LICENSE file for details.
