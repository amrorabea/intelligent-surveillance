# 🔍 Intelligent Surveillance System

A comprehensive AI-powered surveillance platform that processes video footage to enable semantic understanding and natural language querying of visual events. This full-stack solution combines advanced computer vision, natural language processing, and modern web technologies.

## 🚀 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │────│   FastAPI Backend │────│  AI Processing  │
│                 │    │                  │    │                 │
│ • Video Upload  │    │ • REST APIs      │    │ • YOLOv8        │
│ • Search Query  │    │ • Authentication │    │ • BLIP Captions │
│ • Analytics     │    │ • Job Queue      │    │ • Object Track  │
│ • Frame Preview │    │ • Data Control   │    │ • Vector Search │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │                 │
                    │ • ChromaDB      │
                    │ • Redis Queue   │
                    │ • File Storage  │
                    └─────────────────┘
```

## ✨ Key Features

### 🎬 **Video Processing Pipeline**
- 🔍 **YOLOv8 Object Detection**: Real-time detection of people, vehicles, objects with confidence scoring
- 🔁 **Multi-Object Tracking**: Maintains object identity across video frames
- 📝 **AI Scene Captioning (BLIP)**: Generates detailed natural language descriptions of scenes
- 🖼️ **Smart Frame Extraction**: Intelligently selects and saves key frames for analysis
- ⚡ **Asynchronous Processing**: Background job processing with Celery and Redis

### 🔍 **Intelligent Search System**
- 🧠 **Semantic Search**: Natural language queries over processed footage using vector embeddings
- 🎯 **Advanced Filtering**: Filter by date range, object types, confidence thresholds, projects
- 📊 **Search Analytics**: Track search patterns and popular queries
- 🖼️ **Frame Preview**: Direct access to relevant video frames from search results

### 📊 **Analytics Dashboard**
- 📈 **Real-time Metrics**: Videos processed, objects detected, system performance
- 📉 **Trend Analysis**: Detection patterns, confidence distributions, activity timelines
- 🎯 **Object Statistics**: Detailed breakdown of detected object types and frequencies
- � **AI Insights**: Automatically generated observations and system alerts

### 🌐 **Modern Web Interface**
- 🎨 **Streamlit Frontend**: Interactive multi-page web application
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- 🔄 **Real-time Updates**: Auto-refresh capabilities and live status monitoring
- 🛠️ **Debug Tools**: System status, endpoint testing, and troubleshooting utilities

### ⚙️ **Robust Backend Infrastructure**
- 🚀 **FastAPI Framework**: High-performance async REST API with automatic documentation
- 🔐 **Authentication System**: JWT tokens and API key support with role-based access
- 🔄 **Job Queue Management**: Distributed task processing with status tracking
- 🗄️ **Vector Database**: ChromaDB integration for high-speed similarity search
- 📁 **Project Management**: Multi-tenant architecture with isolated project spaces

## 🛠️ Technology Stack

### **Backend**
- **FastAPI**: Modern async web framework with automatic API documentation
- **Celery**: Distributed task queue for background video processing
- **Redis**: Message broker and caching layer
- **ChromaDB**: Vector database for semantic search capabilities
- **PyTorch**: Deep learning framework for AI models

### **AI & Computer Vision**
- **YOLOv8**: State-of-the-art object detection model
- **BLIP**: AI model for generating image captions
- **OpenCV**: Computer vision library for video processing
- **Sentence Transformers**: For semantic text embeddings

### **Frontend**
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization and charting
- **Pandas**: Data manipulation and analysis

### **Infrastructure**
- **Docker**: Containerization support
- **SQLite**: Lightweight database for metadata
- **JWT**: Secure authentication tokens

## 📋 Requirements

- **Python**: 3.10 or later
- **Memory**: 8GB+ RAM recommended for AI models
- **Storage**: SSD recommended for video processing
- **GPU**: Optional but recommended for faster AI inference

## 🚀 Quick Start

### **Method 1: Using Make (Recommended)**

```bash
# Clone the repository
git clone https://github.com/yourusername/intelligent-surveillance.git
cd intelligent-surveillance

# Setup everything (creates virtual environment, installs dependencies, starts services)
make fullstack

# Or start individual components
make backend          # Start FastAPI + Celery + Redis
make frontend         # Start Streamlit interface
make status-fullstack # Check all services status
```

### **Method 2: Manual Installation**

1. **Clone and Setup**
```bash
git clone https://github.com/yourusername/intelligent-surveillance.git
cd intelligent-surveillance
```

2. **Create Virtual Environment**
```bash
python -m venv venv_surveillance
source venv_surveillance/bin/activate  # Linux/Mac
# OR
venv_surveillance\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r src/requirements.txt
pip install -r streamlit/requirements.txt
```

4. **Configure Environment**
```bash
cp src/.env.example src/.env
# Edit src/.env with your configuration
```

5. **Start Services**
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start FastAPI Backend
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 5000

# Terminal 3: Start Celery Worker
cd src && celery -A services.job_queue worker --loglevel=info

# Terminal 4: Start Streamlit Frontend
cd streamlit && streamlit run app.py --server.port 8501
```

6. **Access the Application**
- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:5000/docs
- **API Health Check**: http://localhost:5000/api/surveillance/health

## 🎯 Usage Guide

### **1. Video Processing**
1. Navigate to **📤 Video Processing** page
2. Select a project ID or create new one
3. Upload video file (MP4, AVI, MOV formats supported)
4. Configure processing options:
   - **Sample Rate**: Frame extraction frequency
   - **Detection Threshold**: Minimum confidence for object detection
   - **Enable Tracking**: Multi-object tracking across frames
   - **Enable Captioning**: AI-generated scene descriptions
5. Monitor processing progress in real-time
6. View completed results and extracted frames

### **2. Semantic Search**
1. Go to **🔍 Semantic Search** page
2. Enter natural language queries like:
   - `"person walking with a dog"`
   - `"red car in parking lot"`
   - `"people gathering around table"`
3. Apply filters:
   - **Time Range**: Filter by date/time periods
   - **Confidence Threshold**: Minimum similarity score
   - **Object Types**: Filter by specific detected objects
   - **Project**: Search within specific projects
4. View results with frame previews and metadata
5. Export search results as JSON or CSV

### **3. Analytics Dashboard**
1. Visit **📊 Analytics** page
2. Select analysis time period
3. View comprehensive metrics:
   - **Key Performance**: Videos processed, objects detected
   - **Object Distribution**: Types and frequencies of detections
   - **Confidence Analysis**: Quality metrics for AI predictions
   - **Activity Timeline**: Detection patterns over time
   - **System Performance**: Processing speeds and efficiency
4. Export analytics data for external analysis

### **4. System Monitoring**
1. Check **⚙️ System Status** page
2. Monitor service health:
   - **API Status**: Backend connectivity and response times
   - **Database Health**: ChromaDB and Redis status
   - **Processing Queue**: Active and pending jobs
   - **Storage Usage**: File system and database metrics
3. Use debug tools for troubleshooting
4. View system logs and error diagnostics

## 🔗 API Endpoints

### **Core Surveillance APIs**
- `POST /api/data/upload/{project_id}` - Upload video/image files
- `POST /api/surveillance/process/{project_id}/{file_id}` - Start AI processing pipeline
- `GET /api/surveillance/query` - Semantic search with natural language
- `GET /api/surveillance/jobs/{project_id}` - List processing jobs and status

### **Frame & Media APIs**
- `GET /api/surveillance/frame/{result_id}` - Get specific frame from search results
- `GET /api/surveillance/frame-direct/{result_id}` - Direct frame access
- `GET /api/surveillance/projects/{project_id}/frame` - Project-specific frame access

### **Analytics & Monitoring**
- `GET /api/surveillance/analytics` - Comprehensive system analytics
- `GET /api/surveillance/analytics/summary/{project_id}` - Project-specific analytics
- `GET /api/surveillance/health` - System health and status checks
- `GET /api/surveillance/stats` - Database and performance statistics

### **Authentication**
- All endpoints support JWT token authentication
- Development mode supports `Bearer dev` token for testing
- API key authentication available for programmatic access

## 💡 Example Queries
### **Natural Language Search Queries**
- `"person walking with a dog in the park"`
- `"red car parked near the building entrance"`
- `"people gathering around a table for meeting"`
- `"person carrying a backpack entering the hallway"`
- `"vehicle stopping at the security checkpoint"`
- `"anyone wearing red clothing after midnight"`
- `"person running or moving quickly through the area"`
- `"bicycle or motorcycle in the parking zone"`

### **API Usage Examples**

**Upload and Process Video:**
```bash
# Upload video file
curl -X POST "http://localhost:5000/api/data/upload/my_project" \
  -H "Authorization: Bearer dev" \
  -F "file=@surveillance_video.mp4"

# Start processing
curl -X POST "http://localhost:5000/api/surveillance/process/my_project/file_123" \
  -H "Authorization: Bearer dev" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_rate": 1.0,
    "detection_threshold": 0.5,
    "enable_tracking": true,
    "enable_captioning": true
  }'
```

**Search Processed Footage:**
```bash
# Semantic search
curl -X GET "http://localhost:5000/api/surveillance/query" \
  -H "Authorization: Bearer dev" \
  -G -d "query=person with backpack" \
  -d "max_results=10" \
  -d "confidence_threshold=0.3"
```

**Get Analytics:**
```bash
# System analytics
curl -X GET "http://localhost:5000/api/surveillance/analytics" \
  -H "Authorization: Bearer dev"

# Project-specific analytics
curl -X GET "http://localhost:5000/api/surveillance/analytics/summary/my_project" \
  -H "Authorization: Bearer dev"
```

## 🔧 Configuration

### **Environment Variables (.env)**
```bash
# Database Configuration
DATABASE_URL=sqlite:///./surveillance.db
CHROMA_DB_PATH=./assets/files/chromadb

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your-secret-key-here
DISABLE_AUTH=false

# AI Model Configuration
YOLO_MODEL_PATH=./models/yolov8n.pt
BLIP_MODEL_NAME=Salesforce/blip-image-captioning-base

# File Storage
PROJECT_FILES_DIR=./assets/files
MAX_FILE_SIZE_MB=500

# Processing Configuration
DEFAULT_SAMPLE_RATE=1.0
DEFAULT_DETECTION_THRESHOLD=0.5
ENABLE_GPU=true
```

## 🚀 Deployment Options

### **Local Development**
- Use the provided Makefile for easy setup
- All services run locally with file-based storage
- Perfect for development and testing

### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Scale Celery workers
docker-compose up --scale celery_worker=3
```

### **Cloud Deployment**
- **Frontend**: Deploy Streamlit app to Streamlit Community Cloud
- **Backend**: Deploy FastAPI to Railway, Render, or AWS
- **Database**: Use managed Redis and vector database services
- **Storage**: AWS S3 or similar for video file storage

### **Production Considerations**
- Use PostgreSQL instead of SQLite for multi-user scenarios
- Implement proper authentication and rate limiting
- Set up monitoring and logging (e.g., Prometheus, Grafana)
- Configure load balancing for high-traffic scenarios
- Use CDN for serving video frames and static assets

## 📊 Performance Metrics

### **Processing Capabilities**
- **Video Processing**: 12-15 FPS on CPU, 30+ FPS with GPU
- **Object Detection**: Sub-second inference per frame
- **Search Response**: <500ms for typical queries
- **Concurrent Users**: 10+ simultaneous video processing jobs

### **Scalability**
- **Horizontal Scaling**: Add more Celery workers for increased throughput
- **Storage**: Handles TBs of video data with proper storage backend
- **Search Index**: Millions of indexed frames with sub-second search

## 🛠️ Development

### **Project Structure**
```
intelligent-surveillance/
├── src/                    # Backend source code
│   ├── controllers/        # Business logic controllers
│   ├── models/            # Data models and schemas
│   ├── routes/            # API route definitions
│   ├── services/          # External services (auth, queue)
│   └── helpers/           # Utility functions
├── streamlit/             # Frontend web application
│   ├── pages/             # Multi-page Streamlit app
│   └── utils/             # Frontend utilities
├── tests/                 # Test suites
├── docs/                  # Documentation
├── scripts/               # Setup and utility scripts
└── Makefile              # Development automation
```

### **Adding New Features**
1. **Backend**: Add new controllers in `src/controllers/`
2. **API**: Define routes in `src/routes/`
3. **Frontend**: Create new pages in `streamlit/pages/`
4. **Models**: Add AI models in `src/models/`
5. **Tests**: Write tests in `tests/`

### **Testing**
```bash
# Run backend tests
make test

# Run specific test modules
python -m pytest tests/test_vision_controller.py

# Test API endpoints
make test-api
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms of the MIT license.
