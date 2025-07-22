# 💻 Development Guide

This guide helps developers set up, understand, and contribute to the Intelligent Surveillance System.

## 🚀 Development Environment Setup

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Git 2.25+
- Docker & Docker Compose
- VS Code (recommended) or PyCharm
- Node.js 16+ (for frontend development)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/your-repo/intelligent-surveillance.git
cd intelligent-surveillance

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt
pip install -r dev-requirements.txt

# Install pre-commit hooks
pre-commit install

# Setup AI models
./scripts/setup_ai_models.sh

# Start development services
make dev
```

## 🏗️ Project Structure

```
intelligent-surveillance/
├── 📁 src/                     # Main application source
│   ├── 📁 controllers/         # Business logic controllers
│   ├── 📁 models/             # Data models and schemas
│   ├── 📁 routes/             # API route definitions
│   ├── 📁 services/           # Background services (Celery)
│   ├── 📁 helpers/            # Utility functions and config
│   └── main.py                # FastAPI application entry
├── 📁 streamlit/              # Frontend application
│   ├── 📁 pages/              # Streamlit pages
│   ├── 📁 utils/              # Frontend utilities
│   └── app.py                 # Main Streamlit app
├── 📁 tests/                  # Test suites
├── 📁 docs/                   # Documentation
├── 📁 scripts/                # Setup and utility scripts
├── 📁 monitoring/             # Monitoring configs
└── 📄 Makefile               # Development commands
```

### Key Components

#### Controllers (`src/controllers/`)
- **VisionController**: AI model management (YOLOv8, BLIP)
- **TrackingController**: Object tracking across frames
- **VectorDBController**: Semantic search and embeddings
- **DataController**: File and data management
- **ProcessController**: Video processing pipeline
- **ProjectController**: Project management
- **QueryController**: Search and analytics

#### Routes (`src/routes/`)
- **surveillance.py**: Main API endpoints
- **data.py**: Data management endpoints
- **base.py**: Health checks and system info

#### Services (`src/services/`)
- **job_queue.py**: Celery task definitions
- **auth.py**: Authentication (when enabled)

## 🛠️ Development Workflow

### Code Style & Standards
We follow Python PEP 8 and use these tools:

```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_vision_controller.py

# Run with live logging
pytest -s --log-cli-level=INFO

# Performance testing
pytest tests/performance/ --benchmark-only
```

### Development Commands
```bash
# Start development environment
make dev                    # Start all services in dev mode
make api                   # Start API only
make worker                # Start Celery worker only
make frontend              # Start Streamlit frontend only
make redis                 # Start Redis only

# Development utilities
make test                  # Run test suite
make lint                  # Run linting
make format                # Format code
make clean                 # Clean temporary files
make logs                  # View service logs

# Database operations
make reset-db              # Reset vector database
make backup-db             # Backup current data
make restore-db            # Restore from backup
```

## 🧪 Testing Strategy

### Test Structure
```
tests/
├── unit/                  # Unit tests for individual components
│   ├── controllers/       # Controller unit tests
│   ├── models/           # Model unit tests
│   └── services/         # Service unit tests
├── integration/          # Integration tests
│   ├── api/              # API endpoint tests
│   └── pipeline/         # End-to-end pipeline tests
├── performance/          # Performance and load tests
└── fixtures/             # Test data and fixtures
    ├── videos/           # Sample video files
    └── images/           # Sample image files
```

### Writing Tests

#### Unit Test Example
```python
# tests/unit/controllers/test_vision_controller.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.controllers.VisionController import VisionController

class TestVisionController:
    @pytest.fixture
    def controller(self):
        return VisionController()
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detect_objects_success(self, controller, sample_image):
        """Test successful object detection"""
        detections = controller.detect_objects(sample_image)
        
        assert isinstance(detections, list)
        for detection in detections:
            assert 'class_name' in detection
            assert 'confidence' in detection
            assert 'bbox' in detection
            assert detection['confidence'] >= 0.0
            assert detection['confidence'] <= 1.0
    
    @patch('src.controllers.VisionController.YOLO')
    def test_model_loading_failure(self, mock_yolo, controller):
        """Test handling of model loading failure"""
        mock_yolo.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception) as exc_info:
            controller._setup_yolo_model()
        
        assert "Model loading failed" in str(exc_info.value)
```

#### Integration Test Example
```python
# tests/integration/api/test_surveillance_endpoints.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

class TestSurveillanceAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_upload_video_success(self, client, sample_video_file):
        """Test successful video upload"""
        response = client.post(
            "/surveillance/upload",
            files={"file": ("test.mp4", sample_video_file, "video/mp4")},
            data={"project_id": "test-project"}
        )
        assert response.status_code == 200
        assert "job_id" in response.json()
    
    def test_search_endpoint(self, client):
        """Test semantic search endpoint"""
        response = client.post(
            "/surveillance/search",
            json={
                "query": "person walking",
                "project_id": "test-project",
                "limit": 5
            }
        )
        assert response.status_code == 200
        assert "results" in response.json()
```

#### Performance Test Example
```python
# tests/performance/test_processing_performance.py
import pytest
import time
from src.controllers.VisionController import VisionController

class TestProcessingPerformance:
    @pytest.fixture
    def controller(self):
        return VisionController()
    
    def test_detection_speed(self, controller, sample_image, benchmark):
        """Benchmark object detection speed"""
        result = benchmark(controller.detect_objects, sample_image)
        assert isinstance(result, list)
    
    def test_batch_processing_performance(self, controller):
        """Test batch processing performance"""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                 for _ in range(10)]
        
        start_time = time.time()
        results = [controller.detect_objects(img) for img in images]
        end_time = time.time()
        
        processing_time = end_time - start_time
        fps = len(images) / processing_time
        
        assert fps > 5.0  # At least 5 FPS
        assert len(results) == len(images)
```

### Test Configuration
```python
# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_video_file(test_data_dir):
    """Sample video file for testing"""
    video_path = test_data_dir / "videos" / "sample.mp4"
    if not video_path.exists():
        pytest.skip("Sample video file not found")
    return open(video_path, "rb")

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        "TESTING": True,
        "REDIS_URL": "redis://localhost:6379/15",  # Use test database
        "VECTOR_DB_PATH": "./test_vector_db",
        "UPLOAD_DIR": "./test_uploads"
    }
```

## 🔌 API Development

### Adding New Endpoints

#### 1. Define Route
```python
# src/routes/new_feature.py
from fastapi import APIRouter, Depends, HTTPException
from src.models.schemas import NewFeatureRequest, NewFeatureResponse
from src.controllers.NewFeatureController import NewFeatureController

router = APIRouter(prefix="/new-feature", tags=["new-feature"])

@router.post("/process", response_model=NewFeatureResponse)
async def process_new_feature(
    request: NewFeatureRequest,
    controller: NewFeatureController = Depends()
):
    """Process new feature request"""
    try:
        result = await controller.process(request)
        return NewFeatureResponse(
            success=True,
            result=result,
            message="Processing completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2. Define Models
```python
# src/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class NewFeatureRequest(BaseModel):
    """Request model for new feature"""
    input_data: str = Field(..., description="Input data for processing")
    options: Optional[dict] = Field(default=None, description="Processing options")
    
    class Config:
        schema_extra = {
            "example": {
                "input_data": "sample input",
                "options": {"setting1": "value1"}
            }
        }

class NewFeatureResponse(BaseModel):
    """Response model for new feature"""
    success: bool = Field(..., description="Whether processing succeeded")
    result: Optional[dict] = Field(default=None, description="Processing result")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

#### 3. Implement Controller
```python
# src/controllers/NewFeatureController.py
from typing import Dict, Any
import asyncio
from src.controllers.BaseController import BaseController

class NewFeatureController(BaseController):
    """Controller for new feature functionality"""
    
    def __init__(self):
        super().__init__()
        self.setup_dependencies()
    
    def setup_dependencies(self):
        """Initialize any required dependencies"""
        pass
    
    async def process(self, request: 'NewFeatureRequest') -> Dict[str, Any]:
        """Process new feature request"""
        try:
            # Implement your processing logic here
            result = await self._do_processing(request.input_data, request.options)
            
            return {
                "processed_data": result,
                "processing_time": self._get_processing_time(),
                "status": "completed"
            }
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    async def _do_processing(self, input_data: str, options: dict) -> Dict[str, Any]:
        """Internal processing method"""
        # Your implementation here
        await asyncio.sleep(0.1)  # Simulate async processing
        return {"output": f"processed_{input_data}"}
```

#### 4. Register Route
```python
# src/main.py
from src.routes import new_feature

app.include_router(new_feature.router, prefix="/api")
```

### Error Handling
```python
# src/helpers/exceptions.py
class SurveillanceException(Exception):
    """Base exception for surveillance system"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ProcessingError(SurveillanceException):
    """Exception raised during video processing"""
    pass

class ModelError(SurveillanceException):
    """Exception raised during AI model operations"""
    pass

# src/main.py
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(SurveillanceException)
async def surveillance_exception_handler(request: Request, exc: SurveillanceException):
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )
```

## 🎨 Frontend Development

### Streamlit Development
```python
# streamlit/pages/new_page.py
import streamlit as st
import requests
from utils.api_client import SurveillanceAPIClient

def main():
    st.title("🆕 New Feature Page")
    
    # Initialize API client
    api_client = SurveillanceAPIClient()
    
    with st.form("new_feature_form"):
        input_data = st.text_input("Input Data")
        options = st.json_input("Options (JSON)", value={})
        submitted = st.form_submit_button("Process")
        
        if submitted and input_data:
            with st.spinner("Processing..."):
                try:
                    result = api_client.process_new_feature(input_data, options)
                    st.success("Processing completed!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
```

### API Client Extension
```python
# streamlit/utils/api_client.py
class SurveillanceAPIClient:
    def process_new_feature(self, input_data: str, options: dict = None) -> dict:
        """Process new feature via API"""
        response = requests.post(
            f"{self.base_url}/new-feature/process",
            json={
                "input_data": input_data,
                "options": options or {}
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
```

## 🔧 Configuration Management

### Environment Configuration
```python
# src/helpers/config.py
from pydantic import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Database Configuration
    redis_url: str = "redis://localhost:6379"
    vector_db_path: str = "./vector_db"
    
    # AI Model Configuration
    yolo_model_path: str = "./src/models/yolov8n.pt"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "auto"  # auto, cpu, cuda
    
    # Processing Configuration
    max_upload_size: int = 200 * 1024 * 1024  # 200MB
    batch_size: int = 8
    max_workers: int = 2
    
    # Security Configuration
    secret_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    cors_origins: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

### Feature Flags
```python
# src/helpers/feature_flags.py
from enum import Enum
from typing import Dict, Any

class FeatureFlag(Enum):
    AUTHENTICATION_ENABLED = "auth_enabled"
    GPU_ACCELERATION = "gpu_acceleration"
    ADVANCED_TRACKING = "advanced_tracking"
    BATCH_PROCESSING = "batch_processing"

class FeatureManager:
    """Manage feature flags"""
    
    def __init__(self):
        self.flags: Dict[str, bool] = {
            FeatureFlag.AUTHENTICATION_ENABLED.value: False,
            FeatureFlag.GPU_ACCELERATION.value: True,
            FeatureFlag.ADVANCED_TRACKING.value: False,
            FeatureFlag.BATCH_PROCESSING.value: True,
        }
    
    def is_enabled(self, flag: FeatureFlag) -> bool:
        """Check if feature flag is enabled"""
        return self.flags.get(flag.value, False)
    
    def enable(self, flag: FeatureFlag):
        """Enable feature flag"""
        self.flags[flag.value] = True
    
    def disable(self, flag: FeatureFlag):
        """Disable feature flag"""
        self.flags[flag.value] = False

# Global feature manager
feature_manager = FeatureManager()
```

## 📊 Logging & Monitoring

### Logging Configuration
```python
# src/helpers/logging_config.py
import logging
import logging.config
from pathlib import Path

def setup_logging():
    """Configure application logging"""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/app.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed',
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/error.log',
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 3,
                'formatter': 'detailed',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'celery': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
        }
    }
    
    logging.config.dictConfig(config)
```

### Performance Monitoring
```python
# src/helpers/metrics.py
import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Usage example
@measure_time
async def process_video(video_path: str):
    # Processing logic here
    pass
```

## 🎯 Best Practices

### Code Organization
- **Single Responsibility**: Each class/function should have one clear purpose
- **Dependency Injection**: Use FastAPI's dependency injection system
- **Type Hints**: Always use type hints for better code documentation
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Add appropriate logging at all levels

### Performance
- **Async/Await**: Use async programming for I/O operations
- **Batch Processing**: Process multiple items together when possible
- **Caching**: Cache expensive computations and API responses
- **Connection Pooling**: Reuse database connections
- **Memory Management**: Monitor and optimize memory usage

### Security
- **Input Validation**: Validate all user inputs
- **SQL Injection**: Use parameterized queries
- **Authentication**: Implement proper authentication when needed
- **Rate Limiting**: Protect against abuse
- **HTTPS**: Use HTTPS in production

### Documentation
- **Docstrings**: Document all functions and classes
- **Type Hints**: Use for better IDE support
- **API Documentation**: Keep OpenAPI docs updated
- **README**: Maintain clear setup instructions
- **Changelog**: Document all changes

## 🤝 Contributing

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Review Guidelines
- **Code Quality**: Ensure code follows project standards
- **Tests**: Add tests for new functionality
- **Documentation**: Update documentation as needed
- **Performance**: Consider performance implications
- **Security**: Review for security issues

### Release Process
1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Run full test suite
4. **Documentation**: Update documentation
5. **Tag**: Create git tag for release
6. **Deploy**: Deploy to production

---

**🎉 Happy coding! Your contributions make this project better for everyone.**
