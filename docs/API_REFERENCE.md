# 📚 API Reference

Complete REST API documentation for the Intelligent Surveillance System.

## 🌐 Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://intelligent-surveillance.streamlit.app/`

## 📖 API Overview

The API follows RESTful principles and returns JSON responses. All endpoints support CORS and include comprehensive error handling.

### Authentication
Currently, authentication is disabled for development. In production, use JWT tokens:

```bash
# Get token (when auth is enabled)
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "password"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "http://localhost:8000/surveillance/projects"
```

## 📋 Endpoints Overview

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| **Health** | `/health` | GET | System health check |
| **Upload** | `/surveillance/upload` | POST | Upload video for processing |
| **Search** | `/surveillance/search` | POST | Semantic search in videos |
| **Projects** | `/surveillance/projects` | GET | List all projects |
| **Analytics** | `/surveillance/analytics` | GET | Project analytics |
| **Frames** | `/surveillance/frame/{frame_id}` | GET | Get specific frame |
| **Jobs** | `/surveillance/jobs` | GET | List processing jobs |

## 🏥 Health & Status

### Health Check
Check if the API is running and healthy.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "redis": "connected",
    "worker": "active",
    "database": "operational"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

## 📤 Video Upload & Processing

### Upload Video
Upload a video file for AI processing.

```http
POST /surveillance/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Video file (MP4, AVI, MOV, MKV)
- `project_id` (required): Project identifier
- `description` (optional): Video description

**Response:**
```json
{
  "message": "Video uploaded successfully",
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "project_id": "my-project",
  "filename": "surveillance_video.mp4",
  "status": "queued"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/surveillance/upload" \
     -F "file=@video.mp4" \
     -F "project_id=security-cameras" \
     -F "description=Front door camera footage"
```

**Error Responses:**
```json
// File too large
{
  "detail": "File size exceeds maximum allowed size (200MB)"
}

// Invalid format
{
  "detail": "Unsupported file format. Allowed: mp4, avi, mov, mkv"
}

// Missing project_id
{
  "detail": "project_id is required"
}
```

---

## 🔍 Search & Query

### Semantic Search
Search for content within processed videos using natural language.

```http
POST /surveillance/search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "person wearing red shirt",
  "project_id": "my-project",
  "limit": 10,
  "threshold": 0.7,
  "detected_objects": ["person"],
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-31T23:59:59Z"
}
```

**Parameters:**
- `query` (required): Natural language search query
- `project_id` (optional): Limit search to specific project
- `limit` (optional): Maximum results (default: 10, max: 100)
- `threshold` (optional): Similarity threshold (0.0-1.0, default: 0.7)
- `detected_objects` (optional): Filter by object types
- `start_time` (optional): Filter by time range
- `end_time` (optional): Filter by time range

**Response:**
```json
{
  "results": [
    {
      "frame_id": "frame_001234",
      "video_filename": "surveillance_video.mp4",
      "project_id": "my-project",
      "timestamp": 45.67,
      "similarity_score": 0.89,
      "caption": "A person wearing a red shirt walking down the street",
      "detected_objects": [
        {
          "class": "person",
          "confidence": 0.95,
          "bbox": [100, 200, 150, 300]
        }
      ],
      "frame_path": "/api/surveillance/frame/frame_001234",
      "metadata": {
        "processing_time": "2024-01-15T10:30:00Z",
        "model_version": "yolov8n"
      }
    }
  ],
  "total_results": 25,
  "query_time": 0.045,
  "project_id": "my-project"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/surveillance/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "blue car parking",
       "project_id": "parking-lot",
       "limit": 5
     }'
```

---

## 📊 Projects & Management

### List Projects
Get all available projects with metadata.

```http
GET /surveillance/projects
```

**Query Parameters:**
- `include_stats` (optional): Include statistics (default: false)

**Response:**
```json
{
  "projects": [
    {
      "id": "security-cameras",
      "name": "Security Cameras",
      "description": "Main building security footage",
      "created_at": "2024-01-15T10:00:00Z",
      "video_count": 12,
      "frame_count": 15420,
      "last_processed": "2024-01-15T15:30:00Z",
      "stats": {
        "total_detections": 2341,
        "unique_objects": ["person", "car", "bicycle"],
        "processing_time": "2h 15m"
      }
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/surveillance/projects?include_stats=true"
```

### Project Details
Get detailed information about a specific project.

```http
GET /surveillance/projects/{project_id}
```

**Response:**
```json
{
  "project": {
    "id": "security-cameras",
    "name": "Security Cameras",
    "description": "Main building security footage",
    "created_at": "2024-01-15T10:00:00Z",
    "videos": [
      {
        "filename": "front_door_20240115.mp4",
        "uploaded_at": "2024-01-15T10:30:00Z",
        "status": "completed",
        "frame_count": 1800,
        "duration": 60.0
      }
    ],
    "statistics": {
      "total_frames": 15420,
      "total_detections": 2341,
      "object_distribution": {
        "person": 1256,
        "car": 890,
        "bicycle": 195
      }
    }
  }
}
```

---

## 📈 Analytics & Statistics

### Project Analytics
Get comprehensive analytics for a project or all projects.

```http
GET /surveillance/analytics
```

**Query Parameters:**
- `project_id` (optional): Specific project (omit for all projects)
- `start_date` (optional): Filter by date range
- `end_date` (optional): Filter by date range

**Response:**
```json
{
  "analytics": {
    "overview": {
      "total_projects": 5,
      "total_videos": 47,
      "total_frames": 125680,
      "total_detections": 89432,
      "processing_time": "15h 23m"
    },
    "object_statistics": {
      "person": {
        "count": 45123,
        "percentage": 50.5,
        "trend": "+15%"
      },
      "car": {
        "count": 28901,
        "percentage": 32.3,
        "trend": "+8%"
      },
      "bicycle": {
        "count": 15408,
        "percentage": 17.2,
        "trend": "-3%"
      }
    },
    "temporal_distribution": {
      "hourly": {
        "00": 1245,
        "01": 987,
        "06": 3456,
        "12": 8901,
        "18": 6543
      },
      "daily": {
        "monday": 12567,
        "tuesday": 13890,
        "wednesday": 11234
      }
    },
    "performance_metrics": {
      "avg_processing_time": 45.6,
      "avg_detection_confidence": 0.87,
      "frames_per_second": 23.4
    }
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/surveillance/analytics?project_id=security-cameras"
```

---

## 🖼️ Frame & Media Access

### Get Frame Image
Retrieve a specific video frame as an image.

```http
GET /surveillance/frame/{frame_id}
```

**Response:**
- **Content-Type**: `image/jpeg`
- **Binary image data**

**Example:**
```bash
curl "http://localhost:8000/surveillance/frame/frame_001234" \
     --output frame.jpg
```

### Get Frame Metadata
Get metadata for a specific frame without the image.

```http
GET /surveillance/frame/{frame_id}/metadata
```

**Response:**
```json
{
  "frame_id": "frame_001234",
  "video_filename": "surveillance_video.mp4",
  "project_id": "my-project",
  "timestamp": 45.67,
  "width": 1920,
  "height": 1080,
  "size_bytes": 245760,
  "detected_objects": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 150, 300]
    }
  ],
  "caption": "A person wearing a red shirt walking down the street",
  "processing_metadata": {
    "processed_at": "2024-01-15T10:30:00Z",
    "yolo_version": "8n",
    "blip_version": "base"
  }
}
```

### Get Frame Thumbnail
Get a smaller thumbnail version of a frame.

```http
GET /surveillance/frame/{frame_id}/thumbnail
```

**Query Parameters:**
- `size` (optional): Thumbnail size (default: 150, max: 500)

**Response:**
- **Content-Type**: `image/jpeg`
- **Binary thumbnail data**

---

## ⚙️ Job Management

### List Processing Jobs
Get status of all video processing jobs.

```http
GET /surveillance/jobs
```

**Query Parameters:**
- `status` (optional): Filter by status (queued, processing, completed, failed)
- `project_id` (optional): Filter by project
- `limit` (optional): Maximum results (default: 50)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "12345678-1234-1234-1234-123456789abc",
      "project_id": "security-cameras",
      "filename": "front_door.mp4",
      "status": "completed",
      "progress": 100,
      "started_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:45:00Z",
      "results": {
        "frames_processed": 1800,
        "objects_detected": 234,
        "processing_time": 15.5
      }
    }
  ],
  "total_jobs": 47,
  "active_jobs": 3
}
```

### Get Job Status
Get detailed status of a specific processing job.

```http
GET /surveillance/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "project_id": "security-cameras",
  "filename": "front_door.mp4",
  "status": "processing",
  "progress": 67,
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:45:00Z",
  "current_stage": "object_detection",
  "stages": {
    "frame_extraction": "completed",
    "object_detection": "processing",
    "image_captioning": "queued",
    "embedding_generation": "queued"
  },
  "metrics": {
    "frames_processed": 1200,
    "frames_total": 1800,
    "objects_detected": 156,
    "processing_speed": 23.4
  }
}
```

---

## 🚨 Error Handling

### Error Response Format
All errors follow this structure:

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/surveillance/upload",
  "request_id": "req_12345"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `UNSUPPORTED_FORMAT` | 415 | Unsupported file format |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Error Examples

```bash
# File too large
HTTP 413 Payload Too Large
{
  "detail": "File size exceeds maximum allowed size (200MB)",
  "error_code": "FILE_TOO_LARGE"
}

# Project not found
HTTP 404 Not Found
{
  "detail": "Project 'nonexistent-project' not found",
  "error_code": "NOT_FOUND"
}

# Invalid search query
HTTP 400 Bad Request
{
  "detail": "Search query cannot be empty",
  "error_code": "VALIDATION_ERROR"
}
```

---

## 📊 Rate Limiting

### Limits
- **Upload**: 5 files per minute
- **Search**: 60 requests per minute
- **Analytics**: 30 requests per minute
- **Frame Access**: 300 requests per minute

### Headers
Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642234567
```

---

## 🔧 SDK & Client Libraries

### Python Client
```python
from surveillance_client import SurveillanceClient

client = SurveillanceClient("http://localhost:8000")

# Upload video
result = client.upload_video(
    file_path="video.mp4",
    project_id="my-project"
)

# Search
results = client.search(
    query="person walking",
    project_id="my-project",
    limit=10
)

# Get analytics
analytics = client.get_analytics(project_id="my-project")
```

### JavaScript/Node.js Client
```javascript
import { SurveillanceClient } from '@surveillance/client';

const client = new SurveillanceClient('http://localhost:8000');

// Upload video
const uploadResult = await client.uploadVideo({
  file: videoFile,
  projectId: 'my-project'
});

// Search
const searchResults = await client.search({
  query: 'person walking',
  projectId: 'my-project',
  limit: 10
});
```

---

## 📋 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Interactive Docs**: http://localhost:8000/docs
- **JSON Spec**: http://localhost:8000/openapi.json
- **ReDoc**: http://localhost:8000/redoc

---

## 🧪 Testing

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Upload test video
curl -X POST "http://localhost:8000/surveillance/upload" \
     -F "file=@test_video.mp4" \
     -F "project_id=test"

# Test search
curl -X POST "http://localhost:8000/surveillance/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "project_id": "test"}'
```

### Postman Collection
Download the Postman collection: [surveillance-api.postman_collection.json](postman/surveillance-api.postman_collection.json)

---

**📞 Need Help?**
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [System Status](http://localhost:8501) (when running)
- Open an issue on GitHub
