from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    VIDEO_PROCESSING = "video_processing"
    LIVE_STREAM = "live_stream"
    IMAGE_ANALYSIS = "image_analysis"

# Request Models
class ProcessVideoRequest(BaseModel):
    sample_rate: float = Field(default=1.0, ge=0.1, le=30.0, description="Frames per second to extract")
    detection_threshold: float = Field(default=0.5, ge=0.1, le=1.0, description="Minimum confidence for object detection")
    enable_tracking: bool = Field(default=True, description="Enable object tracking across frames")
    enable_captioning: bool = Field(default=True, description="Generate frame captions")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    project_id: Optional[str] = Field(None, description="Filter results to specific project")
    start_time: Optional[float] = Field(None, ge=0, description="Start timestamp filter")
    end_time: Optional[float] = Field(None, ge=0, description="End timestamp filter")
    object_types: Optional[List[str]] = Field(None, description="Filter by object types")
    min_confidence: Optional[float] = Field(None, ge=0, le=1, description="Minimum detection confidence")

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")

# Response Models
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")

class FrameAnalysis(BaseModel):
    timestamp: float
    frame_path: str
    detections: List[DetectionResult]
    total_objects: int
    caption: Optional[str] = None
    tracking_results: Optional[Dict[str, Any]] = None

class ProcessingJobResponse(BaseModel):
    job_id: str
    project_id: str
    file_id: str
    job_type: JobType
    status: ProcessingStatus
    progress: int = Field(..., ge=0, le=100)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

class ProcessVideoResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    project_id: str
    file_id: str
    estimated_frames: Optional[int] = None

class QueryResult(BaseModel):
    id: str
    caption: str
    file_id: str
    timestamp: float
    frame_path: str
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    detected_objects: List[str] = []

class QueryResponse(BaseModel):
    success: bool
    query: str
    results: List[QueryResult]
    total_results: int
    processing_time: float

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    total_videos: int = 0
    total_processed_frames: int = 0

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool
    vector_db_connected: bool
    disk_usage_mb: float
    memory_usage_mb: float
    active_jobs: int
    version: str

class AnalyticsResponse(BaseModel):
    project_id: str
    total_videos: int
    total_frames: int
    total_detections: int
    most_common_objects: List[Dict[str, Any]]
    activity_by_hour: List[Dict[str, Any]]
    processing_stats: Dict[str, Any]

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

# Error Response Models
class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
