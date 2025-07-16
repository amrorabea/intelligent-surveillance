from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Optional, List
import os
import logging
from datetime import datetime

from helpers.config import get_settings, Settings
from controllers.ProcessController import ProcessController
from controllers.QueryController import QueryController
from controllers.ProjectController import ProjectController
from models.schemas import (
    ProcessVideoRequest, ProcessVideoResponse, QueryResponse,
    ProcessingJobResponse, SystemHealthResponse, AnalyticsResponse,
    APIResponse
)
from services.auth import get_current_user, get_optional_user, check_rate_limit, verify_project_access
from services.job_queue import job_manager

logger = logging.getLogger(__name__)

surveillance_router = APIRouter(
    prefix="/api/surveillance",
    tags=["surveillance"],
    dependencies=[Depends(check_rate_limit)]
)

# Video Processing Endpoints
@surveillance_router.post("/process/{project_id}/{file_id}", response_model=ProcessVideoResponse)
async def process_surveillance_video(
    project_id: str, 
    file_id: str,
    request: ProcessVideoRequest = ProcessVideoRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_optional_user),  # Changed from get_current_user
    app_settings: Settings = Depends(get_settings)
):
    """
    Process a surveillance video through the AI pipeline (background job)
    """
    try:
        # Comment out project access verification for testing
        # await verify_project_access(project_id, user)
        
        # Validate file exists
        project_path = ProjectController().get_project_path(project_id)
        file_path = os.path.join(project_path, file_id)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found in project {project_id}"
            )
        
        # Validate it's a video file
        process_controller = ProcessController(project_id=project_id)
        if not process_controller.is_video_file(file_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file_id} is not a supported video format"
            )
        
        # Submit background job
        job_id = job_manager.submit_video_processing_job(
            project_id=project_id,
            file_id=file_id,
            sample_rate=request.sample_rate,
            detection_threshold=request.detection_threshold,
            enable_tracking=request.enable_tracking,
            enable_captioning=request.enable_captioning
        )
        
        # Estimate frames for progress tracking
        try:
            import cv2
            video = cv2.VideoCapture(file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            estimated_frames = int(frame_count / (fps / request.sample_rate)) if fps > 0 else 0
            video.release()
        except Exception:
            estimated_frames = None
        
        logger.info(f"Started video processing job {job_id} for {project_id}/{file_id}")
        
        return ProcessVideoResponse(
            success=True,
            job_id=job_id,
            message="Video processing started",
            project_id=project_id,
            file_id=file_id,
            estimated_frames=estimated_frames
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start video processing: {str(e)}"
        )

# Job Management Endpoints
@surveillance_router.get("/jobs/{project_id}", response_model=List[ProcessingJobResponse])
async def get_processing_jobs(
    project_id: str,
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    user: dict = Depends(get_optional_user)  # Changed from get_current_user
):
    """Get all processing jobs for a project"""
    try:
        # Comment out project access verification for testing
        # await verify_project_access(project_id, user)
        
        # TODO: Get jobs from database
        # For now, get active jobs from job manager
        active_jobs = job_manager.get_active_jobs()
        
        # Filter by project_id
        project_jobs = [
            job for job in active_jobs 
            if len(job.get('args', [])) > 0 and job['args'][0] == project_id
        ]
        
        return [
            ProcessingJobResponse(
                job_id=job['job_id'],
                project_id=project_id,
                file_id=job['args'][1] if len(job['args']) > 1 else 'unknown',
                job_type='video_processing',
                status='running',
                progress=50,  # TODO: Get actual progress
                created_at=datetime.utcnow()
            )
            for job in project_jobs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting jobs for project {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@surveillance_router.get("/jobs/status/{job_id}")
async def get_job_status(
    job_id: str,
    user: dict = Depends(get_optional_user)
):
    """Get detailed status of a processing job"""
    try:
        status_info = job_manager.get_job_status(job_id)
        return APIResponse(
            success=True,
            data=status_info
        )
        
    except Exception as e:
        logger.error(f"Error getting job status {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@surveillance_router.delete("/jobs/{job_id}")
async def cancel_processing_job(
    job_id: str,
    user: dict = Depends(get_current_user)
):
    """Cancel a running processing job"""
    try:
        success = job_manager.cancel_job(job_id)
        
        if success:
            return APIResponse(
                success=True,
                data={"message": f"Job {job_id} cancelled successfully"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to cancel job {job_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Query Endpoints
@surveillance_router.get("/query", response_model=QueryResponse)
async def query_surveillance(
    query: str = Query(..., description="Natural language query"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results"),
    start_time: Optional[float] = Query(None, description="Start timestamp filter"),
    end_time: Optional[float] = Query(None, description="End timestamp filter"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    user: dict = Depends(get_current_user),
    app_settings: Settings = Depends(get_settings)
):
    """Query surveillance footage using natural language"""
    try:
        start_time_query = datetime.utcnow()
        
        if project_id:
            await verify_project_access(project_id, user)
        
        # Initialize query controller
        query_controller = QueryController()
        
        # Execute query
        results = query_controller.process_query(
            query_text=query,
            max_results=max_results,
            project_id=project_id
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time_query).total_seconds()
        
        # Convert to response format
        from models.schemas import QueryResult
        query_results = [
            QueryResult(
                id=result['id'],
                caption=result['caption'],
                file_id=result['file_id'],
                timestamp=result['timestamp'],
                frame_path=result.get('frame_path', ''),
                score=result['score'],
                detected_objects=result.get('detected_objects', [])
            )
            for result in results.get('results', [])
        ]
        
        return QueryResponse(
            success=True,
            query=query,
            results=query_results,
            total_results=len(query_results),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying surveillance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@surveillance_router.get("/query/timeline/{project_id}")
async def get_timeline_events(
    project_id: str,
    start_time: datetime = Query(..., description="Start time"),
    end_time: datetime = Query(..., description="End time"),
    object_types: Optional[List[str]] = Query(None, description="Filter by object types"),
    user: dict = Depends(get_current_user)
):
    """Get timeline of surveillance events in a time range"""
    try:
        await verify_project_access(project_id, user)
        
        # TODO: Implement timeline query
        # This would query the database for events in the time range
        
        return APIResponse(
            success=True,
            data={
                "project_id": project_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "events": [],  # TODO: Implement actual timeline query
                "note": "TODO: Implement timeline query from database"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting timeline for {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Frame and Media Endpoints
@surveillance_router.get("/frame/{result_id}")
async def get_frame(
    result_id: str,
    user: dict = Depends(get_current_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get a specific frame from query results"""
    try:
        # Initialize query controller
        query_controller = QueryController()
        
        # Get frame path
        frame_path = query_controller.get_frame_for_result(result_id)
        
        if not frame_path or not os.path.exists(frame_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame not found for result {result_id}"
            )
        
        # Return the image file
        return FileResponse(
            frame_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Analytics Endpoints
@surveillance_router.get("/analytics/summary/{project_id}", response_model=AnalyticsResponse)
async def get_analytics_summary(
    project_id: str,
    user: dict = Depends(get_current_user)
):
    """Get analytics summary for a project"""
    try:
        await verify_project_access(project_id, user)
        
        # TODO: Implement analytics from database
        return AnalyticsResponse(
            project_id=project_id,
            total_videos=0,  # TODO: Query from database
            total_frames=0,  # TODO: Query from database  
            total_detections=0,  # TODO: Query from database
            most_common_objects=[],  # TODO: Aggregate from database
            activity_by_hour=[],  # TODO: Aggregate from database
            processing_stats={}  # TODO: Calculate from database
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# System Health Endpoints
@surveillance_router.get("/health", response_model=SystemHealthResponse)
async def health_check(user: dict = Depends(get_optional_user)):
    """System health check"""
    try:
        import psutil
        
        # Check database connection
        db_connected = True  # TODO: Implement actual DB health check
        
        # Check vector database connection
        vector_db_connected = True  # TODO: Implement actual vector DB health check
        
        # Get system stats
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Get active jobs count
        active_jobs = len(job_manager.get_active_jobs())
        
        return SystemHealthResponse(
            status="healthy" if db_connected and vector_db_connected else "degraded",
            timestamp=datetime.utcnow(),
            database_connected=db_connected,
            vector_db_connected=vector_db_connected,
            disk_usage_mb=disk_info.used / (1024 * 1024),
            memory_usage_mb=memory_info.used / (1024 * 1024),
            active_jobs=active_jobs,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return SystemHealthResponse(
            status="error",
            timestamp=datetime.utcnow(),
            database_connected=False,
            vector_db_connected=False,
            disk_usage_mb=0,
            memory_usage_mb=0,
            active_jobs=0,
            version="1.0.0"
        )
