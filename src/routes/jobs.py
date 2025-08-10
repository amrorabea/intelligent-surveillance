from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
import logging
from datetime import datetime

from models.schemas import ProcessingJobResponse, APIResponse
from services.auth import get_optional_user, get_current_user
from services.job_queue import job_manager

logger = logging.getLogger(__name__)

jobs_router = APIRouter(
    prefix="/api/surveillance/jobs",
    tags=["jobs"]
)

@jobs_router.get("/{project_id}", response_model=List[ProcessingJobResponse])
async def get_processing_jobs(
    project_id: str,
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    user: dict = Depends(get_optional_user)
):
    """Get all processing jobs for a project"""
    try:
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
                progress=50,
                created_at=datetime.utcnow()
            )
            for job in project_jobs
        ]
        
    except Exception as e:
        logger.error(f"Error getting jobs for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@jobs_router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    user: dict = Depends(get_optional_user)
):
    """Get detailed status of a processing job"""
    try:
        status_info = job_manager.get_job_status(job_id)
        return APIResponse(success=True, data=status_info)
        
    except Exception as e:
        logger.error(f"Error getting job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@jobs_router.delete("/{job_id}")
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
                status_code=400,
                detail=f"Failed to cancel job {job_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
