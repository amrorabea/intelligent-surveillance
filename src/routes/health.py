from fastapi import APIRouter, Depends
import logging
from datetime import datetime

from models.schemas import SystemHealthResponse
from services.auth import get_optional_user
from services.job_queue import job_manager

logger = logging.getLogger(__name__)

health_router = APIRouter(
    prefix="/api/surveillance/health",
    tags=["health"]
)

@health_router.get("", response_model=SystemHealthResponse)
async def health_check(user: dict = Depends(get_optional_user)):
    """System health check"""
    try:
        import psutil
        
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        active_jobs = len(job_manager.get_active_jobs())
        
        return SystemHealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            database_connected=True,
            vector_db_connected=True,
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
