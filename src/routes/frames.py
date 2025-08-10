from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional
import os
import logging

from services.auth import get_optional_user
from helpers.config import get_settings, Settings
from controllers.QueryController import QueryController

logger = logging.getLogger(__name__)

frames_router = APIRouter(
    prefix="/api/surveillance/frames",
    tags=["frames"]
)

def _find_frame_file(frame_filename: str, project_id: str, app_settings: Settings) -> Optional[str]:
    """Helper function to find frame file in various locations"""
    base_dir = app_settings.project_files_dir
    clean_filename = frame_filename.replace('.mp4_frame_', '_frame_')
    
    # Define search paths
    search_paths = [
        os.path.join(base_dir, project_id, "frames"),
        os.path.join(base_dir, project_id, "extracted_frames"),
        os.path.join(base_dir, "projects", project_id, "frames"),
        os.path.join(base_dir, "test", "frames"),
    ]
    
    # Define filename variants
    filename_variants = [
        frame_filename,
        clean_filename,
        f"{frame_filename}.jpg",
        f"{clean_filename}.jpg",
        f"{frame_filename}.jpeg",
        f"{clean_filename}.jpeg",
    ]
    
    # Search for the file
    for search_path in search_paths:
        for filename in filename_variants:
            full_path = os.path.join(search_path, filename)
            if os.path.exists(full_path):
                return full_path
    
    return None

@frames_router.get("/{frame_filename}")
async def get_frame_by_filename(
    frame_filename: str,
    project_id: Optional[str] = Query("default", description="Project ID"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get a frame image by its filename"""
    try:
        frame_path = _find_frame_file(frame_filename, project_id, app_settings)
        
        if not frame_path:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_filename} not found in project {project_id}"
            )
        
        media_type = "image/png" if frame_filename.lower().endswith('.png') else "image/jpeg"
        
        return FileResponse(
            frame_path,
            media_type=media_type,
            headers={"Cache-Control": "max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame {frame_filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@frames_router.get("/result/{result_id}")
async def get_frame_by_result_id(
    result_id: str,
    project_id: Optional[str] = Query(None, description="Project ID"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get a specific frame from query results"""
    try:
        if not project_id:
            project_id = "default"
        
        # Create query controller without vector controller for now
        # TODO: This should be updated to use pre-initialized controller when available
        query_controller = QueryController(project_id=project_id)
        frame_path = query_controller.get_frame_for_result(result_id)
        
        if not frame_path or not os.path.exists(frame_path):
            raise HTTPException(
                status_code=404,
                detail=f"Frame not found for result {result_id}"
            )
        
        return FileResponse(
            frame_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving frame for {result_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@frames_router.get("/direct/{frame_filename}")
async def get_frame_direct(
    frame_filename: str,
    project_id: Optional[str] = Query("test", description="Project ID"),
    user: dict = Depends(get_optional_user)
):
    """Direct frame access with hardcoded paths for testing"""
    try:
        # Hardcoded paths for testing
        base_paths = [
            f"/home/amro/Desktop/intelligent-surveillance/src/assets/files/{project_id}/frames",
            "/home/amro/Desktop/intelligent-surveillance/src/assets/files/test/frames",
            f"/home/amro/Desktop/intelligent-surveillance/assets/files/{project_id}/frames",
            "/home/amro/Desktop/intelligent-surveillance/assets/files/test/frames",
        ]
        
        # Clean filename (remove .mp4 if present)
        clean_filename = frame_filename.replace('.mp4_frame_', '_frame_')
        
        # Try different extensions and names
        filename_variants = [
            frame_filename,
            clean_filename,
            frame_filename + ".jpg",
            clean_filename + ".jpg",
            frame_filename + ".jpeg", 
            clean_filename + ".jpeg",
        ]
        
        for base_path in base_paths:
            for filename in filename_variants:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    logger.info(f"Found frame at: {full_path}")
                    return FileResponse(
                        full_path,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"}
                    )
        
        # If not found, return debugging info
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found. Tried variants of {frame_filename} in projects {project_id}/test"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame {frame_filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
