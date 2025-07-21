from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers.DataController import DataController
import aiofiles
from models import ResponseSignal
import logging

from controllers.ProcessController import ProcessController
from controllers.ProjectController import ProjectController
from models.schemas import ProcessVideoRequest, ProcessVideoResponse

from services.auth import get_optional_user
from services.job_queue import job_manager


logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    
    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": file_id
            }
        )


# Video Processing Endpoints
@data_router.post("/process/{project_id}/{file_id}", response_model=ProcessVideoResponse)
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
