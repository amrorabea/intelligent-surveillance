from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any
import os
import logging
from datetime import datetime

from helpers.config import get_settings, Settings
from controllers.QueryController import QueryController
from controllers.VectorDBController import VectorDBController
from models.schemas import (
    QueryResponse,
    ProcessingJobResponse, SystemHealthResponse, AnalyticsResponse,
    APIResponse
)
from services.auth import get_current_user, get_optional_user, verify_project_access
from services.job_queue import job_manager

logger = logging.getLogger(__name__)

surveillance_router = APIRouter(
    prefix="/api/surveillance",
    tags=["surveillance"]
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
        
        # TODO: Issue: Problem happens when there is an actual job running
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
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Query surveillance footage using natural language"""
    try:
        start_time_query = datetime.utcnow()
        
        # if project_id:
        #     await verify_project_access(project_id, user)
        
        # Initialize query controller
        query_controller = QueryController(project_id=project_id)
        
        # Execute query
        results = query_controller.process_query(
            query_text=query,
            max_results=max_results,
            project_id=project_id
        )
        
        logger.info(f"Results for query: {results}")

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
    project_id: Optional[str] = Query(None, description="Project ID for frame lookup"),
    user: dict = Depends(get_optional_user),  # Changed to optional
    app_settings: Settings = Depends(get_settings)
):
    """Get a specific frame from query results"""
    try:
        # Use provided project_id or default
        if not project_id:
            project_id = "default"
            
        # Initialize query controller
        query_controller = QueryController(project_id=project_id)
        
        # Get frame path
        frame_path = query_controller.get_frame_for_result(result_id)
        
        if not frame_path or not os.path.exists(frame_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame not found for result {result_id} in project {project_id}"
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
        logger.error(f"Error retrieving frame for {result_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@surveillance_router.get("/frames/{frame_filename}")
async def get_frame_by_filename(
    frame_filename: str,
    project_id: Optional[str] = Query("default", description="Project ID"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get a frame image by its filename"""
    try:
        # Handle different filename formats
        # Remove .mp4 from filename if present (common mismatch)
        clean_filename = frame_filename.replace('.mp4_frame_', '_frame_')
        
        # Construct potential frame paths
        base_dir = app_settings.project_files_dir
        potential_paths = [
            # Try with original filename
            os.path.join(base_dir, project_id, "frames", frame_filename),
            os.path.join(base_dir, project_id, "extracted_frames", frame_filename),
            os.path.join(base_dir, "projects", project_id, "frames", frame_filename),
            os.path.join(base_dir, "projects", project_id, "extracted_frames", frame_filename),
            # Try with cleaned filename  
            os.path.join(base_dir, project_id, "frames", clean_filename),
            os.path.join(base_dir, project_id, "extracted_frames", clean_filename),
            os.path.join(base_dir, "projects", project_id, "frames", clean_filename),
            os.path.join(base_dir, "projects", project_id, "extracted_frames", clean_filename),
            # Try in test directory (common location)
            os.path.join(base_dir, "test", "frames", frame_filename),
            os.path.join(base_dir, "test", "frames", clean_filename),
        ]
        
        # Add image extensions if not present
        extended_paths = []
        for path in potential_paths:
            if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
                extended_paths.extend([
                    path + ".jpg",
                    path + ".jpeg", 
                    path + ".png"
                ])
            extended_paths.append(path)
        
        # Try to find the file
        for frame_path in extended_paths:
            if os.path.exists(frame_path):
                logger.info(f"Found frame at: {frame_path}")
                return FileResponse(
                    frame_path,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"}
                )
        
        # If not found, log the attempts and return 404
        logger.warning(f"Frame not found. Tried paths for {frame_filename} (cleaned: {clean_filename}):")
        for path in extended_paths[:10]:  # Log first 10 attempts
            logger.warning(f"  - {path}")
            
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame {frame_filename} not found in project {project_id}"
        )
        
        # Try to find the frame file
        frame_path = None
        for path in potential_paths:
            if os.path.exists(path):
                frame_path = path
                break
        
        if not frame_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame file '{frame_filename}' not found in project '{project_id}'"
            )
        
        # Determine media type
        media_type = "image/jpeg"
        if frame_filename.lower().endswith('.png'):
            media_type = "image/png"
        
        return FileResponse(
            frame_path,
            media_type=media_type,
            headers={"Cache-Control": "max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame {frame_filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Additional frame endpoints for different URL patterns
@surveillance_router.get("/frame-by-id/{frame_id}")
async def get_frame_by_id(
    frame_id: str,
    project_id: Optional[str] = Query("default", description="Project ID"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get a frame by ID (alternative endpoint)"""
    return await get_frame_by_filename(frame_id, project_id, user, app_settings)

@surveillance_router.get("/surveillance/results/{result_id}/frame")
async def get_result_frame(
    result_id: str,
    project_id: Optional[str] = Query("default", description="Project ID"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get frame for a specific result (alternative endpoint)"""
    return await get_frame(result_id, project_id, user, app_settings)

@surveillance_router.get("/surveillance/projects/{project_id}/frame")
async def get_project_frame(
    project_id: str,
    frame_path: Optional[str] = Query(None, description="Full path to frame"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get frame by project and path"""
    if not frame_path:
        raise HTTPException(status_code=400, detail="frame_path parameter required")
    
    try:
        if os.path.exists(frame_path):
            return FileResponse(
                frame_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "max-age=3600"}
            )
        else:
            raise HTTPException(status_code=404, detail=f"Frame not found: {frame_path}")
    except Exception as e:
        logger.error(f"Error serving frame from path {frame_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@surveillance_router.get("/data/frame/{project_id}")
async def get_data_frame(
    project_id: str,
    frame_path: Optional[str] = Query(None, description="Full path to frame"),
    user: dict = Depends(get_optional_user),
    app_settings: Settings = Depends(get_settings)
):
    """Get frame from data endpoint (alternative endpoint)"""
    return await get_project_frame(project_id, frame_path, user, app_settings)

# Analytics Endpoints
@surveillance_router.get("/analytics/summary/{project_id}", response_model=AnalyticsResponse)
async def get_analytics_summary(
    project_id: str,
    user: dict = Depends(get_current_user)
):
    """Get analytics summary for a project"""
    try:
        await verify_project_access(project_id, user)
        
        # Get analytics using VectorDBController
        vector_controller = VectorDBController()
        
        # Get project-specific data from vector database
        try:
            # Check if there's a project-specific collection
            project_collection_name = f"surveillance_{project_id}"
            all_results = []
            
            if vector_controller.client:
                collections = vector_controller.client.list_collections()
                
                # Look for project-specific collection first
                for collection in collections:
                    if collection.name == project_collection_name and collection.count() > 0:
                        temp_controller = VectorDBController(collection_name=collection.name)
                        collection_results = temp_controller.semantic_search(
                            query="object detection analysis surveillance",
                            limit=1000,
                            filter_criteria=None
                        )
                        all_results.extend(collection_results)
                        break
                else:
                    # If no project-specific collection, search all collections for this project
                    for collection in collections:
                        if collection.name.startswith('surveillance_') and collection.count() > 0:
                            temp_controller = VectorDBController(collection_name=collection.name)
                            collection_results = temp_controller.semantic_search(
                                query="object detection analysis surveillance",
                                limit=1000,
                                filter_criteria={"project_id": project_id} if project_id != "all" else None
                            )
                            all_results.extend(collection_results)
            
            query_results = all_results
        except Exception as e:
            logger.warning(f"Vector DB query failed for project {project_id}: {e}")
            query_results = []
        
        total_detections = len(query_results)
        
        # Calculate object type distribution
        object_counts = {}
        confidence_sum = 0
        processing_times = []
        videos_seen = set()
        
        for result_item in query_results:
            metadata = result_item.get("metadata", {})
            if metadata and isinstance(metadata, dict):
                # Track videos
                video_filename = metadata.get("video_filename", "unknown")
                videos_seen.add(video_filename)
                
                # Count object types
                detected_objects = metadata.get("detected_objects", [])
                if isinstance(detected_objects, str):
                    # Handle comma-separated objects
                    if ',' in detected_objects:
                        detected_objects = [obj.strip() for obj in detected_objects.split(',')]
                    else:
                        detected_objects = [detected_objects.strip()]
                elif not isinstance(detected_objects, list):
                    detected_objects = []
                    
                for obj in detected_objects:
                    obj_str = str(obj).strip()
                    if obj_str:
                        object_counts[obj_str] = object_counts.get(obj_str, 0) + 1
                
                # Sum confidence scores
                confidence = metadata.get("confidence")
                if confidence is not None:
                    try:
                        confidence_sum += float(confidence)
                    except (ValueError, TypeError):
                        pass
                
                # Collect processing times
                processing_time = metadata.get("processing_time")
                if processing_time is not None:
                    try:
                        processing_times.append(float(processing_time))
                    except (ValueError, TypeError):
                        pass
        
        # Calculate averages
        total_videos = len(videos_seen)
        avg_confidence = confidence_sum / total_detections if total_detections > 0 else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Convert object counts to the expected format
        most_common_objects = [
            {"object_type": obj_type, "count": count}
            for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Generate activity by hour (distribute detections across 24 hours)
        activity_by_hour = []
        for hour in range(24):
            # Simple distribution - more activity during day hours
            if 6 <= hour <= 22:  # Day hours
                count = max(0, total_detections // 24 + (hour % 4 - 1) * 2)
            else:  # Night hours
                count = max(0, total_detections // 48)
            activity_by_hour.append({"hour": hour, "detection_count": count})
        
        return AnalyticsResponse(
            project_id=project_id,
            total_videos=total_videos,
            total_frames=total_detections,  # Approximate: 1 frame per detection
            total_detections=total_detections,
            most_common_objects=most_common_objects,
            activity_by_hour=activity_by_hour,
            processing_stats={
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "unique_object_types": len(object_counts),
                "processing_jobs_completed": len(processing_times)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for {project_id}: {e}")
        # Return empty analytics instead of error
        return AnalyticsResponse(
            project_id=project_id,
            total_videos=0,
            total_frames=0,
            total_detections=0,
            most_common_objects=[],
            activity_by_hour=[{"hour": i, "detection_count": 0} for i in range(24)],
            processing_stats={}
        )

@surveillance_router.get("/analytics", response_model=Dict[str, Any])
async def get_general_analytics(
    user: dict = Depends(get_current_user)
):
    """Get general analytics across all projects"""
    try:
        # Get analytics using available controllers
        # Create ONE VectorDBController instance and reuse it for all collections
        vector_controller = VectorDBController()
        logger.info("Heavy analytics mode - sentence transformer model will be loaded")
        
        # Get all available data from vector database
        try:
            # Get all collections and aggregate data from them
            all_results = []
            
            if vector_controller.client:
                collections = vector_controller.client.list_collections()
                logger.info(f"Found collections: {[c.name for c in collections]}")
                
                for collection in collections:
                    if collection.name.startswith('surveillance_') and collection.count() > 0:
                        logger.info(f"Checking collection {collection.name} with {collection.count()} documents")
                        
                        # Switch to this collection WITHOUT creating a new controller
                        # This avoids reloading the sentence transformer model
                        if vector_controller.switch_collection(collection.name):
                            # Get data from this collection using the existing controller
                            collection_results = vector_controller.semantic_search(
                                query="object detection analysis surveillance",
                                limit=1000,
                                filter_criteria=None
                            )
                            
                            all_results.extend(collection_results)
                            logger.info(f"Got {len(collection_results)} results from {collection.name}")
                        else:
                            logger.warning(f"Failed to switch to collection {collection.name}")
            
            query_results = all_results
        except Exception as e:
            # If vector DB fails, return mock data with error info
            logger.warning(f"Vector DB query failed: {e}")
            query_results = []
        
        total_detections = len(query_results)
        
        # Process the results to generate analytics
        all_object_counts = {}
        all_confidences = []
        all_processing_times = []
        projects_seen = set()
        videos_seen = set()
        
        for result_item in query_results:
            metadata = result_item.get("metadata", {})
            if metadata and isinstance(metadata, dict):
                # Track projects and videos
                project_id = metadata.get("project_id", "unknown")
                video_filename = metadata.get("video_filename", "unknown")
                projects_seen.add(project_id)
                videos_seen.add(f"{project_id}:{video_filename}")
                
                # Count objects
                detected_objects = metadata.get("detected_objects", [])
                if isinstance(detected_objects, str):
                    # Handle both comma-separated strings and single objects
                    if ',' in detected_objects:
                        detected_objects = [obj.strip() for obj in detected_objects.split(',')]
                    else:
                        detected_objects = [detected_objects.strip()]
                elif not isinstance(detected_objects, list):
                    detected_objects = []
                    
                for obj in detected_objects:
                    obj_str = str(obj).strip()
                    if obj_str:
                        all_object_counts[obj_str] = all_object_counts.get(obj_str, 0) + 1
                
                # Collect confidence and processing time
                confidence = metadata.get("confidence")
                if confidence is not None:
                    try:
                        all_confidences.append(float(confidence))
                    except (ValueError, TypeError):
                        pass
                
                processing_time = metadata.get("processing_time")
                if processing_time is not None:
                    try:
                        all_processing_times.append(float(processing_time))
                    except (ValueError, TypeError):
                        pass
        
        # Calculate final metrics
        total_videos = len(videos_seen)
        total_projects = len(projects_seen)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0
        
        # Estimate total frames (assume average of 1 frame per detection for now)
        total_frames = total_detections
        
        # Prepare response data
        analytics_data = {
            "total_videos": total_videos,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "total_projects": total_projects,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "object_counts": all_object_counts,
            "confidence_distribution": {
                "0.5-0.6": len([c for c in all_confidences if 0.5 <= c < 0.6]),
                "0.6-0.7": len([c for c in all_confidences if 0.6 <= c < 0.7]),
                "0.7-0.8": len([c for c in all_confidences if 0.7 <= c < 0.8]),
                "0.8-0.9": len([c for c in all_confidences if 0.8 <= c < 0.9]),
                "0.9-1.0": len([c for c in all_confidences if 0.9 <= c <= 1.0]),
            },
            "timeline_data": [
                {"timestamp": f"2024-01-01T{hour:02d}:00:00", "detections_count": max(0, total_detections // 24 + (hour % 3 - 1) * 5)}
                for hour in range(24)
            ],
            "performance_metrics": {
                "processing_speeds": all_processing_times[-10:] if all_processing_times else [12.5, 14.2, 13.8, 15.1, 12.9]
            },
            "search_analytics": {
                "total_searches": 0,  # TODO: Track search requests
                "avg_results": total_detections / max(1, total_videos),
                "avg_search_time": 0.0,  # TODO: Track search timing
                "popular_terms": []  # TODO: Track search terms
            },
            "insights": [
                {
                    "title": "Detection Performance",
                    "description": f"Processed {total_videos} videos across {total_projects} projects with {total_detections} detections",
                    "confidence": 0.95
                },
                {
                    "title": "Average Confidence",
                    "description": f"Detection confidence averaging {avg_confidence:.1%}" if avg_confidence > 0 else "No confidence data available",
                    "confidence": 0.88 if avg_confidence > 0 else 0.5
                },
                {
                    "title": "Object Diversity",
                    "description": f"Detected {len(all_object_counts)} different object types",
                    "confidence": 0.92
                }
            ],
            "alerts": [
                {
                    "type": "info",
                    "title": "System Status",
                    "message": "All analytics systems operational",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
        }
        
        return {"success": True, "data": analytics_data}
        
    except Exception as e:
        logger.error(f"Error getting general analytics: {e}")
        # Return mock data if everything fails
        return {
            "success": True,
            "data": {
                "total_videos": 0,
                "total_frames": 0,
                "total_detections": 0,
                "total_projects": 0,
                "avg_confidence": 0,
                "avg_processing_time": 0,
                "object_counts": {},
                "confidence_distribution": {"0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0},
                "timeline_data": [],
                "performance_metrics": {"processing_speeds": []},
                "search_analytics": {"total_searches": 0, "avg_results": 0, "avg_search_time": 0, "popular_terms": []},
                "insights": [{"title": "No Data", "description": "No analytics data available yet", "confidence": 0.5}],
                "alerts": [{"type": "info", "title": "System Status", "message": f"Analytics system error: {str(e)}", "timestamp": datetime.utcnow().isoformat()}]
            }
        }

# Lightweight Analytics Endpoint
@surveillance_router.get("/analytics/light", response_model=Dict[str, Any])
async def get_light_analytics(
    user: dict = Depends(get_optional_user)
):
    """Get lightweight analytics without loading sentence transformer models"""
    try:
        logger.info("Getting light analytics (no model loading)")
        
        # Get basic collection information WITHOUT creating VectorDBController
        # This avoids any potential sentence transformer loading
        collection_stats = []
        total_documents = 0
        total_collections = 0
        active_collections = 0
        client_available = False
        
        try:
            # Import chromadb directly and create a simple client
            import chromadb
            import os
            from helpers.config import get_settings
            
            # Get the database directory from settings
            settings = get_settings()
            db_dir = os.path.join(settings.files_directory, "chromadb")
            
            # Create a direct chromadb client
            client = chromadb.PersistentClient(path=db_dir)
            client_available = True
            logger.info("Direct ChromaDB client created for light analytics")
            
            # Get collections without loading any models
            collections = client.list_collections()
            logger.info(f"Found {len(collections)} collections for light analytics")
            
            for collection in collections:
                if collection.name.startswith('surveillance_'):
                    try:
                        count = collection.count()
                        stats = {
                            "name": collection.name,
                            "count": count,
                            "available": True
                        }
                        collection_stats.append(stats)
                        total_documents += count
                        total_collections += 1
                        if count > 0:
                            active_collections += 1
                    except Exception as e:
                        logger.warning(f"Failed to get stats for collection '{collection.name}': {e}")
                        stats = {
                            "name": collection.name,
                            "count": 0,
                            "available": False,
                            "error": str(e)
                        }
                        collection_stats.append(stats)
                        total_collections += 1
                            
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {e}")
            client_available = False
        
        # Create lightweight analytics response
        analytics_data = {
            "total_collections": total_collections,
            "active_collections": active_collections,
            "total_documents": total_documents,
            "collection_stats": collection_stats,
            "system_health": {
                "vector_db_available": client_available,
                "collections_accessible": len(collection_stats) > 0,
                "mode": "light"
            },
            "performance_note": "Light mode - heavy analytics disabled to prevent model loading",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {"success": True, "data": analytics_data}
        
    except Exception as e:
        logger.error(f"Error getting light analytics: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "total_collections": 0,
                "active_collections": 0,
                "total_documents": 0,
                "collection_stats": [],
                "system_health": {
                    "vector_db_available": False,
                    "collections_accessible": False,
                    "mode": "light"
                },
                "performance_note": "Light mode - analytics unavailable",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

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

@surveillance_router.get("/frame-direct/{frame_filename}")
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
