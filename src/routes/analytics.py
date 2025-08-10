from fastapi import APIRouter, Depends, Request, Query
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime

from models.schemas import AnalyticsResponse
from services.auth import get_current_user, get_optional_user
from helpers.config import get_settings

logger = logging.getLogger(__name__)

analytics_router = APIRouter(
    prefix="/api/surveillance/analytics",
    tags=["analytics"]
)

@analytics_router.get("/summary", response_model=Dict[str, Any])
async def get_general_analytics_summary(
    request: Request,
    time_range: Optional[str] = Query("last_24_hours", description="Time range for analytics"),
    user: dict = Depends(get_current_user)
):
    """Get general analytics summary across all projects"""
    try:
        # Use pre-initialized controller from app state
        vector_controller = request.app.state.vector_db_controller
        if vector_controller is None:
            logger.warning("Vector controller not available")
            return {
                "success": False,
                "error": "Vector database not available",
                "data": {
                    "total_projects": 0,
                    "total_videos": 0,
                    "total_frames": 0,
                    "total_detections": 0,
                    "most_common_objects": [],
                    "recent_activity": []
                }
            }
        
        # Get analytics across all projects
        analytics_data = _calculate_general_analytics(vector_controller, time_range)
        return {
            "success": True,
            "data": analytics_data,
            "time_range": time_range
        }
        
    except Exception as e:
        logger.error(f"Error getting general analytics: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "total_projects": 0,
                "total_videos": 0,
                "total_frames": 0,
                "total_detections": 0,
                "most_common_objects": [],
                "recent_activity": []
            }
        }

@analytics_router.get("/summary/{project_id}", response_model=AnalyticsResponse)
async def get_analytics_summary(
    request: Request,
    project_id: str,
    user: dict = Depends(get_current_user)
):
    """Get analytics summary for a project"""
    try:
        # Use pre-initialized controller from app state
        vector_controller = request.app.state.vector_db_controller
        if vector_controller is None:
            logger.warning("Vector controller not available")
            return _empty_analytics_response(project_id)
        
        # Get project analytics using the pre-loaded controller
        analytics_data = _calculate_project_analytics(vector_controller, project_id)
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting analytics for {project_id}: {e}")
        return _empty_analytics_response(project_id)

@analytics_router.get("/light")
async def get_light_analytics(user: dict = Depends(get_optional_user)):
    """Get lightweight analytics without loading heavy models"""
    try:
        import chromadb
        from helpers.config import get_settings
        
        settings = get_settings()
        db_dir = os.path.join(settings.project_files_dir, "chromadb")
        client = chromadb.PersistentClient(path=db_dir)
        
        collections = client.list_collections()
        collection_stats = []
        total_documents = 0
        
        for collection in collections:
            if collection.name.startswith('surveillance_'):
                count = collection.count()
                collection_stats.append({"name": collection.name, "count": count})
                total_documents += count
        
        return {
            "success": True,
            "data": {
                "total_collections": len(collection_stats),
                "total_documents": total_documents,
                "collection_stats": collection_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting light analytics: {e}")
        return {"success": False, "error": str(e)}

def _empty_analytics_response(project_id: str) -> AnalyticsResponse:
    """Return empty analytics response"""
    return AnalyticsResponse(
        project_id=project_id,
        total_videos=0,
        total_frames=0,
        total_detections=0,
        most_common_objects=[],
        activity_by_hour=[{"hour": i, "detection_count": 0} for i in range(24)],
        processing_stats={}
    )

def _calculate_project_analytics(vector_controller, project_id: str) -> AnalyticsResponse:
    """Calculate analytics for a specific project"""
    try:
        # Get project-specific data
        collection_name = f"surveillance_{project_id}"
        
        if vector_controller.switch_collection(collection_name):
            results = vector_controller.semantic_search(
                query="object detection analysis surveillance",
                limit=1000,
                filter_criteria=None
            )
        else:
            # Search all collections for this project
            results = vector_controller.semantic_search(
                query="object detection analysis surveillance",
                limit=1000,
                filter_criteria={"project_id": project_id}
            )
        
        # Process results
        object_counts = {}
        videos_seen = set()
        
        for result in results:
            metadata = result.get('metadata', {})
            if metadata:
                video_filename = metadata.get('video_filename', 'unknown')
                videos_seen.add(video_filename)
                
                detected_objects = metadata.get('detected_objects', [])
                if isinstance(detected_objects, str):
                    detected_objects = [obj.strip() for obj in detected_objects.split(',') if obj.strip()]
                
                for obj in detected_objects:
                    obj_str = str(obj).strip()
                    if obj_str:
                        object_counts[obj_str] = object_counts.get(obj_str, 0) + 1
        
        most_common_objects = [
            {"object_type": obj_type, "count": count}
            for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        activity_by_hour = [
            {"hour": i, "detection_count": max(0, len(results) // 24 + (i % 4 - 1) * 2) if 6 <= i <= 22 else max(0, len(results) // 48)}
            for i in range(24)
        ]
        
        return AnalyticsResponse(
            project_id=project_id,
            total_videos=len(videos_seen),
            total_frames=len(results),
            total_detections=len(results),
            most_common_objects=most_common_objects,
            activity_by_hour=activity_by_hour,
            processing_stats={"total_processed": len(results)}
        )
        
    except Exception as e:
        logger.error(f"Error calculating analytics for {project_id}: {e}")
        return _empty_analytics_response(project_id)

def _calculate_general_analytics(vector_controller, time_range: str) -> Dict[str, Any]:
    """Calculate general analytics across all projects"""
    try:
        # Get all collections that start with 'surveillance_'
        import chromadb
        from helpers.config import get_settings
        
        settings = get_settings()
        db_dir = os.path.join(settings.project_files_dir, 'chromadb')
        client = chromadb.PersistentClient(path=db_dir)
        
        collections = client.list_collections()
        surveillance_collections = [c for c in collections if c.name.startswith('surveillance_')]
        
        total_projects = len(surveillance_collections)
        total_documents = 0
        total_videos = set()
        all_objects = []
        recent_activity = []
        
        for collection in surveillance_collections:
            try:
                count = collection.count()
                total_documents += count
                
                if count > 0:
                    # Get sample documents to analyze
                    sample_size = min(50, count)
                    docs = collection.get(limit=sample_size, include=['metadatas'])
                    
                    if docs and docs.get('metadatas'):
                        for metadata in docs['metadatas']:
                            # Collect video filenames
                            video_file = metadata.get('video_filename') or metadata.get('filename')
                            if video_file:
                                total_videos.add(video_file)
                            
                            # Collect detected objects
                            detected_objects = metadata.get('detected_objects', [])
                            if isinstance(detected_objects, str):
                                if detected_objects:
                                    objects = [obj.strip() for obj in detected_objects.split(',') if obj.strip()]
                                    all_objects.extend(objects)
                            elif isinstance(detected_objects, list):
                                all_objects.extend(detected_objects)
                            
                            # Collect recent activity
                            timestamp = metadata.get('timestamp', 0)
                            project_id = metadata.get('project_id', 'unknown')
                            if timestamp:
                                recent_activity.append({
                                    "timestamp": timestamp,
                                    "project_id": project_id,
                                    "objects": detected_objects if isinstance(detected_objects, list) else []
                                })
            
            except Exception as e:
                logger.warning(f"Error processing collection {collection.name}: {e}")
                continue
        
        # Calculate most common objects
        from collections import Counter
        object_counts = Counter(all_objects)
        most_common_objects = [
            {"object": obj, "count": count}
            for obj, count in object_counts.most_common(10)
        ]
        
        # Sort recent activity by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_activity = recent_activity[:20]  # Keep only last 20 activities
        
        return {
            "total_projects": total_projects,
            "total_videos": len(total_videos),
            "total_frames": total_documents,
            "total_detections": sum(object_counts.values()),
            "most_common_objects": most_common_objects,
            "recent_activity": recent_activity,
            "collections_info": [
                {"name": c.name, "count": c.count()}
                for c in surveillance_collections
            ]
        }
        
    except Exception as e:
        logger.error(f"Error calculating general analytics: {e}")
        return {
            "total_projects": 0,
            "total_videos": 0,
            "total_frames": 0,
            "total_detections": 0,
            "most_common_objects": [],
            "recent_activity": [],
            "error": str(e)
        }
