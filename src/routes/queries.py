from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional, List
import logging
from datetime import datetime

from controllers.QueryController import QueryController
from models.schemas import QueryResponse, QueryResult, APIResponse
from services.auth import get_optional_user, get_current_user
from helpers.config import get_settings, Settings

logger = logging.getLogger(__name__)

queries_router = APIRouter(
    prefix="/api/surveillance/query",
    tags=["queries"]
)

@queries_router.get("", response_model=QueryResponse)
async def query_surveillance(
    request: Request,
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
        
        # Use pre-initialized vector controller from app state
        vector_controller = request.app.state.vector_db_controller
        if vector_controller is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Initialize query controller with pre-loaded vector controller
        query_controller = QueryController(
            project_id=project_id,
            vector_controller=vector_controller
        )
        
        # Execute query
        results = query_controller.process_query(
            query_text=query,
            max_results=max_results,
            project_id=project_id
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time_query).total_seconds()
        
        # Convert to response format
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
        raise HTTPException(status_code=500, detail=str(e))

@queries_router.get("/timeline/{project_id}")
async def get_timeline_events(
    project_id: str,
    start_time: datetime = Query(..., description="Start time"),
    end_time: datetime = Query(..., description="End time"),
    object_types: Optional[List[str]] = Query(None, description="Filter by object types"),
    user: dict = Depends(get_current_user)
):
    """Get timeline of surveillance events in a time range"""
    try:
        return APIResponse(
            success=True,
            data={
                "project_id": project_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "events": [],
                "note": "TODO: Implement timeline query from database"
            }
        )
    except Exception as e:
        logger.error(f"Error getting timeline for {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
