from celery import Celery
from celery.result import AsyncResult
from typing import Dict, Any, List
import uuid
import os
import time
import logging
import json

logger = logging.getLogger(__name__)

# Celery configuration with simplified backend
def create_celery_app():
    # Use Redis as broker and backend (fallback to in-memory for development)
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    celery_app = Celery(
        'surveillance_worker',
        broker=redis_url,
        backend=redis_url,
        include=['src.services.job_queue']
    )
    
    # Simplified configuration to avoid serialization issues
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour timeout
        task_soft_time_limit=3300,  # 55 minutes soft timeout
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_ignore_result=False,
        result_expires=3600,
        task_reject_on_worker_lost=True,
        # Add these to fix serialization issues
        result_backend_transport_options={
            'master_name': 'mymaster',
        },
        task_always_eager=False,  # Don't execute tasks eagerly
        task_eager_propagates=False,
        task_store_eager_result=False,
    )
    
    return celery_app

# Global Celery app instance
celery_app = create_celery_app()

class JobManager:
    """Manages background job processing"""
    
    def __init__(self):
        self.celery_app = celery_app
    
    def submit_video_processing_job(self, project_id: str, file_id: str, 
                                  sample_rate: float = 1.0, 
                                  detection_threshold: float = 0.5,
                                  enable_tracking: bool = True,
                                  enable_captioning: bool = True) -> str:
        """
        Submit a video processing job to the queue
        
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Submit the job with simplified parameters
        try:
            task = self.celery_app.send_task(
                'src.services.job_queue.process_video_task',
                args=[project_id, file_id, sample_rate, detection_threshold, enable_tracking, enable_captioning],
                task_id=job_id,
                countdown=1,  # Start after 1 second
                retry=False  # Don't retry on failure
            )
            logger.info(f"Submitted video processing job {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    def submit_live_stream_job(self, project_id: str, stream_url: str, 
                             sample_rate: float = 1.0) -> str:
        """
        Submit a live stream processing job
        
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        
        try:
            self.celery_app.send_task(
                'src.services.job_queue.process_live_stream_task',
                args=[project_id, stream_url, sample_rate],
                task_id=job_id,
                retry=False
            )
            return job_id
        except Exception as e:
            logger.error(f"Failed to submit live stream job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job with better error handling"""
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            
            # Get basic status info
            status = result.status
            ready = result.ready()
            
            status_info = {
                'job_id': job_id,
                'status': status,
                'ready': ready,
                'successful': None,
                'failed': None,
                'result': None,
                'error': None,
                'traceback': None
            }
            
            # Only try to get detailed info if the task is ready
            if ready:
                try:
                    status_info['successful'] = result.successful()
                    status_info['failed'] = result.failed()
                    
                    if result.successful():
                        status_info['result'] = result.result
                    elif result.failed():
                        # Try to get error info safely
                        try:
                            status_info['error'] = str(result.result)
                            status_info['traceback'] = result.traceback
                        except Exception:
                            status_info['error'] = 'Unknown error occurred'
                            
                except Exception as e:
                    logger.warning(f"Could not get detailed status for job {job_id}: {e}")
                    status_info['error'] = f'Status retrieval failed: {str(e)}'
            
            logger.info(f"JOB STATUS: {status_info}")
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return {
                'job_id': job_id,
                'status': 'UNKNOWN',
                'ready': False,
                'error': f'Failed to get job status: {str(e)}'
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        try:
            self.celery_app.control.revoke(job_id, terminate=True)
            logger.info(f"Cancelled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active jobs"""
        try:
            # Get active tasks from all workers
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return []
            
            all_jobs = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    all_jobs.append({
                        'job_id': task['id'],
                        'name': task['name'],
                        'worker': worker,
                        'args': task.get('args', []),
                        'kwargs': task.get('kwargs', {}),
                        'started': task.get('time_start'),
                    })
            
            return all_jobs
            
        except Exception as e:
            logger.error(f"Failed to get active jobs: {e}")
            return []

# Global job manager instance
job_manager = JobManager()

# Task definitions with better error handling
@celery_app.task(bind=True, name='src.services.job_queue.process_video_task', acks_late=True)
def process_video_task(self, project_id: str, file_id: str, sample_rate: float, 
                      detection_threshold: float, enable_tracking: bool, enable_captioning: bool):
    """
    Background task for processing surveillance video with proper result handling
    """
    start_time = time.time()
    task_id = self.request.id
    
    try:
        logger.info(f"[{task_id}] Starting video processing for {project_id}/{file_id}")
        
        # Import here to avoid import issues at module level
        try:
            from src.controllers.ProcessController import ProcessController
        except ImportError:
            from controllers.ProcessController import ProcessController
        
        # Initialize controller
        process_controller = ProcessController(project_id=project_id)
        
        # Process the video
        result = process_controller.process_video(
            file_id=file_id, 
            sample_rate=sample_rate,
            detection_threshold=detection_threshold,
            enable_tracking=enable_tracking,
            enable_captioning=enable_captioning
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create a clean, serializable result with minimal data
        success_result = {
            'success': True,
            'status': 'completed',
            'project_id': str(project_id),
            'file_id': str(file_id),
            'task_id': str(task_id),
            'processing_time': round(processing_time, 2),
            'processed_frames': int(result.get('processed_frames', 0)),
            'total_extracted_frames': int(result.get('total_extracted_frames', 0)),
            'success_rate': round(float(result.get('success_rate', 0)), 2)
        }
        
        # Add minimal analytics to avoid serialization issues
        success_result['analytics'] = result.get('analytics', {})
        
        logger.info(f"[{task_id}] Video processing completed successfully: {success_result['processed_frames']} frames in {processing_time:.2f}s")
        
        return success_result
        
    except Exception as e:
        processing_time = time.time() - start_time 
        error_msg = str(e)[:200]  # Limit error message length
        
        logger.error(f"[{task_id}] Video processing failed: {error_msg}", exc_info=True)
        
        # Return minimal error result
        error_result = {
            'success': False,
            'status': 'failed',
            'error': error_msg,
            'project_id': str(project_id),
            'file_id': str(file_id),
            'task_id': str(task_id),
            'processing_time': round(processing_time, 2)
        }
        
        return error_result

@celery_app.task(bind=True, name='src.services.job_queue.process_live_stream_task', acks_late=True)
def process_live_stream_task(self, project_id: str, stream_url: str, sample_rate: float):
    """
    Background task for processing live surveillance stream
    """
    task_id = self.request.id
    
    try:
        logger.info(f"[{task_id}] Starting live stream processing")
        
        # TODO: Implement live stream processing here
        return {
            'success': True,
            'project_id': str(project_id),
            'stream_url': str(stream_url),
            'status': 'running',
            'task_id': str(task_id)
        }
        
    except Exception as e:
        error_msg = str(e)[:200]
        logger.error(f"[{task_id}] Live stream processing failed: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'status': 'failed',
            'project_id': str(project_id),
            'stream_url': str(stream_url),
            'task_id': str(task_id)
        }
