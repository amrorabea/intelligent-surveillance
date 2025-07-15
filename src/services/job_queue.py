from celery import Celery
from celery.result import AsyncResult
from typing import Dict, Any, List
import uuid
import os

# Celery configuration
def create_celery_app():
    # Use Redis as broker and backend (fallback to in-memory for development)
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    celery_app = Celery(
        'surveillance_worker',
        broker=redis_url,
        backend=redis_url,
        include=['services.job_queue']
    )
    
    # Configuration
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
        worker_disable_rate_limits=False,
        task_default_retry_delay=60,
        task_max_retries=3,
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
        
        # Submit the job
        self.celery_app.send_task(
            'services.job_queue.process_video_task',
            args=[project_id, file_id, sample_rate, detection_threshold, enable_tracking, enable_captioning],
            task_id=job_id,
            countdown=1  # Start after 1 second
        )
        
        return job_id
    
    def submit_live_stream_job(self, project_id: str, stream_url: str, 
                             sample_rate: float = 1.0) -> str:
        """
        Submit a live stream processing job
        
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        
        self.celery_app.send_task(
            'services.job_queue.process_live_stream_task',
            args=[project_id, stream_url, sample_rate],
            task_id=job_id
        )
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job"""
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            
            status_info = {
                'job_id': job_id,
                'status': result.status,
                'ready': result.ready(),
                'successful': result.successful() if result.ready() else None,
                'failed': result.failed() if result.ready() else None,
                'result': result.result if result.ready() and result.successful() else None,
                'error': str(result.result) if result.ready() and result.failed() else None,
                'traceback': result.traceback if result.ready() and result.failed() else None
            }
            
            # Get progress info if available
            if hasattr(result, 'info') and result.info:
                if isinstance(result.info, dict):
                    status_info.update(result.info)
            
            return status_info
            
        except Exception as e:
            return {
                'job_id': job_id,
                'status': 'UNKNOWN',
                'error': f'Failed to get job status: {str(e)}'
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        try:
            self.celery_app.control.revoke(job_id, terminate=True)
            return True
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")
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
            print(f"Failed to get active jobs: {e}")
            return []

# Global job manager instance
job_manager = JobManager()

# Task definitions (these will be imported by Celery workers)
@celery_app.task(bind=True)
def process_video_task(self, project_id: str, file_id: str, sample_rate: float, 
                      detection_threshold: float, enable_tracking: bool, enable_captioning: bool):
    """
    Background task for processing surveillance video
    
    This is a placeholder - you'll implement the actual processing logic
    """
    try:
        # Update task state to indicate it has started
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting video processing...'}
        )
        
        # TODO: Implement actual video processing here
        # This is where you'll call your ProcessController.process_video() method
        # and update progress throughout the processing
        
        # Example progress updates:
        # self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'status': 'Extracting frames...'})
        # self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'status': 'Running object detection...'})
        # self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'status': 'Generating captions...'})
        
        # Placeholder result
        result = {
            'project_id': project_id,
            'file_id': file_id,
            'processed_frames': 0,  # TODO: Update with actual count
            'total_detections': 0,  # TODO: Update with actual count
            'processing_time': 0.0,  # TODO: Calculate actual time
            'status': 'completed'
        }
        
        return result
        
    except Exception as e:
        # Update task state to failed
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'Processing failed'}
        )
        raise

@celery_app.task(bind=True)
def process_live_stream_task(self, project_id: str, stream_url: str, sample_rate: float):
    """
    Background task for processing live surveillance stream
    
    This is a placeholder - you'll implement the actual live stream processing
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': -1, 'status': 'Starting live stream processing...'}
        )
        
        # TODO: Implement live stream processing here
        # This would typically run continuously until stopped
        
        return {
            'project_id': project_id,
            'stream_url': stream_url,
            'status': 'running'
        }
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'status': 'Live stream processing failed'}
        )
        raise
