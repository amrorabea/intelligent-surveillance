try:
    # Try absolute imports first (for Celery running from project root)
    from src.controllers.BaseController import BaseController
    from src.controllers.VisionController import VisionController
    from src.controllers.VectorDBController import VectorDBController
    from src.controllers.TrackingController import TrackingController
    from src.controllers.ProjectController import ProjectController
    from src.models.enums.ProcessingEnum import ProcessingEnum
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from .BaseController import BaseController
    from .VisionController import VisionController
    from .VectorDBController import VectorDBController
    from .TrackingController import TrackingController
    from .ProjectController import ProjectController
    from models.enums.ProcessingEnum import ProcessingEnum

import os
import cv2
import logging
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        
        # Create frames directory for extracted frames
        self.frames_dir = os.path.join(self.project_path, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def is_video_file(self, file_id: str):
        """Check if file is a supported video format"""
        file_ext = self.get_file_extension(file_id)
        return file_ext in [ProcessingEnum.MP4.value, ProcessingEnum.AVI.value, ProcessingEnum.MOV.value]

    def is_image_file(self, file_id: str):
        """Check if file is a supported image format"""
        file_ext = self.get_file_extension(file_id)
        return file_ext in [ProcessingEnum.PNG.value, ProcessingEnum.JPG.value, ProcessingEnum.JPEG.value]

    def extract_frames_from_video(self, video_path, sample_rate=1.0):
        """
        Extract frames from video at specified sample rate
        
        Args:
            video_path (str): Path to video file
            sample_rate (float): Frames per second to extract (1.0 = 1 frame per second)
            
        Returns:
            list: List of extracted frame information with timestamps
        """
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video {video_path}")
                return []
                
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval based on sample rate
            frame_interval = int(fps / sample_rate) if sample_rate > 0 else int(fps)
            
            extracted_frames = []
            frame_pos = 0
            
            while frame_pos < frame_count:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                success, frame = video.read()
                
                if success:
                    timestamp = frame_pos / fps
                    
                    # Save frame to disk
                    frame_path = self._save_frame_to_disk(frame, video_path, timestamp)
                    
                    if frame_path:
                        extracted_frames.append({
                            'timestamp': timestamp,
                            'frame_path': frame_path,
                            'frame_number': frame_pos
                        })
                
                frame_pos += frame_interval
            
            video.release()
            return extracted_frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

    def _save_frame_to_disk(self, frame, video_path, timestamp):
        """
        Save a single frame to disk
        
        Args:
            frame: OpenCV frame array
            video_path (str): Original video path
            timestamp (float): Timestamp of the frame
            
        Returns:
            str: Path to saved frame or None if failed
        """
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = f"{video_name}_frame_{timestamp:.2f}s.jpg"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            
            # Check if frame already exists
            if os.path.exists(frame_path):
                return frame_path
                
            # Save frame
            success = cv2.imwrite(frame_path, frame)
            return frame_path if success else None
            
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None

    def process_video(self, file_id: str, sample_rate=1.0, detection_threshold=0.5, 
                     enable_tracking=True, enable_captioning=True, progress_callback=None):
        """
        Enhanced video processing through the complete surveillance AI pipeline
        
        Args:
            file_id (str): ID of the video file to process
            sample_rate (float): Frames per second to extract and analyze
            detection_threshold (float): Confidence threshold for object detection
            enable_tracking (bool): Whether to enable object tracking
            enable_captioning (bool): Whether to generate captions
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            dict: Comprehensive processing results with analytics
        """
        start_time = time.time()
        file_path = os.path.join(self.project_path, file_id)
        
        # Initialize result structure
        result = {
            "file_id": file_id,
            "start_time": datetime.now().isoformat(),
            "status": "processing",
            "error": None,
            "video_info": {},
            "processing_stats": {},
            "analytics": {},
            "sample_results": [],
            "performance_metrics": {}
        }
        
        try:
            # Validate input file
            if not os.path.exists(file_path):
                result.update({
                    "status": "error",
                    "error": f"Video file {file_id} not found",
                    "processed_frames": 0
                })
                return result
                
            if not self.is_video_file(file_id):
                result.update({
                    "status": "error", 
                    "error": f"File {file_id} is not a supported video format",
                    "processed_frames": 0
                })
                return result
            
            # Extract video metadata
            video_info = self._extract_video_metadata(file_path)
            result["video_info"] = video_info
            
            logger.info(f"Processing video: {file_path}")
            logger.info(f"Video info: {video_info['duration']:.2f}s, {video_info['fps']:.1f} FPS, {video_info['total_frames']} frames")
            
            # Step 1: Extract frames with progress tracking
            if progress_callback:
                progress_callback({"stage": "extraction", "progress": 0, "message": "Starting frame extraction"})
            
            extracted_frames = self.extract_frames_from_video(file_path, sample_rate)
            
            if not extracted_frames:
                result.update({
                    "status": "error",
                    "error": "No frames could be extracted from video",
                    "processed_frames": 0
                })
                return result
            
            logger.info(f"Extracted {len(extracted_frames)} frames for processing")
            
            # Step 2: Initialize AI controllers with enhanced configuration
            vision_controller = VisionController(
                detection_threshold=detection_threshold,
                max_detections=100  # Increase for surveillance scenarios
            )
            
            tracking_controller = TrackingController() if enable_tracking else None
            vector_db = VectorDBController(collection_name=f"surveillance_{self.project_id}")
            
            # Step 3: Enhanced processing pipeline
            processed_results = []
            analytics_data = self._initialize_analytics()
            performance_metrics = {"detection_times": [], "caption_times": [], "storage_times": []}
            
            total_frames = len(extracted_frames)
            
            for frame_idx, frame_info in enumerate(extracted_frames):
                try:
                    frame_start_time = time.time()
                    timestamp = frame_info['timestamp']
                    frame_path = frame_info['frame_path']
                    
                    # Progress update
                    progress = (frame_idx + 1) / total_frames
                    if progress_callback:
                        progress_callback({
                            "stage": "processing",
                            "progress": progress,
                            "frame": frame_idx + 1,
                            "total": total_frames,
                            "message": f"Processing frame at {timestamp:.2f}s"
                        })
                    
                    # A. Enhanced Object Detection with timing
                    detection_start = time.time()
                    detections = vision_controller.detect_objects(frame_path, detection_threshold)
                    detection_time = time.time() - detection_start
                    performance_metrics["detection_times"].append(detection_time)
                    
                    # Update analytics
                    self._update_detection_analytics(analytics_data, detections)
                    
                    logger.debug(f"Frame {timestamp:.2f}s: {detections.get('total_objects', 0)} objects detected")
                    
                    # B. Advanced Object Tracking
                    tracking_results = {}
                    if enable_tracking and tracking_controller:
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            # Extract and format bounding boxes
                            bboxes = self._extract_bboxes_for_tracking(detections)
                            tracking_results = tracking_controller.track_objects(frame, bboxes)
                            
                            # Update analytics with tracking info
                            self._update_tracking_analytics(analytics_data, tracking_results)
                        else:
                            tracking_results = {"tracked_objects": {}, "total_tracked": 0}
                    
                    # C. Enhanced Scene Captioning
                    caption = ""
                    caption_time = 0
                    if enable_captioning:
                        caption_start = time.time()
                        # Use surveillance-focused captioning with detection context
                        caption_result = vision_controller.generate_surveillance_caption(frame_path, detections)
                        caption = caption_result.get('text', '') if isinstance(caption_result, dict) else str(caption_result)
                        caption_time = time.time() - caption_start
                        performance_metrics["caption_times"].append(caption_time)
                    
                    # D. Enhanced Vector Database Storage
                    storage_start = time.time()
                    storage_success = False
                    
                    if caption and len(caption.strip()) > 0:
                        # Enhanced metadata with comprehensive context
                        metadata = self._create_enhanced_metadata(
                            file_id, timestamp, frame_path, frame_info, 
                            detections, tracking_results
                        )
                        
                        try:
                            storage_success = vector_db.store_embedding(
                                document_id=f"{file_id}_frame_{timestamp:.2f}s",
                                text=caption,
                                metadata=metadata
                            )
                        except Exception as e:
                            logger.error(f"Error storing embedding for frame {timestamp:.2f}s: {e}")
                            storage_success = False
                    
                    storage_time = time.time() - storage_start
                    performance_metrics["storage_times"].append(storage_time)
                    
                    # Compile comprehensive frame results
                    frame_result = {
                        "timestamp": timestamp,
                        "frame_path": frame_path,
                        "frame_number": frame_info['frame_number'],
                        "detections": detections,
                        "tracking": tracking_results,
                        "caption": caption,
                        "stored": storage_success,
                        "processing_time": time.time() - frame_start_time,
                        "performance": {
                            "detection_time": detection_time,
                            "caption_time": caption_time,
                            "storage_time": storage_time
                        }
                    }
                    
                    processed_results.append(frame_result)
                    
                    # Store sample results (first 5 and last 5 frames)
                    if len(result["sample_results"]) < 5 or frame_idx >= total_frames - 5:
                        result["sample_results"].append(frame_result)
                    
                except Exception as e:
                    logger.error(f"Error processing frame at {timestamp}s: {e}")
                    analytics_data["errors"].append({
                        "timestamp": timestamp,
                        "error": str(e),
                        "frame_path": frame_path
                    })
                    continue
            
            # Step 4: Generate comprehensive analytics and final results
            processing_time = time.time() - start_time
            
            result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "processing_time": processing_time,
                "processed_frames": len(processed_results),
                "total_extracted_frames": len(extracted_frames),
                "success_rate": len(processed_results) / len(extracted_frames) if extracted_frames else 0,
                "processing_stats": self._generate_processing_stats(processed_results, analytics_data),
                "analytics": self._finalize_analytics(analytics_data, video_info),
                "performance_metrics": self._calculate_performance_metrics(performance_metrics, processing_time)
            })
            
            # Final progress update
            if progress_callback:
                progress_callback({
                    "stage": "completed",
                    "progress": 1.0,
                    "message": f"Processing completed: {len(processed_results)} frames processed"
                })
            
            logger.info(f"Video processing completed: {len(processed_results)}/{len(extracted_frames)} frames processed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in video processing: {e}")
            result.update({
                "status": "error",
                "error": str(e),
                "end_time": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            })
            return result
    
    def _extract_video_metadata(self, video_path):
        """Extract comprehensive video metadata"""
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                return {"error": "Could not open video"}
            
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            video.release()
            
            return {
                "fps": fps,
                "total_frames": frame_count,
                "duration": duration,
                "resolution": {"width": width, "height": height},
                "file_size": os.path.getsize(video_path),
                "format": os.path.splitext(video_path)[1]
            }
        except Exception as e:
            return {"error": f"Failed to extract metadata: {e}"}
    
    def _initialize_analytics(self):
        """Initialize analytics data structure"""
        return {
            "object_detections": {},
            "confidence_scores": [],
            "tracking_performance": [],
            "caption_lengths": [],
            "errors": [],
            "processing_times": []
        }
    
    def _update_detection_analytics(self, analytics_data, detections):
        """Update analytics with detection results"""
        if detections and 'detections' in detections:
            for detection in detections['detections']:
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0)
                
                # Count object classes
                if class_name not in analytics_data["object_detections"]:
                    analytics_data["object_detections"][class_name] = 0
                analytics_data["object_detections"][class_name] += 1
                
                # Track confidence scores
                analytics_data["confidence_scores"].append(confidence)
    
    def _update_tracking_analytics(self, analytics_data, tracking_results):
        """Update analytics with tracking results"""
        if tracking_results:
            total_tracked = tracking_results.get("total_tracked", 0)
            analytics_data["tracking_performance"].append(total_tracked)
    
    def _extract_bboxes_for_tracking(self, detections):
        """Extract bounding boxes in tracking format"""
        bboxes = []
        if detections and 'detections' in detections:
            for detection in detections['detections']:
                if 'bbox' in detection:
                    bbox_data = detection['bbox']
                    bbox = [
                        bbox_data['x1'], bbox_data['y1'], 
                        bbox_data['x2'], bbox_data['y2']
                    ]
                    bboxes.append(bbox)
        return bboxes
    
    def _create_enhanced_metadata(self, file_id, timestamp, frame_path, frame_info, detections, tracking_results):
        """Create comprehensive metadata for vector storage"""
        metadata = {
            "file_id": file_id,
            "project_id": self.project_id,
            "timestamp": timestamp,
            "frame_path": frame_path,
            "frame_number": frame_info['frame_number'],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Add detection information
        if detections and 'detections' in detections:
            metadata.update({
                "detected_objects": [d.get('class', 'unknown') for d in detections['detections']],
                "total_detections": len(detections['detections']),
                "detection_confidence_avg": sum(d.get('confidence', 0) for d in detections['detections']) / len(detections['detections']) if detections['detections'] else 0,
                "high_confidence_objects": [d.get('class') for d in detections['detections'] if d.get('confidence', 0) > 0.8]
            })
        else:
            metadata.update({
                "detected_objects": [],
                "total_detections": 0,
                "detection_confidence_avg": 0,
                "high_confidence_objects": []
            })
        
        # Add tracking information
        if tracking_results:
            metadata.update({
                "tracked_objects": tracking_results.get("total_tracked", 0),
                "new_tracks": tracking_results.get("new_tracks", 0),
                "continued_tracks": tracking_results.get("continued_tracks", 0)
            })
        
        return metadata
    
    def _generate_processing_stats(self, processed_results, analytics_data):
        """Generate comprehensive processing statistics"""
        if not processed_results:
            return {}
        
        return {
            "frames_processed": len(processed_results),
            "frames_with_detections": sum(1 for r in processed_results if r['detections'].get('total_objects', 0) > 0),
            "frames_with_tracking": sum(1 for r in processed_results if r['tracking'].get('total_tracked', 0) > 0),
            "frames_successfully_stored": sum(1 for r in processed_results if r.get('stored', False)),
            "total_objects_detected": sum(r['detections'].get('total_objects', 0) for r in processed_results),
            "average_objects_per_frame": sum(r['detections'].get('total_objects', 0) for r in processed_results) / len(processed_results),
            "error_count": len(analytics_data["errors"])
        }
    
    def _finalize_analytics(self, analytics_data, video_info):
        """Generate final analytics summary"""
        analytics = {
            "object_distribution": analytics_data["object_detections"],
            "most_detected_objects": sorted(analytics_data["object_detections"].items(), key=lambda x: x[1], reverse=True)[:10],
            "confidence_statistics": {},
            "tracking_statistics": {},
            "video_coverage": {}
        }
        
        # Confidence statistics
        if analytics_data["confidence_scores"]:
            scores = analytics_data["confidence_scores"]
            analytics["confidence_statistics"] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "high_confidence_rate": sum(1 for s in scores if s > 0.8) / len(scores)
            }
        
        # Tracking statistics
        if analytics_data["tracking_performance"]:
            track_counts = analytics_data["tracking_performance"]
            analytics["tracking_statistics"] = {
                "average_tracks_per_frame": sum(track_counts) / len(track_counts),
                "max_concurrent_tracks": max(track_counts),
                "frames_with_tracking": sum(1 for t in track_counts if t > 0)
            }
        
        # Video coverage analysis
        if video_info:
            analytics["video_coverage"] = {
                "total_duration": video_info.get("duration", 0),
                "sampling_efficiency": len(analytics_data["processing_times"]) / video_info.get("total_frames", 1) if video_info.get("total_frames") else 0,
                "resolution": video_info.get("resolution", {}),
                "processing_ratio": sum(analytics_data["processing_times"]) / video_info.get("duration", 1) if video_info.get("duration") else 0
            }
        
        return analytics
    
    def _calculate_performance_metrics(self, performance_metrics, total_processing_time):
        """Calculate comprehensive performance metrics"""
        metrics = {
            "total_processing_time": total_processing_time,
            "average_frame_processing_time": 0,
            "detection_performance": {},
            "caption_performance": {},
            "storage_performance": {}
        }
        
        # Detection performance
        if performance_metrics["detection_times"]:
            det_times = performance_metrics["detection_times"]
            metrics["detection_performance"] = {
                "average_time": sum(det_times) / len(det_times),
                "min_time": min(det_times),
                "max_time": max(det_times),
                "total_time": sum(det_times)
            }
        
        # Caption performance
        if performance_metrics["caption_times"]:
            cap_times = performance_metrics["caption_times"]
            metrics["caption_performance"] = {
                "average_time": sum(cap_times) / len(cap_times),
                "min_time": min(cap_times),
                "max_time": max(cap_times),
                "total_time": sum(cap_times)
            }
        
        # Storage performance
        if performance_metrics["storage_times"]:
            stor_times = performance_metrics["storage_times"]
            metrics["storage_performance"] = {
                "average_time": sum(stor_times) / len(stor_times),
                "min_time": min(stor_times),
                "max_time": max(stor_times),
                "total_time": sum(stor_times)
            }
        
        # Calculate efficiency metrics
        all_times = (performance_metrics["detection_times"] + 
                    performance_metrics["caption_times"] + 
                    performance_metrics["storage_times"])
        
        if all_times:
            metrics["average_frame_processing_time"] = sum(all_times) / (len(all_times) // 3)  # Approximate
            metrics["processing_efficiency"] = {
                "frames_per_second": len(performance_metrics["detection_times"]) / total_processing_time if total_processing_time > 0 else 0,
                "bottleneck_analysis": {
                    "detection_percentage": sum(performance_metrics["detection_times"]) / total_processing_time * 100 if total_processing_time > 0 else 0,
                    "caption_percentage": sum(performance_metrics["caption_times"]) / total_processing_time * 100 if total_processing_time > 0 else 0,
                    "storage_percentage": sum(performance_metrics["storage_times"]) / total_processing_time * 100 if total_processing_time > 0 else 0
                }
            }
        
        return metrics

    def get_processing_summary(self, file_id: str):
        """Get a summary of processing results for a specific video"""
        try:
            # This could be implemented to read from a processing log file
            # or database to provide historical processing information
            frames_dir = os.path.join(self.frames_dir)
            if not os.path.exists(frames_dir):
                return {"error": "No processing data found"}
            
            # Count processed frames for this video
            video_name = os.path.splitext(file_id)[0]
            frame_files = [f for f in os.listdir(frames_dir) if f.startswith(video_name)]
            
            return {
                "file_id": file_id,
                "extracted_frames": len(frame_files),
                "frames_directory": frames_dir,
                "last_processed": max([os.path.getmtime(os.path.join(frames_dir, f)) for f in frame_files]) if frame_files else None
            }
        except Exception as e:
            logger.error(f"Error getting processing summary: {e}")
            return {"error": str(e)}

    def cleanup_processed_data(self, file_id: str = None):
        """Clean up processed frames and temporary data"""
        try:
            if file_id:
                # Clean up specific video's data
                video_name = os.path.splitext(file_id)[0]
                frames_to_remove = [f for f in os.listdir(self.frames_dir) if f.startswith(video_name)]
                
                for frame_file in frames_to_remove:
                    frame_path = os.path.join(self.frames_dir, frame_file)
                    os.remove(frame_path)
                
                logger.info(f"Cleaned up {len(frames_to_remove)} frames for video {file_id}")
                return {"cleaned_frames": len(frames_to_remove)}
            else:
                # Clean up all frames
                frame_files = os.listdir(self.frames_dir)
                for frame_file in frame_files:
                    frame_path = os.path.join(self.frames_dir, frame_file)
                    if os.path.isfile(frame_path):
                        os.remove(frame_path)
                
                logger.info(f"Cleaned up {len(frame_files)} frames")
                return {"cleaned_frames": len(frame_files)}
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}
