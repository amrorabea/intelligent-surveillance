from .BaseController import BaseController
from .VisionController import VisionController
from .VectorDBController import VectorDBController
from .TrackingController import TrackingController
from .ProjectController import ProjectController
import os
import cv2
from models.enums.ProcessingEnum import ProcessingEnum

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

    def process_video(self, file_id: str, sample_rate=1.0):
        """
        Process a video file through the complete surveillance AI pipeline
        
        Args:
            file_id (str): ID of the video file to process
            sample_rate (float): Frames per second to extract and analyze
            
        Returns:
            dict: Processing results summary
        """
        file_path = os.path.join(self.project_path, file_id)
        
        if not os.path.exists(file_path):
            return {"error": f"Video file {file_id} not found", "processed_frames": 0}
            
        if not self.is_video_file(file_id):
            return {"error": f"File {file_id} is not a supported video format", "processed_frames": 0}
        
        print(f"Processing video: {file_path}")
        
        # Step 1: Extract frames from video
        extracted_frames = self.extract_frames_from_video(file_path, sample_rate)
        
        if not extracted_frames:
            return {"error": "No frames could be extracted from video", "processed_frames": 0}
        
        # Step 2: Initialize AI controllers
        vision_controller = VisionController()
        tracking_controller = TrackingController()
        vector_db = VectorDBController(collection_name=f"surveillance_{self.project_id}")
        
        # Step 3: Process each frame through AI pipeline
        processed_results = []
        successful_frames = 0
        
        for frame_info in extracted_frames:
            try:
                timestamp = frame_info['timestamp']
                frame_path = frame_info['frame_path']
                
                # A. Object Detection
                detections = vision_controller.detect_objects(frame_path)
                
                # B. Object Tracking (pass the actual frame for tracking)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Extract bounding boxes for tracking
                    bboxes = []
                    if detections and 'detections' in detections:
                        for detection in detections['detections']:
                            if 'box' in detection:
                                bboxes.append(detection['box'])
                    
                    tracking_results = tracking_controller.track_objects(frame, bboxes)
                else:
                    tracking_results = {"tracked_objects": {}, "total_tracked": 0}
                
                # C. Scene Captioning
                caption = vision_controller.generate_caption(frame_path)
                
                # D. Store in Vector Database
                metadata = {
                    "file_id": file_id,
                    "project_id": self.project_id,
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "frame_number": frame_info['frame_number']
                }
                
                # Add detected object classes to metadata
                if detections and 'detections' in detections:
                    metadata["detected_objects"] = [d.get('class', 'unknown') for d in detections['detections']]
                else:
                    metadata["detected_objects"] = []
                
                # Store the caption in vector database for semantic search
                storage_success = vector_db.store_embedding(
                    document_id=f"{file_id}_{timestamp}",
                    text=caption,
                    metadata=metadata
                )
                
                # Compile frame results
                frame_result = {
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "detections": detections,
                    "tracking": tracking_results,
                    "caption": caption,
                    "stored": storage_success
                }
                
                processed_results.append(frame_result)
                successful_frames += 1
                
            except Exception as e:
                print(f"Error processing frame at {timestamp}s: {e}")
                continue
        
        return {
            "file_id": file_id,
            "processed_frames": successful_frames,
            "total_extracted_frames": len(extracted_frames),
            "success_rate": successful_frames / len(extracted_frames) if extracted_frames else 0,
            "sample_results": processed_results[:3]  # Return first 3 for preview
        }
