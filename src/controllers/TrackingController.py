import cv2
import numpy as np
from .BaseController import BaseController

class TrackingController(BaseController):
    def __init__(self):
        super().__init__()
        # Note: We'll use OpenCV's built-in tracker instead of ByteTrack
        # This simplifies dependencies while still providing tracking functionality
        self.trackers = {}
        self.next_id = 0
        self.tracking_history = {}  # Store tracking history for each object
        
    def init_tracker(self, frame, bbox):
        """Initialize a tracker for a new object"""
        tracker = cv2.TrackerKCF_create()  # KCF tracker is a good balance of speed and accuracy
        success = tracker.init(frame, bbox)
        if success:
            object_id = self.next_id
            self.next_id += 1
            self.trackers[object_id] = tracker
            self.tracking_history[object_id] = [bbox]  # Initialize tracking history
            return object_id
        return None
        
    def track_objects(self, frame, detections=None):
        """
        Track objects across frames
        
        Args:
            frame: Current video frame (numpy array)
            detections: Optional list of new detections in format (x1, y1, x2, y2)
        
        Returns:
            dict: Tracking results with object IDs and positions
        """
        if detections is None:
            detections = []
            
        # Initialize new trackers for new detections
        if detections:
            for bbox in detections:
                self.init_tracker(frame, tuple(bbox))
                
        # Update existing trackers
        updated_trackers = {}
        updated_positions = {}
        
        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                updated_trackers[obj_id] = tracker
                updated_positions[obj_id] = bbox
                self.tracking_history[obj_id].append(bbox)  # Update tracking history
                
                # Limit history length to avoid memory issues
                if len(self.tracking_history[obj_id]) > 100:  
                    self.tracking_history[obj_id] = self.tracking_history[obj_id][-100:]
                    
        # Replace old trackers with updated ones
        self.trackers = updated_trackers
        
        return {
            "tracked_objects": updated_positions,
            "total_tracked": len(updated_positions)
        }
        
    def get_object_path(self, object_id):
        """
        Get the movement path of a tracked object
        
        Args:
            object_id: ID of the tracked object
            
        Returns:
            list: List of positions (bboxes) for the object
        """
        if object_id in self.tracking_history:
            return self.tracking_history[object_id]
        return []
        
    def calculate_object_speed(self, object_id, fps=30):
        """
        Estimate the movement speed of an object
        
        Args:
            object_id: ID of the tracked object
            fps: Frames per second of the video
            
        Returns:
            float: Estimated speed in pixels per second
        """
        if object_id not in self.tracking_history or len(self.tracking_history[object_id]) < 2:
            return 0.0
            
        # Get last two positions
        prev_box = self.tracking_history[object_id][-2]
        curr_box = self.tracking_history[object_id][-1]
        
        # Calculate center points
        prev_center = (
            (prev_box[0] + prev_box[2]) / 2,
            (prev_box[1] + prev_box[3]) / 2
        )
        curr_center = (
            (curr_box[0] + curr_box[2]) / 2,
            (curr_box[1] + curr_box[3]) / 2
        )
        
        # Calculate distance moved
        distance = np.sqrt(
            (curr_center[0] - prev_center[0])**2 + 
            (curr_center[1] - prev_center[1])**2
        )
        
        # Convert to speed (pixels per second)
        speed = distance * fps
        
        return speed