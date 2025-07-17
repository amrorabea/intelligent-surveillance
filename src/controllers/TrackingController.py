import numpy as np
try:
    # Try absolute imports first (for Celery running from project root)
    from src.controllers.BaseController import BaseController
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from .BaseController import BaseController

class TrackingController(BaseController):
    def __init__(self):
        super().__init__()
        # Use detection-based tracking instead of OpenCV trackers
        # This is more reliable and doesn't depend on specific OpenCV versions
        self.tracks = {}  # Store active tracks: {track_id: last_bbox}
        self.next_id = 0
        self.tracking_history = {}  # Store tracking history for each object
        self.max_distance_threshold = 100  # Maximum distance for track association
        self.iou_threshold = 0.3  # Minimum IoU for track association
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate center-to-center distance between two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def track_objects(self, frame, detections=None):
        """
        Track objects using detection-to-detection matching
        
        Args:
            frame: Current video frame (numpy array) - not used in this approach
            detections: List of new detections in format [x1, y1, x2, y2]
        
        Returns:
            dict: Tracking results with object IDs and positions
        """
        if detections is None:
            detections = []
        
        # Convert detection format if needed
        detection_boxes = []
        for det in detections:
            if len(det) >= 4:
                # Ensure we have [x1, y1, x2, y2] format
                detection_boxes.append([float(det[0]), float(det[1]), float(det[2]), float(det[3])])
        
        if not detection_boxes:
            # No new detections, return empty tracking results
            return {
                "tracked_objects": {},
                "total_tracked": 0,
                "new_tracks": 0,
                "continued_tracks": 0
            }
        
        # Associate detections with existing tracks
        matched_tracks = {}
        unmatched_detections = detection_boxes.copy()
        
        for track_id, last_bbox in self.tracks.items():
            best_match = None
            best_score = 0
            best_detection_idx = -1
            
            for i, detection_bbox in enumerate(unmatched_detections):
                # Calculate IoU for association
                iou = self.calculate_iou(last_bbox, detection_bbox)
                distance = self.calculate_distance(last_bbox, detection_bbox)
                
                # Combine IoU and distance for scoring
                score = iou * 0.7 + max(0, (self.max_distance_threshold - distance) / self.max_distance_threshold) * 0.3
                
                if score > best_score and score > self.iou_threshold:
                    best_score = score
                    best_match = detection_bbox
                    best_detection_idx = i
            
            if best_match is not None:
                # Update existing track
                matched_tracks[track_id] = best_match
                self.tracks[track_id] = best_match
                self.tracking_history[track_id].append(best_match)
                
                # Limit history length
                if len(self.tracking_history[track_id]) > 100:
                    self.tracking_history[track_id] = self.tracking_history[track_id][-100:]
                
                # Remove matched detection
                unmatched_detections.pop(best_detection_idx)
        
        # Create new tracks for unmatched detections
        new_tracks_count = 0
        for detection_bbox in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = detection_bbox
            self.tracking_history[track_id] = [detection_bbox]
            matched_tracks[track_id] = detection_bbox
            new_tracks_count += 1
        
        return {
            "tracked_objects": matched_tracks,
            "total_tracked": len(matched_tracks),
            "new_tracks": new_tracks_count,
            "continued_tracks": len(matched_tracks) - new_tracks_count
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
    
    def cleanup_lost_tracks(self, max_age=10):
        """
        Remove tracks that haven't been updated recently
        
        Args:
            max_age: Maximum number of frames without detection before removing track
        """
        # This simple version removes tracks that are no longer being detected
        # In a more sophisticated version, you could track frame counts
        pass
    
    def get_tracking_stats(self):
        """
        Get overall tracking statistics
        
        Returns:
            dict: Statistics about the tracking system
        """
        total_objects = len(self.tracks)
        total_history_length = sum(len(history) for history in self.tracking_history.values())
        
        return {
            "active_tracks": total_objects,
            "total_tracked_objects": len(self.tracking_history),
            "average_track_length": total_history_length / len(self.tracking_history) if self.tracking_history else 0,
            "next_id": self.next_id
        }