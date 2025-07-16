import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from controllers.VisionController import VisionController


def test_yolo_setup():
    # Initialize vision controller
    vision = VisionController()
    
    # Check model info
    model_info = vision.get_model_info()
    print("Model Info:", model_info)
    
    # Test with an actual image file
    test_image_path = "./bus.jpg"  # Replace with actual path
    
    if os.path.exists(test_image_path):
        # Run object detection
        results = vision.detect_objects(test_image_path)
        print("Detection Results:", results)
    else:
        print("No test image found")

if __name__ == "__main__":
    test_yolo_setup()