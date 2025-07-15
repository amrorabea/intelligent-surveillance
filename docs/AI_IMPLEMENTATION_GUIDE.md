# AI Model Implementation Guide

This guide provides detailed instructions for implementing the AI models in the VisionController.

## Overview

The VisionController is structured to support two main AI capabilities:
1. **Object Detection** using YOLOv8
2. **Image Captioning** using BLIP (Bootstrapping Language-Image Pre-training)

All AI model implementations are marked as TODOs and ready for your implementation.

## YOLOv8 Object Detection Implementation

### 1. Install Dependencies

```bash
pip install ultralytics torch torchvision
```

### 2. Implement in `_setup_yolo_model()` method

Replace the TODO section in `/src/controllers/VisionController.py`:

```python
def _setup_yolo_model(self) -> None:
    try:
        from ultralytics import YOLO
        
        model_path = os.path.join(self.model_cache_dir, 'yolov8n.pt')
        
        if os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded from {model_path}")
        else:
            # Download model if not found
            self.yolo_model = YOLO('yolov8n.pt')  # Downloads automatically
            # Save to cache directory
            self.yolo_model.save(model_path)
            logger.info("YOLOv8 model downloaded and cached")
        
        # Validate model with test inference
        test_result = self.yolo_model.predict(
            source='https://ultralytics.com/images/bus.jpg', 
            save=False, 
            verbose=False
        )
        logger.info("YOLOv8 model validation successful")
        
        self.model_info['yolo'] = {
            'status': 'loaded',
            'model_path': model_path,
            'model_type': 'YOLOv8n',
            'classes': len(self.yolo_model.names),
            'input_size': 640
        }
        
    except Exception as e:
        logger.error(f"Failed to setup YOLO model: {e}")
        self.model_info['yolo'] = {'status': 'error', 'error': str(e)}
```

### 3. Implement in `detect_objects()` method

Replace the TODO section:

```python
# Check if model is loaded
if not self.yolo_model:
    return {
        'detections': [],
        'total_objects': 0,
        'error': 'YOLO model not loaded',
        'frame_path': frame_path
    }

# Run inference
results = self.yolo_model(frame_path, conf=threshold, verbose=False)

# Process results
detections = []
for result in results:
    if result.boxes is not None:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            if i >= self.max_detections:
                break
                
            # Extract box data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu())
            class_id = int(box.cls[0].cpu())
            class_name = result.names[class_id]
            
            # Calculate box dimensions
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            detection = {
                'id': i,
                'class': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(width),
                    'height': float(height),
                    'area': float(area)
                }
            }
            detections.append(detection)

return {
    'detections': detections,
    'total_objects': len(detections),
    'threshold_used': threshold,
    'frame_path': frame_path
}
```

## BLIP Image Captioning Implementation

### 1. Install Dependencies

```bash
pip install transformers torch torchvision pillow
```

### 2. Implement in `_setup_caption_model()` method

Replace the TODO section:

```python
def _setup_caption_model(self) -> None:
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        model_name = "Salesforce/blip-image-captioning-base"
        cache_dir = os.path.join(self.model_cache_dir, 'blip')
        
        # Load processor and model
        self.caption_processor = BlipProcessor.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_model = self.caption_model.to(device)
        
        logger.info(f"BLIP model loaded successfully on {device}")
        
        self.model_info['blip'] = {
            'status': 'loaded',
            'model_name': model_name,
            'device': device,
            'cache_dir': cache_dir
        }
        
    except Exception as e:
        logger.error(f"Failed to setup caption model: {e}")
        self.caption_processor = None
        self.caption_model = None
        self.model_info['blip'] = {'status': 'error', 'error': str(e)}
```

### 3. Implement in `generate_caption()` method

Replace the TODO section:

```python
# Check if models are loaded
if not self.caption_model or not self.caption_processor:
    return {
        'caption': '',
        'error': 'Caption model not loaded',
        'frame_path': frame_path
    }

# Load and preprocess image
from PIL import Image
import torch

image = Image.open(frame_path).convert('RGB')

# Process image
inputs = self.caption_processor(image, return_tensors="pt")

# Move to appropriate device if using GPU
device = next(self.caption_model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate caption
with torch.no_grad():
    outputs = self.caption_model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
        temperature=0.7
    )

# Decode caption
caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)

# Clean up caption
caption = caption.strip()

# Remove common prefixes
prefixes_to_remove = ['a picture of', 'an image of', 'a photo of']
for prefix in prefixes_to_remove:
    if caption.lower().startswith(prefix):
        caption = caption[len(prefix):].strip()
        break

return {
    'caption': caption,
    'frame_path': frame_path,
    'model_config': {
        'max_length': max_length,
        'min_length': min_length,
        'num_beams': 4
    }
}
```

## Optional Enhancements

### 1. GPU Acceleration

Add GPU detection and optimization:

```python
# In _setup_yolo_model()
if torch.cuda.is_available():
    self.yolo_model.to('cuda')
    logger.info("YOLO model moved to GPU")

# In _setup_caption_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
self.caption_model = self.caption_model.to(device)
```

### 2. Model Variants

Support different YOLO model sizes:

```python
# Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model_size = 'yolov8n.pt'  # Can be configurable
self.yolo_model = YOLO(model_size)
```

### 3. Batch Processing Optimization

Implement parallel processing in `batch_analyze_frames()`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_analyze_frames(self, frame_paths, max_workers=4, **kwargs):
    results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(
                self.analyze_frame, 
                path, 
                **kwargs
            ): path for path in frame_paths
        }
        
        # Collect results
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                if 'error' in result:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    'frame_path': path,
                    'error': str(e),
                    'detections': None,
                    'caption': None
                })
                failed_count += 1
    
    return results
```

## Testing Your Implementation

### 1. Unit Tests

Run the existing tests to ensure your implementation works:

```bash
cd /home/amro/Desktop/intelligent-surveillance
python -m pytest tests/test_vision_controller.py -v
```

### 2. Manual Testing

Create a simple test script:

```python
from src.controllers.VisionController import VisionController
import os

# Initialize controller
vc = VisionController()

# Test with a sample image
test_image = "path/to/your/test/image.jpg"

# Test object detection
detections = vc.detect_objects(test_image)
print("Detections:", detections)

# Test captioning
caption = vc.generate_caption(test_image)
print("Caption:", caption)

# Test full analysis
analysis = vc.analyze_frame(test_image)
print("Full Analysis:", analysis)
```

### 3. Model Information

Check model status:

```python
info = vc.get_model_info()
print("Model Info:", info)

health = vc.health_check()
print("Health Status:", health)
```

## Performance Considerations

1. **Model Loading**: Models are loaded once during initialization for efficiency
2. **GPU Memory**: Monitor GPU memory usage with large batch processing
3. **Image Preprocessing**: Consider resizing large images before processing
4. **Caching**: Downloaded models are cached locally to avoid re-downloading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Fails**: Check internet connection and disk space
3. **Import Errors**: Ensure all dependencies are installed
4. **Performance Issues**: Use GPU acceleration and optimize batch processing

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Integration with Surveillance System

Once implemented, the VisionController will be automatically integrated with:

- **ProcessController**: For video frame extraction and processing
- **VectorDBController**: For storing and searching visual features
- **Surveillance API**: For real-time video analysis endpoints
- **Job Queue**: For background processing of video files

The TODO markers ensure you can implement at your own pace while maintaining the production-ready structure of the surveillance system.
