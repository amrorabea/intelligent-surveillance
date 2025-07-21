"""
Computer Vision Controller for Intelligent Surveillance System

This module handles all AI-powered computer vision tasks including:
- Object detection using YOLOv8
- Image captioning using BLIP
- Frame analysis and batch processing
- Model management and inference
"""

import os
import logging
from datetime import datetime
import time
from typing import Dict, List, Any, Optional

try:
    # Try absolute imports first (for Celery running from project root)
    from src.controllers.BaseController import BaseController
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from .BaseController import BaseController

# Configure logging
logger = logging.getLogger(__name__)


class VisionController(BaseController):
    """
    Production-ready computer vision controller for surveillance video analysis.
    
    Handles object detection, image captioning, and comprehensive frame analysis
    with robust error handling and logging. All AI model implementations are
    left as TODOs for user implementation.
    """
    
    def __init__(self, 
                 detection_threshold: float = 0.5, 
                 max_detections: int = 100,
                 model_cache_dir: Optional[str] = None):
        """
        Initialize the VisionController with configuration parameters.
        
        Args:
            detection_threshold: Minimum confidence score for object detection
            max_detections: Maximum number of objects to detect per frame
            model_cache_dir: Directory to cache downloaded models
        """
        super().__init__()
        
        # Configuration
        self.detection_threshold = detection_threshold
        self.max_detections = max_detections
        self.model_cache_dir = model_cache_dir or os.path.join(self.base_dir, 'models')
        
        # Model instances
        self.yolo_model = None
        self.caption_processor = None
        self.caption_model = None
        
        # Model metadata
        self.models_loaded = False
        self.model_info = {}
        
        # Supported formats
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """
        Initialize and load AI models for computer vision tasks.
        
        This method sets up model paths, checks availability, and loads models.
        """
        try:
            logger.info("Initializing computer vision models...")
            
            # Ensure model cache directory exists
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            # Initialize YOLO model
            self._setup_yolo_model()
            
            # Initialize BLIP captioning model
            self._setup_caption_model()
            
            # Update model status
            self._update_model_status()
            
            logger.info("Vision controller initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision models: {e}")
            self.models_loaded = False
    
    def _setup_yolo_model(self) -> None:
        """
        Setup YOLOv8 model for object detection.
        """
        try:
            model_path = os.path.join(self.model_cache_dir, 'yolov8n.pt')
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics package not found. Install with: pip install ultralytics")
                self.model_info['yolo'] = {
                    'status': 'dependency_missing',
                    'error': 'ultralytics package not installed',
                    'install_command': 'pip install ultralytics'
                }
                return

            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                logger.info(f"YOLOv8 model loaded from {model_path}")
            else:
                logger.info("YOLOv8 model not found locally, downloading...")
                # This will automatically download the model
                self.yolo_model = YOLO('yolov8n.pt')
                # Save to cache directory
                os.makedirs(self.model_cache_dir, exist_ok=True)
                self.yolo_model.save(model_path)
                logger.info(f"YOLOv8 model downloaded and saved to {model_path}")
            
            # Validate model with a test prediction
            try:
                # Test with a small dummy image
                import numpy as np
                from PIL import Image
                
                # Create a test image (3x640x640 RGB)
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                test_pil = Image.fromarray(test_image)
                
                # Run a quick test prediction
                results = self.yolo_model.predict(source=test_pil, save=False, verbose=False, imgsz=640)
                
                logger.info("YOLOv8 model validation successful")
                
                # Update model info
                self.model_info['yolo'] = {
                    'status': 'loaded',
                    'model_path': model_path,
                    'model_type': 'YOLOv8n',
                    'model_size': 'nano',
                    'classes': len(self.yolo_model.names),
                    'class_names': list(self.yolo_model.names.values()),
                    'device': str(self.yolo_model.device),
                    'validation': 'passed'
                }
                
            except Exception as validation_error:
                logger.warning(f"YOLOv8 model validation failed: {validation_error}")
                self.model_info['yolo'] = {
                    'status': 'loaded_unvalidated',
                    'model_path': model_path,
                    'model_type': 'YOLOv8n',
                    'validation_error': str(validation_error)
                }
            
        except Exception as e:
            logger.error(f"Failed to setup YOLO model: {e}")
            self.yolo_model = None
            self.model_info['yolo'] = {
                'status': 'error',
                'error': str(e),
                'model_path': model_path if 'model_path' in locals() else None
            }
    
    def _setup_caption_model(self) -> None:
        """Setup BLIP model using existing cached files"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            model_name = "Salesforce/blip-image-captioning-base"
            cache_dir = os.path.join(self.model_cache_dir, 'blip')
            
            # Set environment variables to use our cache
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HOME"] = cache_dir
            
            logger.info(f"Loading BLIP model from cache: {model_name}")
            logger.info(f"Cache directory: {cache_dir}")
            
            # Check if model files exist in cache
            cached_model_path = os.path.join(cache_dir, "models--Salesforce--blip-image-captioning-base")
            
            if os.path.exists(cached_model_path):
                logger.info("✅ Found existing BLIP model cache, using local files...")
                
                try:
                    # Load processor from cache first
                    self.caption_processor = BlipProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True  # Force use of local files only
                    )
                    logger.info("✅ BLIP processor loaded from cache")
                    
                except Exception as processor_error:
                    logger.warning(f"Processor local load failed: {processor_error}, trying normal load...")
                    self.caption_processor = BlipProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir
                    )
                    logger.info("✅ BLIP processor loaded")
                
                # Determine device and dtype
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                try:
                    # Try to load model from cache
                    self.caption_model = BlipForConditionalGeneration.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=True  # Force use of local files only
                    )
                    logger.info("✅ BLIP model loaded from cache")
                    
                except Exception as model_error:
                    logger.warning(f"Model local load failed: {model_error}, trying normal load...")
                    self.caption_model = BlipForConditionalGeneration.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    )
                    logger.info("✅ BLIP model loaded")
                
                # Move to device and set precision
                self.caption_model = self.caption_model.to(device)
                if torch.cuda.is_available():
                    self.caption_model = self.caption_model.half()
                
                self.caption_model.eval()
                
                # Validate model with existing cache
                try:
                    from PIL import Image
                    test_image = Image.new('RGB', (224, 224), color='blue')
                    
                    # Process inputs and ensure they match model device/dtype
                    inputs = self.caption_processor(test_image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Ensure input dtype matches model
                    if torch.cuda.is_available():
                        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.caption_model.generate(
                            **inputs,
                            max_length=20,
                            num_beams=4,
                            early_stopping=True,
                            do_sample=False
                        )
                    
                    test_caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
                    
                    logger.info(f"🎉 BLIP model loaded and validated successfully on {device}")
                    logger.info(f"Test caption: '{test_caption}'")
                    
                    self.model_info['blip'] = {
                        'status': 'loaded',
                        'model_name': model_name,
                        'device': str(device),
                        'dtype': str(dtype),
                        'cache_dir': cache_dir,
                        'test_caption': test_caption,
                        'validation': 'passed'
                    }
                    
                    return  # Success - exit here
                    
                except Exception as validation_error:
                    logger.warning(f"BLIP model validation failed: {validation_error}")
                    self.model_info['blip'] = {
                        'status': 'loaded_unvalidated',
                        'model_name': model_name,
                        'device': str(device),
                        'validation_error': str(validation_error)
                    }
                    return  # Still loaded, just unvalidated
            
            else:
                logger.warning("No cached BLIP model found, would need to download")
                self.model_info['blip'] = {
                    'status': 'not_found',
                    'error': 'No cached model files found',
                    'cache_dir': cache_dir
                }
                return # Failed - Not loaded
            
        except Exception as e:
            logger.error(f"Failed to setup BLIP caption model: {e}")
            self.caption_processor = None
            self.caption_model = None
            self.model_info['blip'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_model_status(self) -> None:
        """Update the overall model loading status."""
        yolo_loaded = self.yolo_model is not None
        blip_loaded = self.caption_model is not None
        
        self.models_loaded = yolo_loaded or blip_loaded  # At least one model loaded
        
        self.model_info['summary'] = {
            'yolo_loaded': yolo_loaded,
            'blip_loaded': blip_loaded,
            'all_loaded': yolo_loaded and blip_loaded,
            'any_loaded': self.models_loaded,
            'last_updated': datetime.now().isoformat()
        }

    
    def _validate_image_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is a supported image format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Image file not found: {file_path}")
                return False
            
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension not in self.supported_image_formats:
                logger.warning(f"Unsupported image format: {file_extension}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image file {file_path}: {e}")
            return False
    
    def detect_objects(self, 
                      frame_path: str, 
                      confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect objects in a single frame using YOLOv8.
        
        Args:
            frame_path: Path to the image frame
            confidence_threshold: Override default confidence threshold
            
        Returns:
            dict: Detection results with classes, bounding boxes, and confidence scores
        """

        start_time = time.time()

        try:
            # Validate input
            if not self._validate_image_file(frame_path):
                return {
                    'detections': [],
                    'total_objects': 0,
                    'error': 'Invalid or missing image file',
                    'frame_path': frame_path
                }
            
            if not self.yolo_model:
                return {
                    'detections': [],
                    'total_objects': 0,
                    'error': 'YOLO model not loaded',
                    'frame_path': frame_path
                }

            threshold = confidence_threshold or self.detection_threshold
            
            logger.debug(f"Running YOLO detection on {frame_path} with threshold {threshold}")
            
            results = self.yolo_model.predict(
                source=frame_path,
                conf=threshold,
                save=False,
                verbose=False,
                imgsz=640
            )

            # Process results
            detections = []
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        if i >= self.max_detections:
                            break
                        
                        # Extract box data (convert tensors to Python types)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # Calculate box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        center_x = x1 + width / 2
                        center_y = y1 + height / 2
                        
                        detection = {
                            'id': i,
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': round(confidence, 3),
                            'bbox': {
                                'x1': round(x1, 2),
                                'y1': round(y1, 2),
                                'x2': round(x2, 2),
                                'y2': round(y2, 2),
                                'width': round(width, 2),
                                'height': round(height, 2),
                                'area': round(area, 2),
                                'center_x': round(center_x, 2),
                                'center_y': round(center_y, 2)
                            }
                        }
                        detections.append(detection)

            return {
                'detections': detections,
                'total_objects': len(detections),
                'threshold_used': threshold,
                'frame_path': frame_path,
                'processing_time': time.time() - start_time,
                'model_info': {
                    'model_type': 'YOLOv8n',
                    'image_size': 640
                }
            }

        except Exception as e:
            logger.error(f"Error detecting objects in {frame_path}: {e}")
            return {
                'detections': [],
                'total_objects': 0,
                'error': str(e),
                'frame_path': frame_path,
                'processing_time': time.time() - start_time
            }
    
    def generate_caption(self, image_path: str, max_length: int = 50, num_beams: int = 4) -> str:
        """
        Generate a descriptive caption for an image using BLIP model
        
        Args:
            image_path: Path to the image file
            max_length: Maximum caption length (default: 50)
            num_beams: Number of beams for beam search (default: 4)
            
        Returns:
            str: Generated caption describing the image content
        """
        import time
        start_time = time.time()

        try:
            # Validate input
            if not self._validate_image_file(image_path):
                return "Invalid image file"
                
            if not self.caption_model or not self.caption_processor:
                logger.warning('BLIP model not loaded, using placeholder caption')
                return 'Surveillance scene captured - AI captioning system under development'
            
            from PIL import Image
            import torch

            # Load and preprocess image
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Resize if image is too large (for memory efficiency)
                max_size = 1024
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from original size to {image.size}")
                
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                return f"Error loading image: {str(e)}"
            
            # Get device with error handling
            try:
                device = next(self.caption_model.parameters()).device
            except Exception as e:
                logger.error(f"Error getting model device: {e}")
                device = torch.device('cpu')

            # Generate caption
            try:
                # Preprocess image for model
                inputs = self.caption_processor(image, return_tensors='pt')
                
                # Move inputs to the same device as model and match dtype
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Ensure input dtype matches model dtype for CUDA
                if device.type == 'cuda' and hasattr(self.caption_model, 'dtype'):
                    # Convert float tensors to match model precision
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

                # Generate caption with memory optimization
                with torch.no_grad():
                    # Clear GPU cache before generation if using CUDA
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Generate caption with safe parameters
                    # Note: BLIP model handles tokenizer settings internally
                    generation_kwargs = {
                        'max_length': max_length,
                        'num_beams': num_beams,
                        'early_stopping': True,
                        'do_sample': False,
                        'no_repeat_ngram_size': 2,  # Avoid repetitive phrases
                        'length_penalty': 1.0       # Encourage appropriate length
                    }
                    
                    outputs = self.caption_model.generate(**inputs, **generation_kwargs)
            
                # Decode and clean caption
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
                caption = caption.strip()

                # Post-process caption
                if caption and not caption.endswith(('.', '!', '?')):
                    caption += '.'

                # Capitalize first letter
                if caption:
                    caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()

                # Calculate processing time correctly
                processing_time = time.time() - start_time
                logger.debug(f"Generated caption for {image_path} in {processing_time:.3f}s: {caption}")

                return caption if caption else "Unable to generate meaningful caption"

            except Exception as e:
                logger.error(f"Error during caption generation for {image_path}: {e}")
                return f"Caption generation failed: {str(e)}"

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return f"Caption generation error: {str(e)}"
    
    def generate_surveillance_caption(self, image_path: str, detections: Dict = None, 
                                    max_length: int = 60, num_beams: int = 4, new_tracks = None) -> Dict[str, Any]:
        """
        Generate a surveillance-focused caption that incorporates detected objects
        
        Args:
            image_path: Path to the image file
            detections: Optional detection results to enhance caption
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            dict: Caption with metadata and processing info
        """
        start_time = time.time()
        
        try:
            # Generate base caption if there are new tracks
            if new_tracks:
                base_caption = self.generate_caption(image_path, max_length, num_beams)
            else:
                base_caption = ""
            
            # Enhance with detection information if available
            enhanced_caption = base_caption
            context_info = []
            
            if detections and detections.get('detections'):
                detected_objects = detections['detections']
                
                # Count objects by class
                object_counts = {}
                confidence_scores = []
                
                for detection in detected_objects:
                    class_name = detection.get('class', 'object')
                    confidence = detection.get('confidence', 0)
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    confidence_scores.append(confidence)
                
                # Create context string
                if object_counts:
                    object_descriptions = []
                    for obj_class, count in object_counts.items():
                        if count == 1:
                            object_descriptions.append(f"a {obj_class}")
                        else:
                            object_descriptions.append(f"{count} {obj_class}s")
                    
                    if object_descriptions:
                        objects_text = ", ".join(object_descriptions[:-1])
                        if len(object_descriptions) > 1:
                            objects_text += f" and {object_descriptions[-1]}"
                        else:
                            objects_text = object_descriptions[0]
                        
                        # Calculate average confidence
                        avg_confidence = sum(confidence_scores) / len(confidence_scores)
                        confidence_level = "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
                        
                        context_info.append(f"Scene contains {objects_text} (confidence: {confidence_level})")
                        
                        # Enhance base caption with detection context
                        if "surveillance scene" in base_caption.lower() or "AI captioning system" in base_caption:
                            enhanced_caption = f"Surveillance footage showing {objects_text}."
                        else:
                            enhanced_caption = f"{base_caption} The scene contains {objects_text}."
            
            processing_time = time.time() - start_time
            
            return {
                'text': enhanced_caption,
                'base_caption': base_caption,
                'context_info': context_info,
                'processing_time': processing_time,
                'has_detections': bool(detections and detections.get('detections')),
                'detection_count': len(detections.get('detections', [])) if detections else 0,
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating surveillance caption: {e}")
            return {
                'text': f"Caption generation error: {str(e)}",
                'base_caption': None,
                'context_info': [],
                'processing_time': time.time() - start_time,
                'error': str(e),
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }

    def analyze_frame(self, 
                     frame_path: str, 
                     include_caption: bool = True,
                     include_detections: bool = True,
                     confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a video frame, combining detection and captioning.
        
        Args:
            frame_path: Path to the image frame
            include_caption: Whether to generate image caption
            include_detections: Whether to run object detection
            confidence_threshold: Detection confidence threshold
            
        Returns:
            dict: Complete analysis with detections, caption, and metadata
        """
        try:
            start_time = datetime.now()
            
            # Validate input
            if not self._validate_image_file(frame_path):
                return {
                    'frame_path': frame_path,
                    'error': 'Invalid or missing image file',
                    'detections': None,
                    'caption': None,
                    'analysis_timestamp': start_time.isoformat()
                }
            
            # Initialize results
            analysis_result = {
                'frame_path': frame_path,
                'analysis_timestamp': start_time.isoformat(),
                'settings': {
                    'include_caption': include_caption,
                    'include_detections': include_detections,
                    'detection_threshold': confidence_threshold or self.detection_threshold,
                    'max_detections': self.max_detections
                }
            }
            
            # Run object detection if requested
            if include_detections:
                detections = self.detect_objects(frame_path, confidence_threshold)
                analysis_result['detections'] = detections
            else:
                analysis_result['detections'] = None
            
            # Generate caption if requested
            if include_caption:
                caption_text = self.generate_caption(frame_path)
                analysis_result['caption'] = {
                    'text': caption_text,
                    'frame_path': frame_path,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                analysis_result['caption'] = None
            
            # Calculate processing time
            end_time = datetime.now()
            analysis_result['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            # Add summary statistics
            if include_detections and analysis_result['detections'] and 'detections' in analysis_result['detections']:
                analysis_result['summary'] = {
                    'total_objects': analysis_result['detections']['total_objects'],
                    'unique_classes': len(set(d['class'] for d in analysis_result['detections']['detections'])),
                    'has_caption': include_caption and analysis_result['caption'] is not None
                }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_path}: {e}")
            return {
                'frame_path': frame_path,
                'error': str(e),
                'detections': None,
                'caption': None,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    
    def batch_analyze_frames(self, 
                            frame_paths: List[str], 
                            include_caption: bool = True,
                            include_detections: bool = True,
                            confidence_threshold: Optional[float] = None,
                            max_workers: int = 4) -> Dict[str, Any]:
        """
        Analyze multiple frames in batch with optional parallel processing.
        
        Args:
            frame_paths: List of frame paths to analyze
            include_caption: Whether to generate captions
            include_detections: Whether to run object detection
            confidence_threshold: Detection confidence threshold
            max_workers: Maximum number of parallel workers (TODO: implement threading)
            
        Returns:
            dict: Batch analysis results with individual frame results and summary
        """
        try:
            start_time = datetime.now()
            
            if not frame_paths:
                return {
                    'results': [],
                    'summary': {'total_frames': 0, 'successful': 0, 'failed': 0},
                    'processing_time_seconds': 0,
                    'batch_timestamp': start_time.isoformat()
                }
            
            logger.info(f"Starting batch analysis of {len(frame_paths)} frames")
            
            # TODO: Implement parallel processing for better performance
            """
            Example implementation with ThreadPoolExecutor:
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            failed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self.analyze_frame, 
                        path, 
                        include_caption, 
                        include_detections, 
                        confidence_threshold
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
            """
            
            # Sequential processing implementation
            results = []
            failed_count = 0
            
            for i, frame_path in enumerate(frame_paths):
                try:
                    logger.debug(f"Processing frame {i+1}/{len(frame_paths)}: {frame_path}")
                    result = self.analyze_frame(
                        frame_path, 
                        include_caption, 
                        include_detections, 
                        confidence_threshold
                    )
                    results.append(result)
                    
                    if 'error' in result:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process frame {frame_path}: {e}")
                    results.append({
                        'frame_path': frame_path,
                        'error': str(e),
                        'detections': None,
                        'caption': None,
                        'analysis_timestamp': datetime.now().isoformat()
                    })
                    failed_count += 1
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Generate summary statistics
            successful_count = len(frame_paths) - failed_count
            total_objects = 0
            unique_classes = set()
            
            for result in results:
                if (result.get('detections') and 
                    isinstance(result['detections'], dict) and 
                    'detections' in result['detections']):
                    
                    total_objects += result['detections'].get('total_objects', 0)
                    for detection in result['detections'].get('detections', []):
                        if 'class' in detection:
                            unique_classes.add(detection['class'])
            
            summary = {
                'total_frames': len(frame_paths),
                'successful': successful_count,
                'failed': failed_count,
                'success_rate': successful_count / len(frame_paths) if frame_paths else 0,
                'total_objects_detected': total_objects,
                'unique_classes_detected': len(unique_classes),
                'average_processing_time': processing_time / len(frame_paths) if frame_paths else 0
            }
            
            batch_result = {
                'results': results,
                'summary': summary,
                'processing_time_seconds': processing_time,
                'batch_timestamp': start_time.isoformat(),
                'settings': {
                    'include_caption': include_caption,
                    'include_detections': include_detections,
                    'confidence_threshold': confidence_threshold or self.detection_threshold,
                    'max_workers': max_workers
                }
            }
            
            logger.info(f"Batch analysis completed: {successful_count}/{len(frame_paths)} successful")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in batch frame analysis: {e}")
            return {
                'results': [],
                'summary': {'total_frames': len(frame_paths), 'successful': 0, 'failed': len(frame_paths)},
                'error': str(e),
                'batch_timestamp': datetime.now().isoformat()
            }
    
    def extract_objects_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and summarize object detection results from multiple frame analyses.
        
        Args:
            analysis_results: List of frame analysis results
            
        Returns:
            dict: Summary of detected objects across all frames
        """
        try:
            class_counts = {}
            total_objects = 0
            frames_with_objects = 0
            confidence_scores = []
            
            for result in analysis_results:
                if (result.get('detections') and 
                    isinstance(result['detections'], dict) and 
                    'detections' in result['detections']):
                    
                    detections = result['detections']['detections']
                    if detections:
                        frames_with_objects += 1
                    
                    for detection in detections:
                        class_name = detection.get('class', 'unknown')
                        confidence = detection.get('confidence', 0)
                        
                        # Count classes
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        total_objects += 1
                        confidence_scores.append(confidence)
            
            # Calculate statistics
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            detection_rate = frames_with_objects / len(analysis_results) if analysis_results else 0
            
            # Sort classes by frequency
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'total_objects': total_objects,
                'unique_classes': len(class_counts),
                'class_distribution': dict(sorted_classes),
                'most_common_classes': sorted_classes[:10],  # Top 10
                'frames_processed': len(analysis_results),
                'frames_with_objects': frames_with_objects,
                'detection_rate': detection_rate,
                'average_confidence': avg_confidence,
                'min_confidence': min(confidence_scores) if confidence_scores else 0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting objects summary: {e}")
            return {
                'error': str(e),
                'total_objects': 0,
                'unique_classes': 0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about loaded models and system status.
        
        Returns:
            dict: Model information, capabilities, and system status
        """
        try:
            system_info = {
                'vision_controller_version': '1.0.0',
                'models_loaded': self.models_loaded,
                'model_details': self.model_info.copy(),
                'configuration': {
                    'detection_threshold': self.detection_threshold,
                    'max_detections': self.max_detections,
                    'model_cache_dir': self.model_cache_dir,
                    'supported_formats': list(self.supported_image_formats)
                },
                'capabilities': {
                    'object_detection': self.yolo_model is not None,
                    'image_captioning': self.caption_model is not None,
                    'batch_processing': True,
                    'parallel_processing': False  # TODO: Implement
                },
                'last_updated': datetime.now().isoformat()
            }
            
            # Add system resource information if available
            try:
                import psutil
                system_info['system_resources'] = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3)
                }
            except ImportError:
                system_info['system_resources'] = 'psutil not available'
            
            # Add GPU information if available
            try:
                # TODO: Add GPU detection when implementing CUDA support
                """
                import torch
                if torch.cuda.is_available():
                    system_info['gpu_info'] = {
                        'cuda_available': True,
                        'gpu_count': torch.cuda.device_count(),
                        'current_device': torch.cuda.current_device(),
                        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                    }
                else:
                    system_info['gpu_info'] = {'cuda_available': False}
                """
                system_info['gpu_info'] = 'TODO: Implement GPU detection'
            except Exception as e:
                system_info['gpu_info'] = f'Error detecting GPU: {e}'
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'error': str(e),
                'models_loaded': False,
                'vision_controller_version': '1.0.0'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the vision controller and its models.
        
        Returns:
            dict: Health status and diagnostic information
        """
        try:
            start_time = datetime.now()
            
            health_status = {
                'status': 'healthy',
                'timestamp': start_time.isoformat(),
                'checks': {}
            }
            
            # Check model cache directory
            try:
                if os.path.exists(self.model_cache_dir):
                    health_status['checks']['model_cache_dir'] = 'ok'
                else:
                    health_status['checks']['model_cache_dir'] = 'missing'
                    health_status['status'] = 'warning'
            except Exception as e:
                health_status['checks']['model_cache_dir'] = f'error: {e}'
                health_status['status'] = 'error'
            
            # Check models
            health_status['checks']['yolo_model'] = 'not_implemented' if not self.yolo_model else 'loaded'
            health_status['checks']['caption_model'] = 'not_implemented' if not self.caption_model else 'loaded'
            
            # TODO: Add actual model validation tests
            """
            # Test YOLO model if loaded
            if self.yolo_model:
                try:
                    # Run a quick test inference
                    test_result = self.yolo_model.predict(source='path/to/test/image.jpg', save=False, verbose=False)
                    health_status['checks']['yolo_inference'] = 'ok'
                except Exception as e:
                    health_status['checks']['yolo_inference'] = f'failed: {e}'
                    health_status['status'] = 'error'
            
            # Test caption model if loaded
            if self.caption_model and self.caption_processor:
                try:
                    # Run a quick test caption generation
                    from PIL import Image
                    test_image = Image.new('RGB', (224, 224), color='red')
                    inputs = self.caption_processor(test_image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.caption_model.generate(**inputs, max_length=20)
                    health_status['checks']['caption_inference'] = 'ok'
                except Exception as e:
                    health_status['checks']['caption_inference'] = f'failed: {e}'
                    health_status['status'] = 'error'
            """
            
            # Calculate health check duration
            end_time = datetime.now()
            health_status['check_duration_seconds'] = (end_time - start_time).total_seconds()
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            # TODO: Implement proper model cleanup
            # Clear GPU memory, unload models if needed
            if hasattr(self, 'yolo_model') and self.yolo_model:
                del self.yolo_model
                self.yolo_model = None
                
            if hasattr(self, 'caption_model') and self.caption_model:
                del self.caption_model
                self.caption_model = None
                
            if hasattr(self, 'caption_processor') and self.caption_processor:
                del self.caption_processor
                self.caption_processor = None
                
            # Clear GPU cache if using PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
                
            self.models_loaded = False
            self.logger.info("Vision models cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")