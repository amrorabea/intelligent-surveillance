# AI Models Guide

This guide provides the details on how the models are implemented and what to expect from them

## Overview

The project is structured to support 3 main capabilities:
1. **Object Detection** using YOLOv8
2. **Object Tracking** using classical IoU
3. **Image Captioning** using BLIP (Bootstrapping Language-Image Pre-training)

## YOLOv8 Object Detection Details

### 1. Dependencies

```bash
pip intsall ultralytics torch torchvision
```

### 2. Setup

main function `_setup_yolo_model()`

- Responsible for importing our YOLO model to be used, Currently we are using a small model `YOLOv8n` which is a lightweight version of YOLO.

### 3. Object Detection Logic

main function `detect_objects()`

- Takes in a single image and provides detections of objects and their bounding boxes.

### 4. Surveillance Detection Logic

main function `generate_surveillance_caption()`

- This is the part where we use the detected objects from the `detect_objects()` function, **The Logic must be improved** but for now, we add the counts of each detected object, e.g., `Scene Contains 3 Persons, a bus`, no color detection implemented for now but we should do that as it's really important.

## BLIP Image Captioning Details

### 1. Dependencies

```bash
pip install transformers torch torchvision pillow
```

### 2. Setup

main function `_setup_caption_model()`

- Responsible for importing current local BLIP model, if not found then it downloads it.

### 3. Image Captioning Logic

main functions `generate_caption()`, `generate_surveillance_caption()`

- **In/Out**: Takes in the image path, and returns the caption generated for this image.

- **Problems**: BLIP (1-2 FPS) is pretty slow compared to YOLO (30-60 FPS), so to use it we only have to do it on important images, this is where **Object Tracking** Shines, Once we track new objects, this is where we will run BLIP.

- **Advantages**: Provides good captions that would increase our vectordb searching accuracy
