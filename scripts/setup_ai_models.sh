#!/bin/bash

# AI Model Setup Script for Intelligent Surveillance System
# This script helps download and set up AI models for computer vision tasks

set -e  # Exit on any error

echo "🤖 AI Model Setup for Intelligent Surveillance System"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/src/models"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Models directory: $MODELS_DIR${NC}"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package
check_python_package() {
    python -c "import $1" 2>/dev/null
}

echo ""
echo "📋 Checking prerequisites..."

# Check Python
if ! command_exists python; then
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Python found: $(python --version)${NC}"
fi

# Check pip
if ! command_exists pip; then
    echo -e "${RED}❌ pip is not installed or not in PATH${NC}"
    exit 1
else
    echo -e "${GREEN}✅ pip found: $(pip --version)${NC}"
fi

echo ""
echo "🧠 AI Model Setup Options"
echo "========================="
echo "1. Install PyTorch (CPU only)"
echo "2. Install PyTorch (CUDA support)"
echo "3. Install YOLOv8 (Ultralytics)"
echo "4. Install BLIP (Transformers)"
echo "5. Install OpenCV"
echo "6. Install all AI dependencies"
echo "7. Download pre-trained models"
echo "8. Test AI model setup"
echo "9. Show installation status"
echo "0. Exit"

while true; do
    echo ""
    read -p "Choose an option (0-9): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Installing PyTorch (CPU only)...${NC}"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            echo -e "${GREEN}✅ PyTorch (CPU) installed${NC}"
            ;;
        2)
            echo -e "${YELLOW}Installing PyTorch (CUDA support)...${NC}"
            echo "Note: Make sure you have CUDA installed on your system"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            echo -e "${GREEN}✅ PyTorch (CUDA) installed${NC}"
            ;;
        3)
            echo -e "${YELLOW}Installing YOLOv8 (Ultralytics)...${NC}"
            pip install ultralytics
            echo -e "${GREEN}✅ YOLOv8 installed${NC}"
            ;;
        4)
            echo -e "${YELLOW}Installing BLIP (Transformers)...${NC}"
            pip install transformers
            echo -e "${GREEN}✅ BLIP/Transformers installed${NC}"
            ;;
        5)
            echo -e "${YELLOW}Installing OpenCV...${NC}"
            pip install opencv-python
            echo -e "${GREEN}✅ OpenCV installed${NC}"
            ;;
        6)
            echo -e "${YELLOW}Installing all AI dependencies...${NC}"
            echo "This will install: PyTorch, YOLOv8, BLIP, OpenCV, and related packages"
            read -p "Continue? (y/N): " confirm
            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                # Install PyTorch (ask for CUDA preference)
                echo "Choose PyTorch installation:"
                echo "1. CPU only (smaller, works everywhere)"
                echo "2. CUDA (faster, requires NVIDIA GPU with CUDA)"
                read -p "Choice (1/2): " torch_choice
                
                if [[ $torch_choice == "2" ]]; then
                    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
                else
                    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
                fi
                
                # Install other AI packages
                pip install ultralytics transformers opencv-python sentence-transformers pillow
                echo -e "${GREEN}✅ All AI dependencies installed${NC}"
            else
                echo "Installation cancelled"
            fi
            ;;
        7)
            echo -e "${YELLOW}Downloading pre-trained models...${NC}"
            
            # Download YOLOv8 model
            echo "Downloading YOLOv8 nano model..."
            python -c "
from ultralytics import YOLO
import os
model = YOLO('yolov8n.pt')
model_path = os.path.join('$MODELS_DIR', 'yolov8n.pt')
model.save(model_path)
print(f'YOLOv8 model saved to {model_path}')
" 2>/dev/null || echo -e "${RED}❌ Failed to download YOLOv8 model (install ultralytics first)${NC}"
            
            # Create a test script for BLIP model download
            echo "Testing BLIP model download..."
            python -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
cache_dir = os.path.join('$MODELS_DIR', 'blip')
try:
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir=cache_dir)
    print(f'BLIP model cached to {cache_dir}')
except Exception as e:
    print(f'Failed to download BLIP model: {e}')
" 2>/dev/null || echo -e "${RED}❌ Failed to download BLIP model (install transformers first)${NC}"
            
            echo -e "${GREEN}✅ Model download completed${NC}"
            ;;
        8)
            echo -e "${YELLOW}Testing AI model setup...${NC}"
            
            # Test PyTorch
            python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
" 2>/dev/null || echo -e "${RED}❌ PyTorch not installed${NC}"
            
            # Test YOLOv8
            python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8 test successful')
" 2>/dev/null || echo -e "${RED}❌ YOLOv8 not working${NC}"
            
            # Test BLIP
            python -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
print('BLIP imports successful')
" 2>/dev/null || echo -e "${RED}❌ BLIP/Transformers not working${NC}"
            
            # Test OpenCV
            python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
" 2>/dev/null || echo -e "${RED}❌ OpenCV not installed${NC}"
            
            echo -e "${GREEN}✅ AI model testing completed${NC}"
            ;;
        9)
            echo -e "${YELLOW}Checking installation status...${NC}"
            echo ""
            
            # Check each package
            packages=("torch" "ultralytics" "transformers" "cv2" "PIL")
            for package in "${packages[@]}"; do
                if check_python_package "$package"; then
                    echo -e "${GREEN}✅ $package is installed${NC}"
                else
                    echo -e "${RED}❌ $package is NOT installed${NC}"
                fi
            done
            
            echo ""
            echo "Model files in $MODELS_DIR:"
            if [ -d "$MODELS_DIR" ]; then
                ls -la "$MODELS_DIR" 2>/dev/null || echo "No model files found"
            else
                echo "Models directory does not exist"
            fi
            ;;
        0)
            echo "Exiting..."
            break
            ;;
        *)
            echo -e "${RED}Invalid option. Please choose 0-9.${NC}"
            ;;
    esac
done

echo ""
echo -e "${BLUE}🎯 Next Steps:${NC}"
echo "1. After installing AI dependencies, refer to docs/AI_IMPLEMENTATION_GUIDE.md"
echo "2. Implement the TODO sections in src/controllers/VisionController.py"
echo "3. Run tests with: python -m pytest tests/test_vision_controller.py"
echo "4. Test the full system with real video files"
echo ""
echo -e "${GREEN}Setup complete! Happy coding! 🚀${NC}"
