#!/bin/bash

# 🔍 Intelligent Surveillance System - One-Command Setup
# Simple, minimal setup for the entire surveillance system

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "\n${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

show_help() {
    echo "🔍 Intelligent Surveillance System Setup"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install     Install Python dependencies"
    echo "  models      Download AI models"
    echo "  docker      Start with Docker (recommended)"
    echo "  stop        Stop Docker containers"
    echo "  start       Start backend server"
    echo "  frontend    Start Streamlit frontend"
    echo "  full        Complete setup + start everything"
    echo "  clean       Clean up generated files"
    echo "  help        Show this help"
    echo ""
    echo "Quick start:"
    echo "  $0 docker    # Use Docker (easiest)"
    echo "  $0 full      # Manual setup"
}

install_dependencies() {
    print_step "📦 Installing Python dependencies..."
    
    # Check Python version
    if ! python3 --version >/dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Install requirements
    if [ -f "src/requirements.txt" ]; then
        pip3 install -r src/requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

download_models() {
    print_step "🤖 Downloading AI models..."
    
    # Create models directory
    mkdir -p src/models
    
    # Download YOLOv8 model if not exists
    if [ ! -f "src/models/yolov8n.pt" ]; then
        echo "📥 Downloading YOLOv8 model..."
        if command -v curl >/dev/null 2>&1; then
            curl -L -o src/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
        elif command -v wget >/dev/null 2>&1; then
            wget -O src/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
        else
            print_error "Need curl or wget to download models"
            exit 1
        fi
        print_success "YOLOv8 model downloaded"
    else
        print_success "YOLOv8 model already exists"
    fi
    
    print_success "AI models ready (BLIP downloads automatically)"
}

stop_docker() {
    print_step "🛑 Stopping Docker containers..."
    
    # Determine if we need sudo
    if ! docker info >/dev/null 2>&1; then
        if sudo docker info >/dev/null 2>&1; then
            DOCKER_CMD="sudo docker"
        else
            print_error "Cannot access Docker"
            exit 1
        fi
    else
        DOCKER_CMD="docker"
    fi
    
    # Stop and remove containers
    $DOCKER_CMD stop surveillance-backend surveillance-worker surveillance-redis 2>/dev/null || true
    $DOCKER_CMD rm surveillance-backend surveillance-worker surveillance-redis 2>/dev/null || true
    
    # Remove network
    $DOCKER_CMD network rm surveillance-network 2>/dev/null || true
    
    print_success "Docker containers stopped and removed"
}

start_docker() {
    print_step "🐳 Starting with Docker..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker not found. Install with:"
        echo "  sudo apt install docker.io"
        exit 1
    fi
    
    # Check if Docker daemon is running (try with and without sudo)
    if ! docker info >/dev/null 2>&1; then
        if sudo docker info >/dev/null 2>&1; then
            print_warning "Docker requires sudo. Adding user to docker group..."
            echo "Run these commands to fix permissions:"
            echo "  sudo usermod -aG docker \$USER"
            echo "  newgrp docker"
            echo "Or run this script with sudo."
            
            # Ask if user wants to continue with sudo
            echo ""
            echo "Continue with sudo? (y/N)"
            read -r response
            if [[ ! "$response" =~ ^([yY][eS]|[yY])$ ]]; then
                exit 1
            fi
            DOCKER_CMD="sudo docker"
        else
            print_error "Docker daemon is not running. Start it with:"
            echo "  sudo systemctl start docker"
            exit 1
        fi
    else
        DOCKER_CMD="docker"
    fi
    
    # Create .env if needed
    if [ ! -f ".env" ]; then
        if [ -f ".env.docker" ]; then
            cp .env.docker .env
            print_success "Created .env from template"
        fi
    fi
    
    # Manual Docker commands to avoid compose issues
    print_step "Building and starting containers manually..."
    
    # Clean up any existing containers
    print_step "Cleaning up existing containers..."
    $DOCKER_CMD stop surveillance-backend surveillance-worker surveillance-redis 2>/dev/null || true
    $DOCKER_CMD rm surveillance-backend surveillance-worker surveillance-redis 2>/dev/null || true
    
    # Build the application image
    print_step "Building application image..."
    $DOCKER_CMD build -t surveillance-app .
    
    # Create network
    $DOCKER_CMD network create surveillance-network 2>/dev/null || true
    
    # Start Redis
    print_step "Starting Redis..."
    $DOCKER_CMD run -d --name surveillance-redis \
        --network surveillance-network \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine
    
    # Wait a moment for Redis to start
    sleep 3
    
    # Start Backend
    print_step "Starting Backend API..."
    $DOCKER_CMD run -d --name surveillance-backend \
        --network surveillance-network \
        -p 5000:5000 \
        -e REDIS_HOST=surveillance-redis \
        -e REDIS_PORT=6379 \
        -e PYTHONPATH=/app/src \
        -v "$(pwd)/src/data:/app/src/data" \
        -v "$(pwd)/src/assets:/app/src/assets" \
        --restart unless-stopped \
        surveillance-app
    
    # Wait for backend to start
    sleep 3
    
    # Start Worker
    print_step "Starting AI Worker..."
    $DOCKER_CMD run -d --name surveillance-worker \
        --network surveillance-network \
        -e REDIS_HOST=surveillance-redis \
        -e REDIS_PORT=6379 \
        -e PYTHONPATH=/app/src \
        -v "$(pwd)/src/data:/app/src/data" \
        -v "$(pwd)/src/assets:/app/src/assets" \
        --restart unless-stopped \
        surveillance-app \
        celery -A src.services.job_queue worker --loglevel=info --concurrency=1
    
    print_success "Docker services started!"
    echo ""
    echo "🌐 Access points:"
    echo "  • Backend API: http://localhost:5000"
    echo "  • API Docs: http://localhost:5000/docs"
    echo ""
    echo "📊 Check status: $DOCKER_CMD ps"
    echo "📋 View logs: $DOCKER_CMD logs surveillance-backend"
    echo "🛑 Stop all: ./run.sh stop"
}

start_backend() {
    print_step "🚀 Starting backend server..."
    
    if [ ! -f "src/main.py" ]; then
        print_error "main.py not found in src/"
        exit 1
    fi
    
    cd src
    echo "Starting FastAPI on http://localhost:5000"
    python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
}

start_frontend() {
    print_step "🎨 Starting Streamlit frontend..."
    
    if [ ! -f "streamlit/app.py" ]; then
        print_error "Streamlit app not found"
        exit 1
    fi
    
    # Install streamlit if needed
    if ! python3 -c "import streamlit" 2>/dev/null; then
        pip3 install streamlit
    fi
    
    cd streamlit
    echo "Starting Streamlit on http://localhost:8501"
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
}

full_setup() {
    print_step "🔧 Complete setup starting..."
    
    # Install dependencies
    install_dependencies
    
    # Download models
    download_models
    
    print_success "Setup complete!"
    echo ""
    echo "🚀 Next steps:"
    echo "  • Start backend: $0 start"
    echo "  • Start frontend: $0 frontend"
    echo "  • Or use Docker: $0 docker"
}

clean_project() {
    print_step "🧹 Cleaning project..."
    
    # Remove cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove logs
    rm -rf src/data/*.log 2>/dev/null || true
    
    # Remove generated files
    rm -f analytics_output.json 2>/dev/null || true
    
    print_success "Project cleaned"
}

# Main logic
case "${1:-help}" in
    install)
        install_dependencies
        ;;
    models)
        download_models
        ;;
    docker)
        start_docker
        ;;
    stop)
        stop_docker
        ;;
    start)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    full)
        full_setup
        ;;
    clean)
        clean_project
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
