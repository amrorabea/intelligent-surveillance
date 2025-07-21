#!/bin/bash
# demo.sh - Comprehensive demo script for the Intelligent Surveillance System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Intelligent Surveillance System - Interactive Demo${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# Function to print step headers
print_step() {
    echo ""
    echo -e "${PURPLE}📋 Step $1: $2${NC}"
    echo -e "${PURPLE}$(printf '%.0s-' {1..50})${NC}"
}

# Function to wait for user input
wait_for_user() {
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read -r
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    echo -e "${YELLOW}⏳ Waiting for ${service_name} to start...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ ${service_name} is running${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts...${NC}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ ${service_name} failed to start after $max_attempts attempts${NC}"
    return 1
}

# Function to simulate video upload and processing
demo_video_processing() {
    echo -e "${BLUE}🎬 Simulating video upload and processing...${NC}"
    
    # Create a dummy project
    local project_id="demo-$(date +%s)"
    echo -e "${YELLOW}📁 Creating demo project: ${project_id}${NC}"
    
    # Show curl command for video upload (simulated)
    echo -e "${YELLOW}📤 Video upload command (example):${NC}"
    echo "curl -X POST http://localhost:5000/api/data/upload/${project_id} \\"
    echo "     -F 'file=@sample_video.mp4' \\"
    echo "     -H 'Content-Type: multipart/form-data'"
    
    echo ""
    echo -e "${YELLOW}⚙️ Processing configuration (example):${NC}"
    echo "curl -X POST http://localhost:5000/api/surveillance/process/${project_id}/file123 \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{"
    echo "       \"detect_objects\": true,"
    echo "       \"track_objects\": true,"
    echo "       \"generate_captions\": true,"
    echo "       \"confidence_threshold\": 0.5"
    echo "     }'"
    
    echo ""
    echo -e "${GREEN}✅ Video processing would be handled by Celery workers${NC}"
    echo -e "${GREEN}✅ Results stored in ChromaDB for semantic search${NC}"
}

# Function to demonstrate search capabilities
demo_semantic_search() {
    echo -e "${BLUE}🔍 Demonstrating semantic search capabilities...${NC}"
    
    echo -e "${YELLOW}🧠 Example search queries:${NC}"
    echo "  • 'person walking with a dog'"
    echo "  • 'red car in parking lot'"
    echo "  • 'people gathering together'"
    echo "  • 'bicycle on the street'"
    echo ""
    
    echo -e "${YELLOW}🔍 Search API endpoint (example):${NC}"
    echo "curl -X GET 'http://localhost:5000/api/surveillance/query' \\"
    echo "     -G -d 'query=person walking with dog' \\"
    echo "     -d 'max_results=10' \\"
    echo "     -d 'similarity_threshold=0.7'"
    
    echo ""
    echo -e "${GREEN}✅ Results include frame previews, AI captions, and similarity scores${NC}"
}

# Main demo flow
main() {
    print_step "1" "System Overview"
    echo -e "${GREEN}🧠 AI Models:${NC}"
    echo "  • YOLOv8: Real-time object detection (80+ object classes)"
    echo "  • BLIP: Advanced image captioning for scene understanding"
    echo "  • Sentence Transformers: Semantic search capabilities"
    echo ""
    echo -e "${GREEN}⚙️ Backend Technologies:${NC}"
    echo "  • FastAPI: High-performance REST API"
    echo "  • Celery: Distributed task processing"
    echo "  • Redis: Message broker and caching"
    echo "  • ChromaDB: Vector database for semantic search"
    echo ""
    echo -e "${GREEN}🎨 Frontend:${NC}"
    echo "  • Streamlit: Interactive web interface"
    echo "  • Real-time job monitoring"
    echo "  • Analytics dashboard"
    echo "  • System health monitoring"
    
    wait_for_user
    
    print_step "2" "Environment Check"
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✅ Python 3 installed: $(python3 --version)${NC}"
    else
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi
    
    # Check virtual environment
    if [ -d "venv_surveillance" ]; then
        echo -e "${GREEN}✅ Virtual environment found${NC}"
    else
        echo -e "${YELLOW}⚠️ Virtual environment not found. Run 'make setup' first${NC}"
    fi
    
    # Check Redis
    if systemctl is-active --quiet redis-server 2>/dev/null; then
        echo -e "${GREEN}✅ Redis server running${NC}"
    else
        echo -e "${YELLOW}⚠️ Redis server not running. Starting with 'make redis'${NC}"
    fi
    
    wait_for_user
    
    print_step "3" "Starting Backend Services"
    echo -e "${YELLOW}🚀 Starting surveillance system backend...${NC}"
    echo ""
    echo "This will start:"
    echo "  • Redis server (if not running)"
    echo "  • Celery workers for background processing"
    echo "  • FastAPI server with REST endpoints"
    echo ""
    echo -e "${YELLOW}Note: This demo assumes services are already running.${NC}"
    echo -e "${YELLOW}To start services manually, run: make dev${NC}"
    
    # Check if services are running
    if check_service "API Server" "http://localhost:5000/health" 3; then
        echo -e "${GREEN}✅ Backend services are ready${NC}"
    else
        echo -e "${YELLOW}⚠️ Backend services not detected. Please run 'make dev' in another terminal${NC}"
        echo -e "${YELLOW}The demo will continue with API examples...${NC}"
    fi
    
    wait_for_user
    
    print_step "4" "API Endpoints Overview"
    echo -e "${BLUE}📊 Available API endpoints:${NC}"
    echo ""
    echo -e "${GREEN}Data Management:${NC}"
    echo "  POST /api/data/upload/{project_id}     - Upload video files"
    echo "  GET  /api/data/projects                - List all projects"
    echo "  DELETE /api/data/project/{project_id}  - Delete project"
    echo ""
    echo -e "${GREEN}Video Processing:${NC}"
    echo "  POST /api/surveillance/process/{project_id}/{file_id} - Start processing"
    echo "  GET  /api/surveillance/jobs/status/{job_id}           - Check job status"
    echo "  GET  /api/surveillance/jobs/active                    - List active jobs"
    echo ""
    echo -e "${GREEN}Semantic Search:${NC}"
    echo "  GET  /api/surveillance/query           - Search with natural language"
    echo "  GET  /api/surveillance/stats           - Database statistics"
    echo ""
    echo -e "${GREEN}System Health:${NC}"
    echo "  GET  /api/health                       - Overall system health"
    echo "  GET  /api/surveillance/analytics       - System analytics"
    
    echo ""
    echo -e "${YELLOW}📚 Interactive API documentation: http://localhost:5000/docs${NC}"
    
    wait_for_user
    
    print_step "5" "Video Processing Demo"
    demo_video_processing
    
    wait_for_user
    
    print_step "6" "Semantic Search Demo"
    demo_semantic_search
    
    wait_for_user
    
    print_step "7" "Frontend Interface"
    echo -e "${BLUE}🎨 Starting Streamlit frontend...${NC}"
    echo ""
    echo "The frontend provides:"
    echo -e "${GREEN}📤 Video Processing:${NC} Upload and configure video analysis"
    echo -e "${GREEN}🔍 Semantic Search:${NC} Natural language search interface"
    echo -e "${GREEN}📊 Analytics:${NC} Interactive charts and insights"
    echo -e "${GREEN}⚙️ System Status:${NC} Health monitoring and diagnostics"
    echo ""
    echo -e "${YELLOW}To start the frontend manually: make frontend${NC}"
    echo -e "${YELLOW}To start both backend and frontend: make fullstack${NC}"
    echo ""
    echo -e "${YELLOW}Frontend URL: http://localhost:8501${NC}"
    
    if check_service "Streamlit Frontend" "http://localhost:8501" 3; then
        echo -e "${GREEN}✅ Frontend is running and accessible${NC}"
        
        echo ""
        echo -e "${BLUE}🌐 Opening frontend in browser...${NC}"
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8501 2>/dev/null &
        elif command -v open &> /dev/null; then
            open http://localhost:8501 2>/dev/null &
        else
            echo -e "${YELLOW}Please open http://localhost:8501 in your browser${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ Frontend not detected. You can start it with 'make frontend'${NC}"
    fi
    
    wait_for_user
    
    print_step "8" "System Architecture"
    echo -e "${BLUE}🏗️ System Architecture Overview:${NC}"
    echo ""
    echo "┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐"
    echo "│   Streamlit     │    │    FastAPI      │    │     Redis       │"
    echo "│   Frontend      │◄──►│   Backend       │◄──►│  Message Broker │"
    echo "│  (Port 8501)    │    │  (Port 5000)    │    │  (Port 6379)    │"
    echo "└─────────────────┘    └─────────────────┘    └─────────────────┘"
    echo "         │                       │                       ▲"
    echo "         │                       ▼                       │"
    echo "         │              ┌─────────────────┐              │"
    echo "         │              │  Celery Workers │──────────────┘"
    echo "         │              │  (Background    │"
    echo "         │              │   Processing)   │"
    echo "         │              └─────────────────┘"
    echo "         │                       │"
    echo "         │                       ▼"
    echo "         │              ┌─────────────────┐"
    echo "         │              │   AI Models     │"
    echo "         │              │  YOLOv8 + BLIP  │"
    echo "         │              └─────────────────┘"
    echo "         │                       │"
    echo "         │                       ▼"
    echo "         │              ┌─────────────────┐"
    echo "         └─────────────►│    ChromaDB     │"
    echo "                        │ Vector Database │"
    echo "                        └─────────────────┘"
    
    wait_for_user
    
    print_step "9" "Next Steps"
    echo -e "${GREEN}🎯 Getting Started:${NC}"
    echo ""
    echo -e "${YELLOW}1. Setup (if not done):${NC}"
    echo "   make setup          # Complete system setup"
    echo ""
    echo -e "${YELLOW}2. Start Services:${NC}"
    echo "   make dev            # Start backend services"
    echo "   make frontend       # Start frontend (separate terminal)"
    echo "   # OR"
    echo "   make fullstack      # Start everything together"
    echo ""
    echo -e "${YELLOW}3. Test the System:${NC}"
    echo "   • Upload a video through the frontend"
    echo "   • Monitor processing in real-time"
    echo "   • Try semantic search queries"
    echo "   • Explore analytics dashboard"
    echo ""
    echo -e "${YELLOW}4. Development:${NC}"
    echo "   make test           # Run tests"
    echo "   make status         # Check system status"
    echo "   make logs           # View application logs"
    echo ""
    echo -e "${GREEN}📚 Documentation:${NC}"
    echo "   • API Docs: http://localhost:5000/docs"
    echo "   • Frontend: http://localhost:8501"
    echo "   • README files in each directory"
    echo ""
    echo -e "${GREEN}🔧 Useful Commands:${NC}"
    echo "   make help           # Show all available commands"
    echo "   make stop-all       # Stop all services"
    echo "   make clean          # Clean temporary files"
    echo "   make reset          # Complete reset and setup"
    
    echo ""
    echo -e "${BLUE}🎉 Demo Complete!${NC}"
    echo -e "${GREEN}The Intelligent Surveillance System is ready for use.${NC}"
    echo ""
    echo -e "${YELLOW}For questions or issues, check the documentation or logs.${NC}"
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}Demo interrupted. Services may still be running.${NC}"; exit 1' INT

# Run the demo
main "$@"
