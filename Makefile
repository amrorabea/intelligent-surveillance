# ============================================================================
# Intelligent Surveillance System - Makefile
# Automates setup, testing, and running of the surveillance platform
# ============================================================================

# Variables
PYTHON := python3
VENV_NAME := venv_surveillance
VENV_PATH := $(shell pwd)/$(VENV_NAME)
ACTIVATE := . $(VENV_PATH)/bin/activate
SRC_DIR := ./src
REDIS_PORT := 6379
API_PORT := 5000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Default target
.PHONY: help
help:
	@echo "$(BLUE)Intelligent Surveillance System - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  setup           - Complete project setup (venv + dependencies + Redis)"
	@echo "  install         - Install Python dependencies only"
	@echo "  install-redis   - Install and configure Redis"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  dev             - Start all services for development (Redis + Celery + API)"
	@echo "  api             - Start FastAPI server only"
	@echo "  worker          - Start Celery worker only"
	@echo "  redis           - Start Redis server"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@echo "  test            - Run all tests"
	@echo "  test-api        - Test API endpoints"
	@echo "  test-upload     - Test file upload"
	@echo "  test-process    - Test video processing"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@echo "  status          - Check system status"
	@echo "  logs            - Show application logs"
	@echo "  clean           - Clean up temporary files"
	@echo "  reset           - Reset project (clean + setup)"
	@echo "  stop            - Stop all services"

# ============================================================================
# Setup Commands
# ============================================================================

.PHONY: setup
setup: create-venv install install-redis create-dirs
	@echo "$(GREEN)✅ Complete setup finished!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run 'make dev' to start all services"
	@echo "  2. Visit http://localhost:$(API_PORT)/docs for API documentation"

.PHONY: create-venv
create-venv:
	@echo "$(BLUE)🔧 Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_PATH)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)✅ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment already exists$(NC)"; \
	fi

.PHONY: install
install: create-venv
	@echo "$(BLUE)📦 Installing Python dependencies...$(NC)"
	@$(ACTIVATE) && pip install --upgrade pip
	@$(ACTIVATE) && pip install -r $(SRC_DIR)/requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

.PHONY: install-redis
install-redis:
	@echo "$(BLUE)🔧 Setting up Redis...$(NC)"
	@if ! command -v redis-server > /dev/null; then \
		echo "$(YELLOW)Installing Redis...$(NC)"; \
		sudo apt update; \
		sudo apt install -y redis-server; \
	fi
	@sudo systemctl enable redis-server
	@echo "$(GREEN)✅ Redis configured$(NC)"

.PHONY: create-dirs
create-dirs:
	@echo "$(BLUE)📁 Creating project directories...$(NC)"
	@mkdir -p $(SRC_DIR)/assets/files/projects
	@mkdir -p $(SRC_DIR)/assets/files/chromadb
	@mkdir -p $(SRC_DIR)/logs
	@echo "$(GREEN)✅ Directories created$(NC)"

# ============================================================================
# Development Commands
# ============================================================================

.PHONY: dev
dev: check-redis start-services
	@echo "$(GREEN)🚀 Development environment started!$(NC)"
	@echo "$(YELLOW)Services running:$(NC)"
	@echo "  • API Server: http://localhost:$(API_PORT)"
	@echo "  • API Docs: http://localhost:$(API_PORT)/docs"
	@echo "  • Redis: localhost:$(REDIS_PORT)"
	@echo ""
	@echo "$(YELLOW)Press Ctrl+C to stop all services$(NC)"

.PHONY: start-services
start-services:
	@echo "$(BLUE)🚀 Starting all services...$(NC)"
	@trap 'echo "$(RED)🛑 Stopping services...$(NC)"; kill %1 %2 2>/dev/null; exit' INT; \
	(cd $(SRC_DIR) && $(ACTIVATE) && python -m celery -A services.job_queue worker --loglevel=info) & \
	sleep 3; \
	(cd $(SRC_DIR) && $(ACTIVATE) && python -m uvicorn main:app --reload --host 0.0.0.0 --port $(API_PORT)) & \
	wait

.PHONY: api
api: check-venv
	@echo "$(BLUE)🌐 Starting FastAPI server...$(NC)"
	@cd $(SRC_DIR) && $(ACTIVATE) && python -m uvicorn main:app --reload --host 0.0.0.0 --port $(API_PORT)

.PHONY: worker
worker: check-venv check-redis
	@echo "$(BLUE)⚙️  Starting Celery worker...$(NC)"
	@cd $(SRC_DIR) && $(ACTIVATE) && python -m celery -A services.job_queue worker --loglevel=info

.PHONY: redis
redis:
	@echo "$(BLUE)🔧 Starting Redis server...$(NC)"
	@sudo systemctl start redis-server
	@echo "$(GREEN)✅ Redis started$(NC)"

# ============================================================================
# Testing Commands
# ============================================================================

.PHONY: test
test: test-api test-upload
	@echo "$(GREEN)✅ All tests completed$(NC)"

.PHONY: test-api
test-api: check-services
	@echo "$(BLUE)🧪 Testing API endpoints...$(NC)"
	@curl -s http://localhost:$(API_PORT)/health | jq . || echo "$(RED)❌ Health check failed$(NC)"
	@curl -s http://localhost:$(API_PORT)/ | jq . || echo "$(RED)❌ Root endpoint failed$(NC)"

.PHONY: test-upload
test-upload: check-services
	@echo "$(BLUE)🧪 Testing file upload...$(NC)"
	@if [ -f "test_video.mp4" ]; then \
		curl -X POST "http://localhost:$(API_PORT)/api/data/upload/test-project" \
		-H "accept: application/json" \
		-H "Content-Type: multipart/form-data" \
		-F "file=@test_video.mp4;type=video/mp4" | jq .; \
	else \
		echo "$(YELLOW)⚠️  No test_video.mp4 found for upload testing$(NC)"; \
	fi

.PHONY: test-process
test-process: check-services
	@echo "$(BLUE)🧪 Testing video processing...$(NC)"
	@echo "$(YELLOW)Note: Replace YOUR_FILE_ID with actual uploaded file ID$(NC)"
	@curl -X POST "http://localhost:$(API_PORT)/api/surveillance/process/test-project/YOUR_FILE_ID" \
	-H "accept: application/json" \
	-H "Content-Type: application/json" \
	-d '{"sample_rate": 1.0, "detection_threshold": 0.6}' | jq .

# ============================================================================
# Utility Commands
# ============================================================================

.PHONY: status
status:
	@echo "$(BLUE)📊 System Status$(NC)"
	@echo ""
	@echo "$(YELLOW)Virtual Environment:$(NC)"
	@if [ -d "$(VENV_PATH)" ]; then \
		echo "  ✅ $(VENV_NAME) exists"; \
		echo "  📦 Python: $$($(ACTIVATE) && python --version)"; \
	else \
		echo "  ❌ Virtual environment not found"; \
	fi
	@echo ""
	@echo "$(YELLOW)Redis Status:$(NC)"
	@if systemctl is-active --quiet redis-server; then \
		echo "  ✅ Redis is running"; \
		redis-cli ping 2>/dev/null | grep -q PONG && echo "  ✅ Redis responding" || echo "  ❌ Redis not responding"; \
	else \
		echo "  ❌ Redis is not running"; \
	fi
	@echo ""
	@echo "$(YELLOW)API Server:$(NC)"
	@if curl -s http://localhost:$(API_PORT)/health >/dev/null 2>&1; then \
		echo "  ✅ API server responding on port $(API_PORT)"; \
	else \
		echo "  ❌ API server not responding"; \
	fi
	@echo ""
	@echo "$(YELLOW)Project Structure:$(NC)"
	@ls -la $(SRC_DIR)/assets/files/projects/ 2>/dev/null | wc -l | xargs printf "  📁 Projects: %d\n" || echo "  📁 Projects: 0"

.PHONY: logs
logs:
	@echo "$(BLUE)📋 Recent application logs$(NC)"
	@if [ -f "$(SRC_DIR)/logs/app.log" ]; then \
		tail -20 $(SRC_DIR)/logs/app.log; \
	else \
		echo "$(YELLOW)No log file found$(NC)"; \
	fi

.PHONY: clean
clean:
	@echo "$(BLUE)🧹 Cleaning temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*~" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

.PHONY: reset
reset: stop clean
	@echo "$(BLUE)🔄 Resetting project...$(NC)"
	@rm -rf $(VENV_PATH)
	@$(MAKE) setup

.PHONY: stop
stop:
	@echo "$(BLUE)🛑 Stopping all services...$(NC)"
	@pkill -f "uvicorn main:app" 2>/dev/null || true
	@pkill -f "celery.*worker" 2>/dev/null || true
	@sudo systemctl stop redis-server 2>/dev/null || true
	@echo "$(GREEN)✅ All services stopped$(NC)"

# ============================================================================
# Helper Targets (Internal)
# ============================================================================

.PHONY: check-venv
check-venv:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi

.PHONY: check-redis
check-redis:
	@if ! systemctl is-active --quiet redis-server; then \
		echo "$(YELLOW)⚠️  Starting Redis...$(NC)"; \
		sudo systemctl start redis-server; \
	fi
	@sleep 1
	@if ! redis-cli ping >/dev/null 2>&1; then \
		echo "$(RED)❌ Redis is not responding. Check Redis installation.$(NC)"; \
		exit 1; \
	fi

.PHONY: check-services
check-services: check-venv check-redis
	@if ! curl -s http://localhost:$(API_PORT)/health >/dev/null 2>&1; then \
		echo "$(RED)❌ API server not running. Start with 'make dev' or 'make api'$(NC)"; \
		exit 1; \
	fi

# ============================================================================
# Development Shortcuts
# ============================================================================

.PHONY: quick-start
quick-start: check-redis start-services

.PHONY: restart
restart: stop dev

.PHONY: update
update: install
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

# Make all targets silent by default
.SILENT: