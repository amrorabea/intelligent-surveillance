# 🛠️ Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Intelligent Surveillance System.

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Check if all services are running
make status-fullstack

# Test API connectivity
curl http://localhost:8000/health

# Test frontend access
curl http://localhost:8501

# Check Redis connection
redis-cli ping
```

### Log Analysis
```bash
# View application logs
tail -f logs/app.log

# Check Docker logs
docker-compose logs -f

# Monitor system resources
htop
nvidia-smi  # For GPU systems
```

## 🔧 Common Issues & Solutions

### 🚫 Installation Issues

#### Python Version Incompatibility
**Problem**: "Python 3.8+ required" error
```bash
ModuleNotFoundError: No module named 'asyncio'
```

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.8+ (Ubuntu)
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# Create virtual environment with correct Python
python3.9 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

#### Package Installation Errors
**Problem**: "Failed building wheel" errors during pip install

**Solution**:
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu)
sudo apt install build-essential python3-dev

# Install with verbose output to see errors
pip install -v -r src/requirements.txt

# Force reinstall if corrupted
pip install --force-reinstall torch torchvision
```

#### Permission Denied Errors
**Problem**: Permission errors when running scripts

**Solution**:
```bash
# Fix script permissions
chmod +x setup.sh
chmod +x scripts/*.sh
chmod +x fix_permissions.sh

# Run permission fix script
./fix_permissions.sh

# Fix file ownership
sudo chown -R $USER:$USER .
```

### 🌐 Service Startup Issues

#### Port Already in Use
**Problem**: "Address already in use" errors
```bash
Error: [Errno 98] Address already in use: ('0.0.0.0', 8000)
```

**Solution**:
```bash
# Find process using port
sudo lsof -t -i:8000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8000)

# Or use different ports in .env
echo "API_PORT=8001" >> .env
echo "FRONTEND_PORT=8502" >> .env
```

#### Redis Connection Failed
**Problem**: "Connection refused" when connecting to Redis
```bash
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379
```

**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# Start Redis server
redis-server

# Or install Redis if not present
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Check Redis configuration
sudo nano /etc/redis/redis.conf
```

#### Celery Worker Not Starting
**Problem**: Celery worker fails to start or crashes

**Solution**:
```bash
# Check worker status
celery -A src.services.job_queue inspect active

# Start worker manually with verbose logging
cd src && celery -A services.job_queue worker --loglevel=debug

# Clear Redis if tasks are stuck
redis-cli FLUSHALL

# Restart worker
make restart-worker
```

### 🤖 AI Model Issues

#### Model Download Failures
**Problem**: Models fail to download or load
```bash
OSError: Can't load tokenizer from 'Salesforce/blip-image-captioning-base'
```

**Solution**:
```bash
# Check internet connection
ping huggingface.co

# Clear model cache
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Download models manually
python -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
print('BLIP model downloaded successfully')
"

# Run model setup script
./scripts/setup_ai_models.sh
```

#### CUDA/GPU Issues
**Problem**: GPU not detected or CUDA errors
```bash
RuntimeError: CUDA out of memory
UserWarning: CUDA initialization: CUDA unknown error
```

**Solution**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if missing
# Download from: https://developer.nvidia.com/cuda-toolkit

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reduce batch size if out of memory
export TORCH_CUDA_MEMORY_FRACTION=0.7
```

#### Model Loading Errors
**Problem**: Models fail to load or produce errors
```bash
RuntimeError: Error(s) in loading state_dict for YOLOv8
```

**Solution**:
```bash
# Remove corrupted model files
rm src/models/yolov8n.pt

# Re-download models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8 downloaded successfully')
"

# Check model file integrity
ls -la src/models/
file src/models/yolov8n.pt
```

### 📤 Video Processing Issues

#### Video Upload Failures
**Problem**: Videos fail to upload or process
```bash
413 Payload Too Large
415 Unsupported Media Type
```

**Solution**:
```bash
# Check file size (default limit: 200MB)
ls -lh your-video.mp4

# Check video format
file your-video.mp4
ffprobe your-video.mp4

# Convert video if needed
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# Increase upload limit in config
export MAX_UPLOAD_SIZE=500000000  # 500MB
```

#### Processing Jobs Stuck
**Problem**: Video processing jobs remain in "processing" state

**Solution**:
```bash
# Check worker status
make status-fullstack

# Check job queue
redis-cli LLEN celery

# Clear stuck jobs
redis-cli DEL celery

# Restart worker and resubmit
make restart-worker
```

#### Frame Extraction Errors
**Problem**: Error extracting frames from video
```bash
cv2.error: OpenCV(4.5.1) error: (-215:Assertion failed)
```

**Solution**:
```bash
# Install OpenCV with proper codecs
pip uninstall opencv-python
pip install opencv-python-headless

# Check video file integrity
ffplay your-video.mp4  # Should play without errors

# Try alternative extraction method
ffmpeg -i your-video.mp4 -vf fps=1 frame_%04d.jpg
```

### 🔍 Search & Query Issues

#### No Search Results
**Problem**: Semantic search returns no results

**Solution**:
```bash
# Check if video processing completed
curl "http://localhost:8000/surveillance/projects"

# Verify embeddings were generated
curl "http://localhost:8000/surveillance/analytics"

# Try simpler search terms
curl -X POST "http://localhost:8000/surveillance/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "person", "project_id": "your-project"}'

# Check ChromaDB status
ls -la vector_db/
```

#### ChromaDB Errors
**Problem**: Vector database connection or query errors
```bash
chromadb.errors.ChromaDBError: Collection not found
```

**Solution**:
```bash
# Check ChromaDB directory
ls -la vector_db/

# Reset vector database
rm -rf vector_db/
mkdir vector_db/

# Reprocess videos to regenerate embeddings
# Via frontend or API
```

#### Embedding Generation Failures
**Problem**: Error generating semantic embeddings
```bash
RuntimeError: Expected all tensors to be on the same device
```

**Solution**:
```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Or ensure consistent device usage
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
"
```

### 🖥️ Frontend Issues

#### Streamlit Not Loading
**Problem**: Frontend fails to start or load in browser

**Solution**:
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Start manually with debug info
cd streamlit && streamlit run app.py --logger.level=debug

# Check for port conflicts
netstat -tulpn | grep 8501

# Try different port
streamlit run app.py --server.port=8502
```

#### API Connection Errors
**Problem**: Frontend can't connect to backend API

**Solution**:
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check frontend API configuration
grep -r "localhost:8000" streamlit/

# Update API URL if needed
export API_BASE_URL="http://localhost:8000"
```

#### File Upload Issues in Frontend
**Problem**: Cannot upload files through Streamlit interface

**Solution**:
```bash
# Check upload directory permissions
ls -la uploads/
chmod 755 uploads/

# Check disk space
df -h

# Increase Streamlit upload limit
echo "[server]" > ~/.streamlit/config.toml
echo "maxUploadSize = 500" >> ~/.streamlit/config.toml
```

### 🔒 Permission & Security Issues

#### File Permission Errors
**Problem**: Permission denied when accessing files

**Solution**:
```bash
# Fix all permissions
./fix_permissions.sh

# Set proper ownership
sudo chown -R $USER:$USER .

# Fix specific directories
chmod 755 uploads/ vector_db/ logs/
chmod 644 src/models/*.pt
```

#### SSL/TLS Issues
**Problem**: HTTPS/SSL errors in production

**Solution**:
```bash
# Check SSL certificate
openssl x509 -in certificate.crt -text -noout

# Verify certificate chain
openssl verify -CAfile ca-bundle.crt certificate.crt

# Update certificates
sudo apt update && sudo apt install ca-certificates
```

## 🧪 Debug Mode & Logging

### Enable Debug Logging
```bash
# Set debug level in environment
export LOG_LEVEL=DEBUG

# Or modify config
echo "LOG_LEVEL=DEBUG" >> .env

# Restart services to apply
make restart
```

### Verbose Output
```bash
# Run with verbose flags
python -v src/main.py

# Celery debug mode
celery -A src.services.job_queue worker --loglevel=debug

# Streamlit debug mode
streamlit run streamlit/app.py --logger.level=debug
```

### Performance Profiling
```bash
# Profile Python application
python -m cProfile -o profile.stats src/main.py

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
iotop
```

## 🔧 Advanced Debugging

### Database Debugging
```bash
# Check Redis keys
redis-cli KEYS "*"

# Monitor Redis commands
redis-cli MONITOR

# Check ChromaDB collections
python -c "
import chromadb
client = chromadb.PersistentClient('./vector_db')
print(client.list_collections())
"
```

### Network Debugging
```bash
# Check open ports
netstat -tulpn | grep -E ':(8000|8501|6379)'

# Test API endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/surveillance/projects

# Check firewall
sudo ufw status
```

### Memory Debugging
```bash
# Monitor memory usage
watch -n 1 'free -m'

# Check for memory leaks
python -m pympler.asizeof

# GPU memory monitoring
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

## 📋 Diagnostic Commands

### System Information
```bash
# Generate system report
cat > debug_info.txt << EOF
=== System Information ===
$(uname -a)
$(lsb_release -a 2>/dev/null || cat /etc/os-release)

=== Python Environment ===
$(python --version)
$(pip list | grep -E "(torch|opencv|streamlit|fastapi)")

=== GPU Information ===
$(nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected")

=== Service Status ===
$(make status-fullstack 2>&1)

=== Disk Space ===
$(df -h)

=== Memory Usage ===
$(free -m)
EOF

cat debug_info.txt
```

### Log Collection
```bash
# Collect all relevant logs
mkdir -p debug_logs
cp logs/* debug_logs/ 2>/dev/null
docker-compose logs > debug_logs/docker.log 2>&1
journalctl -u redis > debug_logs/redis.log 2>&1

# Create debug archive
tar -czf debug_$(date +%Y%m%d_%H%M%S).tar.gz debug_logs/ debug_info.txt
```

## 📞 Getting Help

### Before Reporting Issues
1. ✅ Check this troubleshooting guide
2. ✅ Search existing GitHub issues
3. ✅ Run diagnostic commands
4. ✅ Collect debug information

### Reporting Issues
When creating a GitHub issue, include:

- **Environment**: OS, Python version, GPU info
- **Error Message**: Complete error traceback
- **Steps to Reproduce**: Detailed reproduction steps
- **Logs**: Relevant log files
- **Configuration**: Any custom settings

### Community Support
- **GitHub Discussions**: General questions and help
- **GitHub Issues**: Bug reports and feature requests  
- **Documentation**: Check all docs in `/docs` folder
- **Stack Overflow**: Tag with `intelligent-surveillance`

## 🛠️ Emergency Recovery

### Complete System Reset
```bash
# Stop all services
make stop

# Clear all data (WARNING: This removes all videos and processing results)
rm -rf uploads/* vector_db/* logs/*
redis-cli FLUSHALL

# Reset models
rm -rf ~/.cache/huggingface ~/.cache/torch
./scripts/setup_ai_models.sh

# Restart system
make fullstack
```

### Backup Recovery
```bash
# Restore from backup
tar -xzf backup_YYYYMMDD.tar.gz
cp -r backup_data/* .

# Restart services
make restart
```

---

**💡 Still having issues? Don't hesitate to reach out through our [GitHub Issues](https://github.com/your-repo/intelligent-surveillance/issues) or check our [FAQ](FAQ.md).**
