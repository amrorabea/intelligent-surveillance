# 📋 System Requirements

This document outlines the hardware and software requirements for running the Intelligent Surveillance System.

## 🖥️ Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0+ GHz (Intel i5/AMD Ryzen 5 equivalent)
- **RAM**: 8GB DDR4
- **Storage**: 20GB free space (SSD recommended)
- **Network**: Broadband internet connection

### Recommended Requirements
- **CPU**: 8 cores, 3.0+ GHz (Intel i7/AMD Ryzen 7 equivalent)
- **RAM**: 16GB+ DDR4
- **Storage**: 100GB+ free space (NVMe SSD)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1660/RTX 3060 or better)
- **Network**: Gigabit ethernet

### Production Requirements
- **CPU**: 16+ cores, 3.5+ GHz (Intel Xeon/AMD EPYC)
- **RAM**: 32GB+ DDR4 ECC
- **Storage**: 500GB+ enterprise SSD with RAID
- **GPU**: NVIDIA RTX 4070/A4000 or better (8GB+ VRAM)
- **Network**: 10 Gigabit ethernet with redundancy

## 💿 Operating System Support

### Officially Supported
- ✅ **Ubuntu 20.04 LTS** (Recommended)
- ✅ **Ubuntu 22.04 LTS**
- ✅ **CentOS 8/RHEL 8**
- ✅ **macOS 11.0+ (Big Sur)**
- ✅ **Windows 10/11** (with WSL2)

### Community Tested
- ⚠️ **Debian 11**
- ⚠️ **Fedora 35+**
- ⚠️ **openSUSE Leap 15**

### Docker Support
- ✅ **Docker Engine 20.10+**
- ✅ **Docker Compose v2.0+**
- ✅ **Kubernetes 1.20+**

## 🐍 Software Dependencies

### Core Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **pip**: 21.0+
- **Git**: 2.25+

### Database & Queue
- **Redis**: 6.0+ (for job queuing and caching)
- **ChromaDB**: Auto-installed (vector database)

### Development Tools (Optional)
- **Node.js**: 16+ (for frontend development)
- **Docker**: 20.10+ (containerization)
- **Make**: 4.0+ (build automation)

## 🚀 Performance Considerations

### Video Processing Performance

| Component | CPU Only | GPU Accelerated |
|-----------|----------|-----------------|
| **Frame Extraction** | ~30 FPS | ~60 FPS |
| **Object Detection** | ~5 FPS | ~30 FPS |
| **Image Captioning** | ~2 FPS | ~15 FPS |
| **Embedding Generation** | ~10 FPS | ~40 FPS |

### Memory Usage Patterns

| Service | Idle | Light Load | Heavy Load |
|---------|------|------------|------------|
| **FastAPI Backend** | 200MB | 500MB | 1GB |
| **Celery Worker** | 300MB | 2GB | 8GB |
| **Redis** | 50MB | 200MB | 1GB |
| **Streamlit Frontend** | 100MB | 300MB | 500MB |
| **AI Models** | 2GB | 3GB | 4GB |

### Storage Requirements

| Data Type | Size per Hour | Notes |
|-----------|---------------|-------|
| **Original Video** | 1-4GB | Depends on resolution/codec |
| **Extracted Frames** | 100-500MB | JPEG compressed |
| **Detection Data** | 10-50MB | JSON metadata |
| **Vector Embeddings** | 50-200MB | High-dimensional vectors |
| **Thumbnails** | 20-100MB | Preview images |

## 🔧 GPU Acceleration (Optional but Recommended)

### NVIDIA GPUs (CUDA)
- **Minimum**: GTX 1060 (6GB VRAM)
- **Recommended**: RTX 3070 (8GB VRAM)
- **Production**: RTX 4080/A5000 (16GB+ VRAM)

### Requirements
- **CUDA Toolkit**: 11.8+
- **cuDNN**: 8.6+
- **NVIDIA Driver**: 520+

### Installation
```bash
# Check GPU compatibility
nvidia-smi

# Install CUDA (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPUs (ROCm)
Limited support through PyTorch ROCm builds:
- **Minimum**: RX 6700 XT
- **ROCm**: 5.4+

## 🌐 Network Requirements

### Bandwidth
- **Upload**: 10 Mbps minimum for video uploads
- **Download**: 5 Mbps for model downloads
- **Internal**: Gigabit for multi-node deployments

### Ports
- **8000**: FastAPI backend
- **8501**: Streamlit frontend
- **6379**: Redis server
- **5555**: Celery Flower (monitoring)

### Firewall Configuration
```bash
# Ubuntu UFW
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
sudo ufw allow 6379/tcp

# CentOS/RHEL firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

## 🔒 Security Requirements

### File System Permissions
- **User**: Non-root user recommended
- **Directories**: 755 permissions
- **Files**: 644 permissions
- **Executables**: 755 permissions

### Network Security
- **TLS/SSL**: Required for production
- **Authentication**: JWT tokens
- **Rate Limiting**: Built-in protection
- **CORS**: Configurable origins

## 📊 Scaling Considerations

### Horizontal Scaling
- **Multiple Workers**: Linear performance scaling
- **Load Balancing**: NGINX/HAProxy recommended
- **Database Sharding**: ChromaDB clustering
- **CDN**: For static assets and thumbnails

### Vertical Scaling
- **CPU**: More cores = faster processing
- **RAM**: Larger models and batch sizes
- **GPU**: Multiple GPUs supported
- **Storage**: NVMe RAID for I/O intensive workloads

## 🐳 Container Requirements

### Docker
```yaml
# Minimum container resources
resources:
  limits:
    memory: "8Gi"
    cpu: "4"
  requests:
    memory: "4Gi"
    cpu: "2"
```

### Kubernetes
```yaml
# Production pod specifications
spec:
  containers:
  - name: surveillance-api
    resources:
      limits:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: 1
      requests:
        memory: "8Gi"
        cpu: "4"
```

## ☁️ Cloud Provider Recommendations

### AWS
- **Instance**: g4dn.xlarge or larger
- **Storage**: EBS gp3 volumes
- **Network**: Enhanced networking enabled
- **Services**: ECS/EKS for orchestration

### Google Cloud
- **Instance**: n1-standard-4 with GPU
- **Storage**: SSD persistent disks
- **Network**: Premium tier
- **Services**: GKE for orchestration

### Azure
- **Instance**: Standard_NC6s_v3
- **Storage**: Premium SSD
- **Network**: Accelerated networking
- **Services**: AKS for orchestration

### Oracle Cloud
- **Instance**: VM.GPU3.1 (free tier available)
- **Storage**: Block volumes
- **Network**: Virtual cloud network

## 🧪 Development Environment

### IDE Recommendations
- **VS Code**: Python extension pack
- **PyCharm**: Professional edition
- **Jupyter**: For notebook development
- **Vim/Neovim**: With Python LSP

### Development Tools
```bash
# Essential development packages
pip install black isort flake8 mypy pytest pytest-cov
```

### Browser Support
- **Chrome**: 90+ (recommended)
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## 📋 Pre-Installation Checklist

### Before Installation
- [ ] Verify CPU and RAM meet minimum requirements
- [ ] Check available storage space (20GB+ free)
- [ ] Install Git and Python 3.8+
- [ ] Configure firewall ports
- [ ] Download GPU drivers (if applicable)

### Optional Optimizations
- [ ] Install SSD for better I/O performance
- [ ] Configure GPU acceleration
- [ ] Set up monitoring tools
- [ ] Plan backup strategy
- [ ] Configure log rotation

### Production Readiness
- [ ] Set up SSL certificates
- [ ] Configure authentication
- [ ] Implement monitoring
- [ ] Plan disaster recovery
- [ ] Set up automated backups

## 🔍 Performance Benchmarks

### Reference System
- **CPU**: Intel i7-10700K (8 cores, 3.8GHz)
- **RAM**: 32GB DDR4-3200
- **GPU**: NVIDIA RTX 3070 (8GB VRAM)
- **Storage**: 1TB NVMe SSD

### Processing Times
- **1080p Video (1 hour)**: ~15 minutes
- **720p Video (1 hour)**: ~8 minutes
- **4K Video (1 hour)**: ~45 minutes
- **Search Query**: <1 second
- **Embedding Generation**: ~2 seconds per frame

## 📞 Support Matrix

| Component | Support Level | Updates |
|-----------|---------------|---------|
| **Ubuntu 20.04/22.04** | ✅ Full | Security + Feature |
| **macOS 11+** | ✅ Full | Security + Feature |
| **Windows 10/11 + WSL2** | ✅ Full | Security + Feature |
| **Docker/Kubernetes** | ✅ Full | Security + Feature |
| **CentOS/RHEL 8** | ⚠️ Community | Security Only |
| **Other Linux** | ⚠️ Community | Best Effort |

---

**💡 Need help with requirements? Check our [Installation Guide](INSTALLATION.md) or [Troubleshooting Guide](TROUBLESHOOTING.md).**
