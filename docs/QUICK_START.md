# 🏃‍♂️ Quick Start Guide

Get up and running with the Intelligent Surveillance System in just 10 minutes!

## 🚀 Prerequisites

Make sure you have completed the [Installation Guide](INSTALLATION.md) first.

## ⚡ 5-Minute Setup

### 1. Start the System
```bash
cd intelligent-surveillance
make fullstack
```

This starts:
- ✅ FastAPI backend (http://localhost:8000)
- ✅ Celery worker for AI processing
- ✅ Redis for job queuing
- ✅ Streamlit frontend (http://localhost:8501)

### 2. Open the Web Interface
Navigate to **http://localhost:8501** in your browser.

### 3. Upload Your First Video
1. Go to the **📤 Video Processing** page
2. Click **"Browse files"** and select a video (MP4, AVI, MOV)
3. Enter a project name (e.g., "my-first-project")
4. Click **"Start Processing"**

### 4. Watch the Magic Happen! ✨
The system will:
- Extract frames from your video
- Detect objects using YOLOv8
- Generate AI captions with BLIP
- Track objects across frames
- Create searchable embeddings

## 🎯 Your First Search

Once processing is complete:

1. Go to **🔍 Semantic Search**
2. Select your project from the dropdown
3. Try searching for:
   - "people walking"
   - "cars on the road"
   - "blue objects"
   - "person with backpack"

## 📊 Explore Your Data

Check out the **📊 Analytics** page to see:
- Detection statistics
- Object distribution charts
- Processing timeline
- Performance metrics

## 🔧 API Playground

Want to use the API directly? Try these examples:

### Upload a Video
```bash
curl -X POST "http://localhost:8000/surveillance/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your-video.mp4" \
     -F "project_id=my-project"
```

### Search for Content
```bash
curl -X POST "http://localhost:8000/surveillance/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "person walking",
       "project_id": "my-project",
       "limit": 5
     }'
```

### Get Analytics
```bash
curl "http://localhost:8000/surveillance/analytics?project_id=my-project"
```

## 🎪 Demo Script

Run our pre-configured demo:

```bash
./demo.sh
```

This will:
1. Download a sample video
2. Process it through the complete pipeline
3. Show example searches and results

## 📚 Understanding the Interface

### Video Processing Page 📤
- **File Upload**: Supports MP4, AVI, MOV, MKV formats
- **Project Management**: Organize videos by project
- **Processing Status**: Real-time progress updates
- **Job Queue**: View all processing jobs

### Semantic Search Page 🔍
- **Natural Language Queries**: Search using plain English
- **Frame Preview**: See matching video frames
- **Relevance Scoring**: Results ranked by AI similarity
- **Metadata Display**: Object details and timestamps

### Analytics Page 📊
- **Detection Stats**: Object counts and distributions
- **Processing Metrics**: Timeline and performance data
- **Project Overview**: Multi-project analytics
- **Export Options**: Download data as CSV/JSON

### System Status Page ⚙️
- **Health Checks**: API and service status
- **Debug Tools**: Technical diagnostics
- **Configuration**: Current system settings
- **Logs**: Recent activity and errors

## 🎯 Common Workflows

### Workflow 1: Security Monitoring
1. Upload surveillance footage
2. Search for "person entering building"
3. Review detected intrusions
4. Export incident reports

### Workflow 2: Traffic Analysis
1. Upload traffic camera footage
2. Search for "red car speeding"
3. Analyze vehicle patterns
4. Generate traffic reports

### Workflow 3: Retail Analytics
1. Upload store camera footage
2. Search for "customer picking up product"
3. Analyze shopping behavior
4. Optimize store layout

## 🔍 Search Tips

### Effective Queries
- ✅ **Good**: "person wearing red shirt"
- ✅ **Good**: "car parked near building"
- ✅ **Good**: "dog running in park"
- ❌ **Avoid**: "show me everything"
- ❌ **Avoid**: "what happened at 3pm"

### Search Techniques
- **Objects**: "person", "car", "bicycle", "phone"
- **Colors**: "red car", "blue shirt", "green bag"
- **Actions**: "walking", "running", "sitting", "talking"
- **Locations**: "near door", "on street", "in room"
- **Combinations**: "person with dog walking"

## ⚡ Performance Tips

### For Better Processing Speed
- Use smaller video files for testing
- Enable GPU acceleration if available
- Process videos during off-peak hours
- Use appropriate video resolution (720p-1080p optimal)

### For Better Search Results
- Use descriptive project names
- Process videos with good lighting
- Ensure objects are clearly visible
- Use diverse vocabulary in searches

## 🛠️ Troubleshooting Quick Fixes

### Video Won't Upload
```bash
# Check file size (max 200MB default)
ls -lh your-video.mp4

# Check format
file your-video.mp4
```

### Processing Stuck
```bash
# Check worker status
make status-fullstack

# Restart worker
make restart-worker
```

### Search Returns No Results
1. Ensure processing completed successfully
2. Try simpler search terms
3. Check project selection
4. Verify video contains detectable objects

### Frontend Not Loading
```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:8501

# Restart frontend
make restart-frontend
```

## 🎓 Learning Path

### Beginner (Day 1)
- ✅ Complete this quick start
- ✅ Upload and process a test video
- ✅ Try basic searches
- ✅ Explore the analytics page

### Intermediate (Week 1)
- 📖 Read [API Reference](API_REFERENCE.md)
- 🔧 Try API calls with curl/Postman
- 📊 Understand analytics metrics
- 🎯 Test different video types

### Advanced (Month 1)
- 🏗️ Study [System Architecture](ARCHITECTURE.md)
- 🤖 Learn about [AI Models](AI_MODELS.md)
- 🚀 Set up production deployment
- 🔧 Customize and extend the system

## 📞 Getting Help

### Quick References
- **API Docs**: http://localhost:8000/docs (when running)
- **System Status**: http://localhost:8501 → System Status page
- **Logs**: Check console output where you ran `make fullstack`

### Common Commands
```bash
# System status
make status-fullstack

# Restart everything
make restart

# View logs
make logs

# Stop everything
make stop
```

## 🎉 What's Next?

Now that you have the system running:

1. **Explore Advanced Features**
   - Multi-project management
   - Batch video processing
   - Custom search filters
   - Analytics export

2. **Learn the Technology**
   - [AI Models Guide](AI_MODELS.md)
   - [Architecture Overview](ARCHITECTURE.md)
   - [Development Guide](DEVELOPMENT.md)

3. **Scale Your Usage**
   - [Deployment Guide](DEPLOYMENT.md)
   - [Performance Tuning](PERFORMANCE.md)
   - [Security Best Practices](SECURITY.md)

4. **Get Involved**
   - [Contributing Guidelines](CONTRIBUTING.md)
   - Report bugs and request features
   - Share your use cases

---

**🚀 Happy Surveilling! You're now ready to harness the power of AI-driven video analysis.**
