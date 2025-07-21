# 🔍 Intelligent Surveillance System - Frontend

Modern, responsive Streamlit frontend for the AI-powered surveillance system.

## 🎯 Features

### 📤 Video Processing
- **File Upload**: Support for MP4, AVI, MOV, MKV, WMV formats
- **Real-time Processing**: Monitor job progress with live updates
- **AI Configuration**: Customize object detection, tracking, and captioning
- **Batch Processing**: Handle multiple videos simultaneously

### 🔍 Semantic Search
- **Natural Language Queries**: Search footage using descriptive text
- **Advanced Filters**: Filter by date range, object types, and projects
- **Interactive Results**: Preview frames, view AI captions, and export data
- **Search History**: Track and repeat previous searches

### 📊 Analytics Dashboard
- **Real-time Metrics**: Videos processed, objects detected, system performance
- **Interactive Charts**: Object detection breakdown, confidence distribution
- **Timeline Analysis**: Activity patterns over time
- **AI Insights**: Automatically generated observations and alerts

### ⚙️ System Status
- **Health Monitoring**: Track all system components and services
- **Resource Usage**: CPU, memory, and disk utilization
- **Service Status**: API health, Redis, Celery workers
- **Background Jobs**: Monitor active processing tasks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FastAPI backend running on `http://localhost:5000`
- Redis server for job queuing
- Celery workers for background processing

### Installation

1. **Install Dependencies**
   ```bash
   cd streamlit
   pip install -r requirements.txt
   ```

2. **Start Frontend**
   ```bash
   # Using the startup script
   ./start_streamlit.sh
   
   # Or directly with Streamlit
   streamlit run app.py
   ```

3. **Access Interface**
   Open your browser to `http://localhost:8501`

## 📁 Directory Structure

```
streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── start_streamlit.sh    # Startup script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── pages/                # Multi-page application
│   ├── 1_📤_Video_Processing.py
│   ├── 2_🔍_Semantic_Search.py
│   ├── 3_📊_Analytics.py
│   └── 4_⚙️_System_Status.py
└── utils/
    └── api_client.py     # Backend API communication
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.28+
- **Visualization**: Plotly, Pandas
- **HTTP Client**: Requests
- **Image Processing**: Pillow, OpenCV
- **Data Analysis**: NumPy, Pandas

## 🎨 UI Features

### Design Elements
- **Modern Interface**: Clean, professional design with custom CSS
- **Responsive Layout**: Adapts to different screen sizes
- **Interactive Components**: Real-time updates and live monitoring
- **Visual Feedback**: Progress bars, status indicators, and alerts

### Navigation
- **Multi-page App**: Organized by functionality
- **Sidebar Navigation**: Easy access to all features
- **Breadcrumbs**: Clear indication of current location
- **Quick Actions**: One-click access to common tasks

## 🔧 Configuration

### Streamlit Settings
Configure in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
port = 8501
address = "0.0.0.0"
```

### API Connection
Update backend URL in `utils/api_client.py`:

```python
class SurveillanceAPIClient:
    def __init__(self, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
```

## 📊 Usage Examples

### Video Processing
1. Navigate to **Video Processing** page
2. Upload a surveillance video file
3. Configure processing options (detection, tracking, captions)
4. Click "Process Video" and monitor progress
5. View results and statistics

### Semantic Search
1. Go to **Semantic Search** page
2. Enter a natural language query (e.g., "person walking with dog")
3. Apply optional filters (date range, object types)
4. Browse results with frame previews and AI captions
5. Export interesting findings

### Analytics
1. Visit **Analytics** page
2. Select time range for analysis
3. View key metrics and interactive charts
4. Review AI-generated insights and alerts
5. Export analytics data

### System Monitoring
1. Access **System Status** page
2. Monitor service health and resource usage
3. Check background job status
4. Run comprehensive health checks
5. Download system reports

## 🔍 Troubleshooting

### Common Issues

**Frontend won't start:**
- Check Python version (3.8+ required)
- Install dependencies: `pip install -r requirements.txt`
- Verify port 8501 is available

**Backend connection failed:**
- Ensure FastAPI server is running on localhost:5000
- Check firewall settings
- Verify API endpoints in browser

**Upload/processing errors:**
- Check file format (MP4, AVI, MOV, MKV, WMV supported)
- Verify file size limits
- Ensure sufficient disk space

**Search returns no results:**
- Check if videos have been processed with AI captioning
- Try lowering similarity threshold
- Verify vector database is populated

### Performance Tips

- **Large Files**: Use frame skipping for faster processing
- **Memory Usage**: Monitor system resources during processing
- **Search Speed**: Use specific queries for better performance
- **Auto-refresh**: Disable when not needed to reduce load

## 🔄 Development

### Adding New Pages
1. Create new file in `pages/` directory
2. Follow naming convention: `N_📄_Page_Name.py`
3. Import and use `SurveillanceAPIClient`
4. Add navigation links in sidebar

### Custom Styling
- Modify CSS in individual page files
- Update theme colors in `config.toml`
- Add custom components as needed

### API Integration
- Extend `api_client.py` for new endpoints
- Add error handling for new API calls
- Update type hints and documentation

## 📝 Notes

- **Real-time Updates**: Uses Streamlit's auto-refresh capabilities
- **State Management**: Session state preserves data across interactions
- **Error Handling**: Graceful degradation with informative messages
- **Responsive Design**: Mobile-friendly interface
- **Accessibility**: Screen reader compatible with proper ARIA labels

## 🚀 Production Deployment

For production deployment:

1. **Environment Variables**
   ```bash
   export STREAMLIT_SERVER_PORT=8501
   export API_BASE_URL=https://your-api-domain.com/api
   ```

2. **Docker Deployment**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

3. **Security Considerations**
   - Use HTTPS in production
   - Implement authentication if needed
   - Configure rate limiting
   - Set up monitoring and logging

---

Built with ❤️ using Streamlit and modern web technologies.
