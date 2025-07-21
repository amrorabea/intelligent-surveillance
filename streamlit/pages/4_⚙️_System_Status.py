# streamlit/pages/4_⚙️_System_Status.py
import streamlit as st
import sys
import os
import time
import psutil
from datetime import datetime
import requests

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

st.set_page_config(
    page_title="System Status - Surveillance System",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .status-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .status-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-good {
        border-left: 4px solid #28a745;
    }
    .status-warning {
        border-left: 4px solid #ffc107;
    }
    .status-error {
        border-left: 4px solid #dc3545;
    }
    .service-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .system-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = SurveillanceAPIClient()

# Initialize session state
if "system_status" not in st.session_state:
    st.session_state.system_status = {}
if "last_check" not in st.session_state:
    st.session_state.last_check = None

st.title("⚙️ System Status")
st.markdown("Monitor the health and performance of your surveillance system")

# Header
st.markdown("""
<div class="status-header">
    <h3>🖥️ System Health Monitor</h3>
    <p>Real-time monitoring of all surveillance system components</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 10 seconds)", value=True)

with col2:
    if st.button("🔄 Refresh Status", type="primary"):
        st.session_state.system_status = {}
        st.session_state.last_check = None

with col3:
    if st.session_state.last_check:
        st.caption(f"Last check: {st.session_state.last_check.strftime('%H:%M:%S')}")

def check_service_health(service_name, url, timeout=5):
    """Check if a service is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "down", "error": "Connection failed"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_system_metrics():
    """Get system resource metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": memory.used / (1024**3),  # GB
            "memory_total": memory.total / (1024**3),  # GB
            "disk_percent": disk.percent,
            "disk_used": disk.used / (1024**3),  # GB
            "disk_total": disk.total / (1024**3),  # GB
        }
    except Exception as e:
        return {"error": str(e)}

# Load system status
if not st.session_state.system_status or auto_refresh:
    with st.spinner("🔍 Checking system status..."):
        status = {}
        
        # Check API services
        services = {
            "FastAPI Server": "http://localhost:5000/health",
            "Surveillance API": "http://localhost:5000/surveillance/health",
            "Data API": "http://localhost:5000/api/data/health"
        }
        
        for service_name, url in services.items():
            status[service_name] = check_service_health(service_name, url)
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            status["Redis"] = {"status": "healthy", "response_time": 0.001}
        except Exception as e:
            status["Redis"] = {"status": "down", "error": str(e)}
        
        # Check Celery
        try:
            from celery import Celery
            app = Celery('surveillance')
            app.config_from_object('src.helpers.config')
            
            # Check if workers are active
            active_workers = app.control.active()
            if active_workers:
                status["Celery Workers"] = {"status": "healthy", "workers": len(active_workers)}
            else:
                status["Celery Workers"] = {"status": "down", "error": "No active workers"}
        except Exception as e:
            status["Celery Workers"] = {"status": "error", "error": str(e)}
        
        # Get system metrics
        status["System Metrics"] = get_system_metrics()
        
        st.session_state.system_status = status
        st.session_state.last_check = datetime.now()

status = st.session_state.system_status

# Overall system health
if status:
    healthy_count = sum(1 for s in status.values() if isinstance(s, dict) and s.get("status") == "healthy")
    total_services = len([s for s in status.values() if isinstance(s, dict) and "status" in s])
    
    if healthy_count == total_services:
        overall_status = "🟢 All Systems Operational"
        overall_color = "status-good"
    elif healthy_count > total_services * 0.7:
        overall_status = "🟡 Some Issues Detected"
        overall_color = "status-warning"
    else:
        overall_status = "🔴 System Issues Detected"
        overall_color = "status-error"
    
    st.markdown(f"""
    <div class="status-card {overall_color}">
        <h3>{overall_status}</h3>
        <p><strong>Services Online:</strong> {healthy_count}/{total_services}</p>
        <p><strong>Last Check:</strong> {st.session_state.last_check.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# Service status grid
st.markdown("### 🔧 Service Status")

if status:
    # Create service status cards
    service_cols = st.columns(2)
    
    col_index = 0
    for service_name, service_status in status.items():
        if service_name == "System Metrics":
            continue  # Handle separately
        
        with service_cols[col_index % 2]:
            if isinstance(service_status, dict) and "status" in service_status:
                status_value = service_status["status"]
                
                # Determine status styling
                if status_value == "healthy":
                    status_icon = "🟢"
                    status_class = "status-good"
                    status_text = "Healthy"
                elif status_value in ["warning", "timeout"]:
                    status_icon = "🟡"
                    status_class = "status-warning"
                    status_text = "Warning"
                else:
                    status_icon = "🔴"
                    status_class = "status-error"
                    status_text = "Error"
                
                # Additional info
                additional_info = ""
                if "response_time" in service_status:
                    additional_info += f"Response: {service_status['response_time']:.3f}s<br>"
                if "workers" in service_status:
                    additional_info += f"Workers: {service_status['workers']}<br>"
                if "error" in service_status:
                    additional_info += f"Error: {service_status['error']}<br>"
                
                st.markdown(f"""
                <div class="status-card {status_class}">
                    <h4>{status_icon} {service_name}</h4>
                    <p><strong>Status:</strong> {status_text}</p>
                    {additional_info}
                </div>
                """, unsafe_allow_html=True)
            
            col_index += 1

# System metrics
st.markdown("### 💻 System Resources")

if "System Metrics" in status:
    metrics = status["System Metrics"]
    
    if "error" not in metrics:
        # Resource usage metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🖥️ CPU Usage",
                f"{metrics['cpu_percent']:.1f}%",
                delta=None
            )
            # CPU usage progress bar
            cpu_color = "🟢" if metrics['cpu_percent'] < 70 else "🟡" if metrics['cpu_percent'] < 90 else "🔴"
            st.progress(metrics['cpu_percent'] / 100)
            st.caption(f"{cpu_color} CPU Status")
        
        with col2:
            st.metric(
                "💾 Memory Usage",
                f"{metrics['memory_used']:.1f} GB / {metrics['memory_total']:.1f} GB",
                delta=f"{metrics['memory_percent']:.1f}%"
            )
            # Memory usage progress bar
            mem_color = "🟢" if metrics['memory_percent'] < 70 else "🟡" if metrics['memory_percent'] < 90 else "🔴"
            st.progress(metrics['memory_percent'] / 100)
            st.caption(f"{mem_color} Memory Status")
        
        with col3:
            st.metric(
                "💽 Disk Usage",
                f"{metrics['disk_used']:.1f} GB / {metrics['disk_total']:.1f} GB",
                delta=f"{metrics['disk_percent']:.1f}%"
            )
            # Disk usage progress bar
            disk_color = "🟢" if metrics['disk_percent'] < 80 else "🟡" if metrics['disk_percent'] < 95 else "🔴"
            st.progress(metrics['disk_percent'] / 100)
            st.caption(f"{disk_color} Disk Status")
        
        # Resource usage alerts
        alerts = []
        if metrics['cpu_percent'] > 90:
            alerts.append("⚠️ High CPU usage detected")
        if metrics['memory_percent'] > 90:
            alerts.append("⚠️ High memory usage detected")
        if metrics['disk_percent'] > 95:
            alerts.append("⚠️ Low disk space available")
        
        if alerts:
            st.markdown("#### 🚨 Resource Alerts")
            for alert in alerts:
                st.warning(alert)
    else:
        st.error(f"❌ Could not get system metrics: {metrics['error']}")

# API Health Details
st.markdown("### 🔌 API Health Check")

with st.expander("📊 Detailed API Status"):
    if st.button("🔍 Run Comprehensive Health Check"):
        with st.spinner("Running detailed health checks..."):
            # Detailed API checks
            api_checks = {
                "Database Connection": "/api/health/database",
                "Vector DB Status": "/api/health/vectordb", 
                "Model Loading": "/api/health/models",
                "File System": "/api/health/filesystem",
                "Queue Status": "/api/health/queue"
            }
            
            for check_name, endpoint in api_checks.items():
                try:
                    full_url = f"http://localhost:5000{endpoint}"
                    response = requests.get(full_url, timeout=10)
                    
                    if response.status_code == 200:
                        st.success(f"✅ {check_name}: Healthy")
                        # Show response details
                        if response.json():
                            with st.expander(f"Details - {check_name}"):
                                st.json(response.json())
                    else:
                        st.error(f"❌ {check_name}: HTTP {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ {check_name}: {str(e)}")

# Background processes
st.markdown("### 🔄 Background Processes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📋 Active Jobs")
    
    # Get active jobs from API
    try:
        jobs_result = api_client.get_active_jobs()
        if jobs_result.get("success", False):
            active_jobs = jobs_result.get("jobs", [])
            
            if active_jobs:
                for job in active_jobs:
                    job_status = job.get("status", "unknown")
                    job_icon = {
                        "PENDING": "🟡",
                        "STARTED": "🔵",
                        "PROGRESS": "🔵",
                        "SUCCESS": "🟢",
                        "FAILURE": "🔴"
                    }.get(job_status, "⚪")
                    
                    st.markdown(f"{job_icon} **{job.get('name', 'Unknown Job')}** - {job_status}")
                    if job.get("progress"):
                        st.progress(job["progress"] / 100)
            else:
                st.info("No active jobs")
        else:
            st.error("Could not fetch job status")
    except Exception as e:
        st.error(f"Error fetching jobs: {str(e)}")

with col2:
    st.markdown("#### 📊 Queue Statistics")
    
    try:
        queue_stats = api_client.get_queue_stats()
        if queue_stats.get("success", False):
            stats = queue_stats.get("stats", {})
            
            st.metric("📥 Pending Jobs", stats.get("pending", 0))
            st.metric("🔄 Active Jobs", stats.get("active", 0))
            st.metric("✅ Completed Jobs", stats.get("completed", 0))
            st.metric("❌ Failed Jobs", stats.get("failed", 0))
        else:
            st.error("Could not fetch queue statistics")
    except Exception as e:
        st.error(f"Error fetching queue stats: {str(e)}")

# System information
with st.expander("ℹ️ System Information"):
    st.markdown("### 🖥️ System Details")
    
    import platform
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Operating System:** {platform.system()} {platform.release()}
        **Python Version:** {platform.python_version()}
        **Architecture:** {platform.machine()}
        **Processor:** {platform.processor() or "Unknown"}
        """)
    
    with col2:
        # Get GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.markdown(f"""
                **CUDA Available:** ✅ Yes
                **GPU:** {gpu_name}
                **GPU Memory:** {gpu_memory:.1f} GB
                **PyTorch Version:** {torch.__version__}
                """)
            else:
                st.markdown("**CUDA Available:** ❌ No")
        except ImportError:
            st.markdown("**PyTorch:** Not installed")

# Action buttons
st.markdown("### 🔧 System Actions")

action_cols = st.columns(4)

with action_cols[0]:
    if st.button("🔄 Restart Services"):
        st.info("Service restart functionality would be implemented here")

with action_cols[1]:
    if st.button("🧹 Clear Cache"):
        st.info("Cache clearing functionality would be implemented here")

with action_cols[2]:
    if st.button("📊 Generate Report"):
        # Generate system status report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_status": status,
            "overall_health": overall_status
        }
        
        st.download_button(
            label="💾 Download Report",
            data=str(report_data),
            file_name=f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with action_cols[3]:
    if st.button("📧 Alert Settings"):
        st.info("Alert configuration would be implemented here")

# Sidebar info
with st.sidebar:
    st.header("⚙️ System Status")
    st.info("""
    Monitor the health and performance of all surveillance system components.
    
    **Monitored Services:**
    - FastAPI backend
    - Redis message broker
    - Celery workers
    - System resources
    - API endpoints
    """)
    
    # Quick stats
    if status:
        healthy_services = sum(1 for s in status.values() if isinstance(s, dict) and s.get("status") == "healthy")
        total_services = len([s for s in status.values() if isinstance(s, dict) and "status" in s])
        
        st.metric("🟢 Healthy Services", f"{healthy_services}/{total_services}")
        
        if "System Metrics" in status and "error" not in status["System Metrics"]:
            metrics = status["System Metrics"]
            st.metric("💾 Memory Usage", f"{metrics['memory_percent']:.1f}%")
            st.metric("🖥️ CPU Usage", f"{metrics['cpu_percent']:.1f}%")

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(10)
    st.rerun()
