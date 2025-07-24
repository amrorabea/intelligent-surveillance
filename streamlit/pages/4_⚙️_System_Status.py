import streamlit as st
import sys
import os
import time
import psutil
from datetime import datetime
import requests
import pandas as pd

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="System Status - Surveillance System",
    page_icon="⚙️",
    layout="wide"
)

# Green-based styling consistent with Analytics.py
st.markdown("""
<style>
    .main .block-container {
        font-family: Arial, sans-serif !important;
        color: #333 !important;
    }
    div.header {
        background: #2e7d32 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        color: white !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    div.header h1 {
        font-size: 1.8rem !important;
        margin: 0 !important;
    }
    div.status-card {
        background: #e8f5e9 !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    div.status-good {
        border-left: 4px solid #28a745 !important;
    }
    div.status-warning {
        border-left: 4px solid #ffc107 !important;
    }
    div.status-error {
        border-left: 4px solid #dc3545 !important;
    }
    button[kind="primary"], button[kind="secondary"] {
        width: 100% !important;
        border-radius: 8px !important;
        background: #2e7d32 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stMetric"] {
        background: #e8f5e9 !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        border: 1px solid #c8e6c9 !important;
        margin: 0.5rem 0 !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 4px !important;
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client and session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = SurveillanceAPIClient()
if 'system_status' not in st.session_state:
    st.session_state.system_status = {}
if 'last_check' not in st.session_state:
    st.session_state.last_check = None
if 'mode' not in st.session_state:
    st.session_state.mode = None

# Demo data
def get_demo_data(status_type):
    """Get demo data for system status"""
    services = {
        "healthy": [
            {"name": "FastAPI Backend", "status": "✅ Running", "details": "Port 8000", "memory": "2.3 GB RAM", "info": "HTTP 200"},
            {"name": "Celery Worker", "status": "✅ Active", "details": "2 workers", "memory": "4.1 GB RAM", "info": "Processing queue"},
            {"name": "Redis Server", "status": "✅ Connected", "details": "Port 6379", "memory": "256 MB RAM", "info": "0ms latency"},
            {"name": "AI Models", "status": "✅ Loaded", "details": "YOLOv8 + BLIP", "memory": "3.2 GB VRAM", "info": "Ready"},
            {"name": "Vector Database", "status": "✅ Operational", "details": "ChromaDB", "memory": "1.8 GB", "info": "Index ready"}
        ],
        "warning": [
            {"name": "FastAPI Backend", "status": "✅ Running", "details": "Port 8000", "memory": "2.3 GB RAM", "info": "HTTP 200"},
            {"name": "Celery Worker", "status": "⚠️ High Load", "details": "2 workers", "memory": "6.8 GB RAM", "info": "Queue: 45 jobs"},
            {"name": "Redis Server", "status": "✅ Connected", "details": "Port 6379", "memory": "512 MB RAM", "info": "5ms latency"},
            {"name": "AI Models", "status": "⚠️ Memory Warning", "details": "YOLOv8 + BLIP", "memory": "7.1/8 GB VRAM", "info": "Near capacity"},
            {"name": "Vector Database", "status": "✅ Operational", "details": "ChromaDB", "memory": "2.1 GB", "info": "Index ready"}
        ],
        "maintenance": [
            {"name": "FastAPI Backend", "status": "🔧 Maintenance", "details": "Port 8000", "memory": "1.2 GB RAM", "info": "HTTP 503"},
            {"name": "Celery Worker", "status": "⏸️ Paused", "details": "0 workers", "memory": "0 GB RAM", "info": "Not processing"},
            {"name": "Redis Server", "status": "✅ Connected", "details": "Port 6379", "memory": "128 MB RAM", "info": "1ms latency"},
            {"name": "AI Models", "status": "⏸️ Unloaded", "details": "Updating", "memory": "0 GB VRAM", "info": "Model sync"},
            {"name": "Vector Database", "status": "🔧 Backup", "details": "ChromaDB", "memory": "1.8 GB", "info": "Read-only mode"}
        ]
    }
    metrics = {
        "cpu_percent": 34.0,
        "memory_percent": 51.3,
        "memory_used": 8.2,
        "memory_total": 16.0,
        "disk_percent": 31.2,
        "disk_used": 156.0,
        "disk_total": 500.0
    }
    perf_data = {
        "Component": ["Video Processing", "Object Detection", "Image Captioning", "Semantic Search", "Database Queries"],
        "Avg Response Time": ["3.2 min", "45 ms", "120 ms", "23 ms", "8 ms"],
        "Throughput": ["18 videos/hour", "25 FPS", "8 FPS", "2,500 queries/min", "15,000 ops/sec"],
        "Success Rate": ["99.2%", "98.7%", "97.1%", "99.8%", "99.9%"]
    }
    log_entries = [
        "✅ 14:23:15 - Video processing completed: security_cam_001.mp4",
        "🔍 14:22:48 - Search query executed: 'person with backpack' (23 results)",
        "📊 14:21:32 - Analytics refresh completed (1,250 new detections)",
        "⚡ 14:20:15 - Celery worker scaled: 2 → 3 workers",
        "🎬 14:19:03 - Video upload started: parking_lot_cam.mp4"
    ]
    config_data = {
        "backend": {
            "API Host": "localhost:8000",
            "Workers": "3 active",
            "Redis": "localhost:6379",
            "Upload Limit": "500 MB",
            "Processing Timeout": "30 min"
        },
        "ai_models": {
            "YOLOv8": "Nano (6.2 MB)",
            "BLIP": "Base model (990 MB)",
            "Confidence Threshold": "0.5",
            "Batch Size": "8 frames",
            "Device": "CUDA (GPU 0)"
        },
        "storage": {
            "Upload Directory": "/app/uploads",
            "Vector DB": "/app/vector_db",
            "Model Cache": "/app/models",
            "Log Files": "/app/logs",
            "Backup Location": "/backup"
        }
    }
    return {
        "services": services[status_type],
        "metrics": metrics,
        "performance": perf_data,
        "logs": log_entries,
        "config": config_data
    }

# Shared display functions
def display_service_status(services, title="🔧 Service Status"):
    """Display service status cards"""
    st.markdown(f"#### {title}")
    for service in services:
        status_value = service["status"]
        css_class = "status-good" if "✅" in status_value else "status-warning" if "⚠️" in status_value or "🔧" in status_value or "⏸️" in status_value else "status-error"
        st.markdown(f"""
        <div class="status-card {css_class}">
            <h5>{service['name']}</h5>
            <p><strong>Status:</strong> {service['status']} | <strong>Details:</strong> {service['details']} | <strong>Memory:</strong> {service['memory']} | <strong>Info:</strong> {service['info']}</p>
        </div>
        """, unsafe_allow_html=True)
    healthy_count = sum(1 for s in services if "✅" in s["status"])
    total_services = len(services)
    overall_status = "🟢 All Systems Operational" if healthy_count == total_services else "🟡 Some Issues Detected" if healthy_count > total_services * 0.7 else "🔴 System Issues Detected"
    css_class = "status-good" if healthy_count == total_services else "status-warning" if healthy_count > total_services * 0.7 else "status-error"
    st.markdown(f"""
    <div class="status-card {css_class}">
        <h3>{overall_status}</h3>
        <p><strong>Services Online:</strong> {healthy_count}/{total_services}</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_metrics(metrics, title="💻 System Resources"):
    """Display system resource metrics"""
    st.markdown(f"#### {title}")
    col1, col2, col3 = st.columns(3)
    with col1:
        cpu_percent = metrics.get("cpu_percent", 0)
        st.metric("🖥️ CPU Usage", f"{cpu_percent:.1f}%")
        cpu_color = "🟢" if cpu_percent < 70 else "🟡" if cpu_percent < 90 else "🔴"
        st.progress(cpu_percent / 100)
        st.markdown(f"<div>{cpu_color} CPU Status</div>", unsafe_allow_html=True)
    with col2:
        memory_used = metrics.get("memory_used", 0)
        memory_total = metrics.get("memory_total", 16.0)
        memory_percent = metrics.get("memory_percent", 0)
        st.metric("💾 Memory Usage", f"{memory_used:.1f}/{memory_total:.1f} GB", delta=f"{memory_percent:.1f}%")
        mem_color = "🟢" if memory_percent < 70 else "🟡" if memory_percent < 90 else "🔴"
        st.progress(memory_percent / 100)
        st.markdown(f"<div>{mem_color} Memory Status</div>", unsafe_allow_html=True)
    with col3:
        disk_used = metrics.get("disk_used", 0)
        disk_total = metrics.get("disk_total", 500.0)
        disk_percent = metrics.get("disk_percent", 0)
        st.metric("💽 Disk Usage", f"{disk_used:.1f}/{disk_total:.1f} GB", delta=f"{disk_percent:.1f}%")
        disk_color = "🟢" if disk_percent < 80 else "🟡" if disk_percent < 95 else "🔴"
        st.progress(disk_percent / 100)
        st.markdown(f"<div>{disk_color} Disk Status</div>", unsafe_allow_html=True)
    alerts = []
    if cpu_percent > 90:
        alerts.append("⚠️ High CPU usage detected")
    if memory_percent > 90:
        alerts.append("⚠️ High memory usage detected")
    if disk_percent > 95:
        alerts.append("⚠️ Low disk space available")
    if alerts:
        st.markdown("##### 🚨 Resource Alerts")
        for alert in alerts:
            st.warning(alert)

def display_performance_overview(perf_data, title="⚡ Performance Overview"):
    """Display performance overview table"""
    st.markdown(f"#### {title}")
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True)

def display_activity_log(log_entries, title="📋 Recent Activity Log"):
    """Display recent activity log"""
    st.markdown(f"#### {title}")
    for entry in log_entries:
        st.text(entry)

def display_system_configuration(config_data, title="⚙️ System Configuration"):
    """Display system configuration"""
    st.markdown(f"#### {title}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔧 Backend Configuration:**")
        for key, value in config_data.get("backend", {}).items():
            st.markdown(f"<div>- {key}: {value}</div>", unsafe_allow_html=True)
        st.markdown("**🧠 AI Model Settings:**")
        for key, value in config_data.get("ai_models", {}).items():
            st.markdown(f"<div>- {key}: {value}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("**💾 Storage Configuration:**")
        for key, value in config_data.get("storage", {}).items():
            st.markdown(f"<div>- {key}: {value}</div>", unsafe_allow_html=True)

def display_complete_status(data, status_type="healthy"):
    """Display complete system status"""
    st.markdown(f"#### {'✅ All Systems Operational' if status_type == 'healthy' else '⚠️ System Issues Detected' if status_type == 'warning' else '🔧 Maintenance Mode Active'}")
    if "services" in data:
        display_service_status(data["services"])
    if "metrics" in data:
        display_system_metrics(data["metrics"])
    if "performance" in data:
        display_performance_overview(data["performance"])
    if "logs" in data:
        display_activity_log(data["logs"])
    if "config" in data:
        display_system_configuration(data["config"])

# Main header
st.markdown('<div class="header"><h1>System Status</h1></div>', unsafe_allow_html=True)
st.markdown("Monitor system health and performance of surveillance components.")

# Mode selection
if not st.session_state.mode:
    st.markdown("### Choose Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Demo Mode", type="primary"):
            st.session_state.mode = "demo"
            st.rerun()
    with col2:
        if st.button("⚙️ Real Status", type="primary"):
            st.session_state.mode = "real"
            st.rerun()
else:
    st.markdown(f"### {'📊 Demo Mode' if st.session_state.mode == 'demo' else '⚙️ Real Status'}")
    if st.button("🧹 Back to Mode Selection", type="secondary"):
        st.session_state.mode = None
        for key in ['show_demo_status', 'demo_status']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Demo Mode
if st.session_state.mode == "demo":
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Healthy", type="primary"):
            st.session_state.demo_status = "healthy"
            st.session_state.show_demo_status = True
    with col2:
        if st.button("⚠️ Warning", type="secondary"):
            st.session_state.demo_status = "warning"
            st.session_state.show_demo_status = True
    with col3:
        if st.button("🔧 Maintenance", type="secondary"):
            st.session_state.demo_status = "maintenance"
            st.session_state.show_demo_status = True

    if st.session_state.get('show_demo_status', False):
        status_type = st.session_state.get('demo_status', 'healthy')
        demo_data = get_demo_data(status_type)
        display_complete_status(demo_data, status_type)
        if st.button("🧹 Clear Demo", type="secondary"):
            for key in ['show_demo_status', 'demo_status']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Real Status
if st.session_state.mode == "real":
    col1, col2 = st.columns([2, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 10 seconds)", value=True)
    with col2:
        if st.button("🔄 Refresh Status", type="primary"):
            st.session_state.system_status = {}
            st.session_state.last_check = None
            st.rerun()

    if st.session_state.last_check:
        st.caption(f"Last check: {st.session_state.last_check.strftime('%H:%M:%S')}")

    if not st.session_state.system_status or auto_refresh:
        with st.spinner("🔍 Checking system status..."):
            try:
                # Check API service health with correct endpoints
                status = {}
                services = {
                    "FastAPI Server": "http://localhost:5000",  # Base server check
                    "Surveillance API": "http://localhost:5000/api/surveillance/health",  # Correct surveillance endpoint
                    "Analytics API": "http://localhost:5000/api/surveillance/analytics/light"  # Analytics endpoint
                }
                for service_name, url in services.items():
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            status[service_name] = {"status": "healthy", "response_time": response.elapsed.total_seconds()}
                        else:
                            status[service_name] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                    except requests.exceptions.ConnectionError:
                        # For base server, try alternative endpoints
                        if service_name == "FastAPI Server":
                            try:
                                # Try the API base endpoint
                                alt_response = requests.get("http://localhost:5000/api", timeout=5)
                                if alt_response.status_code in [200, 404]:  # 404 is ok, means server is running
                                    status[service_name] = {"status": "healthy", "response_time": alt_response.elapsed.total_seconds()}
                                else:
                                    status[service_name] = {"status": "down", "error": "Connection failed"}
                            except Exception:
                                status[service_name] = {"status": "down", "error": "Connection failed"}
                        else:
                            status[service_name] = {"status": "down", "error": "Connection failed"}
                    except requests.exceptions.Timeout:
                        status[service_name] = {"status": "timeout", "error": "Request timeout"}
                    except Exception as e:
                        status[service_name] = {"status": "error", "error": str(e)}
                try:
                    import redis
                    r = redis.Redis(host='localhost', port=6379, db=0)
                    r.ping()
                    status["Redis"] = {"status": "healthy", "response_time": 0.001}
                except Exception as e:
                    status["Redis"] = {"status": "down", "error": str(e)}
                # Check Celery workers with better error handling
                try:
                    # Try multiple approaches to check Celery workers
                    celery_status = {"status": "unknown", "error": "Could not detect Celery"}
                    
                    # Method 1: Try to import and check Celery
                    try:
                        from celery import Celery
                        
                        # Try different config approaches
                        app = None
                        config_attempts = [
                            lambda: Celery('surveillance'),  # Basic instance
                            lambda: Celery('surveillance', broker='redis://localhost:6379/0'),  # With broker
                        ]
                        
                        for attempt in config_attempts:
                            try:
                                app = attempt()
                                # Try to get worker stats
                                inspect = app.control.inspect()
                                active_workers = inspect.active()
                                if active_workers:
                                    worker_count = len(active_workers)
                                    celery_status = {"status": "healthy", "workers": worker_count}
                                    break
                                else:
                                    celery_status = {"status": "warning", "error": "No active workers found"}
                            except Exception:
                                continue
                                
                    except ImportError:
                        celery_status = {"status": "down", "error": "Celery not installed"}
                    
                    # Method 2: If Celery import/config failed, try Redis-based detection
                    if celery_status["status"] == "unknown":
                        try:
                            import redis
                            r = redis.Redis(host='localhost', port=6379, db=0)
                            
                            # Check for Celery-related keys in Redis
                            celery_keys = r.keys('celery*') + r.keys('*worker*')
                            if celery_keys:
                                celery_status = {"status": "healthy", "workers": "Redis-detected"}
                            else:
                                celery_status = {"status": "down", "error": "No Celery activity in Redis"}
                        except Exception:
                            pass
                    
                    # Method 3: Process-based detection as fallback
                    if celery_status["status"] == "unknown":
                        try:
                            import subprocess
                            # Look for celery processes
                            result = subprocess.run(['pgrep', '-f', 'celery'], 
                                                  capture_output=True, text=True, timeout=5)
                            if result.returncode == 0 and result.stdout.strip():
                                process_count = len(result.stdout.strip().split('\n'))
                                celery_status = {"status": "healthy", "workers": f"{process_count} processes"}
                            else:
                                celery_status = {"status": "down", "error": "No celery processes found"}
                        except Exception:
                            celery_status = {"status": "down", "error": "Could not detect workers"}
                    
                    status["Celery Workers"] = celery_status
                    
                except Exception as e:
                    status["Celery Workers"] = {"status": "error", "error": f"Detection failed: {str(e)}"}
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    status["System Metrics"] = {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used": memory.used / (1024**3),
                        "memory_total": memory.total / (1024**3),
                        "disk_percent": disk.percent,
                        "disk_used": disk.used / (1024**3),
                        "disk_total": disk.total / (1024**3)
                    }
                except Exception as e:
                    status["System Metrics"] = {"error": str(e)}
                # Check for active jobs through API
                try:
                    # Try to get job status through health endpoint instead
                    health_result = st.session_state.api_client.health_check()
                    if health_result.get("success", False):
                        status["Active Jobs"] = {"status": "healthy", "info": "Backend responsive"}
                    else:
                        status["Active Jobs"] = {"status": "warning", "error": "Backend not responding"}
                except Exception as e:
                    status["Active Jobs"] = {"status": "error", "error": f"API check failed: {str(e)}"}
                
                # Check queue statistics through Redis if available
                try:
                    if "Redis" in status and status["Redis"]["status"] == "healthy":
                        import redis
                        r = redis.Redis(host='localhost', port=6379, db=0)
                        
                        # Get Redis info
                        redis_info = r.info()
                        queue_info = {
                            "connected_clients": redis_info.get('connected_clients', 0),
                            "used_memory_human": redis_info.get('used_memory_human', 'N/A'),
                            "total_commands_processed": redis_info.get('total_commands_processed', 0)
                        }
                        status["Queue Stats"] = queue_info
                    else:
                        status["Queue Stats"] = {"error": "Redis not available"}
                except Exception as e:
                    status["Queue Stats"] = {"error": f"Queue check failed: {str(e)}"}
                st.session_state.system_status = status
                st.session_state.last_check = datetime.now()
                st.success("✅ Status check completed!")
            except Exception as e:
                st.error(f"❌ Error checking status: {str(e)}")
                if st.session_state.system_status:
                    st.info("📊 Showing cached status")
                else:
                    st.warning("⚠️ No status data available. Try Demo Mode.")
                    st.stop()

    status = st.session_state.system_status
    real_data = {"services": [], "metrics": {}, "performance": {}, "logs": [], "config": {}}
    
    for service_name, service_status in status.items():
        if service_name == "System Metrics":
            real_data["metrics"] = service_status
        elif service_name == "Active Jobs":
            if isinstance(service_status, list):
                real_data["logs"] = [f"{job.get('status', '⚪')} {datetime.now().strftime('%H:%M:%S')} - {job.get('name', 'Unknown Job')} ({job.get('status', 'unknown')})" for job in service_status]
            elif isinstance(service_status, dict) and "error" not in service_status:
                real_data["logs"] = [f"✅ {datetime.now().strftime('%H:%M:%S')} - Backend API health check passed"]
        elif service_name == "Queue Stats":
            if isinstance(service_status, dict) and "error" not in service_status:
                # Handle Redis info display
                connected_clients = service_status.get('connected_clients', 0)
                memory_usage = service_status.get('used_memory_human', 'N/A')
                total_commands = service_status.get('total_commands_processed', 0)
                
                real_data["performance"] = {
                    "Component": ["Redis Queue", "Redis Memory", "Redis Commands"],
                    "Current Value": [f"{connected_clients} clients", memory_usage, f"{total_commands:,} total"],
                    "Status": ["Connected", "Running", "Processing"],
                    "Info": ["Active connections", "Memory usage", "Commands processed"]
                }
        elif isinstance(service_status, dict) and "status" in service_status:
            status_text = "✅ Running" if service_status["status"] == "healthy" else "⚠️ Warning" if service_status["status"] in ["warning", "timeout"] else "🔴 Error"
            details = f"Response: {service_status.get('response_time', 0):.3f}s" if "response_time" in service_status else service_status.get("error", "Unknown")
            memory = f"{service_status.get('workers', 0)} workers" if "workers" in service_status else "N/A"
            info_text = service_status.get("error", "Operational") if service_status["status"] != "healthy" else "Operational"
            real_data["services"].append({
                "name": service_name,
                "status": status_text,
                "details": details,
                "memory": memory,
                "info": info_text
            })
    
    real_data["config"] = {
        "backend": {
            "API Host": "localhost:8000",
            "Workers": "N/A",
            "Redis": "localhost:6379"
        },
        "ai_models": {
            "YOLOv8": "N/A",
            "BLIP": "N/A"
        },
        "storage": {
            "Upload Directory": "/app/uploads",
            "Vector DB": "/app/vector_db"
        }
    }
    
    display_complete_status(real_data)
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()
