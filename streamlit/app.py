import streamlit as st
import sys
import os
from datetime import datetime
import time
import json

# Add src to path for importing controllers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import API client
try:
    from utils.api_client import SurveillanceAPIClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Simplified page config
st.set_page_config(
    page_title="Surveillance System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto"
)

# Simplified CSS for better readability
st.markdown("""
<style>
    body {
        font-family: Arial, sans-serif;
        color: #333;
    }
    .header {
        background: #4a6ee0;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .header h1 {
        font-size: 1.8rem;
        margin: 0;
    }
    .card {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    .card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    .status {
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .status-online {
        background: #e6ffe6;
        color: #2e7d32;
    }
    .status-offline {
        background: #ffe6e6;
        color: #d32f2f;
    }
    .status-demo {
        background: #fff3e0;
        color: #ef6c00;
    }
    .stButton > button {
        background: #4a6ee0;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .status-details {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_status' not in st.session_state:
    st.session_state.system_status = 'checking'
if 'last_status_check' not in st.session_state:
    st.session_state.last_status_check = 0
if 'status_details' not in st.session_state:
    st.session_state.status_details = None

# Check system status and return full response
def check_system_status():
    if not API_AVAILABLE:
        return False, {"status": "API client not available"}
    try:
        import requests
        response = requests.get("http://localhost:5000/api/surveillance/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"status": f"Failed with status code {response.status_code}"}
    except Exception as e:
        return False, {"status": f"Error: {str(e)}"}

# Periodic status check
current_time = time.time()
if current_time - st.session_state.last_status_check > 30:
    is_online, details = check_system_status()
    st.session_state.system_status = 'online' if is_online else 'offline'
    st.session_state.status_details = details
    st.session_state.last_status_check = current_time

# Main header
st.markdown("""
<div class="header">
    <h1>Surveillance System</h1>
</div>
""", unsafe_allow_html=True)

# Status indicator
status_text = {
    'online': '🟢 System Online',
    'offline': '🔴 Backend Offline',
    'checking': '🟡 Demo Mode'
}
st.markdown(f'<div class="status status-{st.session_state.system_status}">{status_text[st.session_state.system_status]}</div>', unsafe_allow_html=True)

# Core features
st.markdown("### Features")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>Video Processing</h3>
        <p>Upload and analyze surveillance videos with object detection.</p>
    </div>
    <div class="card">
        <h3>Analytics</h3>
        <p>View detection statistics and generate reports.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>Search</h3>
        <p>Search videos using natural language descriptions.</p>
    </div>
    <div class="card">
        <h3>Monitoring</h3>
        <p>Track system health and performance in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

# Demo and production modes
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Demo Mode")
    st.markdown("Try features without setup.")
    if st.button("Explore Demo", use_container_width=True):
        st.success("Use demo buttons on any page!")

with col2:
    st.markdown("### Production Mode")
    if st.session_state.system_status == 'online':
        st.markdown("Backend is ready for full functionality.")
    else:
        st.markdown("Start backend for full features.")
    if st.button("Check Status", use_container_width=True):
        is_online, details = check_system_status()
        st.session_state.system_status = 'online' if is_online else 'offline'
        st.session_state.status_details = details
        st.session_state.last_status_check = current_time
        # Display detailed status
        if st.session_state.status_details:
            st.markdown("#### System Status Details")
            st.markdown(f"""
            <div class="status-details">
                <p><strong>Status:</strong> {details.get('status', 'N/A')}</p>
                <p><strong>Timestamp:</strong> {details.get('timestamp', 'N/A')}</p>
                <p><strong>Database Connected:</strong> {details.get('database_connected', 'N/A')}</p>
                <p><strong>Vector DB Connected:</strong> {details.get('vector_db_connected', 'N/A')}</p>
                <p><strong>Disk Usage (MB):</strong> {details.get('disk_usage_mb', 'N/A')}</p>
                <p><strong>Memory Usage (MB):</strong> {details.get('memory_usage_mb', 'N/A')}</p>
                <p><strong>Active Jobs:</strong> {details.get('active_jobs', 'N/A')}</p>
                <p><strong>Version:</strong> {details.get('version', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("No detailed status information available.")

# Simplified sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("""
    <div class="card">
        <strong>Video Processing</strong><br>
        Upload and analyze videos
    </div>
    <div class="card">
        <strong>Search</strong><br>
        Find video content
    </div>
    <div class="card">
        <strong>Analytics</strong><br>
        View insights
    </div>
    <div class="card">
        <strong>Status</strong><br>
        System monitoring
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### System Info")
    st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    st.markdown(f"**Status:** {st.session_state.system_status.capitalize()}")
    st.markdown(f"**API:** {'Available' if API_AVAILABLE else 'Not Available'}")
    
    if st.button("Refresh", use_container_width=True):
        st.session_state.last_status_check = 0
        st.rerun()