import streamlit as st
import sys
import os
import time
import uuid
import re

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Simplified page config
st.set_page_config(
    page_title="Video Processing",
    page_icon="📤",
    layout="wide"
)

# Simplified CSS
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
    .status-success {
        background: #e6ffe6;
        color: #2e7d32;
    }
    .status-error {
        background: #ffe6e6;
        color: #d32f2f;
    }
    .status-processing {
        background: #fff3e0;
        color: #ef6c00;
    }
    .upload-zone {
        border: 2px dashed #4a6ee0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        background: #f9f9f9;
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

# Initialize API client and session state
if API_AVAILABLE:
    api_client = SurveillanceAPIClient()
else:
    api_client = None

if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = str(uuid.uuid4())
if "current_project_name" not in st.session_state:
    st.session_state.current_project_name = f"surveillance-{st.session_state.current_project_id[:8]}"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "processing_jobs" not in st.session_state:
    st.session_state.processing_jobs = {}
if "mode" not in st.session_state:
    st.session_state.mode = None

# Helper functions
def sanitize_project_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', name.replace(' ', '-'))
    return sanitized.lower()

def get_job_status_info(api_client, job_id):
    if not api_client:
        return None, "UNKNOWN", False
    try:
        status_result = api_client.get_job_status(job_id)
        if not status_result.get("success", False):
            return None, "UNKNOWN", False
        status_data = status_result.get("status", {})
        actual_status = status_data.get("data", status_data)
        current_status = actual_status.get("status", "UNKNOWN")
        job_ready = actual_status.get("ready", False)
        return actual_status, current_status, job_ready
    except Exception as e:
        return None, f"ERROR: {str(e)}", False

def is_job_active(current_status):
    completed_states = ["SUCCESS", "COMPLETED", "FAILURE", "FAILED", "ERROR"]
    return current_status not in completed_states

# Main header
st.markdown("""
<div class="header">
    <h1>Video Processing</h1>
</div>
""", unsafe_allow_html=True)

# System status check
backend_available = False
if API_AVAILABLE and api_client:
    try:
        health_check = api_client.health_check()
        backend_available = health_check.get("success", False)
    except:
        backend_available = False

# Mode selection
if not st.session_state.mode:
    st.markdown("### Choose Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Demo Mode", use_container_width=True):
            st.session_state.mode = "demo"
            st.rerun()
    with col2:
        if st.button("Real Processing", use_container_width=True, disabled=not backend_available):
            st.session_state.mode = "real"
            st.rerun()
    if not backend_available:
        st.markdown('<div class="status status-error">🔴 Backend Offline: Real Processing unavailable</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status status-{"success" if st.session_state.mode == "real" and backend_available else "processing"}">{"🟢 Real Processing" if st.session_state.mode == "real" else "🟡 Demo Mode"}</div>', unsafe_allow_html=True)
    if st.button("Back to Mode Selection", use_container_width=True):
        st.session_state.mode = None
        for key in ['show_demo_upload', 'show_demo_processing', 'show_demo_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Demo Mode
if st.session_state.mode == "demo":
    st.markdown("### Demo Mode")
    st.markdown("Try video processing without backend setup.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Upload Demo", use_container_width=True):
            st.session_state.show_demo_upload = True
    with col2:
        if st.button("Processing Demo", use_container_width=True):
            st.session_state.show_demo_processing = True
    with col3:
        if st.button("Results Demo", use_container_width=True):
            st.session_state.show_demo_results = True

    if st.session_state.get('show_demo_upload', False):
        st.markdown("#### Demo Upload")
        st.markdown("""
        <div class="card">
            <h3>Sample Video</h3>
            <p>Name: lobby_surveillance_001.mp4<br>Size: 45.2 MB<br>Duration: 2:15 min<br>Project: building-security-demo</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="status status-success">✅ Video uploaded!</div>', unsafe_allow_html=True)

    if st.session_state.get('show_demo_processing', False):
        st.markdown("#### Demo Processing")
        st.markdown('<div class="card"><h3>Processing Stages</h3></div>', unsafe_allow_html=True)
        for stage, status in [
            ("Frame Extraction", "✅ Complete"),
            ("Object Detection", "🔄 63% complete"),
            ("Object Tracking", "⏳ Queued"),
            ("Scene Captioning", "⏳ Queued"),
            ("Search Indexing", "⏳ Queued")
        ]:
            st.markdown(f'<div class="card">{stage}: {status}</div>', unsafe_allow_html=True)
        st.progress(63, text="Progress: 63%")
        st.markdown('<div class="status status-processing">Objects: person (45), car (12), bicycle (3)</div>', unsafe_allow_html=True)

    if st.session_state.get('show_demo_results', False):
        st.markdown("#### Demo Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Results</h3>
                <p>Time: 4m 23s<br>Frames: 1,350<br>Detections: 234<br>Tracks: 47</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="card"><h3>Objects</h3><p>People: 156<br>Vehicles: 45<br>Bicycles: 12</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Scenes</h3>
                <p>1. Person walking to entrance<br>2. People at reception<br>3. Delivery worker</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="status status-success">🔍 Ready for search</div>', unsafe_allow_html=True)

    if any(st.session_state.get(k, False) for k in ['show_demo_upload', 'show_demo_processing', 'show_demo_results']):
        if st.button("Clear Demo", use_container_width=True):
            for key in ['show_demo_upload', 'show_demo_processing', 'show_demo_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Real Processing
if st.session_state.mode == "real" and backend_available:
    st.markdown("### Real Processing")
    st.markdown("#### Project Setup")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_project_name = st.text_input("Project Name", value=st.session_state.current_project_name)
        if new_project_name != st.session_state.current_project_name:
            st.session_state.current_project_name = new_project_name
            st.session_state.current_project_id = sanitize_project_name(new_project_name) or str(uuid.uuid4())[:8]
    with col2:
        if st.button("New Project", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_project_id = new_id
            st.session_state.current_project_name = f"surveillance-{new_id[:8]}"
            st.session_state.uploaded_files = {}
            st.session_state.processing_jobs = {}
            st.rerun()
    st.markdown(f'<div class="card">Project ID: {st.session_state.current_project_id}</div>', unsafe_allow_html=True)

    st.markdown("#### Video Upload")
    uploaded_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov', 'mkv', 'wmv'])
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>File Details</h3>
                <p>Name: {uploaded_file.name}<br>Size: {file_size_mb:.1f} MB<br>Type: {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            if file_size_mb <= 500:
                st.video(uploaded_file)
            else:
                st.markdown('<div class="status status-error">File too large (max 500MB)</div>', unsafe_allow_html=True)
        with col2:
            if file_size_mb <= 500 and st.button("Upload", use_container_width=True):
                with st.spinner("Uploading..."):
                    uploaded_file.seek(0)
                    result = api_client.upload_file(
                        project_id=st.session_state.current_project_id,
                        file_content=uploaded_file.read(),
                        filename=uploaded_file.name
                    )
                    if result.get("success", False):
                        file_id = result.get("data", {}).get("file_id", uploaded_file.name)
                        st.session_state.uploaded_files[file_id] = {
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                            "upload_time": time.time(),
                            "project_id": st.session_state.current_project_id,
                            "project_name": st.session_state.current_project_name
                        }
                        st.markdown(f'<div class="status status-success">✅ Upload successful! File ID: {file_id[:12]}</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="status status-error">❌ Upload failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_files:
        st.markdown("#### Video Processing")
        for file_id, file_info in st.session_state.uploaded_files.items():
            st.markdown(f"""
            <div class="card">
                <h3>{file_info['name']}</h3>
                <p>Project: {file_info.get('project_name', 'Unknown')}<br>Size: {file_info['size'] / (1024*1024):.1f} MB<br>Uploaded: {time.ctime(file_info['upload_time'])}</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown("**AI Features**")
                detect_objects = st.checkbox("Object Detection", value=True, key=f"detect_{file_id}")
                track_objects = st.checkbox("Object Tracking", value=True, key=f"track_{file_id}")
                generate_captions = st.checkbox("Scene Captioning", value=True, key=f"caption_{file_id}")
            with col2:
                st.markdown("**Settings**")
                sample_rate = st.slider("Sample Rate (fps)", 0.1, 2.0, 1.0, 0.1, key=f"sample_{file_id}")
                detection_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, key=f"conf_{file_id}")
            with col3:
                st.markdown("**Action**")
                if st.button("Process", key=f"process_{file_id}", use_container_width=True):
                    processing_params = {
                        "sample_rate": sample_rate,
                        "detection_threshold": detection_threshold,
                        "enable_tracking": track_objects,
                        "enable_captioning": generate_captions
                    }
                    with st.spinner("Starting..."):
                        result = api_client.process_video(
                            project_id=file_info.get('project_id', st.session_state.current_project_id),
                            file_id=file_id,
                            **processing_params
                        )
                        if result.get("success", False):
                            job_id = result.get("data", {}).get("job_id", str(uuid.uuid4()))
                            st.session_state.processing_jobs[job_id] = {
                                "file_id": file_id,
                                "file_name": file_info['name'],
                                "project_id": file_info.get('project_id', st.session_state.current_project_id),
                                "project_name": file_info.get('project_name', 'Unknown'),
                                "start_time": time.time(),
                                "status": "STARTED",
                                "options": processing_params
                            }
                            st.markdown(f'<div class="status status-success">✅ Processing started! Job: {job_id[:12]}</div>', unsafe_allow_html=True)
                            st.rerun()
                        else:
                            st.markdown(f'<div class="status status-error">❌ Failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

    if st.session_state.processing_jobs:
        st.markdown("#### Active Jobs")
        col1, col2 = st.columns([3, 1])
        with col1:
            auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        for job_id, job_info in st.session_state.processing_jobs.items():
            st.markdown(f"""
            <div class="card">
                <h3>{job_info['file_name']} Processing</h3>
                <p>Job ID: {job_id[:12]}<br>Project: {job_info.get('project_name', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
            if backend_available:
                actual_status, current_status, job_ready = get_job_status_info(api_client, job_id)
                if actual_status is not None and current_status != job_info.get("status"):
                    st.session_state.processing_jobs[job_id]["status"] = current_status
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    status_icons = {
                        "PENDING": "🟡 Queued",
                        "STARTED": "🔵 Processing",
                        "PROGRESS": "🔵 Processing",
                        "SUCCESS": "🟢 Complete",
                        "COMPLETED": "🟢 Complete",
                        "FAILURE": "🔴 Failed",
                        "FAILED": "🔴 Failed",
                        "ERROR": "🔴 Error"
                    }
                    status_display = status_icons.get(current_status, f"⚪ {current_status}")
                    st.markdown(f'<div class="status status-{"success" if current_status in ["SUCCESS", "COMPLETED"] else "error" if current_status in ["FAILURE", "FAILED", "ERROR"] else "processing"}">{status_display}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Started:** {time.ctime(job_info['start_time'])}")
                    if current_status == "PENDING" and time.time() - job_info['start_time'] > 30:
                        st.markdown('<div class="status status-error">⚠️ Long queue time</div>', unsafe_allow_html=True)
                with col2:
                    elapsed_time = time.time() - job_info['start_time']
                    st.markdown(f"**Elapsed:** {elapsed_time:.0f}s")
                    if current_status in ["SUCCESS", "COMPLETED", "FAILURE", "FAILED", "ERROR"]:
                        st.markdown("**Finished:** ✅")
                with col3:
                    if st.button("Check Status", key=f"check_{job_id}", use_container_width=True):
                        with st.spinner("Checking status..."):
                            status_result = api_client.get_job_status(job_id)
                            if status_result.get("success"):
                                st.markdown("#### Job Status Details")
                                st.markdown(f"""
                                <div class="status-details">
                                    <p><strong>Job ID:</strong> {job_id}</p>
                                    <p><strong>Status:</strong> {status_result.get('status', {}).get('status', 'N/A')}</p>
                                    <p><strong>Ready:</strong> {status_result.get('status', {}).get('ready', 'N/A')}</p>
                                    <p><strong>Progress:</strong> {status_result.get('status', {}).get('progress', 'N/A')}</p>
                                    <p><strong>Details:</strong> {status_result.get('status', {}).get('details', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="status status-error">❌ Status check failed: {status_result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
                    if st.button("Remove", key=f"remove_{job_id}", use_container_width=True):
                        del st.session_state.processing_jobs[job_id]
                        st.rerun()
            else:
                st.markdown('<div class="status status-error">🔌 Backend required for job status</div>', unsafe_allow_html=True)

        if auto_refresh and backend_available:
            active_jobs = [job_id for job_id, job_info in st.session_state.processing_jobs.items()
                           if is_job_active(job_info.get("status"))]
            if active_jobs:
                st.markdown(f'<div class="status status-processing">🔄 Auto-refreshing... {len(active_jobs)} job(s) active</div>', unsafe_allow_html=True)
                time.sleep(10)
                st.rerun()
            else:
                st.markdown('<div class="status status-success">✅ All jobs completed!</div>', unsafe_allow_html=True)

    if st.session_state.processing_jobs and backend_available:
        with st.expander("Debug Info"):
            for job_id, job_info in st.session_state.processing_jobs.items():
                st.markdown(f"""
                <div class="card">
                    <h3>Job: {job_id[:12]}</h3>
                    <p>File: {job_info['file_name']}<br>Project: {job_info.get('project_name', 'Unknown')}<br>Started: {time.ctime(job_info['start_time'])}<br>Elapsed: {time.time() - job_info['start_time']:.0f}s</p>
                </div>
                """, unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Check Status", key=f"debug_status_{job_id}", use_container_width=True):
                        with st.spinner("Checking..."):
                            status_result = api_client.get_job_status(job_id)
                            st.markdown("**Raw Response:**")
                            st.json(status_result)
                with col2:
                    if st.button(f"Force Remove", key=f"debug_remove_{job_id}", use_container_width=True):
                        del st.session_state.processing_jobs[job_id]
                        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### Video Processing")
    st.markdown(f'<div class="status status-{"success" if backend_available else "error"}">{"🟢 Online" if backend_available else "🔴 Offline"}</div>', unsafe_allow_html=True)
    st.markdown("### Project")
    st.markdown(f"**Name:** {st.session_state.current_project_name}<br>**ID:** {st.session_state.current_project_id}")
    if st.session_state.uploaded_files:
        st.markdown(f"**Files:** {len(st.session_state.uploaded_files)}")
    if st.session_state.processing_jobs:
        active_count = sum(1 for job_id, job_info in st.session_state.processing_jobs.items()
                          if is_job_active(job_info.get("status")))
        completed_count = sum(1 for job_id, job_info in st.session_state.processing_jobs.items()
                             if job_info.get("status") in ["SUCCESS", "COMPLETED"])
        st.markdown(f"**Active Jobs:** {active_count}<br>**Completed:** {completed_count}")
    st.markdown("### Tips")
    st.markdown("- Sample Rate: Higher = more accurate\n- Confidence: Lower = more detections\n- Formats: MP4, AVI, MOV\n- Max Size: 500MB")
    if st.session_state.processing_jobs and st.button("Clear Completed", use_container_width=True):
        completed_jobs = [job_id for job_id, job_info in st.session_state.processing_jobs.items()
                         if job_info.get("status") in ["SUCCESS", "COMPLETED"]]
        for job_id in completed_jobs:
            del st.session_state.processing_jobs[job_id]
        st.rerun()
    if st.button("Clear All", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.processing_jobs = {}
        st.rerun()