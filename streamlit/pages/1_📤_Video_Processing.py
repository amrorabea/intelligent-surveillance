# streamlit/pages/1_📤_Video_Processing.py
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

st.set_page_config(
    page_title="📤 Video Processing - Surveillance System",
    page_icon="📤",
    layout="wide"
)

# Modern CSS styling consistent with app.py
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    
    .page-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .page-header p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.9;
    }
    
    /* Card styles */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .demo-card {
        background: linear-gradient(145deg, #f8f9ff 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: #f0f4ff;
    }
    
    /* Status indicators */
    .status-success {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Processing steps */
    .process-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: #f8f9fa;
    }
    
    .process-step.completed {
        background: #dcfce7;
    }
    
    .process-step.active {
        background: #fef3c7;
    }
    
    /* Quick stats */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .page-header {
            padding: 1.5rem;
        }
        
        .page-header h1 {
            font-size: 1.8rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client and session state
if API_AVAILABLE:
    api_client = SurveillanceAPIClient()
else:
    api_client = None

# Initialize session state
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = str(uuid.uuid4())
if "current_project_name" not in st.session_state:
    st.session_state.current_project_name = f"surveillance-{st.session_state.current_project_id[:8]}"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "processing_jobs" not in st.session_state:
    st.session_state.processing_jobs = {}

# Helper functions
def sanitize_project_name(name: str) -> str:
    """Convert project name to valid project ID"""
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', name.replace(' ', '-'))
    return sanitized.lower()

def get_job_status_info(api_client, job_id):
    """Get normalized job status information"""
    if not api_client:
        return None, "UNKNOWN", False
    
    status_result = api_client.get_job_status(job_id)
    
    if not status_result.get("success", False):
        return None, "UNKNOWN", False
    
    status_data = status_result.get("status", {})
    
    # Handle different response formats
    if "data" in status_data:
        actual_status = status_data["data"]
    else:
        actual_status = status_data
    
    current_status = actual_status.get("status", "UNKNOWN")
    job_ready = actual_status.get("ready", False)
    
    return actual_status, current_status, job_ready

def is_job_active(current_status):
    """Check if a job is still active (not completed or failed)"""
    completed_states = ["SUCCESS", "COMPLETED", "FAILURE", "FAILED"]
    return current_status not in completed_states

# Main header
st.markdown("""
<div class="page-header">
    <h1>📤 Video Processing</h1>
    <p>Upload and analyze surveillance footage with AI-powered object detection and scene understanding</p>
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

# Status indicator
if backend_available:
    st.markdown("""
    <div class="status-success">
        🟢 <strong>Backend Online</strong> - Ready for real-time processing
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-error">
        🔴 <strong>Backend Offline</strong> - Demo mode available below
    </div>
    """, unsafe_allow_html=True)

# Demo Section
st.markdown("## � Demo Mode")
st.info("**Experience the full video processing pipeline without backend setup**")

demo_col1, demo_col2, demo_col3 = st.columns(3)

with demo_col1:
    if st.button("🎥 Upload Demo", type="primary", use_container_width=True):
        st.session_state.show_demo_upload = True

with demo_col2:
    if st.button("⚡ Processing Demo", type="secondary", use_container_width=True):
        st.session_state.show_demo_processing = True

with demo_col3:
    if st.button("📊 Results Demo", type="secondary", use_container_width=True):
        st.session_state.show_demo_results = True

# Demo Upload Example
if st.session_state.get('show_demo_upload', False):
    st.markdown("### 🎬 Demo: Video Upload")
    with st.container():
        st.markdown("""
        <div class="demo-card">
            <h4>📁 Sample Video: "lobby_surveillance_001.mp4"</h4>
            <p><strong>📏 Size:</strong> 45.2 MB | <strong>⏱️ Duration:</strong> 2:15 minutes | <strong>📺 Resolution:</strong> 1080p</p>
            <p><strong>📂 Project:</strong> building-security-demo</p>
            <p><strong>📝 Description:</strong> Main entrance surveillance with pedestrians and vehicles</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Video uploaded successfully! Processing can now begin.")

# Demo Processing Example  
if st.session_state.get('show_demo_processing', False):
    st.markdown("### ⚡ Demo: Live Processing Pipeline")
    
    # Processing stages with modern styling
    stages = [
        ("🎬 Frame Extraction", "✅ Complete", "1,350 frames extracted at 1fps"),
        ("🎯 Object Detection", "🔄 Processing", "856/1,350 frames (63% complete)"),
        ("🔗 Object Tracking", "⏳ Queued", "Waiting for detection completion"),
        ("📝 Scene Captioning", "⏳ Queued", "BLIP model ready for descriptions"),
        ("🔍 Search Indexing", "⏳ Queued", "Vector embeddings preparation")
    ]
    
    for stage, status, detail in stages:
        col1, col2 = st.columns([1, 2])
        with col1:
            if "✅" in status:
                st.markdown(f'<div class="process-step completed">{stage}</div>', unsafe_allow_html=True)
            elif "🔄" in status:
                st.markdown(f'<div class="process-step active">{stage}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="process-step">{stage}</div>', unsafe_allow_html=True)
        with col2:
            st.write(f"**{status}** - {detail}")
    
    # Progress visualization
    st.progress(63, text="🚀 Overall Progress: 63% complete")
    
    st.markdown("""
    <div class="status-processing">
        <strong>🎯 Live Detection Results:</strong> person (45), car (12), bicycle (3), backpack (8), handbag (15)
    </div>
    """, unsafe_allow_html=True)

# Demo Results Example
if st.session_state.get('show_demo_results', False):
    st.markdown("### 📊 Demo: Processing Results")
    
    results_col1, results_col2 = st.columns(2)
    
    with results_col1:
        st.markdown("""
        <div class="status-success">
            <h4>✅ Processing Complete!</h4>
            <p><strong>⏱️ Total Time:</strong> 4 minutes 23 seconds</p>
            <p><strong>🎬 Frames:</strong> 1,350 processed</p>
            <p><strong>🎯 Detections:</strong> 234 objects found</p>
            <p><strong>🔗 Tracks:</strong> 47 unique paths</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detection breakdown with modern styling
        st.markdown("**�️ Object Categories:**")
        detection_data = {
            "👤 People": 156,
            "🚗 Vehicles": 45,
            "🚲 Bicycles": 12, 
            "🎒 Bags": 39,
            "🐕 Animals": 3,
            "🚦 Infrastructure": 14
        }
        
        for obj, count in detection_data.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"**{obj}**")
            with col_b:
                st.write(f"{count} detected")
    
    with results_col2:
        st.markdown("**� Generated Scene Descriptions:**")
        example_captions = [
            "Person in business attire walking toward main entrance",
            "Multiple people gathering near reception desk during daytime", 
            "Delivery worker carrying packages approaching front doors",
            "Security personnel conducting routine lobby patrol",
            "Visitors with luggage waiting in seating area"
        ]
        
        for i, caption in enumerate(example_captions, 1):
            st.write(f"{i}. *{caption}*")
        
        st.markdown("""
        <div class="status-success">
            <strong>🔍 Search Ready:</strong> All frames indexed for natural language queries
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Try searching:** 'person with briefcase', 'delivery worker', 'people waiting'")

# Demo control
if any(st.session_state.get(k, False) for k in ['show_demo_upload', 'show_demo_processing', 'show_demo_results']):
    if st.button("🧹 Clear Demo", use_container_width=True):
        for key in ['show_demo_upload', 'show_demo_processing', 'show_demo_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown("---")
# Real Processing Section
st.markdown("## � Real Processing")

if backend_available:
    st.success("**Backend connected** - Ready for real-time video analysis")
else:
    st.warning("**Backend offline** - Start the FastAPI server to enable processing")

# Project Configuration
st.markdown("### 📁 Project Setup")
project_col1, project_col2 = st.columns([3, 1])

with project_col1:
    new_project_name = st.text_input(
        "Project Name",
        value=st.session_state.current_project_name,
        help="Name for your surveillance analysis project"
    )
    
    if new_project_name != st.session_state.current_project_name:
        st.session_state.current_project_name = new_project_name
        st.session_state.current_project_id = sanitize_project_name(new_project_name)
        if not st.session_state.current_project_id:
            st.session_state.current_project_id = str(uuid.uuid4())[:8]

with project_col2:
    if st.button("🔄 New Project", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_project_id = new_id
        st.session_state.current_project_name = f"surveillance-{new_id[:8]}"
        st.session_state.uploaded_files = {}
        st.session_state.processing_jobs = {}
        st.rerun()

st.info(f"📊 **Project ID:** `{st.session_state.current_project_id}`")

# File Upload Section
st.markdown("### 📹 Video Upload")

if backend_available:
    uploaded_file = st.file_uploader(
        "Choose surveillance video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV (Max 500MB)"
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # File information display
        file_col1, file_col2 = st.columns([2, 1])
        
        with file_col1:
            st.markdown(f"""
            <div class="feature-card">
                <h4>📄 File Details</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size_mb:.1f} MB</p>
                <p><strong>Type:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if file_size_mb <= 500:
                st.video(uploaded_file)
            else:
                st.error("❌ File too large! Please upload files smaller than 500MB.")
        
        with file_col2:
            if file_size_mb <= 500:
                if st.button("⬆️ Upload Video", type="primary", use_container_width=True):
                    with st.spinner("Uploading video..."):
                        uploaded_file.seek(0)
                        file_content = uploaded_file.read()
                        
                        if api_client:
                            result = api_client.upload_file(
                                project_id=st.session_state.current_project_id,
                                file_content=file_content,
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
                                st.success(f"✅ Upload successful! File ID: {file_id[:12]}...")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                        else:
                            st.error("❌ API client not available")
else:
    st.markdown("""
    <div class="upload-zone">
        <h3>🔌 Backend Required</h3>
        <p>Start the FastAPI backend server to enable video uploads</p>
        <p><code>cd src && python main.py</code></p>
    </div>
    """, unsafe_allow_html=True)

# Processing Section
if st.session_state.uploaded_files:
    st.markdown("### ⚙️ Video Processing")
    
    for file_id, file_info in st.session_state.uploaded_files.items():
        with st.container():
            st.markdown(f"""
            <div class="feature-card">
                <h4>📹 {file_info['name']}</h4>
                <p><strong>Project:</strong> {file_info.get('project_name', 'Unknown')} | 
                   <strong>Size:</strong> {file_info['size'] / (1024*1024):.1f} MB | 
                   <strong>Uploaded:</strong> {time.ctime(file_info['upload_time'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Processing options
            proc_col1, proc_col2, proc_col3 = st.columns([2, 2, 1])
            
            with proc_col1:
                st.markdown("**🎯 AI Features:**")
                detect_objects = st.checkbox("Object Detection (YOLOv8)", value=True, key=f"detect_{file_id}")
                track_objects = st.checkbox("Object Tracking", value=True, key=f"track_{file_id}")
                generate_captions = st.checkbox("Scene Captioning (BLIP)", value=True, key=f"caption_{file_id}")
            
            with proc_col2:
                st.markdown("**⚙️ Settings:**")
                sample_rate = st.slider(
                    "Sample Rate (fps)", 
                    0.1, 2.0, 1.0, 0.1,
                    key=f"sample_{file_id}",
                    help="Frames per second to process"
                )
                
                detection_threshold = st.slider(
                    "Detection Confidence", 
                    0.1, 1.0, 0.5, 0.05,
                    key=f"conf_{file_id}",
                    help="Minimum confidence for detections"
                )
            
            with proc_col3:
                st.markdown("**🚀 Action:**")
                
                if backend_available and api_client:
                    if st.button("▶️ Process", key=f"process_{file_id}", type="primary", use_container_width=True):
                        processing_params = {
                            "sample_rate": sample_rate,
                            "detection_threshold": detection_threshold,
                            "enable_tracking": track_objects,
                            "enable_captioning": generate_captions
                        }
                        
                        with st.spinner("Starting processing..."):
                            project_id_to_use = file_info.get('project_id', st.session_state.current_project_id)
                            
                            result = api_client.process_video(
                                project_id=project_id_to_use,
                                file_id=file_id,
                                **processing_params
                            )
                            
                            if result.get("success", False):
                                job_id = result.get("data", {}).get("job_id", str(uuid.uuid4()))
                                
                                st.session_state.processing_jobs[job_id] = {
                                    "file_id": file_id,
                                    "file_name": file_info['name'],
                                    "project_id": project_id_to_use,
                                    "project_name": file_info.get('project_name', 'Unknown'),
                                    "start_time": time.time(),
                                    "status": "STARTED",
                                    "options": processing_params
                                }
                                st.success(f"✅ Processing started! Job: {job_id[:12]}...")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                else:
                    st.button("🔌 Backend Required", disabled=True, use_container_width=True)
            
            st.divider()

# Processing Jobs Status
if st.session_state.processing_jobs:
    st.markdown("### 📊 Active Jobs")
    
    # Auto-refresh controls
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=True)
    with status_col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    for job_id, job_info in st.session_state.processing_jobs.items():
        with st.container():
            st.markdown(f"""
            <div class="feature-card">
                <h4>⚙️ {job_info['file_name']} Processing</h4>
                <p><strong>Job ID:</strong> {job_id[:12]}... | <strong>Project:</strong> {job_info.get('project_name', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get job status
            if backend_available and api_client:
                actual_status, current_status, job_ready = get_job_status_info(api_client, job_id)
                
                # Update cached status in session state if we got a new status
                if actual_status is not None and current_status != job_info.get("status"):
                    st.session_state.processing_jobs[job_id]["status"] = current_status
                
                if actual_status is not None:
                    job_col1, job_col2, job_col3 = st.columns([2, 1, 1])
                    
                    with job_col1:
                        status_icons = {
                            "PENDING": "🟡 Queued",
                            "STARTED": "🔵 Processing", 
                            "PROGRESS": "🔵 Processing",
                            "SUCCESS": "🟢 Complete",
                            "COMPLETED": "🟢 Complete",
                            "FAILURE": "🔴 Failed",
                            "FAILED": "🔴 Failed"
                        }
                        
                        status_display = status_icons.get(current_status, f"⚪ {current_status}")
                        st.write(f"**Status:** {status_display}")
                        st.write(f"**Started:** {time.ctime(job_info['start_time'])}")
                        
                        if current_status == "PENDING":
                            elapsed = time.time() - job_info['start_time']
                            if elapsed > 30:
                                st.warning("⚠️ Long queue time - check worker status")
                        
                        # Show completion message for completed jobs
                        if current_status in ["SUCCESS", "COMPLETED"]:
                            st.success("🎉 Processing completed successfully!")
                    
                    with job_col2:
                        elapsed_time = time.time() - job_info['start_time']
                        st.metric("⏱️ Elapsed", f"{elapsed_time:.0f}s")
                        
                        # Show completion time for finished jobs
                        if current_status in ["SUCCESS", "COMPLETED", "FAILURE", "FAILED"]:
                            st.metric("🏁 Finished", "✅")
                    
                    with job_col3:
                        if st.button("🗑️ Remove", key=f"remove_{job_id}"):
                            del st.session_state.processing_jobs[job_id]
                            st.rerun()
                else:
                    st.warning(f"⚠️ Cannot retrieve status for {job_id[:12]}...")
            else:
                st.info("🔌 Backend required for job monitoring")
            
            st.divider()
    
    # Auto-refresh logic
    if auto_refresh and backend_available:
        active_jobs = []
        completed_jobs = []
        
        for job_id in st.session_state.processing_jobs:
            if api_client:
                _, current_status, _ = get_job_status_info(api_client, job_id)
                # Update cached status
                if current_status:
                    st.session_state.processing_jobs[job_id]["status"] = current_status
                
                if is_job_active(current_status):
                    active_jobs.append(job_id)
                elif current_status in ["SUCCESS", "COMPLETED"]:
                    completed_jobs.append(job_id)
        
        if active_jobs:
            st.info(f"🔄 Auto-refreshing... {len(active_jobs)} job(s) active")
            time.sleep(10)
            st.rerun()
        else:
            if completed_jobs:
                st.success(f"✅ All jobs completed! ({len(completed_jobs)} finished)")
                
                # Auto-cleanup completed jobs after 30 seconds
                cleanup_time = 30
                oldest_completed = min([
                    st.session_state.processing_jobs[job_id]["start_time"] 
                    for job_id in completed_jobs
                ])
                
                if time.time() - oldest_completed > cleanup_time:
                    if st.button("🧹 Auto-cleanup completed jobs"):
                        for job_id in completed_jobs:
                            del st.session_state.processing_jobs[job_id]
                        st.success("Cleaned up completed jobs!")
                        st.rerun()
                else:
                    remaining_time = cleanup_time - (time.time() - oldest_completed)
                    st.info(f"⏰ Auto-cleanup in {remaining_time:.0f}s or click 'Clear All' in sidebar")
            else:
                st.success("✅ All jobs completed!")

# Debug Section - Add this after the processing jobs section
if st.session_state.processing_jobs and backend_available:
    with st.expander("🔧 Debug: Job Status Details", expanded=False):
        st.markdown("### 🔍 Detailed Job Information")
        
        for job_id, job_info in st.session_state.processing_jobs.items():
            st.markdown(f"**Job ID:** `{job_id}`")
            st.write(f"**File:** {job_info['file_name']}")
            st.write(f"**Project:** {job_info.get('project_name', 'Unknown')}")
            st.write(f"**Started:** {time.ctime(job_info['start_time'])}")
            st.write(f"**Elapsed:** {time.time() - job_info['start_time']:.0f} seconds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🔄 Check Status", key=f"debug_status_{job_id}"):
                    if api_client:
                        with st.spinner("Checking job status..."):
                            status_result = api_client.get_job_status(job_id)
                            st.write("**Raw Status Response:**")
                            st.json(status_result)
                            
                            if status_result.get("success"):
                                status_data = status_result.get("status", {})
                                st.write("**Parsed Status:**")
                                st.json(status_data)
                            else:
                                st.error(f"Status check failed: {status_result.get('error', 'Unknown error')}")
            
            with col2:
                if st.button(f"🗑️ Force Remove", key=f"debug_remove_{job_id}"):
                    del st.session_state.processing_jobs[job_id]
                    st.success("Job removed from tracking")
                    st.rerun()
            
            # Try all possible job status endpoints
            if st.button(f"🔍 Test All Endpoints", key=f"debug_endpoints_{job_id}"):
                with st.spinner("Testing all job status endpoints..."):
                    endpoints_to_test = [
                        f"{api_client.base_url}/surveillance/jobs/status/{job_id}",
                        f"{api_client.base_url}/jobs/status/{job_id}",
                        f"{api_client.base_url}/surveillance/job/{job_id}",
                        f"{api_client.base_url}/job/{job_id}/status",
                        f"{api_client.base_url}/status/{job_id}",
                        f"{api_client.base_url}/surveillance/jobs/{job_info.get('project_id', 'test')}"
                    ]
                    
                    st.write("**Endpoint Test Results:**")
                    for endpoint in endpoints_to_test:
                        try:
                            import requests
                            response = requests.get(endpoint, timeout=5)
                            if response.status_code == 200:
                                st.success(f"✅ {endpoint} - Status: {response.status_code}")
                                try:
                                    data = response.json()
                                    st.json(data)
                                except:
                                    st.text(response.text[:200])
                            elif response.status_code == 404:
                                st.warning(f"❌ {endpoint} - Not Found (404)")
                            else:
                                st.error(f"❌ {endpoint} - Status: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ {endpoint} - Error: {str(e)}")
            
            st.divider()
        
        # Backend connectivity test
        st.markdown("### 🌐 Backend Connectivity Test")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Test Backend Health"):
                with st.spinner("Testing backend..."):
                    try:
                        import requests
                        # Test multiple health endpoints
                        health_endpoints = [
                            "http://localhost:5000/health",
                            "http://localhost:5000/api/surveillance/health",
                            "http://localhost:5000/api/health"
                        ]
                        
                        for endpoint in health_endpoints:
                            try:
                                response = requests.get(endpoint, timeout=5)
                                if response.status_code == 200:
                                    st.success(f"✅ {endpoint}")
                                    st.json(response.json())
                                    break
                                else:
                                    st.warning(f"❌ {endpoint} - Status: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ {endpoint} - Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Health check failed: {str(e)}")
        
        with col2:
            if st.button("📋 List All Jobs"):
                with st.spinner("Fetching all jobs..."):
                    try:
                        import requests
                        # Try to get all jobs for the current project
                        project_id = st.session_state.current_project_id
                        jobs_endpoints = [
                            f"http://localhost:5000/api/surveillance/jobs/{project_id}",
                            f"http://localhost:5000/api/jobs/{project_id}",
                            f"http://localhost:5000/api/surveillance/jobs"
                        ]
                        
                        for endpoint in jobs_endpoints:
                            try:
                                response = requests.get(endpoint, timeout=5)
                                if response.status_code == 200:
                                    st.success(f"✅ Jobs from {endpoint}")
                                    st.json(response.json())
                                    break
                                else:
                                    st.warning(f"❌ {endpoint} - Status: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ {endpoint} - Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Jobs list failed: {str(e)}")

# Sidebar information
with st.sidebar:
    st.markdown("### 📤 Video Processing")
    
    if backend_available:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
    
    st.markdown("### 📊 Current Project")
    st.write(f"**Name:** {st.session_state.current_project_name}")
    st.write(f"**ID:** {st.session_state.current_project_id}")
    
    # Statistics
    if st.session_state.uploaded_files:
        st.metric("📁 Files", len(st.session_state.uploaded_files))
    
    if st.session_state.processing_jobs:
        active_count = 0
        completed_count = 0
        
        # Get real-time status for each job instead of using cached status
        for job_id, job_info in st.session_state.processing_jobs.items():
            if backend_available and api_client:
                # Get current status from backend
                _, current_status, _ = get_job_status_info(api_client, job_id)
                
                if current_status in ["SUCCESS", "COMPLETED"]:
                    completed_count += 1
                elif current_status in ["FAILURE", "FAILED"]:
                    # Don't count failed jobs as active or completed for sidebar stats
                    pass
                else:
                    # PENDING, STARTED, PROGRESS, etc.
                    active_count += 1
            else:
                # Fall back to cached status if backend unavailable
                status = job_info.get("status", "UNKNOWN")
                if status in ["SUCCESS", "COMPLETED"]:
                    completed_count += 1
                elif status not in ["FAILURE", "FAILED"]:
                    active_count += 1
        
        st.metric("⚙️ Active Jobs", active_count)
        st.metric("✅ Completed", completed_count)
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **YOLOv8:** Real-time object detection
    - **Tracking:** Multi-object path analysis  
    - **BLIP:** AI scene descriptions
    - **Search:** Natural language queries
    """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Sample Rate:** Higher = more accurate
    - **Confidence:** Lower = more detections
    - **Formats:** MP4, AVI, MOV, MKV, WMV
    - **Max Size:** 500MB per file
    """)
    
    st.markdown("### 🚀 Quick Actions")
    
    # Clear completed jobs button
    if st.session_state.processing_jobs:
        completed_jobs = []
        if backend_available and api_client:
            for job_id, job_info in st.session_state.processing_jobs.items():
                _, current_status, _ = get_job_status_info(api_client, job_id)
                if current_status in ["SUCCESS", "COMPLETED"]:
                    completed_jobs.append(job_id)
        
        if completed_jobs:
            if st.button("🧹 Clear Completed", use_container_width=True):
                for job_id in completed_jobs:
                    del st.session_state.processing_jobs[job_id]
                st.success(f"Cleared {len(completed_jobs)} completed jobs!")
                st.rerun()
    
    if st.button("🧹 Clear All", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.processing_jobs = {}
        st.success("Cleared!")
        st.rerun()