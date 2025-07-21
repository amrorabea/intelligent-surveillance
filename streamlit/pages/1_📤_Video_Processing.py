# streamlit/pages/1_📤_Video_Processing.py
import streamlit as st
import sys
import os
import time
import uuid
import json
import re

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

st.set_page_config(
    page_title="Video Processing - Surveillance System",
    page_icon="📤",
    layout="wide"
)

# Custom CSS (keeping your existing styles)
st.markdown("""
<style>
    .upload-box {
        border: 2px dashed #FF6B35;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .job-status {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .processing-options {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = SurveillanceAPIClient()

# Initialize session state
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = str(uuid.uuid4())
if "current_project_name" not in st.session_state:
    st.session_state.current_project_name = f"surveillance-{st.session_state.current_project_id[:8]}"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "processing_jobs" not in st.session_state:
    st.session_state.processing_jobs = {}

def sanitize_project_name(name: str) -> str:
    """Convert project name to valid project ID"""
    # Remove special characters, replace spaces with hyphens, lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', name.replace(' ', '-'))
    return sanitized.lower()

def get_job_status_info(api_client, job_id):
    """Get normalized job status information"""
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

st.title("📤 Video Processing")
st.markdown("Upload surveillance footage and process it with AI models")

# Debug info
with st.expander("🔧 Debug Info"):
    st.write(f"**API Base URL:** {api_client.base_url}")
    st.write(f"**Current Project ID:** {st.session_state.current_project_id}")
    st.write(f"**Current Project Name:** {st.session_state.current_project_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test backend connection
        if st.button("🔍 Test Backend Connection"):
            with st.spinner("Testing connection..."):
                try:
                    import requests
                    response = requests.get("http://localhost:5000/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Backend connection successful!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Backend returned status {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
    
    with col2:
        # Discover available endpoints
        if st.button("🔍 Discover API Endpoints"):
            with st.spinner("Discovering endpoints..."):
                result = api_client.get_available_endpoints()
                if result.get("success"):
                    st.success("✅ API accessible!")
                    st.json(result)
                    
                    # Try to get the actual API documentation
                    try:
                        import requests
                        docs_response = requests.get("http://localhost:5000/docs", timeout=5)
                        if docs_response.status_code == 200:
                            st.info("📚 Full API documentation available at: http://localhost:5000/docs")
                    except Exception:
                        pass
                else:
                    st.error(f"❌ API discovery failed: {result.get('error')}")
    
    # Show current uploaded files info
    if st.session_state.uploaded_files:
        st.markdown("**📁 Uploaded Files Debug:**")
        for file_id, file_info in st.session_state.uploaded_files.items():
            st.write(f"- **{file_info['name']}** (ID: `{file_id}`)")
            st.write(f"  Project: {file_info.get('project_id', 'Unknown')}")

# Project selection
st.markdown("### 📁 Project Configuration")
col1, col2 = st.columns([2, 1])

with col1:
    # Project name input that actually gets used
    new_project_name = st.text_input(
        "Project Name",
        value=st.session_state.current_project_name,
        help="Enter a name for this surveillance project (will be used as project ID)"
    )
    
    # Update project name and ID when changed
    if new_project_name != st.session_state.current_project_name:
        st.session_state.current_project_name = new_project_name
        # Create a sanitized project ID from the name
        st.session_state.current_project_id = sanitize_project_name(new_project_name)
        if not st.session_state.current_project_id:  # If sanitization results in empty string
            st.session_state.current_project_id = str(uuid.uuid4())[:8]

with col2:
    if st.button("🔄 New Project"):
        new_id = str(uuid.uuid4())
        st.session_state.current_project_id = new_id
        st.session_state.current_project_name = f"surveillance-{new_id[:8]}"
        st.session_state.uploaded_files = {}
        st.session_state.processing_jobs = {}
        st.rerun()

# Show the actual project ID that will be used
if st.session_state.current_project_id != sanitize_project_name(st.session_state.current_project_name):
    st.info(f"📊 Project ID (used internally): `{st.session_state.current_project_id}`")
else:
    st.info(f"📊 Project ID: `{st.session_state.current_project_id}`")

# File upload section
st.markdown("### 📹 Video Upload")

uploaded_file = st.file_uploader(
    "Choose surveillance video file",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
    help="Upload video files (MP4, AVI, MOV, MKV, WMV) - Max 500MB"
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    # Show file info
    st.markdown(f"""
    **📄 File Details:**
    - **Name:** {uploaded_file.name}
    - **Size:** {file_size_mb:.1f} MB
    - **Type:** {uploaded_file.type}
    """)
    
    # Check file size
    if file_size_mb > 500:
        st.error("❌ File too large! Please upload files smaller than 500MB.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            if st.button("⬆️ Upload File", type="primary"):
                with st.spinner("Uploading file..."):
                    # Read file content
                    uploaded_file.seek(0)  # Reset file pointer
                    file_content = uploaded_file.read()
                    
                    # Upload file using the current project ID
                    result = api_client.upload_file(
                        project_id=st.session_state.current_project_id,
                        file_content=file_content,
                        filename=uploaded_file.name
                    )
                    
                    # Check if upload was successful based on your backend response format
                    if result.get("success", False):
                        # Extract file_id from response
                        if "data" in result and "file_id" in result["data"]:
                            file_id = result["data"]["file_id"]
                        elif "file_id" in result:
                            file_id = result["file_id"]
                        else:
                            # Fallback: create file_id from filename
                            file_id = uploaded_file.name
                        
                        st.session_state.uploaded_files[file_id] = {
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                            "upload_time": time.time(),
                            "project_id": st.session_state.current_project_id,
                            "project_name": st.session_state.current_project_name
                        }
                        st.success(f"✅ File uploaded successfully! File ID: {file_id}")
                        st.balloons()
                    else:
                        st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                        
                        # Show detailed error info
                        with st.expander("🔍 Error Details"):
                            st.json(result)

# Show uploaded files and processing options
if st.session_state.uploaded_files:
    st.markdown("### 📁 Uploaded Files")
    
    for file_id, file_info in st.session_state.uploaded_files.items():
        st.markdown(f"#### 📹 {file_info['name']}")
        st.markdown(f"**Project:** {file_info.get('project_name', 'Unknown')} | **Size:** {file_info['size'] / (1024*1024):.1f} MB | **Uploaded:** {time.ctime(file_info['upload_time'])}")
        
        # Create a container for processing options
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("**📊 File Info:**")
                st.write(f"**Project ID:** {file_info.get('project_id', 'Unknown')}")
                st.write(f"**File ID:** {file_id[:20]}...")
            
            with col2:
                st.markdown("**🔧 Processing Options:**")
                
                # Main options
                detect_objects = st.checkbox("🎯 Object Detection (YOLOv8)", value=True, key=f"detect_{file_id}")
                track_objects = st.checkbox("🔁 Object Tracking", value=True, key=f"track_{file_id}")
                generate_captions = st.checkbox("📝 AI Captions (BLIP)", value=True, key=f"caption_{file_id}")
                
                # Advanced settings (not nested in expander)
                st.markdown("**⚙️ Advanced Settings:**")
                sample_rate = st.slider(
                    "Sample Rate (frames/second)", 
                    0.1, 2.0, 1.0, 0.1,
                    key=f"sample_{file_id}",
                    help="Higher = more frames processed = better accuracy but slower"
                )
                
                detection_threshold = st.slider(
                    "Detection Confidence Threshold", 
                    0.1, 1.0, 0.5, 0.05,
                    key=f"conf_{file_id}",
                    help="Lower = more detections but potentially more false positives"
                )
            
            with col3:
                st.markdown("**🚀 Actions:**")
                
                if st.button("▶️ Process Video", key=f"process_{file_id}", type="primary"):
                    # Prepare processing parameters
                    processing_params = {
                        "sample_rate": sample_rate,
                        "detection_threshold": detection_threshold,
                        "enable_tracking": track_objects,
                        "enable_captioning": generate_captions
                    }
                    
                    with st.spinner("Starting video processing..."):
                        # Use the project ID from when the file was uploaded
                        project_id_to_use = file_info.get('project_id', st.session_state.current_project_id)
                        
                        result = api_client.process_video(
                            project_id=project_id_to_use,
                            file_id=file_id,
                            **processing_params
                        )
                        
                        if result.get("success", False):
                            # Extract job_id from response
                            if "data" in result and "job_id" in result["data"]:
                                job_id = result["data"]["job_id"]
                            elif "job_id" in result:
                                job_id = result["job_id"]
                            else:
                                job_id = str(uuid.uuid4())  # Fallback
                            
                            st.session_state.processing_jobs[job_id] = {
                                "file_id": file_id,
                                "file_name": file_info['name'],
                                "project_id": project_id_to_use,
                                "project_name": file_info.get('project_name', 'Unknown'),
                                "start_time": time.time(),
                                "status": "STARTED",
                                "options": processing_params
                            }
                            st.success(f"✅ Processing started! Job ID: {job_id[:12]}...")
                        else:
                            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                            
                            # Show detailed error - not nested expander
                            st.markdown("**🔍 Error Details:**")
                            st.json(result)
        
        st.divider()

# Show processing jobs
if st.session_state.processing_jobs:
    st.markdown("### ⚙️ Processing Jobs")
    
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    with col1:
        # Check if auto-refresh should be turned off
        default_auto_refresh = True
        if "auto_refresh_off" in st.session_state and st.session_state.auto_refresh_off:
            default_auto_refresh = False
        
        auto_refresh = st.checkbox("🔄 Auto-refresh status (every 10 seconds)", value=default_auto_refresh)
        
        # Reset the auto-refresh off state if user manually enables it
        if auto_refresh and "auto_refresh_off" in st.session_state:
            del st.session_state.auto_refresh_off
            
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    for job_id, job_info in st.session_state.processing_jobs.items():
        with st.container():
            st.markdown(f"**📹 {job_info['file_name']}** (Job: {job_id[:12]}...) - Project: {job_info.get('project_name', 'Unknown')}")
            
            # Get job status using helper function
            actual_status, current_status, job_ready = get_job_status_info(api_client, job_id)
            
            # Enhanced status display
            if actual_status is not None:
                # Show detailed status info
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    status_icon = {
                        "PENDING": "🟡 Queued, waiting for worker",
                        "STARTED": "🔵 Processing in progress", 
                        "PROGRESS": "🔵 Processing in progress",
                        "SUCCESS": "🟢 Completed successfully",
                        "COMPLETED": "🟢 Completed successfully",
                        "FAILURE": "🔴 Failed",
                        "FAILED": "🔴 Failed"
                    }.get(current_status, f"⚪ {current_status}")
                    
                    st.write(f"**Status:** {status_icon}")
                    st.write(f"🕐 **Started:** {time.ctime(job_info['start_time'])}")
                    
                    # Show helpful message for PENDING status
                    if current_status == "PENDING":
                        elapsed = time.time() - job_info['start_time']
                        if elapsed > 30:  # More than 30 seconds
                            st.warning("⚠️ Job has been pending for a while. Check if Celery worker is running.")
                            if st.button("🔧 Debug Celery", key=f"debug_{job_id}"):
                                # Try to get Celery status
                                try:
                                    import requests
                                    debug_response = requests.get("http://localhost:5000/api/debug/celery-status", timeout=5)
                                    if debug_response.status_code == 200:
                                        st.json(debug_response.json())
                                    else:
                                        st.error("Could not get Celery status")
                                except Exception as e:
                                    st.error(f"Celery debug failed: {e}")
                
                with col2:
                    elapsed_time = time.time() - job_info['start_time']
                    st.metric("⏱️ Elapsed", f"{elapsed_time:.0f}s")
                    
                    # Show queue position or worker info if available
                    if "position" in actual_status:
                        st.metric("📍 Queue Position", actual_status["position"])
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{job_id}"):
                        del st.session_state.processing_jobs[job_id]
                        st.rerun()
            
            else:
                st.warning(f"⚠️ Could not retrieve status for job {job_id[:12]}...")
            
            st.divider()
    
    # Auto-refresh logic - only refresh if there are active jobs
    if auto_refresh and st.session_state.processing_jobs:
        # Check if any jobs are still running (not completed or failed)
        active_jobs = []
        for job_id, job_info in st.session_state.processing_jobs.items():
            actual_status, current_status, job_ready = get_job_status_info(api_client, job_id)
            
            if actual_status is not None and is_job_active(current_status):
                active_jobs.append(job_id)
        
        # Only auto-refresh if there are active jobs
        if active_jobs:
            st.info(f"🔄 Auto-refreshing... {len(active_jobs)} job(s) still processing")
            time.sleep(10)
            st.rerun()
        else:
            st.success("✅ All processing jobs completed! Auto-refresh stopped.")
            # Optionally turn off auto-refresh checkbox
            if st.button("🔄 Turn off auto-refresh"):
                st.session_state.auto_refresh_off = True
                st.rerun()

# Help section
with st.expander("❓ Help & Troubleshooting"):
    st.markdown("""
    ### 🎬 Video Processing Help
    
    **Common Issues:**
    
    1. **Upload fails with 400 error:**
       - Check if backend is running (`make dev`)
       - Verify file format is supported
       - Ensure file size is under 500MB
       - Check the Debug Info section above
    
    2. **Processing fails:**
       - Check System Status page for backend health
       - Verify enough disk space and memory
       - Try with smaller video or lower settings
    
    **Supported Formats:**
    - MP4, AVI, MOV, MKV, WMV
    - H.264 codec recommended
    - Max file size: 500MB
    
    **Processing Features:**
    - **Object Detection**: YOLOv8 detects 80+ object types
    - **Object Tracking**: Advanced IoU + distance-based tracking
    - **AI Captions**: BLIP generates scene descriptions
    
    **Performance Tips:**
    - Higher sample rate = better accuracy but slower processing
    - Lower confidence threshold = more detections but more false positives
    - Processing time: approximately 1-3 minutes per minute of video
    
    **Project Names:**
    - Project names are automatically converted to valid IDs
    - Special characters are removed, spaces become hyphens
    - Each project keeps uploaded files separate
    """)

# Sidebar info
with st.sidebar:
    st.header("📤 Video Processing")
    st.info("""
    Upload surveillance videos and process them with AI models.
    
    **Current Features:**
    - ✅ YOLOv8 object detection
    - ✅ Multi-object tracking
    - ✅ BLIP scene captioning
    - ✅ Real-time job monitoring
    - ✅ Project management
    """)
    
    # Show current project info
    st.markdown("### 📁 Current Project")
    st.write(f"**Name:** {st.session_state.current_project_name}")
    st.write(f"**ID:** {st.session_state.current_project_id}")
    
    # Show stats
    if st.session_state.uploaded_files:
        st.metric("📁 Uploaded Files", len(st.session_state.uploaded_files))
    
    if st.session_state.processing_jobs:
        active_jobs = len([j for j in st.session_state.processing_jobs.values() 
                          if j.get("status") not in ["SUCCESS", "COMPLETED", "FAILURE", "FAILED"]])
        completed_jobs = len([j for j in st.session_state.processing_jobs.values() 
                             if j.get("status") in ["SUCCESS", "COMPLETED"]])
        
        st.metric("⚙️ Active Jobs", active_jobs)
        st.metric("✅ Completed", completed_jobs)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    if st.button("🧹 Clear All Data"):
        st.session_state.uploaded_files = {}
        st.session_state.processing_jobs = {}
        st.success("Cleared all data!")
        st.rerun()
