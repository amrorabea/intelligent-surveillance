import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Simplified page config
st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
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
    .search-container {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .stButton > button {
        background: #4a6ee0;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client and session state
if API_AVAILABLE:
    api_client = SurveillanceAPIClient()
else:
    api_client = None

if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_results" not in st.session_state:
    st.session_state.current_results = []
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = ""
if "mode" not in st.session_state:
    st.session_state.mode = None

# Main header
st.markdown("""
<div class="header">
    <h1>Semantic Search</h1>
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
        if st.button("Real Search", use_container_width=True, disabled=not backend_available):
            st.session_state.mode = "real"
            st.rerun()
    if not backend_available:
        st.markdown('<div class="status status-error">🔴 Backend Offline: Real Search unavailable</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status status-{"success" if st.session_state.mode == "real" and backend_available else "processing"}">{"🟢 Real Search" if st.session_state.mode == "real" else "🟡 Demo Mode"}</div>', unsafe_allow_html=True)
    if st.button("Back to Mode Selection", use_container_width=True):
        st.session_state.mode = None
        for key in ['show_demo_search', 'demo_query']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Demo Mode
if st.session_state.mode == "demo":
    st.markdown("### Demo Mode")
    st.markdown("Try semantic search with sample data.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("'person walking'", use_container_width=True):
            st.session_state.demo_query = "person walking"
            st.session_state.show_demo_search = True
    with col2:
        if st.button("'blue car parking'", use_container_width=True):
            st.session_state.demo_query = "blue car parking"
            st.session_state.show_demo_search = True
    with col3:
        if st.button("'delivery person'", use_container_width=True):
            st.session_state.demo_query = "delivery person"
            st.session_state.show_demo_search = True

    if st.session_state.get('show_demo_search', False):
        query = st.session_state.get('demo_query', 'person walking')
        st.markdown(f"#### Demo Results: '{query}'")
        st.markdown("""
        <div class="card">
            <h3>Search Performance</h3>
            <p>Query Time: 0.045s<br>Frames Searched: 1,350<br>Matches: 23</p>
        </div>
        """, unsafe_allow_html=True)
        demo_results = []
        if "person walking" in query.lower():
            demo_results = [
                {"caption": "Person walking to entrance", "timestamp": "00:01:23", "objects": ["person"], "confidence": 0.89},
                {"caption": "Guard on patrol", "timestamp": "00:03:45", "objects": ["person"], "confidence": 0.92}
            ]
        elif "blue car" in query.lower():
            demo_results = [
                {"caption": "Blue sedan parking", "timestamp": "00:01:56", "objects": ["car"], "confidence": 0.94}
            ]
        else:
            demo_results = [
                {"caption": "Delivery worker with packages", "timestamp": "00:02:34", "objects": ["person", "package"], "confidence": 0.91}
            ]
        for i, result in enumerate(demo_results, 1):
            st.markdown(f"""
            <div class="card">
                <h3>Result #{i}</h3>
                <p>Scene: {result['caption']}<br>Time: {result['timestamp']}<br>Confidence: {result['confidence']:.1%}<br>Objects: {', '.join(result['objects'])}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div class="status status-success">✅ Found {len(demo_results)} matches</div>', unsafe_allow_html=True)
        if st.button("Clear Demo", use_container_width=True):
            for key in ['show_demo_search', 'demo_query']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Real Search
if st.session_state.mode == "real" and backend_available:
    st.markdown("### Real Search")
    st.markdown("#### Project Selection")
    col1, col2 = st.columns([3, 1])
    with col1:
        project_id = st.text_input("Project ID", value=st.session_state.selected_project_id, placeholder="e.g., surveillance-abc123")
        if project_id != st.session_state.selected_project_id:
            st.session_state.selected_project_id = project_id
            st.session_state.current_results = []
    with col2:
        if project_id.strip() and st.button("Validate", use_container_width=True):
            with st.spinner("Validating..."):
                try:
                    test_result = api_client.semantic_search("test", project_id=project_id.strip(), max_results=1, confidence_threshold=0.1)
                    if test_result.get("success", False) and len(test_result.get("data", {}).get("results", [])) > 0:
                        st.markdown('<div class="status status-success">✅ Project validated</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status status-error">⚠️ No searchable content</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="status status-error">❌ Validation error: {str(e)}</div>', unsafe_allow_html=True)
    
    search_enabled = bool(project_id.strip() and len(project_id.strip()) >= 3 and project_id.replace('-', '').replace('_', '').isalnum())
    if not search_enabled:
        st.markdown(f'<div class="status status-error">⚠️ {"Enter a valid project ID" if not project_id.strip() else "Invalid project ID format"}</div>', unsafe_allow_html=True)

    if search_enabled:
        st.markdown(f'<div class="status status-success">✅ Project: {project_id}</div>', unsafe_allow_html=True)
        st.markdown("#### Search")
        with st.form("search_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input("Search Query", placeholder="e.g., person walking, red car")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Search", use_container_width=True)
        
        if submitted and query.strip():
            if len(query.strip()) < 3:
                st.markdown('<div class="status status-error">⚠️ Query must be at least 3 characters</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Searching..."):
                    search_params = {"max_results": 20, "confidence_threshold": 0.3, "project_id": project_id.strip()}
                    result = api_client.semantic_search(query.strip(), **search_params)
                    if result.get("success", False):
                        results = result.get("data", {}).get("results", []) if isinstance(result.get("data"), dict) else []
                        st.session_state.current_results = results
                        st.session_state.search_history.append({
                            "query": query.strip(),
                            "timestamp": datetime.now(),
                            "result_count": len(results),
                            "project_id": project_id
                        })
                        st.markdown(f"""
                        <div class="card">
                            <h3>Search Results</h3>
                            <p>Query: {query.strip()}<br>Project: {project_id}<br>Matches: {len(results)}<br>Time: {result.get("data", {}).get("processing_time", 0):.2f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status status-error">❌ Search failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

    if st.session_state.current_results:
        st.markdown("#### Results")
        for i, result in enumerate(st.session_state.current_results, 1):
            st.markdown(f"""
            <div class="card">
                <h3>Result #{i}</h3>
                <p>Similarity: {result.get('score', 0):.2%}<br>Timestamp: {result.get('timestamp', 'N/A')}<br>Caption: {result.get('caption', 'N/A')}<br>Objects: {', '.join(result.get('detected_objects', []))}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Details", key=f"detail_{i}", use_container_width=True):
                st.markdown('<div class="card"><h3>Full Result</h3></div>', unsafe_allow_html=True)
                st.json(result)

# Sidebar
with st.sidebar:
    st.markdown("### Semantic Search")
    st.markdown(f'<div class="status status-{"success" if backend_available else "error"}">{"🟢 Online" if backend_available else "🔴 Offline"}</div>', unsafe_allow_html=True)
    st.markdown("### Project")
    st.markdown(f"**ID:** {st.session_state.selected_project_id or 'None'}")
    if st.session_state.search_history:
        st.markdown(f"**Searches:** {len(st.session_state.search_history)}")
    if st.session_state.current_results:
        st.markdown(f"**Results:** {len(st.session_state.current_results)}")
    st.markdown("### Tips")
    st.markdown("- Use descriptive queries\n- Include colors or actions\n- Ensure videos have captions\n- Try simple terms")
    if st.session_state.current_results and st.button("Clear Results", use_container_width=True):
        st.session_state.current_results = []
        st.rerun()
    if st.session_state.search_history and st.button("Clear History", use_container_width=True):
        st.session_state.search_history = []
        st.rerun()
