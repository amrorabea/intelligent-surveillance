# streamlit/pages/2_🔍_Semantic_Search.py
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

st.set_page_config(
    page_title="🔍 Semantic Search - Surveillance System",
    page_icon="🔍",
    layout="wide"
)

# Modern CSS styling consistent with other pages
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
    
    /* Search box */
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
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
    
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
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
    
    .status-info {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Search stats */
    .search-stats {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Example queries */
    .example-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Project selection */
    .project-selector {
        background: #f8f9ff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .page-header {
            padding: 1.5rem;
        }
        
        .page-header h1 {
            font-size: 1.8rem;
        }
        
        .search-container {
            padding: 1.5rem;
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
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_results" not in st.session_state:
    st.session_state.current_results = []
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = ""

# Main header
st.markdown("""
<div class="page-header">
    <h1>🔍 Semantic Search</h1>
    <p>Search surveillance footage using natural language - AI understands context to find specific scenes and activities</p>
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
        🟢 <strong>Backend Online</strong> - Search engine ready
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
st.info("**Experience semantic search with sample data - no backend required**")

demo_col1, demo_col2, demo_col3 = st.columns(3)

with demo_col1:
    if st.button("� 'person walking'", type="primary", use_container_width=True):
        st.session_state.demo_query = "person walking"
        st.session_state.show_demo_search = True

with demo_col2:
    if st.button("🚗 'blue car parking'", type="secondary", use_container_width=True):
        st.session_state.demo_query = "blue car parking"
        st.session_state.show_demo_search = True

with demo_col3:
    if st.button("📦 'delivery person'", type="secondary", use_container_width=True):
        st.session_state.demo_query = "delivery person"
        st.session_state.show_demo_search = True

# Demo Search Results
if st.session_state.get('show_demo_search', False):
    query = st.session_state.get('demo_query', 'person walking')
    
    st.markdown(f"### 🎯 Demo Results: *'{query}'*")
    
    # Mock search stats
    st.markdown("""
    <div class="search-stats">
        <h4>📊 Search Performance</h4>
        <p><strong>⚡ Query Time:</strong> 0.045 seconds</p>
        <p><strong>🖼️ Frames Searched:</strong> 1,350</p>
        <p><strong>🎯 Matches Found:</strong> 23 relevant frames</p>
        <p><strong>🎚️ Confidence Threshold:</strong> 0.75</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create example results based on query
    if "person walking" in query.lower():
        demo_results = [
            {
                "similarity": 0.94,
                "caption": "Person in business attire walking toward main entrance",
                "timestamp": "00:01:23",
                "objects": ["person", "building", "entrance"],
                "confidence": 0.89,
                "project": "lobby-security-demo"
            },
            {
                "similarity": 0.91,
                "caption": "Security guard walking patrol route in hallway",
                "timestamp": "00:03:45",
                "objects": ["person", "hallway", "uniform"],
                "confidence": 0.92,
                "project": "lobby-security-demo"
            },
            {
                "similarity": 0.87,
                "caption": "Employee walking past reception with briefcase",
                "timestamp": "00:02:10",
                "objects": ["person", "briefcase", "reception"],
                "confidence": 0.85,
                "project": "lobby-security-demo"
            }
        ]
    elif "blue car" in query.lower():
        demo_results = [
            {
                "similarity": 0.96,
                "caption": "Blue sedan maneuvering into visitor parking space",
                "timestamp": "00:01:56",
                "objects": ["car", "parking", "blue"],
                "confidence": 0.94,
                "project": "parking-demo"
            },
            {
                "similarity": 0.88,
                "caption": "Blue SUV arriving at main entrance drop-off zone",
                "timestamp": "00:04:12",
                "objects": ["car", "entrance", "blue"],
                "confidence": 0.87,
                "project": "parking-demo"
            }
        ]
    else:  # delivery person
        demo_results = [
            {
                "similarity": 0.93,
                "caption": "Delivery worker carrying packages to front entrance",
                "timestamp": "00:02:34",
                "objects": ["person", "package", "delivery", "entrance"],
                "confidence": 0.91,
                "project": "delivery-demo"
            },
            {
                "similarity": 0.89,
                "caption": "Postal service worker at building mailbox area",
                "timestamp": "00:05:01",
                "objects": ["person", "mailbox", "uniform"],
                "confidence": 0.86,
                "project": "delivery-demo"
            }
        ]
    
    # Display results with modern styling
    for i, result in enumerate(demo_results, 1):
        st.markdown(f"""
        <div class="result-card">
            <h4>🎬 Result #{i} - Similarity: {result['similarity']:.1%}</h4>
            <p><strong>📝 Scene:</strong> {result['caption']}</p>
            <p><strong>⏰ Time:</strong> {result['timestamp']} | <strong>🎯 Confidence:</strong> {result['confidence']:.1%}</p>
            <p><strong>🏷️ Objects:</strong> {', '.join(result['objects'])}</p>
            <p><strong>📂 Project:</strong> {result['project']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate frame preview
        st.info(f"🖼️ Frame: surveillance_frame_{i:04d}.jpg")
    
    st.success(f"✅ Found {len(demo_results)} relevant frames matching your query!")
    
    # Interactive demo examples
    st.markdown("### 💡 Try More Examples:")
    example_queries = [
        "person with backpack", "cars in parking lot", "security guard patrol", "people talking",
        "someone at door", "bicycle near entrance", "person sitting", "delivery truck"
    ]
    
    query_cols = st.columns(4)
    for i, example in enumerate(example_queries):
        with query_cols[i % 4]:
            if st.button(f"🔍 {example}", key=f"example_{i}"):
                st.session_state.demo_query = example
                st.rerun()

# Demo control
if st.session_state.get('show_demo_search', False):
    if st.button("🧹 Clear Demo", use_container_width=True):
        for key in ['show_demo_search', 'demo_query']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown("---")
# Real Search Section
st.markdown("## 🚀 Real Search")

if backend_available:
    st.success("**Backend connected** - Search your processed videos")
else:
    st.warning("**Backend offline** - Start the FastAPI server to enable search")

# Project Selection (Required) - Enhanced
st.markdown("### 📁 Project Selection")
st.markdown("""
<div class="project-selector">
    <h4>🎯 Select Surveillance Project (Required)</h4>
    <p>Choose which project's videos to search through. Only processed videos with AI captions can be searched.</p>
</div>
""", unsafe_allow_html=True)

# Project selection with enhanced UI
project_col1, project_col2, project_col3 = st.columns([2, 1, 1])

with project_col1:
    project_id = st.text_input(
        "Project ID *",
        value=st.session_state.selected_project_id,
        placeholder="Enter project ID (e.g., surveillance-abc123)",
        help="Required: Enter the ID of the project containing processed videos",
        key="project_id_input"
    )
    
    if project_id != st.session_state.selected_project_id:
        st.session_state.selected_project_id = project_id
        # Clear previous results when project changes
        st.session_state.current_results = []

with project_col2:
    if backend_available and api_client and st.button("🔍 Validate", use_container_width=True, help="Check if project exists and has data"):
        if project_id.strip():
            with st.spinner("Validating project..."):
                try:
                    # Try to get project statistics
                    stats_result = api_client.get_database_stats()
                    if stats_result.get("success", False):
                        stats = stats_result.get("stats", {})
                        st.success("✅ Backend accessible")
                        
                        # Try a test search to validate project
                        test_result = api_client.semantic_search(
                            "test", 
                            project_id=project_id.strip(),
                            max_results=1,
                            confidence_threshold=0.1
                        )
                        
                        if test_result.get("success", False):
                            results_count = len(test_result.get("data", {}).get("results", []))
                            if results_count > 0:
                                st.success(f"✅ Project validated - Found searchable data")
                            else:
                                st.warning(f"⚠️ Project exists but no searchable content found")
                        else:
                            st.warning(f"⚠️ Project may not exist or has no processed videos")
                    else:
                        st.error("❌ Backend connection failed")
                except Exception as e:
                    st.error(f"❌ Validation error: {str(e)}")
        else:
            st.warning("Please enter a project ID first")

with project_col3:
    if st.button("💡 Examples", use_container_width=True, help="Common project ID patterns"):
        st.session_state.show_project_examples = not st.session_state.get('show_project_examples', False)

# Project examples and common patterns
if st.session_state.get('show_project_examples', False):
    st.markdown("""
    <div class="status-info">
        <h4>💡 Common Project ID Patterns</h4>
        <ul>
            <li><strong>surveillance-YYYYMMDD</strong> (e.g., surveillance-20241201)</li>
            <li><strong>security-LOCATION</strong> (e.g., security-lobby, security-parking)</li>
            <li><strong>camera-ID</strong> (e.g., camera-001, camera-main)</li>
            <li><strong>test-project</strong> (for development/testing)</li>
        </ul>
        <p><strong>Tip:</strong> Project IDs are usually created when you upload and process videos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-discovery attempt
    if backend_available and api_client:
        if st.button("🔍 Try Auto-Discovery", help="Attempt to find available projects"):
            with st.spinner("Searching for projects..."):
                try:
                    discovery_result = api_client.discover_projects()
                    if discovery_result.get("success", False):
                        projects = discovery_result.get("projects", [])
                        if len(projects) > 0:
                            st.success(f"✅ Found {len(projects)} projects with data!")
                            
                            st.markdown("**🎯 Available Projects:**")
                            
                            # Display discovered projects in columns
                            project_cols = st.columns(min(len(projects), 4))
                            for i, project in enumerate(projects):
                                with project_cols[i % 4]:
                                    project_id = project["project_id"]
                                    if st.button(f"📂 {project_id}", key=f"discovered_{i}"):
                                        st.session_state.selected_project_id = project_id
                                        st.rerun()
                                    
                                    # Show project info
                                    if project.get("type") == "verified":
                                        st.caption(f"✅ {project.get('data_count', 0)} items")
                                    else:
                                        st.caption("🔍 Discovered")
                        else:
                            st.warning("⚠️ No projects with searchable data found")
                            st.info("📊 Database accessible but no processed videos found. Try uploading and processing videos first.")
                            
                            # Suggest some common test patterns anyway
                            st.markdown("**💡 Try these common patterns:**")
                            common_patterns = ["test-project", "surveillance-demo", "camera-01", "security-main"]
                            
                            pattern_cols = st.columns(len(common_patterns))
                            for i, pattern in enumerate(common_patterns):
                                with pattern_cols[i]:
                                    if st.button(f"📂 {pattern}", key=f"pattern_{i}"):
                                        st.session_state.selected_project_id = pattern
                                        st.rerun()
                    else:
                        st.error(f"❌ Auto-discovery failed: {discovery_result.get('error', 'Unknown error')}")
                        
                        # Fall back to manual suggestions
                        st.markdown("**💡 Try these common patterns manually:**")
                        common_patterns = ["test-project", "surveillance-demo", "camera-01", "security-main"]
                        
                        pattern_cols = st.columns(len(common_patterns))
                        for i, pattern in enumerate(common_patterns):
                            with pattern_cols[i]:
                                if st.button(f"📂 {pattern}", key=f"fallback_pattern_{i}"):
                                    st.session_state.selected_project_id = pattern
                                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Auto-discovery failed: {str(e)}")
                    
                    # Provide manual options
                    st.markdown("**💡 Try these common patterns:**")
                    common_patterns = ["test-project", "surveillance-demo", "camera-01", "security-main"]
                    
                    pattern_cols = st.columns(len(common_patterns))
                    for i, pattern in enumerate(common_patterns):
                        with pattern_cols[i]:
                            if st.button(f"📂 {pattern}", key=f"manual_pattern_{i}"):
                                st.session_state.selected_project_id = pattern
                                st.rerun()

# Enhanced project validation with clear status
if not project_id.strip():
    st.markdown("""
    <div class="status-error">
        ⚠️ <strong>Project ID Required</strong><br>
        Please enter a project ID to enable search functionality. This should match the project used when processing videos.
    </div>
    """, unsafe_allow_html=True)
    search_enabled = False
else:
    # Validate project ID format
    if len(project_id.strip()) < 3:
        st.markdown("""
        <div class="status-error">
            ⚠️ <strong>Invalid Project ID</strong><br>
            Project ID must be at least 3 characters long.
        </div>
        """, unsafe_allow_html=True)
        search_enabled = False
    elif not project_id.replace('-', '').replace('_', '').isalnum():
        st.markdown("""
        <div class="status-error">
            ⚠️ <strong>Invalid Project ID Format</strong><br>
            Project ID should contain only letters, numbers, hyphens (-), and underscores (_).
        </div>
        """, unsafe_allow_html=True)
        search_enabled = False
    else:
        st.markdown("""
        <div class="status-success">
            ✅ <strong>Project Selected:</strong> {}<br>
            Search will be limited to videos processed under this project.
        </div>
        """.format(project_id.strip()), unsafe_allow_html=True)
        search_enabled = True

# Search Interface (only show if project is selected)
if search_enabled:
    st.markdown("### 🔍 Natural Language Search")
    
    st.markdown("""
    <div class="search-container">
        <h4>🧠 AI-Powered Video Search</h4>
        <p>Describe what you're looking for in natural language. Our AI searches through scene descriptions to find relevant moments.</p>
        <p><strong>🎯 Searching in project:</strong> <code>{}</code></p>
    </div>
    """.format(project_id.strip()), unsafe_allow_html=True)

    # Search form
    with st.form("search_form"):
        search_col1, search_col2 = st.columns([4, 1])
        
        with search_col1:
            query = st.text_input(
                "What are you looking for?",
                placeholder="e.g., 'person walking with a dog', 'red car in parking lot', 'people gathered around table'",
                help="Use natural language to describe scenes, objects, or activities"
            )
        
        with search_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    # Quick search examples for the selected project
    st.markdown("### 💡 Quick Search Examples")
    st.markdown("**Click any example to search immediately:**")
    
    quick_example_cols = st.columns(4)
    quick_examples = [
        "person walking", "car parking", "security guard", "people talking",
        "someone at door", "delivery person", "bicycle", "suspicious activity"
    ]
    
    for i, example in enumerate(quick_examples):
        with quick_example_cols[i % 4]:
            if st.button(f"🔍 {example}", key=f"quick_example_{i}", use_container_width=True):
                # Set the query and trigger search
                st.session_state.quick_search_query = example
                st.session_state.quick_search_triggered = True
                st.rerun()

    # Advanced Search Options
    with st.expander("⚙️ Advanced Search Options", expanded=False):
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        
        with opt_col1:
            st.markdown("**🎚️ Search Parameters**")
            max_results = st.slider("Maximum Results", 5, 100, 20)
            similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3, 0.1)
            st.caption("Lower threshold = more results, potentially less relevant")
        
        with opt_col2:
            st.markdown("**📅 Time Range Filter**")
            date_filter = st.checkbox("Enable Date Filter")
            if date_filter:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
                end_date = st.date_input("End Date", value=datetime.now())
            else:
                start_date = end_date = None
        
        with opt_col3:
            st.markdown("**🏷️ Object Filter**")
            object_types = st.multiselect(
                "Object Types",
                ["person", "car", "truck", "bicycle", "motorcycle", "bus", "dog", "cat", "backpack", "handbag"],
                help="Filter results by detected objects"
            )
            st.caption("Leave empty to search all object types")

else:
    # Show placeholder when no project selected
    st.markdown("### � Search Interface")
    st.markdown("""
    <div class="search-container">
        <h4>🔍 Search Interface</h4>
        <p>Please select a project above to enable search functionality.</p>
        <div style="text-align: center; padding: 2rem; background: #f8f9ff; border-radius: 10px; margin: 1rem 0;">
            <h3>🎯 Project Required</h3>
            <p>Enter a valid project ID above to unlock semantic search capabilities.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Handle quick search
if st.session_state.get('quick_search_triggered', False):
    query = st.session_state.get('quick_search_query', '')
    submitted = True
    # Clear the trigger
    st.session_state.quick_search_triggered = False

# Perform search
if submitted and query and query.strip() and search_enabled and backend_available:
    if len(query.strip()) < 3:
        st.warning("⚠️ Please enter a search query with at least 3 characters.")
    else:
        with st.spinner("🔍 Searching through footage..."):
            search_params = {
                "max_results": max_results,
                "confidence_threshold": similarity_threshold,
                "project_id": project_id.strip()  # Use the required project ID
            }
            
            # Add optional filters with validation
            if date_filter and start_date and end_date:
                search_params["start_date"] = start_date.isoformat()
                search_params["end_date"] = end_date.isoformat()
            
            if object_types and len(object_types) > 0:
                search_params["object_types"] = object_types
            
            # Debug info
            st.write("**🔍 Search Parameters:**")
            st.json({
                "query": query.strip(),
                **search_params
            })
            
            # Call API
            if api_client:
                result = api_client.semantic_search(query.strip(), **search_params)
                
                # Debug response
                st.write("**📡 API Response:**")
                st.json(result)
                
                if result.get("success", False):
                    # Handle different response formats
                    if "data" in result:
                        results = result["data"].get("results", []) if isinstance(result["data"], dict) else []
                    elif "results" in result:
                        results = result["results"]
                    else:
                        results = []
                    
                    st.session_state.current_results = results
                    
                    # Add to search history
                    st.session_state.search_history.append({
                        "query": query.strip(),
                        "timestamp": datetime.now(),
                        "result_count": len(results),
                        "project_id": project_id,
                        "filters": search_params
                    })
                    
                    # Show search stats
                    search_time = result.get("data", {}).get("processing_time", 0)
                    total_searched = result.get("total_documents", len(results))
                    
                    st.markdown(f"""
                    <div class="search-stats">
                        <h4>📊 Search Results</h4>
                        <p><strong>🔍 Query:</strong> "{query.strip()}"</p>
                        <p><strong>📂 Project:</strong> {project_id}</p>
                        <p><strong>🎯 Results:</strong> {len(results)} matches from {total_searched} documents</p>
                        <p><strong>⚡ Search Time:</strong> {search_time:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if len(results) == 0:
                        st.markdown("""
                        <div class="status-info">
                            <h4>🔍 Search Completed - No Matches Found</h4>
                            <p><strong>What this means:</strong></p>
                            <ul>
                                <li>✅ Search engine is working correctly</li>
                                <li>✅ Project database is accessible</li>
                                <li>❌ No frames match your query criteria</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**💡 Try these suggestions:**")
                        suggestions_col1, suggestions_col2 = st.columns(2)
                        
                        with suggestions_col1:
                            st.markdown("""
                            **Query Improvements:**
                            - Use simpler terms (e.g., "person" vs "individual")
                            - Try broader descriptions
                            - Lower the similarity threshold
                            - Remove object type filters
                            """)
                        
                        with suggestions_col2:
                            st.markdown("""
                            **Project Status:**
                            - Verify videos were processed with AI captions
                            - Check if this project has processed videos
                            - Ensure BLIP captioning was enabled during processing
                            """)
                        
                        # Quick project check
                        if st.button("🔍 Check Project Status", key="check_project_status"):
                            with st.spinner("Checking project data..."):
                                try:
                                    stats_result = api_client.get_database_stats()
                                    if stats_result.get("success", False):
                                        stats = stats_result.get("stats", {})
                                        st.success("✅ Database accessible")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("📄 Total Documents", stats.get("total_documents", 0))
                                        with col2:
                                            st.metric("🎬 Total Videos", stats.get("total_videos", 0))
                                        with col3:
                                            st.metric("🖼️ Total Frames", stats.get("total_frames", 0))
                                        
                                        if stats.get("total_documents", 0) == 0:
                                            st.warning("💡 Database appears empty - process some videos first!")
                                    else:
                                        st.error(f"❌ Database check failed: {stats_result.get('error')}")
                                except Exception as e:
                                    st.error(f"❌ Error checking database: {str(e)}")
                
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.markdown(f"""
                    <div class="status-error">
                        <h4>❌ Search Failed</h4>
                        <p><strong>Error:</strong> {error_msg}</p>
                        <p><strong>Query:</strong> "{query.strip()}"</p>
                        <p><strong>Project:</strong> {project_id}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ API client not available")

elif submitted and query and query.strip() and not search_enabled:
    st.warning("⚠️ Please select a project ID first.")

elif submitted and (not query or not query.strip()):
    st.warning("⚠️ Please enter a search query.")

elif submitted and not backend_available:
    st.error("🔌 Backend required for search functionality.")

# Display search results
if st.session_state.current_results:
    st.markdown("### 🎯 Search Results")
    
    # Sort options
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Relevance", "Timestamp", "Confidence"],
            help="Choose how to sort the results"
        )
    
    results = st.session_state.current_results.copy()
    
    # Apply sorting
    if sort_by == "Timestamp":
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    elif sort_by == "Confidence":
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    # Relevance is default order
    
    # Display results in a modern grid
    for i, result in enumerate(results):
        st.markdown(f"""
        <div class="result-card">
            <h4>🎬 Result #{i+1}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Frame preview
            frame_displayed = False
            result_id = result.get("result_id") or result.get("id")
            current_project_id = st.session_state.selected_project_id
            
            if result_id and current_project_id and api_client:
                try:
                    frame_data = api_client.get_frame(result_id=result_id, project_id=current_project_id)
                    if frame_data:
                        st.image(frame_data, caption="Frame Preview", use_column_width=True)
                        frame_displayed = True
                except Exception:
                    pass
            
            if not frame_displayed:
                st.info("📷 Frame preview not available")
        
        with col2:
            # Result details
            st.markdown(f"**🎯 Similarity:** {result.get('score', 0):.2%}")
            st.markdown(f"**⏰ Timestamp:** {result.get('timestamp', 'Unknown')}")
            st.markdown(f"**� Project:** {current_project_id}")
            st.markdown(f"**🎬 Video:** {result.get('video_filename', 'Unknown')}")
            st.markdown(f"**🖼️ Frame:** #{result.get('frame_number', 'Unknown')}")
            
            # AI caption
            caption = result.get("caption", "No caption available")
            st.markdown(f"**📝 AI Caption:** {caption}")
            
            # Detected objects
            objects = result.get("detected_objects", [])
            if objects and isinstance(objects, list):
                object_list = ", ".join([
                    f"{obj.get('class', 'Unknown')} ({obj.get('confidence', 0):.2f})" 
                    for obj in objects if isinstance(obj, dict)
                ])
                if object_list:
                    st.markdown(f"**�️ Objects:** {object_list}")
        
        with col3:
            # Actions
            st.markdown("**🔧 Actions:**")
            
            if st.button("👁️ Details", key=f"detail_{i}", use_container_width=True):
                with st.expander(f"📋 Detailed Info - Result #{i+1}", expanded=True):
                    st.json(result)
            
            if st.button("📤 Export", key=f"export_{i}", use_container_width=True):
                export_data = {
                    "query": st.session_state.search_history[-1]["query"] if st.session_state.search_history else "Unknown",
                    "result": result,
                    "project_id": current_project_id,
                    "export_time": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"search_result_{i+1}.json",
                    mime="application/json",
                    key=f"download_{i}"
                )
        
        st.divider()

# Search history
if st.session_state.search_history:
    with st.expander("📜 Search History"):
        st.markdown("### Recent Searches")
        
        for i, search in enumerate(reversed(st.session_state.search_history[-10:])):  # Last 10 searches
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{search['query']}**")
            
            with col2:
                st.write(f"{search['result_count']} results")
            
            with col3:
                st.write(search['timestamp'].strftime("%H:%M:%S"))
            
            with col4:
                if st.button("🔄 Repeat", key=f"repeat_{i}"):
                    # Trigger a rerun with the old query
                    st.session_state.repeat_query = search['query']
                    st.rerun()
        
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()

# Handle repeated query
if "repeat_query" in st.session_state:
    query = st.session_state.repeat_query
    del st.session_state.repeat_query
    submitted = True

# Tips and help
with st.expander("💡 Search Tips & Help"):
    st.markdown("""
    ### 🔍 How to Search Effectively
    
    **Natural Language Queries:**
    - Use descriptive language: "person walking with a dog"
    - Include colors: "red car in parking lot"
    - Specify actions: "people running", "vehicle stopping"
    - Describe scenes: "crowded area", "empty hallway"
    
    **Common Issues:**
    - **No results found**: Try lowering similarity threshold or different terms
    - **HTTP 500 errors**: Usually indicates backend database/configuration issues
    - **Slow searches**: Large databases may take longer to search
    
    **Advanced Features:**
    - **Similarity Threshold**: Higher values = more precise matches
    - **Date Filtering**: Search within specific time ranges
    - **Object Filtering**: Focus on specific object types
    - **Project Filtering**: Search within specific surveillance projects
    
    **Best Practices:**
    - Start with simple queries and refine as needed
    - Use the example queries to learn effective patterns
    - Process videos with AI captioning enabled for better search results
    - Check that ChromaDB is running and accessible
    """)

# Sidebar info
with st.sidebar:
    st.markdown("### 🔍 Semantic Search")
    
    if backend_available:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
    
    # Project status
    st.markdown("### 📂 Current Project")
    if st.session_state.selected_project_id:
        st.markdown(f"**ID:** `{st.session_state.selected_project_id}`")
        
        # Project validation status
        if search_enabled:
            st.success("✅ Project Valid")
        else:
            st.error("❌ Project Invalid")
    else:
        st.warning("⚠️ No Project Selected")
    
    # Search statistics
    if st.session_state.search_history:
        st.metric("📜 Total Searches", len(st.session_state.search_history))
        
        # Show searches for current project
        project_searches = [
            s for s in st.session_state.search_history 
            if s.get('project_id') == st.session_state.selected_project_id
        ]
        if project_searches:
            st.metric("🎯 Project Searches", len(project_searches))
    
    if st.session_state.current_results:
        st.metric("📋 Current Results", len(st.session_state.current_results))
    
    # Requirements
    st.markdown("### ✅ Requirements")
    requirements = [
        ("Backend Running", backend_available),
        ("Project Selected", bool(st.session_state.selected_project_id.strip())),
        ("Valid Project ID", search_enabled),
        ("AI Captions", True)  # We assume this is needed
    ]
    
    for req_name, req_met in requirements:
        icon = "✅" if req_met else "❌"
        st.markdown(f"{icon} {req_name}")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    if st.session_state.current_results:
        if st.button("🧹 Clear Results", use_container_width=True):
            st.session_state.current_results = []
            st.rerun()
    
    if st.session_state.search_history:
        if st.button("📜 Clear History", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    
    if backend_available and api_client:
        if st.button("� Check Database", use_container_width=True):
            with st.spinner("Checking database..."):
                stats_result = api_client.get_database_stats()
                if stats_result.get("success", False):
                    stats = stats_result.get("stats", {})
                    st.success("✅ Database accessible")
                    
                    # Show key stats if available
                    if isinstance(stats, dict):
                        if "total_documents" in stats:
                            st.metric("📄 Documents", stats.get("total_documents", 0))
                        if "total_videos" in stats:
                            st.metric("🎬 Videos", stats.get("total_videos", 0))
                        if "total_frames" in stats:
                            st.metric("🖼️ Frames", stats.get("total_frames", 0))
                else:
                    st.error(f"❌ Database check failed: {stats_result.get('error')}")
    
    # Help section
    st.markdown("### 💡 Tips")
    st.markdown("""
    **Best Practices:**
    - Use descriptive language
    - Include colors and actions
    - Try different similarity thresholds
    - Start with simple queries
    
    **Troubleshooting:**
    - Ensure videos were processed with captions
    - Check project ID is correct
    - Verify backend is running
    - Try broader search terms
    """)
    
    # Project suggestions
    if not st.session_state.selected_project_id.strip():
        st.markdown("### 🎯 Common Projects")
        common_projects = ["test-project", "surveillance-demo", "camera-01", "security-main"]
        
        for project in common_projects:
            if st.button(f"📂 {project}", key=f"sidebar_{project}"):
                st.session_state.selected_project_id = project
                st.rerun()
