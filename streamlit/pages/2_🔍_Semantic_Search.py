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
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

st.set_page_config(
    page_title="Semantic Search - Surveillance System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .search-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card:hover {
        border-color: #FF6B35;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .search-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .example-queries {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
    }
    .error-details {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = SurveillanceAPIClient()

# Initialize session state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_results" not in st.session_state:
    st.session_state.current_results = []

st.title("🔍 Semantic Search")
st.markdown("Search through processed surveillance footage using natural language")

# Debug section
with st.expander("🔧 Debug Info"):
    st.write(f"**API Base URL:** {api_client.base_url}")
    
    # Test search endpoint connectivity
    if st.button("🔍 Test Search Endpoint"):
        with st.spinner("Testing search endpoints..."):
            test_query = "test"
            result = api_client.semantic_search(test_query, max_results=1)
            
            if result.get("success"):
                st.success("✅ Search endpoint accessible!")
                st.json(result)
            else:
                st.error(f"❌ Search endpoint failed: {result.get('error')}")
                
                # Show more debugging info
                st.markdown("**🔍 Debugging Information:**")
                try:
                    import requests
                    # Try direct endpoint test
                    test_url = f"{api_client.base_url}/surveillance/query"
                    test_params = {"query": "test", "max_results": 1}
                    
                    st.write(f"Testing URL: {test_url}")
                    st.write(f"Test params: {test_params}")
                    
                    response = requests.get(test_url, params=test_params, timeout=5)
                    st.write(f"Status: {response.status_code}")
                    st.write(f"Response: {response.text[:1000]}")
                    
                except Exception as e:
                    st.error(f"Direct test failed: {str(e)}")
    
    # Test frame endpoints
    if st.button("🖼️ Check Frame Endpoints"):
        with st.spinner("Checking frame endpoints..."):
            result = api_client.check_frame_endpoints()
            
            if result["total_found"] > 0:
                st.success(f"✅ Found {result['total_found']} frame endpoints")
                for endpoint_info in result["available_endpoints"]:
                    st.write(f"- **{endpoint_info['endpoint']}** (Status: {endpoint_info['status']}) - {endpoint_info['note']}")
            else:
                st.warning("❌ No frame endpoints found")
                st.info("This means frame preview won't work. You may need to implement frame serving endpoints in your backend.")

# Search interface
st.markdown("""
<div class="search-box">
    <h3>🧠 AI-Powered Search</h3>
    <p>Describe what you're looking for in natural language. Our AI will search through all processed footage and find relevant scenes.</p>
</div>
""", unsafe_allow_html=True)

# Search form
with st.form("search_form"):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., 'person walking with a dog', 'red car in parking lot', 'people gathered around table'",
            help="Use natural language to describe scenes, objects, or activities",
            value=""  # Ensure empty string default
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

# Search filters
with st.expander("🔧 Advanced Search Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_results = st.slider("Maximum Results", 5, 100, 20)
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3, 0.1)
    
    with col2:
        # Date range filter
        date_filter = st.checkbox("Filter by Date Range")
        if date_filter:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", value=datetime.now())
        else:
            start_date = end_date = None
    
    with col3:
        # Project filter
        project_filter = st.text_input("Project ID Filter (optional)", help="Search within specific project", value="")
        
        # Object type filter
        object_types = st.multiselect(
            "Filter by Object Types",
            ["person", "car", "truck", "bicycle", "motorcycle", "bus", "dog", "cat", "bird"],
            help="Filter results to specific object types",
            default=[]
        )

# Example queries
st.markdown("""
<div class="example-queries">
    <h4>💡 Example Queries</h4>
    <p>Try these example searches to get started:</p>
</div>
""", unsafe_allow_html=True)

example_cols = st.columns(3)
example_queries = [
    ("👥 People gathering", "people gathering together"),
    ("🚗 Vehicle detection", "car or truck in the scene"),
    ("🏃 Motion activity", "person running or walking fast")
]

# Handle example query clicks
for i, (button_text, example_query) in enumerate(example_queries):
    with example_cols[i]:
        if st.button(button_text, key=f"example_{i}"):
            # Set query and trigger search
            query = example_query
            submitted = True

# Perform search
if submitted and query and query.strip():
    # Validate inputs
    query = query.strip()
    
    if len(query) < 3:
        st.warning("⚠️ Please enter a search query with at least 3 characters.")
    else:
        with st.spinner("🔍 Searching through footage..."):
            search_params = {
                "max_results": max_results,
                "confidence_threshold": similarity_threshold
            }
            
            # Add optional filters with validation
            if date_filter and start_date and end_date:
                search_params["start_date"] = start_date.isoformat()
                search_params["end_date"] = end_date.isoformat()
            
            if project_filter and project_filter.strip():
                search_params["project_id"] = project_filter.strip()
            
            if object_types and len(object_types) > 0:
                search_params["object_types"] = object_types
            
            # Debug info
            st.write("**🔍 Search Parameters:**")
            st.json({
                "query": query,
                **search_params
            })
            
            # Call API
            result = api_client.semantic_search(query, **search_params)
            
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
                    "query": query,
                    "timestamp": datetime.now(),
                    "result_count": len(results),
                    "filters": search_params
                })
                
                # Show search stats
                search_time = result.get("data", {}).get("processing_time", 0)
                total_searched = result.get("total_documents", len(results))
                
                st.markdown(f"""
                <div class="search-stats">
                    <h4>📊 Search Results</h4>
                    <p><strong>Query:</strong> "{query}"</p>
                    <p><strong>Results:</strong> {len(results)} matches from {total_searched} documents</p>
                    <p><strong>Search Time:</strong> {search_time:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(results) == 0:
                    st.info("""
                    🔍 **Search successful but no results found.**
                    
                    This usually means:
                    - ✅ Search endpoint is working
                    - ✅ Database connection is active  
                    - ❌ No processed videos with AI captions in the database
                    
                    **Next steps:**
                    1. Process some videos using the **Video Processing** page
                    2. Make sure **AI Captions (BLIP)** is enabled during processing
                    3. Wait for processing to complete successfully
                    4. Try your search again
                    
                    **Debug info:**
                    - Processing time: {:.2f} seconds
                    - Total documents searched: {}
                    """.format(search_time, total_searched))
                    
                    # Add button to check database status
                    if st.button("🔍 Check Database Status", key="check_db_status"):
                        with st.spinner("Checking database..."):
                            stats_result = api_client.get_database_stats()
                            if stats_result.get("success", False):
                                stats = stats_result.get("stats", {})
                                st.success("✅ Database accessible")
                                st.metric("📄 Documents", stats.get("total_documents", 0))
                                st.metric("🎬 Videos", stats.get("total_videos", 0))
                                st.metric("🖼️ Frames", stats.get("total_frames", 0))
                                
                                if stats.get("total_documents", 0) == 0:
                                    st.warning("💡 Database is empty. Process some videos first!")
                            else:
                                st.error(f"❌ Database check failed: {stats_result.get('error')}")
                
            else:
                error_msg = result.get('error', 'Unknown error')
                st.markdown(f"""
                <div class="error-details">
                    <h4>❌ Search Failed</h4>
                    <p><strong>Error:</strong> {error_msg}</p>
                    <p><strong>Query:</strong> "{query}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show debugging information
                with st.expander("🔍 Debugging Information"):
                    st.markdown("**Possible causes:**")
                    st.markdown("""
                    1. **Backend database not initialized** - No processed videos in the database
                    2. **ChromaDB connection issue** - Vector database not accessible
                    3. **Missing AI captions** - Videos processed without BLIP captioning
                    4. **Backend configuration** - Search service not properly configured
                    5. **Parameter validation** - Invalid search parameters
                    """)
                    
                    st.markdown("**Solutions:**")
                    st.markdown("""
                    1. Process some videos first using the Video Processing page
                    2. Check if ChromaDB is running and accessible
                    3. Ensure videos were processed with captioning enabled
                    4. Check backend logs for detailed error messages
                    5. Try a simpler query without filters
                    """)

elif submitted and (not query or not query.strip()):
    st.warning("⚠️ Please enter a search query.")

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
    
    # Display results in a grid
    for i, result in enumerate(results):
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <h4>🎬 Result #{i+1}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Frame preview (if available)
                frame_displayed = False
                
                # Try multiple ways to get frame data
                frame_data = None
                
                # Method 1: Use result_id if available
                result_id = result.get("result_id") or result.get("id")
                project_id = result.get("project_id", "default")
                
                if result_id:
                    try:
                        frame_data = api_client.get_frame(result_id=result_id, project_id=project_id)
                    except Exception as e:
                        st.caption(f"Frame error: {str(e)[:30]}...")
                
                # Method 2: Use frame_path if available
                if not frame_data:
                    frame_path = result.get("frame_path")
                    if frame_path and project_id:
                        try:
                            frame_data = api_client.get_frame(frame_path=frame_path, project_id=project_id)
                        except Exception as e:
                            st.caption(f"Path error: {str(e)[:30]}...")
                
                # Display frame if we got data
                if frame_data:
                    try:
                        st.image(frame_data, caption="Frame Preview", use_column_width=True)
                        frame_displayed = True
                    except Exception as e:
                        st.error(f"Display error: {str(e)[:30]}...")
                
                # Fallback display
                if not frame_displayed:
                    st.info("📷 Frame preview not available")
                    if st.button("🔍 Debug Frame", key=f"debug_frame_{i}"):
                        st.write(f"**Result ID:** {result_id}")
                        st.write(f"**Frame Path:** {result.get('frame_path', 'None')}")
                        st.write(f"**Project ID:** {project_id}")
                        
                        # Check available endpoints
                        endpoints_info = api_client.check_frame_endpoints()
                        st.json(endpoints_info)
                    
                    # Debug info for frame retrieval
                    with st.expander("🔍 Frame Debug Info"):
                        st.write("**Available frame data:**")
                        st.write(f"- Result ID: {result.get('result_id', 'None')}")
                        st.write(f"- Frame Path: {result.get('frame_path', 'None')}")
                        st.write(f"- Project ID: {result.get('project_id', 'None')}")
                        st.write(f"- ID: {result.get('id', 'None')}")
                        
                        if result.get("metadata"):
                            st.write("**Metadata:**")
                            st.json(result["metadata"])
            
            with col2:
                # Result details
                st.markdown(f"**🎯 Similarity Score:** {result.get('score', 0):.3f}")
                st.markdown(f"**⏰ Timestamp:** {result.get('timestamp', 'Unknown')}")
                st.markdown(f"**📁 Project:** {project_filter}")
                st.markdown(f"**🎬 Video:** {result.get('video_filename', 'Unknown')}")
                st.markdown(f"**🖼️ Frame:** {result.get('frame_number', 'Unknown')}")
                
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
                        st.markdown(f"**🎯 Objects:** {object_list}")
            
            with col3:
                # Actions
                st.markdown("**🔧 Actions:**")
                
                if st.button("👁️ View Details", key=f"detail_{i}"):
                    with st.expander(f"📋 Detailed Information - Result #{i+1}", expanded=True):
                        st.json(result)
                
                if st.button("📤 Export", key=f"export_{i}"):
                    # Create export data
                    export_data = {
                        "query": query if "query" in locals() else "Unknown",
                        "result": result,
                        "export_time": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="💾 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"search_result_{i+1}.json",
                        mime="application/json",
                        key=f"download_{i}"
                    )
            
            st.markdown("---")

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
    st.header("🔍 Semantic Search")
    st.info("""
    Search through processed surveillance footage using natural language queries.
    
    **Requirements:**
    - Videos must be processed with AI captioning
    - ChromaDB must be running
    - Search index must be built
    """)
    
    if st.session_state.search_history:
        st.metric("📜 Total Searches", len(st.session_state.search_history))
    
    if st.session_state.current_results:
        st.metric("🎯 Current Results", len(st.session_state.current_results))
    
    # Quick actions
    st.header("🚀 Quick Actions")
    if st.button("🧹 Clear Results"):
        st.session_state.current_results = []
        st.rerun()
    
    if st.button("📊 Check Database"):
        with st.spinner("Checking database..."):
            stats_result = api_client.get_database_stats()
            if stats_result.get("success", False):
                stats = stats_result.get("stats", {})
                st.success("✅ Database accessible")
                st.metric("📄 Documents", stats.get("total_documents", 0))
                st.metric("🎬 Videos", stats.get("total_videos", 0))
                st.metric("🖼️ Frames", stats.get("total_frames", 0))
            else:
                st.error(f"❌ Database check failed: {stats_result.get('error')}")
