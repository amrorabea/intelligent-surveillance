import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Analytics Dashboard - Surveillance System",
    page_icon="📊",
    layout="wide"
)

# Updated styling with more specific selectors and !important
st.markdown("""
<style>
    /* Target the main app container for general styling */
    .main .block-container {
        font-family: Arial, sans-serif !important;
        color: #333 !important;
    }
    /* Header styling */
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
    /* Button styling */
    button[kind="primary"], button[kind="secondary"] {
        width: 100% !important;
        border-radius: 8px !important;
        background: #2e7d32 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    /* Metric styling */
    div[data-testid="stMetric"] {
        background: #e8f5e9 !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        border: 1px solid #c8e6c9 !important;
        margin: 0.5rem 0 !important;
    }
    /* Status messages (success, info, warning, error) */
    div[data-testid="stAlert"] {
        border-radius: 4px !important;
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Shared functions for analytics display
def get_demo_data():
    """Get standardized demo data for analytics"""
    return {
        "system_health": {
            "database_connected": True,
            "vector_db_connected": True,
            "active_jobs": 0,
            "disk_usage_mb": 43384.8,
            "memory_usage_mb": 8404.6,
            "version": "1.0.0"
        },
        "surveillance_stats": {
            "videos_processed": 47,
            "frames_processed": 125680,
            "objects_detected": 15234,
            "avg_confidence": 0.87
        },
        "detection_counts": {
            'person': 1245,
            'car': 567,
            'bicycle': 123,
            'motorcycle': 89,
            'bus': 45,
            'truck': 78,
            'backpack': 234,
            'handbag': 189
        },
        "processing_stats": {
            "total_videos": 47,
            "total_frames": 125680,
            "avg_processing_time": 2.45,
            "success_rate": 94.2
        },
        "project_breakdown": [
            {"project_id": "building-security", "detections": 3456, "videos": 12},
            {"project_id": "parking-lot", "detections": 2987, "videos": 8},
            {"project_id": "entrance-main", "detections": 1876, "videos": 6},
            {"project_id": "warehouse-cam", "detections": 2543, "videos": 9},
            {"project_id": "office-floors", "detections": 1734, "videos": 5}
        ],
        "timeline_data": [
            {"timestamp": f"2025-07-22T{i:02d}:00:00", "detections_count": max(10, 200 - abs(12 - i) * 10 + (i % 3) * 5)}
            for i in range(24)
        ],
        "insights": [
            {
                "title": "Peak Activity Detection",
                "description": "Highest detection activity occurs between 12:00-14:00, suggesting lunch hour foot traffic patterns",
                "confidence": 0.89
            },
            {
                "title": "Object Type Analysis",
                "description": "Person detections account for 45% of all objects, indicating primary focus on human activity monitoring",
                "confidence": 0.92
            },
            {
                "title": "Processing Efficiency",
                "description": "Average processing time of 2.45s per video demonstrates optimal system performance",
                "confidence": 0.85
            },
            {
                "title": "Project Distribution",
                "description": "Building security project shows highest activity with 3456 total detections across 12 videos",
                "confidence": 0.78
            }
        ]
    }

def display_system_health_metrics(db_data, title="📈 System Health & Database Status"):
    """Display system health metrics"""
    st.markdown(f"#### {title}")
    
    # Debug: Show what health data we actually have
    st.write("🔍 **Debug - Health data structure:**", db_data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        database_connected = db_data.get("database_connected", db_data.get("db_connected", False))
        status_icon = "✅" if database_connected else "❌"
        st.metric("🗄️ Database", f"{status_icon} {'Connected' if database_connected else 'Disconnected'}")
    with col2:
        vector_db_connected = db_data.get("vector_db_connected", db_data.get("vectordb_connected", False))
        vector_icon = "✅" if vector_db_connected else "❌"
        st.metric("🔍 Vector DB", f"{vector_icon} {'Connected' if vector_db_connected else 'Disconnected'}")
    with col3:
        active_jobs = db_data.get("active_jobs", db_data.get("jobs", 0))
        st.metric("⚡ Active Jobs", active_jobs)
    with col4:
        disk_usage = db_data.get("disk_usage_mb", db_data.get("disk_usage", 0))
        if disk_usage == 0:
            st.metric("💾 Disk Usage", "N/A")
        else:
            st.metric("💾 Disk Usage", f"{disk_usage:.1f} MB")
            
    system_col1, system_col2 = st.columns(2)
    with system_col1:
        memory_usage = db_data.get("memory_usage_mb", db_data.get("memory_usage", 0))
        if memory_usage == 0:
            st.metric("🧠 Memory Usage", "N/A")
        else:
            st.metric("🧠 Memory Usage", f"{memory_usage:.1f} MB")
    with system_col2:
        version = db_data.get("version", db_data.get("app_version", "Unknown"))
        st.metric("📋 System Version", version)
        
    # Show overall status
    if database_connected and vector_db_connected:
        st.success("🎉 All systems operational!")
    elif database_connected or vector_db_connected:
        st.warning("⚠️ Partial system connectivity")
    else:
        # Check if we have any health indicators at all
        if any(key in db_data for key in ['status', 'health', 'message']):
            st.info(f"📊 Backend Status: {db_data.get('status', db_data.get('health', db_data.get('message', 'Running')))}")
        else:
            st.error("❌ System connectivity issues")

def display_surveillance_metrics(surv_data, title="🎯 Surveillance Statistics"):
    """Display surveillance metrics"""
    st.markdown(f"#### {title}")
    
    # Check if we have actual surveillance data or just health data
    has_real_data = any(key in surv_data for key in ['videos_processed', 'total_videos', 'frames_processed', 'total_frames', 'objects_detected', 'total_detections'])
    
    if not has_real_data:
        st.warning("⚠️ No surveillance statistics available. This appears to be health/status data only.")
        st.info("💡 Try processing some videos first to generate surveillance statistics.")
        return 0, 0, 0, 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        videos_processed = surv_data.get("videos_processed", surv_data.get("total_videos", 0))
        st.metric("🎬 Videos Processed", videos_processed)
    with col2:
        frames_processed = surv_data.get("frames_processed", surv_data.get("total_frames", 0))
        st.metric("🖼️ Frames Processed", frames_processed)
    with col3:
        objects_detected = surv_data.get("objects_detected", surv_data.get("total_detections", 0))
        st.metric("🔍 Objects Detected", objects_detected)
    with col4:
        avg_confidence = surv_data.get("avg_confidence", 0)
        formatted_confidence = f"{avg_confidence:.1%}" if 0 < avg_confidence <= 1 else f"{avg_confidence:.2f}" if avg_confidence > 0 else "N/A"
        st.metric("📊 Avg Confidence", formatted_confidence)
    return videos_processed, frames_processed, objects_detected, avg_confidence

def display_surveillance_overview_chart(videos_processed, frames_processed, objects_detected):
    """Display surveillance overview chart"""
    surv_metrics = []
    if videos_processed > 0:
        surv_metrics.append(('Videos Processed', videos_processed))
    if frames_processed > 0:
        surv_metrics.append(('Frames Processed', frames_processed))
    if objects_detected > 0:
        surv_metrics.append(('Objects Detected', objects_detected))
    if len(surv_metrics) > 1:
        st.markdown("##### 📊 Surveillance Processing Overview")
        surv_chart_data = pd.DataFrame({
            'Metric': [metric[0] for metric in surv_metrics],
            'Count': [metric[1] for metric in surv_metrics]
        })
        st.bar_chart(surv_chart_data.set_index('Metric'))

def display_processing_insights(videos_processed, frames_processed, objects_detected, avg_confidence):
    """Display processing insights"""
    if videos_processed > 0 and frames_processed > 0:
        st.markdown("##### 💡 Processing Insights")
        col1, col2 = st.columns(2)
        with col1:
            avg_frames_per_video = frames_processed / videos_processed
            st.markdown(f"<div>📹 <b>{avg_frames_per_video:.1f}</b> average frames per video</div>", unsafe_allow_html=True)
            if objects_detected > 0:
                detection_rate = objects_detected / frames_processed
                st.markdown(f"<div>🎯 <b>{detection_rate:.3f}</b> detections per frame</div>", unsafe_allow_html=True)
        with col2:
            if objects_detected > 0:
                avg_detections_per_video = objects_detected / videos_processed
                st.markdown(f"<div>🔍 <b>{avg_detections_per_video:.1f}</b> detections per video</div>", unsafe_allow_html=True)
            if avg_confidence > 0:
                confidence_percentage = avg_confidence * 100 if avg_confidence <= 1 else avg_confidence
                st.markdown(f"<div>📊 <b>{confidence_percentage:.1f}%</b> average confidence</div>", unsafe_allow_html=True)

def display_detection_counts(detection_counts):
    """Display detection counts and chart"""
    if not detection_counts or not isinstance(detection_counts, dict):
        return
    st.markdown("##### 🎯 Detection Counts")
    det_cols = st.columns(min(4, len(detection_counts)))
    for i, (obj_type, count) in enumerate(detection_counts.items()):
        with det_cols[i % 4]:
            st.metric(f"🔍 {obj_type.title()}", count)
    if len(detection_counts) > 1:
        st.markdown("##### 📊 Detection Distribution Chart")
        df_detections = pd.DataFrame(list(detection_counts.items()), columns=['Object Type', 'Count'])
        st.bar_chart(df_detections.set_index('Object Type'))

def display_confidence_distribution(confidence_dist):
    """Display confidence distribution chart"""
    if not confidence_dist or not isinstance(confidence_dist, dict):
        return
    st.markdown("##### 📊 Confidence Distribution")
    
    # Convert to DataFrame for better visualization
    conf_data = []
    for range_key, count in confidence_dist.items():
        conf_data.append({"Confidence Range": range_key, "Count": count})
    
    if conf_data:
        df_confidence = pd.DataFrame(conf_data)
        st.bar_chart(df_confidence.set_index('Confidence Range')['Count'])
        
        # Show summary stats
        total_with_confidence = sum(confidence_dist.values())
        if total_with_confidence > 0:
            st.info(f"📈 Total detections with confidence scores: {total_with_confidence}")
        else:
            st.warning("⚠️ No confidence data available for detections")

def display_performance_metrics(perf_metrics):
    """Display performance metrics"""
    if not perf_metrics or not isinstance(perf_metrics, dict):
        return
    
    st.markdown("##### ⚡ Performance Metrics")
    
    processing_speeds = perf_metrics.get("processing_speeds", [])
    if processing_speeds:
        # Convert to proper format for chart
        if isinstance(processing_speeds, dict):
            speeds_data = list(processing_speeds.values())
        else:
            speeds_data = processing_speeds
            
        st.markdown("**Processing Speeds (fps)**")
        speed_df = pd.DataFrame({
            'Video': [f"Video {i+1}" for i in range(len(speeds_data))],
            'Speed (fps)': speeds_data
        })
        st.bar_chart(speed_df.set_index('Video')['Speed (fps)'])
        
        # Show average
        avg_speed = sum(speeds_data) / len(speeds_data)
        st.metric("📊 Average Processing Speed", f"{avg_speed:.1f} fps")

def display_search_analytics(search_analytics):
    """Display search analytics"""
    if not search_analytics or not isinstance(search_analytics, dict):
        return
        
    st.markdown("##### 🔍 Search Analytics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_searches = search_analytics.get("total_searches", 0)
        st.metric("🔍 Total Searches", total_searches)
    with col2:
        avg_results = search_analytics.get("avg_results", 0)
        st.metric("📊 Avg Results", f"{avg_results:.1f}" if avg_results > 0 else "N/A")
    with col3:
        avg_search_time = search_analytics.get("avg_search_time", 0)
        st.metric("⏱️ Avg Search Time", f"{avg_search_time:.2f}s" if avg_search_time > 0 else "N/A")
    
    # Show popular terms if available
    popular_terms = search_analytics.get("popular_terms", [])
    if popular_terms:
        st.markdown("**Popular Search Terms:**")
        terms_text = ", ".join(popular_terms[:10])  # Show top 10
        st.write(terms_text)

def display_alerts(alerts):
    """Display system alerts"""
    if not alerts or not isinstance(alerts, list):
        return
        
    st.markdown("##### 🚨 System Alerts")
    for alert in alerts:
        if isinstance(alert, dict):
            alert_type = alert.get("type", "info")
            title = alert.get("title", "Alert")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "")
            
            if alert_type == "error":
                st.error(f"**{title}**: {message}")
            elif alert_type == "warning":
                st.warning(f"**{title}**: {message}")
            elif alert_type == "success":
                st.success(f"**{title}**: {message}")
            else:
                st.info(f"**{title}**: {message}")
                
            if timestamp:
                st.caption(f"Time: {timestamp}")

def display_detection_timeline(timeline_data):
    """Display detection timeline chart"""
    if not timeline_data:
        return
    st.markdown("##### ⏰ Detection Timeline")
    
    # Handle both list and dict formats
    if isinstance(timeline_data, dict):
        # Convert dict to list format
        timeline_list = []
        for key, value in timeline_data.items():
            if isinstance(value, dict):
                timeline_list.append(value)
    else:
        timeline_list = timeline_data
    
    timeline_df = pd.DataFrame(timeline_list)
    if not timeline_df.empty and 'timestamp' in timeline_df.columns and 'detections_count' in timeline_df.columns:
        # Convert timestamp to datetime and extract hour
        timeline_df['datetime'] = pd.to_datetime(timeline_df['timestamp'])
        timeline_df['hour'] = timeline_df['datetime'].dt.hour
        
        # Create chart data
        chart_data = timeline_df.set_index('hour')['detections_count']
        st.line_chart(chart_data)
        
        # Show summary stats
        total_detections = timeline_df['detections_count'].sum()
        peak_hour = timeline_df.loc[timeline_df['detections_count'].idxmax(), 'hour']
        st.info(f"📊 Total timeline detections: {total_detections} | Peak activity at hour: {peak_hour}")
    else:
        st.warning("⚠️ Timeline data format not recognized")

def display_processing_statistics(proc_stats):
    """Display processing statistics"""
    # Check for any meaningful data
    meaningful_keys = ['total_videos', 'total_frames', 'total_items', 'total_documents', 'total_collections', 'avg_processing_time', 'success_rate']
    has_data = any(proc_stats.get(key, 0) for key in meaningful_keys)
    
    if not proc_stats or not has_data:
        st.warning("⚠️ No processing statistics available")
        return
        
    st.markdown("##### ⚡ Processing Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_videos = proc_stats.get("total_videos", 0)
        total_items = proc_stats.get("total_items", 0)
        display_value = total_videos if total_videos > 0 else total_items
        st.metric("🎬 Videos/Items", display_value)
        
    with col2:
        total_frames = proc_stats.get("total_frames", 0)
        total_docs = proc_stats.get("total_documents", 0)
        display_value = total_frames if total_frames > 0 else total_docs
        st.metric("🖼️ Frames/Docs", display_value)
        
    with col3:
        total_collections = proc_stats.get("total_collections", 0)
        avg_processing_time = proc_stats.get("avg_processing_time", 0)
        if avg_processing_time > 0:
            st.metric("⏱️ Avg Process Time", f"{avg_processing_time:.2f}s")
        else:
            st.metric("📂 Collections", total_collections)
            
    with col4:
        success_rate = proc_stats.get("success_rate", 0)
        if success_rate > 0:
            st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("📊 Status", "Operational")

def display_processing_efficiency(proc_stats):
    """Display processing efficiency insights"""
    if not proc_stats or not any(proc_stats.get(key, 0) for key in ['total_videos', 'total_frames', 'avg_processing_time']):
        return
        
    st.markdown("##### 💡 Processing Efficiency")
    col1, col2 = st.columns(2)
    with col1:
        total_videos = proc_stats.get("total_videos", 0)
        total_frames = proc_stats.get("total_frames", 0)
        if total_videos > 0 and total_frames > 0:
            frames_per_video = total_frames / total_videos
            st.markdown(f"<div>📹 <b>{frames_per_video:.0f}</b> frames per video on average</div>", unsafe_allow_html=True)
    with col2:
        avg_processing_time = proc_stats.get("avg_processing_time", 0)
        if avg_processing_time > 0:
            st.markdown(f"<div>⚡ <b>{avg_processing_time:.2f}s</b> average processing time per video</div>", unsafe_allow_html=True)

def display_project_breakdown(project_data):
    """Display project breakdown table and insights"""
    if not project_data or not isinstance(project_data, list):
        return
    st.markdown("##### 📁 Project Breakdown")
    projects_df = pd.DataFrame(project_data)
    st.dataframe(projects_df, use_container_width=True)
    st.markdown("##### 💡 Project Insights")
    total_detections = sum(p.get("detections", 0) for p in project_data)
    total_videos = sum(p.get("videos", 0) for p in project_data)
    col1, col2 = st.columns(2)
    with col1:
        avg_detections = total_detections / len(project_data) if len(project_data) > 0 else 0
        st.markdown(f"<div>📊 <b>{avg_detections:.0f}</b> average detections per project</div>", unsafe_allow_html=True)
    with col2:
        avg_videos = total_videos / len(project_data) if len(project_data) > 0 else 0
        st.markdown(f"<div>🎬 <b>{avg_videos:.1f}</b> average videos per project</div>", unsafe_allow_html=True)

def display_ai_insights(insights):
    """Display AI insights"""
    if not insights or not isinstance(insights, list):
        return
    st.markdown("##### 💡 AI Insights")
    for insight in insights:
        if isinstance(insight, dict):
            title = insight.get("title", "Insight")
            description = insight.get("description", "No description")
            confidence = insight.get("confidence", 0)
            if confidence > 0.8:
                st.success(f"**{title}**: {description} (Confidence: {confidence:.1%})")
            elif confidence > 0.6:
                st.info(f"**{title}**: {description} (Confidence: {confidence:.1%})")
            else:
                st.warning(f"**{title}**: {description} (Confidence: {confidence:.1%})")

def display_complete_analytics(data, mode_title="Analytics"):
    """Display complete analytics dashboard"""
    if "system_health" in data:
        display_system_health_metrics(data["system_health"])
    if "surveillance_stats" in data:
        videos, frames, detections, confidence = display_surveillance_metrics(data["surveillance_stats"])
        display_surveillance_overview_chart(videos, frames, detections)
        display_processing_insights(videos, frames, detections, confidence)
    if "detection_counts" in data:
        display_detection_counts(data["detection_counts"])
    if "timeline_data" in data:
        display_detection_timeline(data["timeline_data"])
    if "processing_stats" in data:
        display_processing_statistics(data["processing_stats"])
        display_processing_efficiency(data["processing_stats"])
    if "project_breakdown" in data:
        display_project_breakdown(data["project_breakdown"])
    if "insights" in data:
        display_ai_insights(data["insights"])

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = SurveillanceAPIClient()
if 'analytics_cache' not in st.session_state:
    st.session_state.analytics_cache = {}
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = datetime.now() - timedelta(hours=1)

# Main header
st.markdown('<div class="header"><h1>Analytics Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown("View detection statistics, patterns, and insights from your surveillance data.")

# Mode selection
if 'mode' not in st.session_state:
    st.session_state.mode = None

if not st.session_state.mode:
    st.markdown("### Choose Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Demo Mode", type="primary"):
            st.session_state.mode = "demo"
            st.rerun()
    with col2:
        if st.button("📈 Real Analytics", type="primary"):
            st.session_state.mode = "real"
            st.rerun()
else:
    st.markdown(f"### {'📈 Demo Mode' if st.session_state.mode == 'demo' else '📊 Real Analytics'}")
    if st.button("🧹 Back to Mode Selection", type="secondary"):
        st.session_state.mode = None
        for key in ['show_demo_overview', 'show_demo_trends', 'show_demo_complete']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Demo Mode
if st.session_state.mode == "demo":
    demo_data = get_demo_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Database Stats", type="primary"):
            st.session_state.show_demo_overview = True
    with col2:
        if st.button("🎯 Surveillance Stats", type="secondary"):
            st.session_state.show_demo_trends = True
    with col3:
        if st.button("📈 Complete Analytics", type="secondary"):
            st.session_state.show_demo_complete = True

    if st.session_state.get('show_demo_overview', False):
        display_system_health_metrics(demo_data["system_health"])
    if st.session_state.get('show_demo_trends', False):
        videos, frames, detections, confidence = display_surveillance_metrics(demo_data["surveillance_stats"])
        display_surveillance_overview_chart(videos, frames, detections)
        display_processing_insights(videos, frames, detections, confidence)
    if st.session_state.get('show_demo_complete', False):
        st.markdown("#### 📈 Complete Analytics")
        display_complete_analytics(demo_data)

    if any(st.session_state.get(k, False) for k in ['show_demo_overview', 'show_demo_trends', 'show_demo_complete']):
        if st.button("🧹 Clear Demo", type="secondary"):
            for key in ['show_demo_overview', 'show_demo_trends', 'show_demo_complete']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Real Analytics
if st.session_state.mode == "real":
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        time_range = st.selectbox("📅 Select time range:", ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])
    with col2:
        # Heavy vs Light analytics toggle
        analytics_mode = st.selectbox("🔬 Analytics Mode:", ["Light (Fast)", "Heavy (AI Insights)"])
        heavy_mode = analytics_mode.startswith("Heavy")
    with col3:
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.force_refresh = True
            st.rerun()

    # Show mode explanation
    if heavy_mode:
        st.info("🧠 **Heavy Mode**: Loads AI models for advanced insights (slower, more detailed)")
    else:
        st.info("⚡ **Light Mode**: Fast analytics without AI model loading (faster, basic stats)")

    # Simple caching
    cache_duration = 30
    current_time = datetime.now()
    time_since_cache = (current_time - st.session_state.cache_timestamp).total_seconds()
    should_fetch = time_since_cache >= cache_duration or st.session_state.get('force_refresh', False)

    if should_fetch:
        with st.spinner("📊 Loading analytics data..."):
            try:
                # Make API calls and track what data we actually get
                db_stats = st.session_state.api_client.get_database_stats()
                surveillance_stats = st.session_state.api_client.get_surveillance_stats()
                analytics_data = st.session_state.api_client.get_analytics(
                    heavy_mode=heavy_mode,
                    time_range=time_range.lower().replace(" ", "_")
                )
                
                st.session_state.analytics_cache = {
                    'db_stats': db_stats,
                    'surveillance_stats': surveillance_stats,
                    'analytics_data': analytics_data
                }
                st.session_state.cache_timestamp = current_time
                st.session_state.force_refresh = False
                
                # Show what we actually got
                success_count = sum([
                    1 for result in [db_stats, surveillance_stats, analytics_data] 
                    if result.get("success", False)
                ])
                
                if success_count > 0:
                    st.success(f"✅ {success_count}/3 API endpoints responded successfully!")
                    
                    # Debug: Show response structure for analytics data
                    if analytics_data.get("success", False):
                        analytics_content = analytics_data.get("data", {})
                        if analytics_content:
                            st.info(f"📊 Analytics data keys: {list(analytics_content.keys())}")
                        else:
                            st.warning("📊 Analytics endpoint returned success but no data content")
                else:
                    st.error("❌ All API endpoints failed to return data")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                if not st.session_state.analytics_cache:
                    st.warning("⚠️ No cached data available. Try Demo Mode.")
                    st.stop()

    # Display data analysis
    cached_data = st.session_state.analytics_cache
    has_real_data = False
    
    # Check each data source and display if available
    if cached_data.get('db_stats', {}).get("success", False):
        db_data = cached_data['db_stats'].get("stats", {})
        # Only show if it contains meaningful database info, not just health status
        if any(key in db_data for key in ['database_connected', 'vector_db_connected', 'disk_usage_mb', 'memory_usage_mb']):
            display_system_health_metrics(db_data)
            has_real_data = True
        else:
            st.info("📊 Database stats: Health endpoint responded but no detailed metrics available")

    if cached_data.get('surveillance_stats', {}).get("success", False):
        surv_data = cached_data['surveillance_stats'].get("stats", {})
        videos, frames, detections, confidence = display_surveillance_metrics(surv_data)
        if videos > 0 or frames > 0 or detections > 0:
            has_real_data = True
            display_surveillance_overview_chart(videos, frames, detections)
            display_processing_insights(videos, frames, detections, confidence)

    if cached_data.get('analytics_data', {}).get("success", False):
        analytics_content = cached_data['analytics_data'].get("data", {})
        sections_displayed = 0
        
        # Debug: Show what analytics data we actually have
        st.write("🔍 **Debug - Analytics data structure:**", analytics_content)
        
        # Check if this is an empty analytics response (no processed data)
        total_documents = analytics_content.get("total_documents", 0)
        total_collections = analytics_content.get("total_collections", 0)
        total_videos = analytics_content.get("total_videos", 0)
        total_detections = analytics_content.get("total_detections", 0)
        is_light_mode = analytics_content.get("system_health", {}).get("mode") == "light"
        
        # Check if we have real analytics data (from heavy mode)
        has_heavy_analytics = any([
            total_videos > 0,
            total_detections > 0,
            analytics_content.get("object_counts"),
            analytics_content.get("timeline_data"),
            analytics_content.get("insights")
        ])
        
        # Only show "no data" message if we truly have no analytics data
        if total_documents == 0 and total_collections == 0 and not has_heavy_analytics:
            st.info("📊 **Analytics Status**: Backend is healthy but no processed video data found")
            st.markdown("""
            **Current State:**
            - ✅ Analytics endpoints are working
            - ✅ Database connections are healthy
            - ❌ No videos have been processed yet
            - 📊 Total documents in database: **0**
            
            **Next Steps:**
            1. 🎬 **Process some videos** - Upload and process videos first
            2. 🔄 **Return here** - Analytics will show real data after processing
            """)
            
            if is_light_mode:
                st.warning("⚡ **Light Mode Active** - Switch to Heavy Analytics for full AI insights after processing videos")
            
            if st.button("🎬 Go to Video Processing", type="primary"):
                st.switch_page("pages/1_📤_Video_Processing.py")
            
            # Don't process further analytics if no data
            st.stop()
        
        # If we have heavy analytics data, show a summary
        if has_heavy_analytics:
            st.success("🎉 **Real Analytics Data Found!** Heavy mode analytics loaded successfully")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎬 Videos", total_videos)
            with col2:
                st.metric("🔍 Detections", total_detections)
            with col3:
                total_frames = analytics_content.get("total_frames", 0)
                st.metric("🖼️ Frames", total_frames)
            with col4:
                total_projects = analytics_content.get("total_projects", 0)
                st.metric("📁 Projects", total_projects)
        
        # Try to extract real data from various possible structures
        real_analytics_data = {}
        
        # Check for direct data fields from heavy analytics
        for key in ['object_counts', 'detection_counts', 'timeline_data', 'total_videos', 'total_frames', 
                   'project_breakdown', 'insights', 'confidence_distribution', 'performance_metrics', 
                   'search_analytics', 'alerts', 'avg_confidence', 'avg_processing_time']:
            if key in analytics_content:
                real_analytics_data[key] = analytics_content[key]
        
        # Check for nested data structures
        if 'stats' in analytics_content:
            stats_data = analytics_content['stats']
            for key in ['object_counts', 'detection_counts', 'timeline_data', 'total_videos', 'total_frames']:
                if key in stats_data:
                    real_analytics_data[key] = stats_data[key]
        
        # Check for collection data
        if 'collection_stats' in analytics_content:
            collection_data = analytics_content['collection_stats']
            if isinstance(collection_data, list) and len(collection_data) > 0:
                # Process collection stats to extract meaningful data
                total_items = sum(col.get('count', 0) for col in collection_data if isinstance(col, dict))
                if total_items > 0:
                    real_analytics_data['total_items'] = total_items
            elif isinstance(collection_data, dict) and 'count' in collection_data:
                real_analytics_data['total_items'] = collection_data['count']
        
        # Use document count as a metric if available
        if total_documents > 0:
            real_analytics_data['total_items'] = total_documents
        
        # Display sections based on available data
        if real_analytics_data.get("object_counts") or real_analytics_data.get("detection_counts"):
            counts_data = real_analytics_data.get("object_counts", real_analytics_data.get("detection_counts"))
            display_detection_counts(counts_data)
            sections_displayed += 1
            
        if real_analytics_data.get("confidence_distribution"):
            display_confidence_distribution(real_analytics_data["confidence_distribution"])
            sections_displayed += 1
            
        if real_analytics_data.get("timeline_data"):
            display_detection_timeline(real_analytics_data["timeline_data"])
            sections_displayed += 1
            
        if real_analytics_data.get("performance_metrics"):
            display_performance_metrics(real_analytics_data["performance_metrics"])
            sections_displayed += 1
            
        if real_analytics_data.get("search_analytics"):
            display_search_analytics(real_analytics_data["search_analytics"])
            sections_displayed += 1
            
        # Show processing stats if we have meaningful data
        processing_data = {
            "total_videos": real_analytics_data.get("total_videos", total_videos),
            "total_frames": real_analytics_data.get("total_frames", analytics_content.get("total_frames", 0)),
            "total_detections": analytics_content.get("total_detections", 0),
            "total_projects": analytics_content.get("total_projects", 0),
            "total_items": real_analytics_data.get("total_items", 0),
            "total_documents": total_documents,
            "total_collections": total_collections,
            "avg_processing_time": real_analytics_data.get("avg_processing_time", analytics_content.get("avg_processing_time", 0)),
            "avg_confidence": real_analytics_data.get("avg_confidence", analytics_content.get("avg_confidence", 0)),
            "success_rate": real_analytics_data.get("success_rate", 0)
        }
        
        # If we have any processing data, display it
        if any(processing_data.values()):
            if processing_data.get("total_items") and not processing_data.get("total_videos"):
                # Use total_items as a proxy for processed content
                processing_data["total_videos"] = processing_data["total_items"]
            display_processing_statistics(processing_data)
            display_processing_efficiency(processing_data)
            sections_displayed += 1
            
        if real_analytics_data.get("project_breakdown"):
            display_project_breakdown(real_analytics_data["project_breakdown"])
            sections_displayed += 1
            
        if real_analytics_data.get("insights"):
            display_ai_insights(real_analytics_data["insights"])
            sections_displayed += 1
            
        if real_analytics_data.get("alerts"):
            display_alerts(real_analytics_data["alerts"])
            sections_displayed += 1
            
        if sections_displayed > 0:
            has_real_data = True
            st.success(f"📊 Displayed {sections_displayed} analytics sections")
        else:
            st.warning("📊 Analytics endpoint responded but no recognizable data structure found")
    
    # Show status if no real data found
    if not has_real_data:
        st.warning("⚠️ No real analytics data available")
        
        # Show detailed debug information
        with st.expander("🔍 API Response Details"):
            st.write("**Database Stats Response:**")
            st.json(cached_data.get('db_stats', {}))
            st.write("**Surveillance Stats Response:**")
            st.json(cached_data.get('surveillance_stats', {}))
            st.write("**Analytics Response:**")
            st.json(cached_data.get('analytics_data', {}))
        
        st.markdown("""
        **Possible reasons:**
        - No videos have been processed yet
        - Backend analytics endpoints are returning health data instead of analytics
        - Database doesn't contain processed video data
        
        **Solutions:**
        1. 🎬 **Process some videos first** - Go to "Video Processing" page and upload/process videos
        2. 📊 **Use Demo Mode** - See example analytics with sample data
        3. 🔧 **Check backend** - Ensure analytics endpoints return actual data, not just health status
        """)
        
        if st.button("🎬 Go to Video Processing", type="primary"):
            st.switch_page("pages/1_📤_Video_Processing.py")
