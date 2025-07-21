# streamlit/pages/3_📊_Analytics.py
import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.api_client import SurveillanceAPIClient
except ImportError:
    st.error("Could not import API client. Please check the utils directory.")
    st.stop()

st.set_page_config(
    page_title="Analytics - Surveillance System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .alert-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = SurveillanceAPIClient()

def generate_mock_data():
    """Generate mock analytics data for demonstration"""
    return {
        "total_videos": 15,
        "total_frames": 25000,
        "total_detections": 3500,
        "avg_confidence": 0.78,
        "avg_processing_time": 45.2,
        "object_counts": {
            "person": 1200,
            "car": 850,
            "bicycle": 120,
            "motorcycle": 80,
            "truck": 200,
            "bus": 45,
            "dog": 30,
            "bird": 15
        },
        "confidence_distribution": {
            "0.5-0.6": 200,
            "0.6-0.7": 500,
            "0.7-0.8": 1200,
            "0.8-0.9": 1100,
            "0.9-1.0": 500
        },
        "timeline_data": [
            {"timestamp": "2024-01-01T00:00:00", "detections_count": 45},
            {"timestamp": "2024-01-01T01:00:00", "detections_count": 52},
            {"timestamp": "2024-01-01T02:00:00", "detections_count": 38},
            {"timestamp": "2024-01-01T03:00:00", "detections_count": 29},
            {"timestamp": "2024-01-01T04:00:00", "detections_count": 15},
            {"timestamp": "2024-01-01T05:00:00", "detections_count": 22},
        ],
        "performance_metrics": {
            "processing_speeds": [12.5, 14.2, 13.8, 15.1, 12.9, 14.5]
        },
        "search_analytics": {
            "total_searches": 89,
            "avg_results": 12.5,
            "avg_search_time": 0.34,
            "popular_terms": [
                {"query": "person walking", "count": 15},
                {"query": "car parking", "count": 12},
                {"query": "bicycle", "count": 8},
                {"query": "people gathering", "count": 6},
                {"query": "dog", "count": 4}
            ]
        },
        "insights": [
            {
                "title": "Peak Activity Hours",
                "description": "Most surveillance activity occurs between 9 AM and 5 PM",
                "confidence": 0.85
            },
            {
                "title": "High Person Detection Rate",
                "description": "Person detection accuracy has improved by 12% this week",
                "confidence": 0.92
            }
        ],
        "alerts": [
            {
                "type": "info",
                "title": "Processing Performance",
                "message": "All systems operating normally",
                "timestamp": "2024-01-01T12:00:00"
            }
        ]
    }

# Initialize session state
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

st.title("📊 Analytics Dashboard")
st.markdown("Comprehensive analytics and insights from your surveillance system")

# Header
st.markdown("""
<div class="analytics-header">
    <h3>🎯 Surveillance Analytics</h3>
    <p>Real-time insights, trends, and patterns from your AI-powered surveillance system</p>
</div>
""", unsafe_allow_html=True)

# Refresh controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 30 seconds)", value=False)

with col2:
    if st.button("🔄 Refresh Data", type="primary"):
        st.session_state.analytics_data = None
        st.session_state.last_refresh = None

with col3:
    if st.session_state.last_refresh:
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# Time range selector
st.markdown("### 📅 Analysis Period")
time_range = st.selectbox(
    "Select time range for analysis:",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time", "Custom range"]
)

if time_range == "Custom range":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
else:
    start_date = end_date = None

# Load analytics data
if st.session_state.analytics_data is None or auto_refresh:
    with st.spinner("📊 Loading analytics data..."):
        # Prepare time filter
        time_filter = {}
        if time_range == "Last 24 hours":
            time_filter["hours"] = 24
        elif time_range == "Last 7 days":
            time_filter["days"] = 7
        elif time_range == "Last 30 days":
            time_filter["days"] = 30
        elif time_range == "Custom range" and start_date and end_date:
            time_filter["start_date"] = start_date.isoformat()
            time_filter["end_date"] = end_date.isoformat()
        
        # Get analytics data from API
        result = api_client.get_analytics(**time_filter)
        
        if result.get("success", False):
            st.session_state.analytics_data = result.get("data", {})
            st.session_state.last_refresh = datetime.now()
        else:
            st.error(f"❌ Failed to load analytics: {result.get('error', 'Unknown error')}")
            # Use mock data for demonstration
            st.session_state.analytics_data = generate_mock_data()
            st.warning("⚠️ Using mock data for demonstration")

analytics = st.session_state.analytics_data

if analytics:
    # Key metrics overview
    st.markdown("### 🎯 Key Metrics")
    
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        total_videos = analytics.get("total_videos", 0)
        st.metric("🎬 Videos Processed", total_videos)
    
    with metric_cols[1]:
        total_frames = analytics.get("total_frames", 0)
        st.metric("🖼️ Frames Analyzed", f"{total_frames:,}")
    
    with metric_cols[2]:
        total_detections = analytics.get("total_detections", 0)
        st.metric("🎯 Objects Detected", f"{total_detections:,}")
    
    with metric_cols[3]:
        avg_confidence = analytics.get("avg_confidence", 0)
        st.metric("📈 Avg Confidence", f"{avg_confidence:.1%}")
    
    with metric_cols[4]:
        processing_time = analytics.get("avg_processing_time", 0)
        st.metric("⏱️ Avg Processing", f"{processing_time:.1f}s")
    
    # Charts and visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Object detection breakdown
        st.markdown("#### 🎯 Object Detection Breakdown")
        object_counts = analytics.get("object_counts", {})
        
        if object_counts:
            # Create pie chart
            labels = list(object_counts.keys())
            values = list(object_counts.values())
            
            fig_pie = px.pie(
                values=values,
                names=labels,
                title="Objects Detected by Type"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No object detection data available")
    
    with col2:
        # Detection confidence distribution
        st.markdown("#### 📊 Confidence Score Distribution")
        confidence_data = analytics.get("confidence_distribution", {})
        
        if confidence_data:
            confidence_ranges = list(confidence_data.keys())
            confidence_counts = list(confidence_data.values())
            
            fig_bar = px.bar(
                x=confidence_ranges,
                y=confidence_counts,
                title="Detection Confidence Distribution",
                labels={"x": "Confidence Range", "y": "Number of Detections"}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No confidence distribution data available")
    
    # Timeline analysis
    st.markdown("#### ⏰ Activity Timeline")
    
    timeline_data = analytics.get("timeline_data", [])
    if timeline_data:
        # Convert to DataFrame
        df_timeline = pd.DataFrame(timeline_data)
        
        if not df_timeline.empty and 'timestamp' in df_timeline.columns:
            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
            
            # Create timeline chart
            fig_timeline = px.line(
                df_timeline,
                x='timestamp',
                y='detections_count',
                title="Detection Activity Over Time",
                labels={"timestamp": "Time", "detections_count": "Detections per Hour"}
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No timeline data available")
    else:
        st.info("No timeline data available")
    
    # Detailed statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Performance Metrics")
        
        perf_metrics = analytics.get("performance_metrics", {})
        
        # Processing speed chart
        if "processing_speeds" in perf_metrics:
            speeds = perf_metrics["processing_speeds"]
            
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(
                x=list(range(len(speeds))),
                y=speeds,
                mode='lines+markers',
                name='Frames per Second',
                line=dict(color='#FF6B35')
            ))
            
            fig_speed.update_layout(
                title="Processing Speed Over Time",
                xaxis_title="Processing Job",
                yaxis_title="Frames per Second",
                height=300
            )
            st.plotly_chart(fig_speed, use_container_width=True)
        else:
            st.info("No performance metrics available")
    
    with col2:
        st.markdown("#### 🔍 Search Analytics")
        
        search_stats = analytics.get("search_analytics", {})
        
        if search_stats:
            st.metric("🔍 Total Searches", search_stats.get("total_searches", 0))
            st.metric("📊 Avg Results per Search", search_stats.get("avg_results", 0))
            st.metric("⚡ Avg Search Time", f"{search_stats.get('avg_search_time', 0):.2f}s")
            
            # Popular search terms
            popular_terms = search_stats.get("popular_terms", [])
            if popular_terms:
                st.markdown("**🔥 Popular Search Terms:**")
                for i, term in enumerate(popular_terms[:5], 1):
                    st.write(f"{i}. {term['query']} ({term['count']} searches)")
        else:
            st.info("No search analytics available")
    
    # Insights and alerts
    st.markdown("#### 💡 AI Insights")
    
    insights = analytics.get("insights", [])
    alerts = analytics.get("alerts", [])
    
    if insights or alerts:
        col1, col2 = st.columns(2)
        
        with col1:
            if insights:
                st.markdown("**🧠 Generated Insights:**")
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h5>{insight.get('title', 'Insight')}</h5>
                        <p>{insight.get('description', 'No description available')}</p>
                        <small>Confidence: {insight.get('confidence', 0):.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No insights available")
        
        with col2:
            if alerts:
                st.markdown("**⚠️ System Alerts:**")
                for alert in alerts:
                    alert_type = alert.get('type', 'info')
                    alert_color = {
                        'warning': '#ffc107',
                        'error': '#dc3545',
                        'info': '#17a2b8'
                    }.get(alert_type, '#17a2b8')
                    
                    st.markdown(f"""
                    <div class="alert-card" style="border-color: {alert_color};">
                        <h5>⚠️ {alert.get('title', 'Alert')}</h5>
                        <p>{alert.get('message', 'No message available')}</p>
                        <small>{alert.get('timestamp', 'Unknown time')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts")
    else:
        st.info("No insights or alerts available")
    
    # Raw data export
    with st.expander("📄 Export Analytics Data"):
        st.markdown("**Download analytics data for external analysis**")
        
        export_format = st.selectbox("Export format:", ["JSON", "CSV"])
        
        if st.button("💾 Download Data"):
            if export_format == "JSON":
                data_str = json.dumps(analytics, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=data_str,
                    file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:  # CSV
                # Convert key metrics to CSV format
                csv_data = []
                for key, value in analytics.items():
                    if isinstance(value, (int, float, str)):
                        csv_data.append({"metric": key, "value": value})
                
                if csv_data:
                    df_export = pd.DataFrame(csv_data)
                    csv_str = df_export.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_str,
                        file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No data available for CSV export")

else:
    st.error("❌ No analytics data available")

# Help section
with st.expander("❓ Analytics Help"):
    st.markdown("""
    ### 📊 Understanding Your Analytics
    
    **Key Metrics:**
    - **Videos Processed**: Total number of surveillance videos analyzed
    - **Frames Analyzed**: Total video frames processed by AI models
    - **Objects Detected**: Total number of objects found across all videos
    - **Avg Confidence**: Average confidence score of object detections
    - **Avg Processing**: Average time to process each video
    
    **Charts Explained:**
    - **Object Detection Breakdown**: Shows the distribution of different object types
    - **Confidence Distribution**: Shows how confident the AI was in its detections
    - **Activity Timeline**: Shows detection activity over time
    - **Performance Metrics**: Shows processing speed and efficiency
    
    **Insights & Alerts:**
    - **AI Insights**: Automatically generated observations about your data
    - **System Alerts**: Important notifications about system performance
    
    **Tips for Better Analytics:**
    - Process more videos to get richer insights
    - Use consistent time periods for comparison
    - Monitor processing speed to optimize performance
    - Pay attention to confidence scores for quality assessment
    """)

# Sidebar info
with st.sidebar:
    st.header("📊 Analytics")
    st.info("""
    Comprehensive insights from your surveillance system.
    
    **Features:**
    - Real-time metrics
    - Object detection analytics
    - Performance monitoring
    - Search analytics
    - AI-generated insights
    """)
    
    # Quick actions
    st.header("🔧 Quick Actions")
    
    if st.button("📈 Performance Report"):
        st.info("Feature coming soon!")
    
    if st.button("📧 Email Report"):
        st.info("Feature coming soon!")
    
    if st.button("⚙️ Configure Alerts"):
        st.info("Feature coming soon!")
    
    # Auto-refresh timer
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
