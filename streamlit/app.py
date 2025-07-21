# streamlit/app.py
import streamlit as st
import sys
import os

# Add src to path for importing your controllers if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="🔍 Intelligent Surveillance System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍 Intelligent Surveillance System</h1>
    <p>AI-powered video analysis with YOLOv8, BLIP, and advanced object tracking</p>
</div>
""", unsafe_allow_html=True)

# Welcome section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🚀 Welcome to Your AI Surveillance Platform")
    
    st.markdown("""
    This system combines cutting-edge AI technologies to provide comprehensive video surveillance analysis:
    
    **🧠 Powered by State-of-the-Art AI:**
    - **YOLOv8**: Real-time object detection with 80+ object classes
    - **BLIP**: Advanced image captioning for scene understanding
    - **Custom Tracking**: IoU + distance-based multi-object tracking
    - **Vector Search**: Semantic search using ChromaDB and sentence transformers
    """)
    
    st.markdown("### 📋 Quick Start Guide:")
    st.markdown("""
    1. **📤 Upload Video**: Go to Video Processing to upload surveillance footage
    2. **🔄 AI Analysis**: Let our AI models analyze objects, generate captions, and track movement
    3. **🔍 Smart Search**: Use natural language to search through processed footage
    4. **📊 View Analytics**: Check detection statistics and insights
    """)

with col2:
    st.markdown("### 🎯 System Features")
    
    features = [
        ("🎬", "Video Processing", "Upload and analyze surveillance footage"),
        ("🔍", "Semantic Search", "Natural language queries over footage"),
        ("📊", "Analytics Dashboard", "Detection statistics and insights"),
        ("🔁", "Object Tracking", "Multi-object tracking across frames"),
        ("⚙️", "System Status", "Backend health monitoring")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{icon} {title}</h4>
            <p style="margin: 0; color: #666;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Technology stack
st.markdown("---")
st.markdown("## 🛠️ Technology Stack")

tech_cols = st.columns(4)
with tech_cols[0]:
    st.markdown("""
    **🧠 AI Models**
    - YOLOv8 (Object Detection)
    - BLIP (Image Captioning)
    - Sentence Transformers
    """)

with tech_cols[1]:
    st.markdown("""
    **⚙️ Backend**
    - FastAPI (REST API)
    - Celery (Background Jobs)
    - Redis (Message Broker)
    """)

with tech_cols[2]:
    st.markdown("""
    **🗄️ Data Storage**
    - SQLite (Metadata)
    - ChromaDB (Vector Search)
    - File System (Media)
    """)

with tech_cols[3]:
    st.markdown("""
    **🎨 Frontend**
    - Streamlit (UI)
    - Python (Integration)
    - Responsive Design
    """)

# Main page content
st.title("🔍 Intelligent Surveillance System")
st.markdown("""
Welcome to the **Intelligent Surveillance System** powered by AI!

### 🧠 Key Features:
- 🎬 **Video Processing**: Upload and analyze surveillance footage with YOLOv8 + BLIP
- 🔍 **Semantic Search**: Query footage using natural language
- 📊 **Analytics Dashboard**: View detection statistics and insights
- 🔁 **Object Tracking**: Advanced multi-object tracking capabilities

### 📋 Quick Start:
1. Navigate to **Video Processing** to upload and analyze footage
2. Use **Semantic Search** to find specific events or objects
3. Check **Analytics** for detailed insights
4. Monitor **System Status** for backend health

---
*Select a page from the sidebar to get started!*
""")

# Sidebar info
with st.sidebar:
    st.header("🎯 Navigation")
    st.info("""
    **Video Processing**: Upload videos for AI analysis
    
    **Semantic Search**: Search footage with natural language
    
    **Analytics**: View detection statistics
    
    **System Status**: Backend health monitoring
    """)
    
    st.header("🔧 System Info")
    st.write("Backend API:", "http://localhost:5000")
    st.write("Models:", "YOLOv8 + BLIP")
    st.write("Tracking:", "IoU + Distance-based")