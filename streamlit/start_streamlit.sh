#!/bin/bash
# streamlit/start_streamlit.sh

# Start the Streamlit surveillance frontend
echo "🚀 Starting Intelligent Surveillance System Frontend..."

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Install dependencies
echo "📥 Installing Streamlit dependencies..."
pip install -r requirements.txt

# Start Streamlit
echo "🎯 Starting Streamlit on http://localhost:8501"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
