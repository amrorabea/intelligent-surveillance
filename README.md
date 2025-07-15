# Intelligent Surveillance System

This project is a modular, real-time surveillance platform that processes video footage using AI to enable semantic understanding of visual events. Users can query stored CCTV footage using natural language.

## 🧠 Key Features

- 🔍 **YOLOv8 Object Detection**: Detects people, bags, vehicles, and more from live or recorded video.
- 🔁 **Object Tracking**: Maintains identity across frames.
- 📝 **Scene Captioning with BLIP**: Generates rich natural language descriptions for selected keyframes.
- 🧠 **Semantic Search Interface**: Allows free-form natural language queries over stored footage.
- ⚙️ **Modular FastAPI Backend**: Provides clean APIs to analyze video, search events, or extract keyframes.
- 🖼️ **Frame Extraction Engine**: Extracts relevant image frames for processing and future lookup.
- 🧠 **Vector Database Integration**: Embeds event captions into a vector DB for high-speed similarity search.

## Requirements

- Python 3.10 or later
- OpenCV
- PyTorch
- FastAPI
- YOLOv8
- BLIP (Bootstrapped Language-Image Pre-training)
- ChromaDB

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/intelligent-surveillance.git
cd intelligent-surveillance/src
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run the FastAPI server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## API Endpoints

- **POST /api/data/upload/{project_id}** - Upload video or image files
- **POST /api/surveillance/process/{project_id}/{file_id}** - Process a video through the AI pipeline
- **GET /api/surveillance/query** - Search footage with natural language queries
- **GET /api/surveillance/frame/{result_id}** - Get a specific frame from query results

## Usage Examples

Query examples:
- "Who entered the hallway carrying a backpack?"
- "Show me people wearing red after midnight."
- "Was anyone near the car for more than 10 seconds?"

## License

This project is licensed under the terms of the MIT license.
