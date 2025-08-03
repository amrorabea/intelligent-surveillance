from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from contextlib import asynccontextmanager

from routes import base, data, surveillance
from models.database import db_manager

# Import AI controllers
from controllers.VisionController import VisionController
from controllers.VectorDBController import VectorDBController

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("🚀 Starting Intelligent Surveillance System...")
    
    # Initialize database
    try:
        db_manager.create_tables()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
    
    # Initialize AI models
    try:
        vision_controller = VisionController()
        vector_db_controller = VectorDBController()
        
        app.state.vision_controller = vision_controller
        app.state.vector_db_controller = vector_db_controller
        
        logger.info("✅ AI models ready")
    except Exception as e:
        logger.warning(f"⚠️  AI models will load on-demand: {e}")
        app.state.vision_controller = None
        app.state.vector_db_controller = None
    
    logger.info("✅ System ready")
    yield
    
    logger.info("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Intelligent Surveillance System",
    description="AI-powered surveillance video processing and querying",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(surveillance.surveillance_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Surveillance System",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
