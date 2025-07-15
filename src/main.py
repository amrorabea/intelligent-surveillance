from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import time
import logging
import uuid
from contextlib import asynccontextmanager

from routes import base, data, surveillance
from models.database import db_manager
from helpers.config import get_settings

from dotenv import load_dotenv
load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Intelligent Surveillance System...")
    
    # Initialize database
    try:
        db_manager.create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # TODO: Initialize AI models here if you want them loaded at startup
    # This could include loading YOLO and BLIP models
    
    logger.info("Application startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down Intelligent Surveillance System...")

app = FastAPI(
    title="Intelligent Surveillance System",
    description="""
    A modular, real-time surveillance platform that processes video footage using AI 
    to enable semantic understanding of visual events.
    
    ## Features
    
    * **Video Processing**: Upload and process surveillance videos through AI pipeline
    * **Object Detection**: Detect people, objects, and activities using YOLOv8
    * **Scene Captioning**: Generate natural language descriptions of video frames
    * **Semantic Search**: Query footage using natural language
    * **Real-time Processing**: Background job processing for large videos
    * **Analytics**: Get insights and statistics from processed footage
    
    ## API Usage
    
    1. Upload videos using `/api/data/upload/{project_id}`
    2. Process videos with `/api/surveillance/process/{project_id}/{file_id}`
    3. Query footage with `/api/surveillance/query`
    4. Monitor processing jobs with `/api/surveillance/jobs/{project_id}`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request ID middleware for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled exception in request {request_id}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not found",
            "detail": f"Path {request.url.path} not found",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Include routers
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(surveillance.surveillance_router)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Intelligent Surveillance API",
        version="1.0.0",
        description="API for AI-powered surveillance video processing and querying",
        routes=app.routes,
    )
    
    # Add authentication scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API_KEY"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with system information"""
    settings = get_settings()
    return {
        "message": "Welcome to the Intelligent Surveillance System",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/surveillance/health",
        "status": "running"
    }

# Health check endpoint at root level
@app.get("/health", tags=["system"])
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "intelligent-surveillance"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )