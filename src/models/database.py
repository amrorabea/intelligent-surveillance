from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class ProcessedVideo(Base):
    __tablename__ = 'processed_videos'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(String(255), nullable=False)
    project_id = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(500))
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    total_frames = Column(Integer)
    processed_frames = Column(Integer, default=0)
    sample_rate = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
class DetectedObject(Base):
    __tablename__ = 'detected_objects'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, nullable=False)  # Foreign key to ProcessedVideo
    frame_timestamp = Column(Float, nullable=False)
    frame_path = Column(String(500))
    object_class = Column(String(100))
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    tracking_id = Column(Integer)  # For object tracking across frames
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessingJob(Base):
    __tablename__ = 'processing_jobs'
    
    id = Column(String(36), primary_key=True)  # UUID
    project_id = Column(String(255), nullable=False)
    file_id = Column(String(255), nullable=False)
    job_type = Column(String(50), nullable=False)  # video_processing, live_stream, etc.
    status = Column(String(50), default='queued')  # queued, running, completed, failed, cancelled
    progress = Column(Integer, default=0)  # 0-100
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    metadata = Column(JSON)  # Additional job-specific data
    created_at = Column(DateTime, default=datetime.utcnow)

class FrameCaption(Base):
    __tablename__ = 'frame_captions'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, nullable=False)
    frame_timestamp = Column(Float, nullable=False)
    frame_path = Column(String(500))
    caption = Column(Text)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Database connection and session management
class DatabaseManager:
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default to SQLite for development
            db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{os.path.join(db_dir, 'surveillance.db')}"
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
        
    def close_session(self, session):
        """Close a database session"""
        session.close()

# Global database manager instance
db_manager = DatabaseManager()
