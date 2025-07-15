from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Intelligent Surveillance System"
    APP_VERSION: str = "1.0.0"

    # File upload settings for surveillance files
    FILE_ALLOWED_TYPES: List[str] = [
        "video/mp4",
        "video/avi", 
        "video/x-msvideo",  # Alternative AVI MIME type
        "video/quicktime",  # .mov files
        "video/webm",
        "video/mkv",
        "image/jpeg", 
        "image/jpg", 
        "image/png"
    ]
    FILE_MAX_SIZE: int = 500  # MB - larger for video files
    FILE_DEFAULT_CHUNK_SIZE: int = 1048576  # 1MB chunks for upload

    JWT_SECRET_KEY: str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()