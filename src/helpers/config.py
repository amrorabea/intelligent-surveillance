from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Intelligent Surveillance System"
    APP_VERSION: str = "1.0.0"

    # File upload settings for surveillance files
    FILE_ALLOWED_TYPES: List[str] = [
        "mp4", 
        "avi", 
        "quicktime",  # .mov files
        "jpeg", 
        "jpg", 
        "png"
    ]
    FILE_MAX_SIZE: int = 500  # MB - larger for video files
    FILE_DEFAULT_CHUNK_SIZE: int = 1048576  # 1MB chunks for upload

    JWT_SECRET_KEY: str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()