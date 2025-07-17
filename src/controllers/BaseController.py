try:
    # Try absolute imports first (for Celery running from project root)
    from src.helpers.config import get_settings, Settings
except ImportError:
    # Fall back to relative imports (for FastAPI running from src/)
    from helpers.config import get_settings, Settings

import os
import random
import string

class BaseController:
    
    def __init__(self):

        self.app_settings = get_settings()
        
        self.base_dir = os.path.dirname( os.path.dirname(__file__) )
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
        )
        
    def generate_random_string(self, length: int=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))