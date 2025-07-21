from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import secrets
import os
from helpers.config import get_settings

settings = get_settings()

# Security configuration
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class AuthManager:
    """Handles authentication and authorization"""
    
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def create_api_key(self, user_id: str, name: str) -> str:
        """Create an API key for a user"""
        # Generate a secure random key
        api_key = secrets.token_urlsafe(32)
        
        # In production, you'd store this in the database with user_id and name
        # For now, this is a placeholder
        return f"api_{api_key}"
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify an API key"""
        # TODO: In production, verify against database
        # For now, accept any key that starts with "api_"
        if api_key.startswith("api_"):
            return {
                "user_id": "demo_user",
                "permissions": ["read", "write"],
                "rate_limit": 1000
            }
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

# Global auth manager
auth_manager = AuthManager()

# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get current authenticated user
    Supports both JWT tokens and API keys
    """
    # DEVELOPMENT MODE: Allow bypass with 'dev' token
    token = credentials.credentials
    
    if token == "dev":
        return {
            "user_id": "dev_user",
            "username": "developer",
            "role": "admin",
            "projects": ["*"]  # Access to all projects
        }
    
    # Check if it's an API key
    if token.startswith("api_"):
        return auth_manager.verify_api_key(token)
    
    # Otherwise, treat as JWT token
    payload = auth_manager.verify_token(token)
    return payload

async def get_optional_user(request: Request):
    """Optional authentication - doesn't fail if no auth provided"""
    try:
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
            
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
            
        if token.startswith("api_"):
            return auth_manager.verify_api_key(token)
        else:
            return auth_manager.verify_token(token)
    except Exception:
        return None

# Rate limiting
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
        
    def is_allowed(self, key: str, limit: int, window: int = 3600) -> bool:
        """
        Check if request is allowed based on rate limit
        
        Args:
            key: Unique identifier (user_id, ip_address, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds (default: 1 hour)
        """
        now = datetime.utcnow().timestamp()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if now - req_time < window
        ]
        
        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False

# Global rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(request: Request, user: dict = Depends(get_optional_user)):
    """Rate limiting middleware"""
    # Get identifier for rate limiting
    if user:
        identifier = user.get("user_id", "anonymous")
        limit = user.get("rate_limit", 100)
    else:
        # Use IP address for anonymous users
        identifier = request.client.host
        limit = 20  # Lower limit for anonymous users
    
    if not rate_limiter.is_allowed(identifier, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Project access control
async def verify_project_access(project_id: str, user: dict = Depends(get_current_user)):
    """Verify user has access to a specific project"""
    # TODO: In production, check project permissions in database
    # For now, allow all authenticated users access to all projects
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User ID not found in token"
        )
    
    return True

# Input validation and sanitization
def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames"""
    import re
    
    # Remove path separators and other dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def validate_file_size(file_size: int, max_size_mb: int = 500) -> bool:
    """Validate uploaded file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def validate_file_type(filename: str, allowed_types: list) -> bool:
    """Validate file type by extension"""
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in allowed_types
