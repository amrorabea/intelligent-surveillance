# 🔒 Security Guide

This guide covers security best practices, authentication, authorization, and protection mechanisms for the Intelligent Surveillance System.

## 🛡️ Security Overview

### Security Principles
- **Defense in Depth**: Multiple layers of security controls
- **Least Privilege**: Minimal access rights for users and services
- **Zero Trust**: Never trust, always verify
- **Security by Design**: Built-in security from the ground up
- **Data Protection**: Encryption at rest and in transit

### Threat Model
- **Data Breaches**: Unauthorized access to video data
- **Code Injection**: SQL injection, XSS attacks
- **Denial of Service**: Resource exhaustion attacks
- **Privilege Escalation**: Unauthorized system access
- **Data Tampering**: Modification of processing results

## 🔐 Authentication & Authorization

### JWT-Based Authentication

#### Enable Authentication
```python
# src/helpers/config.py
class Settings(BaseSettings):
    # Authentication
    enable_auth: bool = True
    jwt_secret: str = "your-super-secret-jwt-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    
    # Password hashing
    password_hash_algorithm: str = "bcrypt"
    password_salt_rounds: int = 12
```

#### Authentication Implementation
```python
# src/services/auth.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
import bcrypt
from fastapi import HTTPException, status
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    roles: List[str] = []
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCredentials(BaseModel):
    username: str
    password: str

class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256"):
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.users_db = {}  # In production, use proper database
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, username: str, password: str, email: str, roles: List[str] = None) -> User:
        """Create new user"""
        if username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        hashed_password = self.hash_password(password)
        user = User(
            username=username,
            email=email,
            roles=roles or ["user"],
            created_at=datetime.utcnow()
        )
        
        self.users_db[username] = {
            "user": user,
            "password_hash": hashed_password
        }
        
        return user
    
    def authenticate_user(self, credentials: UserCredentials) -> Optional[User]:
        """Authenticate user with credentials"""
        user_data = self.users_db.get(credentials.username)
        if not user_data:
            return None
        
        if not self.verify_password(credentials.password, user_data["password_hash"]):
            return None
        
        user = user_data["user"]
        if not user.is_active:
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        return user
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode = {
            "sub": user.username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "roles": user.roles,
            "email": user.email
        }
        
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def get_current_user(self, token: str) -> User:
        """Get current user from token"""
        payload = self.verify_token(token)
        username = payload.get("sub")
        user_data = self.users_db.get(username)
        
        if user_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user_data["user"]
```

#### FastAPI Integration
```python
# src/dependencies/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.services.auth import AuthService, User
from src.helpers.config import settings

security = HTTPBearer()
auth_service = AuthService(settings.jwt_secret)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency to get current authenticated user"""
    if not settings.enable_auth:
        # Return default user when auth is disabled
        return User(
            username="anonymous",
            email="anonymous@example.com",
            roles=["admin"],
            created_at=datetime.utcnow()
        )
    
    return auth_service.get_current_user(credentials.credentials)

def require_roles(required_roles: List[str]):
    """Dependency factory for role-based access control"""
    def check_roles(current_user: User = Depends(get_current_user)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return check_roles

# Usage in routes
@router.post("/admin/users")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_roles(["admin"]))
):
    """Admin-only endpoint"""
    pass
```

#### Authentication Routes
```python
# src/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from src.services.auth import AuthService, UserCredentials, User
from src.dependencies.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint"""
    credentials = UserCredentials(
        username=form_data.username,
        password=form_data.password
    )
    
    user = auth_service.authenticate_user(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_service.create_access_token(user)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {
            "username": user.username,
            "email": user.email,
            "roles": user.roles
        }
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """User logout endpoint"""
    # In a real implementation, you might blacklist the token
    return {"message": "Successfully logged out"}

@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    # Verify old password
    user_data = auth_service.users_db.get(current_user.username)
    if not auth_service.verify_password(old_password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    new_hash = auth_service.hash_password(new_password)
    user_data["password_hash"] = new_hash
    
    return {"message": "Password updated successfully"}
```

## 🛡️ Input Validation & Sanitization

### Request Validation
```python
# src/models/schemas.py
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re

class VideoUploadRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('project_id')
    def validate_project_id(cls, v):
        # Only allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid project ID format')
        return v.lower()
    
    @validator('description')
    def validate_description(cls, v):
        if v:
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\']', '', v)
            return sanitized.strip()
        return v

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    project_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @validator('query')
    def validate_query(cls, v):
        # Basic SQL injection prevention
        dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER']
        query_upper = v.upper()
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                raise ValueError('Invalid query content')
        return v.strip()
```

### File Upload Security
```python
# src/helpers/file_security.py
import os
import magic
from typing import List, Optional
from fastapi import UploadFile, HTTPException

class FileValidator:
    """Secure file upload validation"""
    
    ALLOWED_VIDEO_TYPES = [
        'video/mp4',
        'video/avi', 
        'video/mov',
        'video/mkv',
        'video/webm'
    ]
    
    ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    @staticmethod
    def validate_file_type(file: UploadFile) -> bool:
        """Validate file MIME type"""
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in FileValidator.ALLOWED_EXTENSIONS:
            return False
        
        # Check MIME type using python-magic
        file_content = file.file.read(2048)  # Read first 2KB
        file.file.seek(0)  # Reset file pointer
        
        detected_type = magic.from_buffer(file_content, mime=True)
        return detected_type in FileValidator.ALLOWED_VIDEO_TYPES
    
    @staticmethod
    def validate_file_size(file: UploadFile) -> bool:
        """Validate file size"""
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        return size <= FileValidator.MAX_FILE_SIZE
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize uploaded filename"""
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        return f"{name}{ext}"
    
    @staticmethod
    def validate_upload(file: UploadFile) -> str:
        """Complete file validation"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not FileValidator.validate_file_type(file):
            raise HTTPException(status_code=415, detail="Unsupported file type")
        
        if not FileValidator.validate_file_size(file):
            raise HTTPException(status_code=413, detail="File too large")
        
        return FileValidator.sanitize_filename(file.filename)
```

## 🔐 Data Protection

### Encryption at Rest
```python
# src/helpers/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    """Handle data encryption and decryption"""
    
    def __init__(self, password: str):
        self.key = self._derive_key(password)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        salt = os.urandom(16)  # In production, store this salt securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt entire file"""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)
    
    def decrypt_file(self, encrypted_path: str, output_path: str):
        """Decrypt entire file"""
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)
```

### Database Security
```python
# src/models/database_security.py
import hashlib
import secrets
from typing import Dict, Any

class DatabaseSecurity:
    """Database security utilities"""
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str = None) -> Dict[str, str]:
        """Hash sensitive data with salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}"
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        
        return {
            "hash": hashed,
            "salt": salt
        }
    
    @staticmethod
    def verify_hash(data: str, stored_hash: str, salt: str) -> bool:
        """Verify hashed data"""
        result = DatabaseSecurity.hash_sensitive_data(data, salt)
        return result["hash"] == stored_hash
    
    @staticmethod
    def anonymize_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize user data for analytics"""
        anonymized = user_data.copy()
        
        # Hash or remove PII
        if 'email' in anonymized:
            anonymized['email_hash'] = hashlib.sha256(
                anonymized['email'].encode()
            ).hexdigest()[:8]
            del anonymized['email']
        
        if 'ip_address' in anonymized:
            # Mask last octet of IP
            ip_parts = anonymized['ip_address'].split('.')
            if len(ip_parts) == 4:
                anonymized['ip_subnet'] = f"{'.'.join(ip_parts[:3])}.0"
            del anonymized['ip_address']
        
        return anonymized
```

## 🌐 Network Security

### HTTPS Configuration
```python
# src/helpers/ssl_config.py
import ssl
from fastapi import FastAPI
import uvicorn

def configure_ssl(app: FastAPI, cert_file: str, key_file: str):
    """Configure SSL/TLS for production"""
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    
    # Security configurations
    ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    ssl_context.options |= ssl.OP_NO_SSLv2
    ssl_context.options |= ssl.OP_NO_SSLv3
    ssl_context.options |= ssl.OP_NO_TLSv1
    ssl_context.options |= ssl.OP_NO_TLSv1_1
    
    return ssl_context

def run_secure_server(app: FastAPI, host: str = "0.0.0.0", port: int = 443):
    """Run server with SSL"""
    ssl_context = configure_ssl(app, "cert.pem", "key.pem")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_context=ssl_context,
        ssl_ciphers="ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM"
    )
```

### Rate Limiting
```python
# src/middleware/rate_limiting.py
import time
import asyncio
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        while (self.clients[client_ip] and 
               current_time - self.clients[client_ip][0] > self.period):
            self.clients[client_ip].popleft()
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.period)}
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        response = await call_next(request)
        return response

# Advanced rate limiting with different limits per endpoint
class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with endpoint-specific limits"""
    
    def __init__(self, app):
        super().__init__(app)
        self.limits = {
            "/surveillance/upload": {"calls": 5, "period": 60},
            "/surveillance/search": {"calls": 60, "period": 60},
            "/auth/login": {"calls": 10, "period": 300},  # 5 minutes
        }
        self.clients = defaultdict(lambda: defaultdict(deque))
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        endpoint = request.url.path
        
        # Get rate limit for endpoint
        limit_config = self.limits.get(endpoint, {"calls": 100, "period": 60})
        
        current_time = time.time()
        client_endpoint_requests = self.clients[client_ip][endpoint]
        
        # Clean old requests
        while (client_endpoint_requests and 
               current_time - client_endpoint_requests[0] > limit_config["period"]):
            client_endpoint_requests.popleft()
        
        # Check rate limit
        if len(client_endpoint_requests) >= limit_config["calls"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {endpoint}",
                headers={"Retry-After": str(limit_config["period"])}
            )
        
        # Add current request
        client_endpoint_requests.append(current_time)
        
        response = await call_next(request)
        return response
```

### CORS Security
```python
# src/middleware/cors_security.py
from fastapi.middleware.cors import CORSMiddleware
from src.helpers.config import settings

def configure_cors(app):
    """Configure CORS with security considerations"""
    
    # Production CORS settings
    if settings.env == "production":
        allowed_origins = [
            "https://yourdomain.com",
            "https://app.yourdomain.com",
        ]
    else:
        # Development settings
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )
```

## 🔍 Security Monitoring

### Audit Logging
```python
# src/middleware/audit_logging.py
import json
import logging
from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

audit_logger = logging.getLogger("audit")

class AuditMiddleware(BaseHTTPMiddleware):
    """Log security-relevant events"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        
        # Extract user info
        user_id = "anonymous"
        if hasattr(request.state, 'user'):
            user_id = request.state.user.username
        
        # Sensitive endpoints to monitor
        sensitive_endpoints = [
            "/surveillance/upload",
            "/auth/login",
            "/admin/",
        ]
        
        is_sensitive = any(request.url.path.startswith(ep) for ep in sensitive_endpoints)
        
        try:
            response = await call_next(request)
            
            if is_sensitive or response.status_code >= 400:
                audit_log = {
                    "timestamp": start_time.isoformat(),
                    "user_id": user_id,
                    "client_ip": request.client.host,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "user_agent": request.headers.get("user-agent", ""),
                    "is_sensitive": is_sensitive,
                }
                
                audit_logger.info(json.dumps(audit_log))
            
            return response
            
        except Exception as e:
            # Log exceptions
            audit_log = {
                "timestamp": start_time.isoformat(),
                "user_id": user_id,
                "client_ip": request.client.host,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "is_error": True,
            }
            
            audit_logger.error(json.dumps(audit_log))
            raise
```

### Intrusion Detection
```python
# src/security/intrusion_detection.py
import re
import time
from collections import defaultdict
from typing import Dict, List, Pattern

class IntrusionDetector:
    """Detect potential security threats"""
    
    def __init__(self):
        self.suspicious_patterns = [
            re.compile(r'union\s+select', re.IGNORECASE),
            re.compile(r'<script[^>]*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'../../', re.IGNORECASE),
            re.compile(r'cmd\.exe', re.IGNORECASE),
            re.compile(r'/etc/passwd', re.IGNORECASE),
        ]
        
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
    
    def check_request(self, request_data: str, client_ip: str) -> Dict[str, any]:
        """Check request for suspicious patterns"""
        threats_found = []
        
        for pattern in self.suspicious_patterns:
            if pattern.search(request_data):
                threats_found.append(pattern.pattern)
        
        if threats_found:
            self.record_threat(client_ip, threats_found)
            return {
                "is_threat": True,
                "threats": threats_found,
                "action": "block" if client_ip in self.blocked_ips else "monitor"
            }
        
        return {"is_threat": False}
    
    def record_threat(self, client_ip: str, threats: List[str]):
        """Record security threat"""
        current_time = time.time()
        self.failed_attempts[client_ip].append({
            "timestamp": current_time,
            "threats": threats
        })
        
        # Clean old attempts (older than 1 hour)
        cutoff = current_time - 3600
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt["timestamp"] > cutoff
        ]
        
        # Block IP if too many threats
        if len(self.failed_attempts[client_ip]) >= 5:
            self.blocked_ips.add(client_ip)
            logging.warning(f"Blocked IP {client_ip} due to repeated threats")
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked"""
        return client_ip in self.blocked_ips
    
    def unblock_ip(self, client_ip: str):
        """Unblock IP address"""
        self.blocked_ips.discard(client_ip)
        if client_ip in self.failed_attempts:
            del self.failed_attempts[client_ip]
```

## 🔧 Security Headers

### Security Headers Middleware
```python
# src/middleware/security_headers.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "media-src 'self'; "
            "frame-ancestors 'none';"
        )
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        
        # Remove server header for security
        response.headers.pop("Server", None)
        
        return response
```

## 🛡️ Production Security Checklist

### Deployment Security
- [ ] **HTTPS Enabled**: SSL/TLS certificates configured
- [ ] **Strong Passwords**: All default passwords changed
- [ ] **Firewall Rules**: Only necessary ports open
- [ ] **Regular Updates**: OS and dependencies updated
- [ ] **Backup Encryption**: Backups are encrypted
- [ ] **Log Monitoring**: Security logs monitored
- [ ] **Access Control**: Principle of least privilege applied
- [ ] **Network Segmentation**: Services properly segmented

### Application Security
- [ ] **Authentication**: Strong authentication implemented
- [ ] **Authorization**: Role-based access control
- [ ] **Input Validation**: All inputs validated and sanitized
- [ ] **SQL Injection Prevention**: Parameterized queries used
- [ ] **XSS Prevention**: Output encoding implemented
- [ ] **CSRF Protection**: CSRF tokens implemented
- [ ] **File Upload Security**: File types and sizes validated
- [ ] **Rate Limiting**: DDoS protection in place

### Data Security
- [ ] **Encryption at Rest**: Sensitive data encrypted
- [ ] **Encryption in Transit**: All communications encrypted
- [ ] **Key Management**: Encryption keys properly managed
- [ ] **Data Minimization**: Only necessary data collected
- [ ] **Data Retention**: Data retention policies implemented
- [ ] **Anonymization**: PII properly anonymized
- [ ] **Backup Security**: Backups encrypted and tested
- [ ] **Audit Logging**: All access logged and monitored

### Infrastructure Security
- [ ] **Container Security**: Containers scanned for vulnerabilities
- [ ] **Network Security**: Proper network segmentation
- [ ] **Monitoring**: Security monitoring implemented
- [ ] **Incident Response**: Response plan in place
- [ ] **Vulnerability Management**: Regular security scans
- [ ] **Compliance**: Relevant compliance requirements met
- [ ] **Documentation**: Security documentation updated
- [ ] **Training**: Team trained on security practices

---

**🔒 Security is everyone's responsibility. Regular security assessments and staying updated with the latest threats are crucial for maintaining a secure system.**
