from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import hmac

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests"""
    
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or [
            "/docs", "/redoc", "/openapi.json", "/health", 
            "/auth/login", "/auth/register", "/", "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request"""
        
        # Skip authentication for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            if self.validate_api_key(api_key):
                # Add API key info to request state
                request.state.auth_type = "api_key"
                request.state.api_key = api_key
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"}
                )
        
        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                scheme, token = auth_header.split()
                if scheme.lower() != "bearer":
                    raise ValueError("Invalid authentication scheme")
                
                payload = self.decode_jwt_token(token)
                if payload:
                    # Add user info to request state
                    request.state.auth_type = "jwt"
                    request.state.user = payload
                    return await call_next(request)
                else:
                    raise ValueError("Invalid token")
                    
            except Exception:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )
        
        # No valid authentication found
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"}
        )
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        
        # In production, check against database
        # For now, check against environment variable
        valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
        return api_key in valid_api_keys
    
    def decode_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None

class JWTManager:
    """JWT token management utilities"""
    
    @staticmethod
    def create_access_token(user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        
        payload = {
            "user_id": user_data.get("user_id"),
            "username": user_data.get("username"),
            "role": user_data.get("role", "user"),
            "permissions": user_data.get("permissions", []),
            "iat": datetime.utcnow().timestamp(),
            "exp": (datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)).timestamp()
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """Decode JWT token"""
        
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def refresh_token(token: str) -> Optional[str]:
        """Refresh JWT token"""
        
        payload = JWTManager.decode_token(token)
        if not payload:
            return None
        
        # Create new token with same user data
        user_data = {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "role": payload.get("role"),
            "permissions": payload.get("permissions", [])
        }
        
        return JWTManager.create_access_token(user_data)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    
    try:
        payload = JWTManager.decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return payload
        
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def require_permission(required_permission: str):
    """Decorator to require specific permission"""
    
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        user_role = current_user.get("role", "")
        
        if required_permission not in user_permissions and user_role != "admin":
            raise HTTPException(
                status_code=403, 
                detail=f"Permission required: {required_permission}"
            )
        
        return current_user
    
    return permission_checker

def require_role(required_role: str):
    """Decorator to require specific role"""
    
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "")
        
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Role required: {required_role}"
            )
        
        return current_user
    
    return role_checker
