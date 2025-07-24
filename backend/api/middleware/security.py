from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import hashlib
import hmac
import time
from typing import Set, Dict, Any
import os
import re

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for additional protection"""
    
    def __init__(self, app):
        super().__init__(app)
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS attempts
            r'union\s+select',             # SQL injection
            r'exec\s*\(',                  # Code execution
            r'eval\s*\(',                  # Code evaluation
            r'__import__',                 # Python imports
        ]
        
    async def dispatch(self, request: Request, call_next):
        """Apply security checks"""
        
        client_ip = self._get_client_ip(request)
        
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        # Check for suspicious patterns in URL and headers
        if self._detect_suspicious_activity(request):
            self._block_ip(client_ip)
            return JSONResponse(
                status_code=403,
                content={"detail": "Suspicious activity detected"}
            )
        
        # Add security headers
        response = await call_next(request)
        self._add_security_headers(response)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def _detect_suspicious_activity(self, request: Request) -> bool:
        """Detect suspicious patterns in request"""
        
        # Check URL path
        url_path = str(request.url.path).lower()
        
        # Check query parameters
        query_string = str(request.url.query).lower()
        
        # Check headers
        user_agent = request.headers.get("user-agent", "").lower()
        
        # Combine all text to check
        combined_text = f"{url_path} {query_string} {user_agent}"
        
        # Check against suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True
        
        return False
    
    def _block_ip(self, ip: str):
        """Block an IP address"""
        
        self.blocked_ips.add(ip)
        
        # Log the blocking
        import logging
        logger = logging.getLogger("ai_ethics_toolkit.security")
        logger.warning(f"Blocked IP {ip} due to suspicious activity")
    
    def _add_security_headers(self, response):
        """Add security headers to response"""
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS (only for HTTPS)
        if os.getenv("HTTPS_ENABLED", "false").lower() == "true":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

class RequestSignatureValidator:
    """Validate request signatures for webhook security"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def validate_signature(self, payload: bytes, signature: str) -> bool:
        """Validate HMAC signature"""
        
        if not signature:
            return False
        
        # Remove signature prefix if present (e.g., "sha256=")
        if "=" in signature:
            signature = signature.split("=", 1)[1]
        
        # Calculate expected signature
        expected = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected)
    
    def generate_signature(self, payload: bytes) -> str:
        """Generate HMAC signature for payload"""
        
        signature = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"

class InputSanitizer:
    """Sanitize and validate input data"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\r', '\n']
        for char in dangerous_chars:
            value = value.replace(char, '')
        
        return value.strip()
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security"""
        
        if not filename or len(filename) > 255:
            return False
        
        # Check for directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_without_ext = filename.split('.')[0].upper()
        if name_without_ext in reserved_names:
            return False
        
        return True
    
    @staticmethod
    def sanitize_json_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON input recursively"""
        
        if isinstance(data, dict):
            return {
                key: InputSanitizer.sanitize_json_input(value)
                for key, value in data.items()
                if key and len(str(key)) < 100  # Limit key length
            }
        elif isinstance(data, list):
            return [
                InputSanitizer.sanitize_json_input(item)
                for item in data[:100]  # Limit list size
            ]
        elif isinstance(data, str):
            return InputSanitizer.sanitize_string(data)
        else:
            return data

# Security configuration
SECURITY_CONFIG = {
    "max_request_size": 100 * 1024 * 1024,  # 100MB
    "max_file_uploads": 10,
    "allowed_file_types": [".pkl", ".joblib", ".csv", ".json", ".txt"],
    "rate_limit_enabled": True,
    "signature_validation_enabled": False,  # Enable for webhooks
    "ip_blocking_enabled": True
}

def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    
    return SECURITY_CONFIG.copy()
