from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
import uuid
import json
from datetime import datetime
from typing import Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger("ai_ethics_toolkit")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses"""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            await self._log_response(request, response, process_time, request_id)
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request {request_id} failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": process_time,
                    "error": str(exc),
                    "error_type": type(exc).__name__
                }
            )
            
            raise exc
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Get user agent
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # Log request
        logger.info(
            f"Request {request_id} started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length"),
                "auth_type": getattr(request.state, "auth_type", None)
            }
        )
    
    async def _log_response(self, request: Request, response: Response, 
                           process_time: float, request_id: str):
        """Log response details"""
        
        logger.info(
            f"Request {request_id} completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": process_time,
                "content_length": response.headers.get("content-length"),
                "content_type": response.headers.get("content-type")
            }
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

class AuditLogger:
    """Specialized logger for audit events"""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_ethics_audit")
        
        # Create file handler for audit logs
        handler = logging.FileHandler("logs/audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_audit_event(self, event_type: str, user_id: str, 
                       details: dict, request_id: str = None):
        """Log audit-specific events"""
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "request_id": request_id,
            "details": details
        }
        
        self.logger.info(
            f"Audit Event: {event_type}",
            extra=audit_entry
        )
    
    def log_bias_audit(self, user_id: str, model_info: dict, results: dict):
        """Log bias audit completion"""
        
        self.log_audit_event(
            event_type="bias_audit_completed",
            user_id=user_id,
            details={
                "model_type": model_info.get("type"),
                "dataset_size": model_info.get("dataset_size"),
                "bias_score": results.get("overall_bias_score"),
                "issues_found": results.get("issues_count", 0)
            }
        )
    
    def log_privacy_analysis(self, user_id: str, analysis_type: str, results: dict):
        """Log privacy analysis completion"""
        
        self.log_audit_event(
            event_type="privacy_analysis_completed",
            user_id=user_id,
            details={
                "analysis_type": analysis_type,
                "privacy_score": results.get("overall_score"),
                "pii_detected": results.get("pii_count", 0),
                "risk_level": results.get("risk_level")
            }
        )

# Global audit logger instance
audit_logger = AuditLogger()
