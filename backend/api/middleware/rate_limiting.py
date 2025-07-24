from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
import os

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent API abuse"""
    
    def __init__(self, app, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        
        # Storage for rate limit data
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup task
        asyncio.create_task(self._cleanup_old_requests())
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests"""
        
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Check rate limits
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit-Minute": str(self.calls_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.calls_per_hour),
                    "X-RateLimit-Remaining-Minute": str(max(0, self.calls_per_minute - len(self.minute_requests[client_id]))),
                    "X-RateLimit-Remaining-Hour": str(max(0, self.calls_per_hour - len(self.hour_requests[client_id])))
                }
            )
        
        # Record the request
        self._record_request(client_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit-Minute"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.calls_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, self.calls_per_minute - len(self.minute_requests[client_id])))
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, self.calls_per_hour - len(self.hour_requests[client_id])))
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting"""
        
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Use user ID from JWT if available
        if hasattr(request.state, "user"):
            user_id = request.state.user.get("user_id")
            if user_id:
                return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limits"""
        
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(client_id, current_time)
        
        # Check minute limit
        if len(self.minute_requests[client_id]) >= self.calls_per_minute:
            return False
        
        # Check hour limit
        if len(self.hour_requests[client_id]) >= self.calls_per_hour:
            return False
        
        return True
    
    def _record_request(self, client_id: str):
        """Record a request for rate limiting"""
        
        current_time = time.time()
        
        self.minute_requests[client_id].append(current_time)
        self.hour_requests[client_id].append(current_time)
    
    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove old requests outside the time windows"""
        
        # Clean minute requests (older than 60 seconds)
        minute_cutoff = current_time - 60
        minute_queue = self.minute_requests[client_id]
        while minute_queue and minute_queue[0] < minute_cutoff:
            minute_queue.popleft()
        
        # Clean hour requests (older than 3600 seconds)
        hour_cutoff = current_time - 3600
        hour_queue = self.hour_requests[client_id]
        while hour_queue and hour_queue[0] < hour_cutoff:
            hour_queue.popleft()
    
    async def _cleanup_old_requests(self):
        """Periodic cleanup of old request data"""
        
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            current_time = time.time()
            
            # Clean up all client data
            for client_id in list(self.minute_requests.keys()):
                self._clean_old_requests(client_id, current_time)
                
                # Remove empty entries
                if not self.minute_requests[client_id]:
                    del self.minute_requests[client_id]
                if not self.hour_requests[client_id]:
                    del self.hour_requests[client_id]

class AdaptiveRateLimiter:
    """Advanced rate limiter with adaptive limits based on system load"""
    
    def __init__(self, base_limit: int = 60):
        self.base_limit = base_limit
        self.current_load = 0.0
        self.adaptive_factor = 1.0
    
    def update_system_load(self, cpu_percent: float, memory_percent: float):
        """Update system load metrics"""
        
        self.current_load = (cpu_percent + memory_percent) / 2
        
        # Adjust rate limit based on system load
        if self.current_load > 80:
            self.adaptive_factor = 0.5  # Reduce to 50% of base limit
        elif self.current_load > 60:
            self.adaptive_factor = 0.75  # Reduce to 75% of base limit
        else:
            self.adaptive_factor = 1.0  # Full rate limit
    
    def get_current_limit(self) -> int:
        """Get current rate limit adjusted for system load"""
        
        return int(self.base_limit * self.adaptive_factor)

# Rate limiting configurations for different endpoints
RATE_LIMIT_CONFIG = {
    "/api/v1/bias/analyze": {"calls_per_minute": 10, "calls_per_hour": 100},
    "/api/v1/privacy/analyze": {"calls_per_minute": 15, "calls_per_hour": 150},
    "/api/v1/explainability/analyze": {"calls_per_minute": 5, "calls_per_hour": 50},
    "/api/v1/hallucination/analyze": {"calls_per_minute": 20, "calls_per_hour": 200},
    "default": {"calls_per_minute": 60, "calls_per_hour": 1000}
}

def get_rate_limit_for_endpoint(endpoint: str) -> Dict[str, int]:
    """Get rate limit configuration for specific endpoint"""
    
    return RATE_LIMIT_CONFIG.get(endpoint, RATE_LIMIT_CONFIG["default"])
