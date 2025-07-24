from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("ai_ethics_toolkit.errors")

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors globally"""
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            # Let FastAPI handle HTTP exceptions normally
            raise exc
            
        except ValidationError as exc:
            # Handle validation errors
            return await self._handle_validation_error(request, exc)
            
        except DatabaseError as exc:
            # Handle database errors
            return await self._handle_database_error(request, exc)
            
        except ExternalServiceError as exc:
            # Handle external service errors
            return await self._handle_external_service_error(request, exc)
            
        except Exception as exc:
            # Handle unexpected errors
            return await self._handle_unexpected_error(request, exc)
    
    async def _handle_validation_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle validation errors"""
        
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.warning(
            f"Validation error in request {request_id}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "error": str(exc),
                "error_type": "validation_error"
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "message": str(exc),
                "error_type": "validation_error",
                "request_id": request_id
            }
        )
    
    async def _handle_database_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle database errors"""
        
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.error(
            f"Database error in request {request_id}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "error": str(exc),
                "error_type": "database_error"
            }
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Database service unavailable",
                "message": "A database error occurred. Please try again later.",
                "error_type": "database_error",
                "request_id": request_id
            }
        )
    
    async def _handle_external_service_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle external service errors"""
        
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.error(
            f"External service error in request {request_id}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "error": str(exc),
                "error_type": "external_service_error"
            }
        )
        
        return JSONResponse(
            status_code=502,
            content={
                "detail": "External service error",
                "message": "An external service is temporarily unavailable.",
                "error_type": "external_service_error",
                "request_id": request_id
            }
        )
    
    async def _handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors"""
        
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log full traceback for unexpected errors
        logger.error(
            f"Unexpected error in request {request_id}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "error": str(exc),
                "error_type": "unexpected_error",
                "traceback": traceback.format_exc()
            }
        )
        
        # Don't expose internal error details in production
        error_detail = "Internal server error"
        error_message = "An unexpected error occurred. Please try again later."
        
        # In development, provide more details
        if sys.gettrace() is not None:  # Debug mode
            error_detail = str(exc)
            error_message = f"Debug: {str(exc)}"
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": error_detail,
                "message": error_message,
                "error_type": "internal_error",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Custom exception classes
class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail"""
    pass

class ExternalServiceError(Exception):
    """Raised when external service calls fail"""
    pass

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass

class AnalysisError(Exception):
    """Raised when analysis operations fail"""
    pass

# Error handlers for specific exceptions
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "message": str(exc),
            "error_type": "validation_error"
        }
    )

async def database_error_handler(request: Request, exc: DatabaseError):
    """Handle database errors"""
    
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database unavailable", 
            "message": "Database service is temporarily unavailable",
            "error_type": "database_error"
        }
    )

async def model_load_error_handler(request: Request, exc: ModelLoadError):
    """Handle model loading errors"""
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Model loading failed",
            "message": str(exc),
            "error_type": "model_load_error"
        }
    )

async def analysis_error_handler(request: Request, exc: AnalysisError):
    """Handle analysis errors"""
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Analysis failed",
            "message": str(exc),
            "error_type": "analysis_error"
        }
    )

# Error reporting utility
class ErrorReporter:
    """Utility for reporting and tracking errors"""
    
    def __init__(self):
        self.error_counts = {}
    
    def report_error(self, error_type: str, error_message: str, 
                    request_info: Dict[str, Any] = None):
        """Report an error occurrence"""
        
        # Count error occurrences
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error details
        logger.error(
            f"Error reported: {error_type}",
            extra={
                "error_type": error_type,
                "error_message": error_message,
                "error_count": self.error_counts[error_type],
                "request_info": request_info or {}
            }
        )
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counters"""
        
        self.error_counts.clear()

# Global error reporter instance
error_reporter = ErrorReporter()
