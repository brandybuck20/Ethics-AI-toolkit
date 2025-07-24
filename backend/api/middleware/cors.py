from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

def setup_cors(app: FastAPI):
    """Configure CORS middleware for the FastAPI application"""
    
    # Get allowed origins from environment variables
    allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    
    # Development vs Production settings
    if os.getenv("ENVIRONMENT", "development") == "development":
        # More permissive CORS for development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins + ["http://localhost:3000", "http://localhost:8080"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Total-Count", "X-Request-ID"]
        )
    else:
        # Stricter CORS for production
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-Requested-With",
                "X-API-Key",
                "X-Request-ID"
            ],
            expose_headers=["X-Total-Count", "X-Request-ID"]
        )

class CORSConfig:
    """CORS configuration class"""
    
    @staticmethod
    def get_cors_config() -> dict:
        """Get CORS configuration based on environment"""
        
        environment = os.getenv("ENVIRONMENT", "development")
        
        base_config = {
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Authorization",
                "Content-Type", 
                "X-Requested-With",
                "X-API-Key",
                "X-Request-ID"
            ],
            "expose_headers": ["X-Total-Count", "X-Request-ID"]
        }
        
        if environment == "development":
            base_config.update({
                "allow_origins": [
                    "http://localhost:8501",
                    "http://127.0.0.1:8501", 
                    "http://localhost:3000",
                    "http://localhost:8080"
                ],
                "allow_headers": ["*"]  # More permissive for development
            })
        else:
            # Production origins from environment
            origins = os.getenv("CORS_ORIGINS", "").split(",")
            base_config["allow_origins"] = [origin.strip() for origin in origins if origin.strip()]
        
        return base_config

    @staticmethod
    def is_origin_allowed(origin: str) -> bool:
        """Check if origin is allowed"""
        
        config = CORSConfig.get_cors_config()
        allowed_origins = config.get("allow_origins", [])
        
        return origin in allowed_origins
