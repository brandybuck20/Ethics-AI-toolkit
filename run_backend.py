#!/usr/bin/env python3
"""
Run script for the AI Ethics Toolkit backend.
"""
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    """Run the FastAPI backend server."""
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 12000))
    
    # Run server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()