#!/usr/bin/env python3
"""
Run script for the AI Ethics Toolkit frontend.
"""
import streamlit.web.cli as stcli
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    """Run the Streamlit frontend server."""
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 12001))
    
    # Set API URL environment variable
    os.environ["API_URL"] = os.environ.get("API_URL", f"http://localhost:12000")
    
    # Run Streamlit
    sys.argv = [
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "true",
        "--server.enableXsrfProtection", "false",
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    stcli.main()

if __name__ == "__main__":
    main()