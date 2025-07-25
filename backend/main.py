# ai_ethics_toolkit/backend/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from backend.api.routes import (
    bias,
    privacy,
    explainability,
    hallucination,
    reports,
)

app = FastAPI(
    title="AI Ethics Toolkit API (Demo)",
    description="Minimal API for demo purposes.",
    version="0.1-demo",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bias.router, prefix="/api/v1/bias", tags=["Bias Detection"])
app.include_router(privacy.router, prefix="/api/v1/privacy", tags=["Privacy Analysis"])
app.include_router(explainability.router, prefix="/api/v1/explainability", tags=["Explainability"])
app.include_router(hallucination.router, prefix="/api/v1/hallucination", tags=["Hallucination Detection"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

import uuid

@app.get("/api/v1/session")
async def get_session_id():
    """Generate and return a unique session ID."""
    return {"session_id": str(uuid.uuid4())}

@app.get("/")
async def root():
    return {
        "message": "AI Ethics Toolkit API (Demo)",
        "version": "0.1-demo",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=12000, reload=True)
