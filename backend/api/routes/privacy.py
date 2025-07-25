from fastapi import APIRouter, Request, Query
from backend.api.models.requests import PrivacyAnalysisRequest

router = APIRouter()

@router.post("/analyze")
async def analyze_privacy(request: Request, payload: PrivacyAnalysisRequest, is_demo: bool = Query(False)):
    # Always return static demo results
    return {
        "overall_score": 9.2,
        "findings": [
            {"type": "PII", "description": "Email address detected", "severity": "medium"}
        ],
        "summary": {
            "risk_level": "Low",
            "recommendations": ["Remove PII from dataset", "Mask sensitive fields"]
        },
        "timestamp": "2025-07-25T08:00:00Z"
    }
