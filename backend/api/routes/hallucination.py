from fastapi import APIRouter, Request, Query
from backend.api.models.requests import HallucinationRequest

router = APIRouter()

@router.post("/detect")
async def detect_hallucinations(request: Request, payload: HallucinationRequest, is_demo: bool = Query(False)):
    # Always return static demo results
    return {
        "hallucinations": [],
        "overall_factuality_score": 0.95,
        "summary": "No hallucinations detected in the provided text.",
        "timestamp": "2025-07-25T08:00:00Z"
    }
