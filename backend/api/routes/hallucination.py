from fastapi import APIRouter, Request, Query
from backend.api.models.requests import HallucinationRequest

router = APIRouter()

@router.post("/detect")
async def detect_hallucinations(request: Request, payload: HallucinationRequest, is_demo: bool = Query(False)):
    if is_demo:
        return {
            "hallucinations": [],
            "overall_factuality_score": 0.95,
            "summary": "No hallucinations detected in the provided text.",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    # Placeholder for actual hallucination detection logic
    return {"message": "Hallucination detection initiated (actual logic not implemented yet). Set is_demo=true for demo data."}
