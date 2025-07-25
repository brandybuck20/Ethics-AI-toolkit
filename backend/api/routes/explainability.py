from fastapi import APIRouter, Request, Query
from backend.api.models.requests import ExplainabilityRequest

router = APIRouter()

@router.post("/explain")
async def generate_explanations(request: Request, payload: ExplainabilityRequest, is_demo: bool = Query(False)):
    # Always return static demo results
    return {
        "method": "shap",
        "top_features": [
            {"feature": "age", "importance": 0.32},
            {"feature": "income", "importance": 0.21},
            {"feature": "loan_amount", "importance": 0.18}
        ],
        "explanation": "The model's prediction is most influenced by age, income, and loan amount.",
        "sample_size": 100,
        "timestamp": "2025-07-25T08:00:00Z"
    }
