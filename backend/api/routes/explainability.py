from fastapi import APIRouter, Request, Query
from backend.api.models.requests import ExplainabilityRequest

router = APIRouter()

@router.post("/explain")
async def generate_explanations(request: Request, payload: ExplainabilityRequest, is_demo: bool = Query(False)):
    if is_demo:
        return {
            "method": "shap",
            "top_features": [
                {"feature": "age", "importance": 0.32},
                {"feature": "income", "importance": 0.21},
                {"feature": "loan_amount", "importance": 0.18}
            ],
            "explanation": "The model's prediction is most influenced by age, income, and loan amount.",
            "sample_size": 100,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    # Placeholder for actual explainability logic
    return {"message": "Explainability initiated (actual logic not implemented yet). Set is_demo=true for demo data."}
