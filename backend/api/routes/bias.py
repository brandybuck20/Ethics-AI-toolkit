from fastapi import APIRouter, Request, Query
from backend.api.models.requests import BiasAuditRequest

router = APIRouter()

@router.post("/audit")
async def run_bias_audit(request: Request, payload: BiasAuditRequest, is_demo: bool = Query(False)):
    # Always return static demo results
    return {
        "overall_bias_score": 0.12,
        "ethics_score": 8.8,
        "bias_detected": True,
        "primary_bias_type": "demographic_parity",
        "recommendations": [
            "Review data collection",
            "Balance dataset",
            "Monitor model predictions"
        ],
        "is_sample": True,
        "sample_size": 100,
        "timestamp": "2025-07-25T08:00:00Z"
    }

@router.get("/metrics")
async def get_available_metrics(is_demo: bool = Query(False)):
   # Always return static demo results
   return {
       "available_metrics": {
           "demographic_parity": {"name": "Demographic Parity", "description": "Measures difference in positive outcome rates between groups."},
           "equalized_odds": {"name": "Equalized Odds", "description": "Equality of true positive and false positive rates across groups."},
           "equal_opportunity": {"name": "Equal Opportunity", "description": "Equality of true positive rates across groups."}
       }
   }

@router.post("/quick-check")
async def quick_bias_check(payload: BiasAuditRequest, is_demo: bool = Query(False)):
   # Always return static demo results
   return {
       "overall_bias_score": 0.15,
       "ethics_score": 8.5,
       "bias_detected": True,
       "primary_bias_type": "demographic_parity",
       "recommendations": [
           "Re-sample data for underrepresented groups",
           "Apply post-processing bias mitigation techniques"
       ],
       "sample_size": 200,
       "timestamp": "2025-07-25T08:00:00Z"
   }
