from fastapi import APIRouter, Request, Query
from backend.api.models.requests import ReportGenerationRequest

router = APIRouter()

@router.post("/generate")
async def generate_audit_report(request: Request, payload: ReportGenerationRequest, is_demo: bool = Query(False)):
    # Always return static demo results
    return {
        "report_id": "mock_report_001",
        "filename": "mock_report_001.txt",
        "format": "txt",
        "size": 1024,
        "generated_at": "2025-07-25T08:00:00Z",
        "download_url": "/api/v1/reports/download/mock_report_001.txt",
        "summary": "This is a mock audit report for prototyping purposes."
    }
