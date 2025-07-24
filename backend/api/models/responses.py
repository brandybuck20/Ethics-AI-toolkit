from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class BiasAuditResponse(BaseModel):
    task_id: Optional[str] = Field(None, description="Background task ID")
    status: Optional[str] = Field(None, description="Task status")
    message: Optional[str] = Field(None, description="Status message")
    overall_bias_score: Optional[float] = Field(None, description="Overall bias score")
    ethics_score: Optional[float] = Field(None, description="Ethics score (0-10)")
    protected_attribute_analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis per protected attribute")
    bias_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of bias findings")
    recommendations: Optional[List[str]] = Field(None, description="Actionable recommendations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Analysis metadata")

class PrivacyAnalysisResponse(BaseModel):
    overall_score: float = Field(..., description="Overall privacy score")
    findings: List[Dict[str, Any]] = Field(..., description="Privacy violation findings")
    summary: Dict[str, Any] = Field(..., description="Analysis summary")
    compliance: Optional[Dict[str, Any]] = Field(None, description="Compliance assessment")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")

class ExplainabilityResponse(BaseModel):
    method: str = Field(..., description="Explanation method used")
    global_importance: Optional[Dict[str, Any]] = Field(None, description="Global feature importance")
    local_explanations: Optional[List[Dict[str, Any]]] = Field(None, description="Local explanations")
    overall_interpretability_score: float = Field(..., description="Interpretability score")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")

class HallucinationResponse(BaseModel):
    overall_factuality_score: float = Field(..., description="Overall factuality score")
    hallucinations: List[Dict[str, Any]] = Field(..., description="Detected hallucinations")
    fact_checks: List[Dict[str, Any]] = Field(..., description="Fact-checking results")
    summary: Dict[str, Any] = Field(..., description="Analysis summary")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")

class ReportResponse(BaseModel):
    report_id: str = Field(..., description="Unique report identifier")
    filename: str = Field(..., description="Generated report filename")
    format: str = Field(..., description="Report format")
    size: int = Field(..., description="File size in bytes")
    generated_at: str = Field(..., description="Generation timestamp")
    download_url: str = Field(..., description="Download URL")
