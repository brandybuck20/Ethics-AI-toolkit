from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BiasAuditRequest(BaseModel):
    protected_attributes: List[str] = Field(..., description="List of protected attribute column names")
    target_column: str = Field(..., description="Target variable column name")
    bias_threshold: float = Field(0.1, description="Bias threshold for detection")
    fairness_metrics: List[str] = Field(["demographic_parity", "equalized_odds"], description="Fairness metrics to calculate")

class PrivacyAnalysisRequest(BaseModel):
    pii_types: List[str] = Field(["names", "emails", "phone"], description="Types of PII to detect")
    sensitivity_level: str = Field("balanced", description="Detection sensitivity level")
    compliance_frameworks: List[str] = Field(["gdpr"], description="Compliance frameworks to check")
    mask_findings: bool = Field(True, description="Whether to mask detected PII")

class ExplainabilityRequest(BaseModel):
    method: str = Field("shap", description="Explanation method to use")
    target_column: str = Field(..., description="Target variable column name")
    sample_size: int = Field(200, description="Sample size for analysis")
    num_features: int = Field(10, description="Number of top features to explain")

class HallucinationRequest(BaseModel):
    detection_types: List[str] = Field(["factual", "urls", "citations"], description="Types of hallucinations to detect")
    sensitivity_level: str = Field("balanced", description="Detection sensitivity")
    confidence_threshold: float = Field(0.6, description="Minimum confidence for flagging")
    verify_links: bool = Field(True, description="Whether to verify URL existence")

class ReportGenerationRequest(BaseModel):
    audit_type: str = Field(..., description="Type of audit report")
    report_format: str = Field("pdf", description="Output format for report")
    include_visualizations: bool = Field(True, description="Include charts and graphs")
    include_recommendations: bool = Field(True, description="Include actionable recommendations")
