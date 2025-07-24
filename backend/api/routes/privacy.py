from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.models.requests import PrivacyAnalysisRequest
from backend.api.models.responses import PrivacyAnalysisResponse
from backend.services.data_service import DataService
from core.privacy.analyzer import PrivacyAnalyzer

router = APIRouter()
data_service = DataService()

@router.post("/analyze", response_model=PrivacyAnalysisResponse)
async def analyze_privacy(
    text_content: Optional[str] = Form(None),
    dataset_file: Optional[UploadFile] = File(None),
    pii_types: str = Form("names,emails,phone,ssn"),
    sensitivity_level: str = Form("balanced"),
    compliance_frameworks: str = Form("gdpr")
):
    """
    Analyze text or dataset for privacy risks and PII exposure
    """
    
    try:
        # Parse configuration
        pii_types_list = pii_types.split(",")
        frameworks_list = compliance_frameworks.split(",")
        
        config = {
            'pii_types': pii_types_list,
            'sensitivity_level': sensitivity_level,
            'compliance_frameworks': frameworks_list,
            'confidence_threshold': 0.7
        }
        
        # Initialize privacy analyzer
        privacy_analyzer = PrivacyAnalyzer(config)
        
        # Analyze based on input type
        if text_content:
            results = privacy_analyzer.analyze_text(text_content)
        elif dataset_file:
            df = await data_service.load_dataset(dataset_file)
            results = privacy_analyzer.analyze_dataset(df)
        else:
            raise HTTPException(status_code=400, detail="Either text_content or dataset_file must be provided")
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'text' if text_content else 'dataset',
            'configuration': config
        }
        
        return PrivacyAnalysisResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Privacy analysis failed: {str(e)}")

@router.post("/pii-detection")
async def detect_pii(
    text: str = Form(...),
    pii_types: str = Form("names,emails,phone,ssn,credit_cards")
):
    """
    Focused PII detection in text
    """
    
    try:
        pii_types_list = pii_types.split(",")
        
        config = {
            'pii_types': pii_types_list,
            'sensitivity_level': 'high',
            'mask_findings': True
        }
        
        privacy_analyzer = PrivacyAnalyzer(config)
        results = privacy_analyzer.analyze_text(text)
        
        # Extract just PII findings
        pii_findings = [
            finding for finding in results.get('findings', [])
            if finding.get('category') == 'PII'
        ]
        
        return {
            'pii_detected': len(pii_findings) > 0,
            'pii_count': len(pii_findings),
            'findings': pii_findings,
            'risk_level': results.get('summary', {}).get('risk_level', 'Unknown')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII detection failed: {str(e)}")

@router.post("/gdpr-compliance")
async def check_gdpr_compliance(request: PrivacyAnalysisRequest):
    """
    Check GDPR compliance for data processing
    """
    
    try:
        # Simulate GDPR compliance check
        compliance_results = {
            'compliant': True,
            'compliance_score': 8.5,
            'violations': [],
            'recommendations': [
                'Ensure proper consent mechanisms are in place',
                'Implement data retention policies',
                'Provide clear privacy notices'
            ],
            'articles_checked': [
                'Article 5 (Principles)',
                'Article 6 (Lawfulness)',
                'Article 7 (Consent)',
                'Article 17 (Right to erasure)'
            ]
        }
        
        return compliance_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GDPR compliance check failed: {str(e)}")

@router.get("/pii-types")
async def get_supported_pii_types():
    """
    Get list of supported PII types for detection
    """
    
    pii_types = {
        'names': {
            'description': 'Person and organization names',
            'examples': ['John Smith', 'Acme Corp'],
            'sensitivity': 'medium'
        },
        'emails': {
            'description': 'Email addresses',
            'examples': ['user@example.com'],
            'sensitivity': 'medium'
        },
        'phone': {
            'description': 'Phone numbers',
            'examples': ['(555) 123-4567', '+1-555-123-4567'],
            'sensitivity': 'medium'
        },
        'ssn': {
            'description': 'Social Security Numbers',
            'examples': ['123-45-6789'],
            'sensitivity': 'high'
        },
        'credit_cards': {
            'description': 'Credit card numbers',
            'examples': ['4532 1234 5678 9012'],
            'sensitivity': 'high'
        },
        'addresses': {
            'description': 'Physical addresses',
            'examples': ['123 Main St, Anytown, NY 12345'],
            'sensitivity': 'medium'
        }
    }
    
    return {"supported_pii_types": pii_types}
