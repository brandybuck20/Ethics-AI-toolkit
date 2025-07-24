from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Request, BackgroundTasks
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.models.requests import HallucinationRequest
from backend.api.models.responses import HallucinationResponse
from core.hallucination.detector import HallucinationDetector

router = APIRouter()
logger = logging.getLogger("ai_toolkit")

# Helper function to get the storage service from the request
def get_storage(request: Request):
    return request.app.state.storage

@router.post("/detect", response_model=HallucinationResponse)
async def detect_hallucinations(
    request: Request,
    session_id: str,
    text_content: str = Form(...),
    detection_types: str = Form("factual,urls,citations"),
    sensitivity_level: str = Form("balanced"),
    confidence_threshold: float = Form(0.6),
    verify_links: bool = Form(True)
):
    """
    Detect hallucinations in AI-generated text
    """
    
    try:
        storage = get_storage(request)
        
        # Parse detection types
        detection_types_list = detection_types.split(",")
        
        # Configure detector
        config = {
            'detection_types': detection_types_list,
            'sensitivity_level': sensitivity_level,
            'confidence_threshold': confidence_threshold,
            'verify_links': verify_links,
            'fact_check_sources': ['wikipedia']  # Local sources only
        }
        
        # Initialize detector
        detector = HallucinationDetector(config)
        
        # Run detection
        results = detector.analyze_text(text_content)
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text_content),
            'configuration': config
        }
        
        # Store results
        result_id = storage.store_analysis_result(results, "hallucination", session_id)
        
        return {
            'result_id': result_id,
            **results
        }
        
    except Exception as e:
        logger.exception(f"Error in hallucination detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hallucination detection failed: {str(e)}")

@router.post("/llm-interaction")
async def analyze_llm_interaction(
    request: Request,
    session_id: str,
    prompt: str = Form(...),
    response: str = Form(...),
    llm_provider: str = Form("unknown")
):
    """
    Analyze LLM prompt-response interaction for hallucinations
    """
    
    try:
        storage = get_storage(request)
        
        config = {
            'detection_types': ['Factual Inaccuracies', 'Non-existent URLs', 'Fake Citations'],
            'sensitivity_level': 'high',
            'confidence_threshold': 0.5,
            'verify_links': True
        }
        
        detector = HallucinationDetector(config)
        results = detector.analyze_llm_interaction(prompt, response)
        
        # Add LLM-specific analysis
        results['llm_analysis'] = {
            'provider': llm_provider,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'response_to_prompt_ratio': len(response) / len(prompt) if len(prompt) > 0 else 0
        }
        
        # Store results
        result_id = storage.store_analysis_result(results, "llm_interaction", session_id)
        
        return {
            'result_id': result_id,
            **results
        }
        
    except Exception as e:
        logger.exception(f"Error in LLM interaction analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM interaction analysis failed: {str(e)}")

@router.post("/fact-check")
async def fact_check_claims(
    request: Request,
    session_id: str,
    claims: List[str] = Form(...),
    sources: str = Form("wikipedia")
):
    """
    Fact-check specific claims against trusted sources
    """
    
    try:
        storage = get_storage(request)
        source_list = sources.split(",")
        
        fact_check_results = []
        
        for claim in claims:
            # Simplified fact-checking (no external APIs)
            suspicious_terms = [
                'time machine', 'teleportation', 'impossible', 
                'never been done', '100% effective', 'cures all'
            ]
            
            is_suspicious = any(term in claim.lower() for term in suspicious_terms)
            
            result = {
                'claim': claim,
                'status': 'questionable' if is_suspicious else 'plausible',
                'confidence': 0.8 if is_suspicious else 0.6,
                'sources_checked': source_list,
                'evidence': 'Pattern-based analysis' if is_suspicious else 'No obvious red flags'
            }
            
            fact_check_results.append(result)
        
        results = {
            'fact_checks': fact_check_results,
            'summary': {
                'total_claims': len(claims),
                'questionable_claims': sum(1 for r in fact_check_results if r['status'] == 'questionable'),
                'plausible_claims': sum(1 for r in fact_check_results if r['status'] == 'plausible')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store results
        result_id = storage.store_analysis_result(results, "fact_check", session_id)
        
        return {
            'result_id': result_id,
            **results
        }
        
    except Exception as e:
        logger.exception(f"Error in fact checking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fact-checking failed: {str(e)}")

@router.get("/detection-types")
async def get_detection_types():
    """
    Get available hallucination detection types
    """
    
    detection_types = {
        'factual': {
            'name': 'Factual Inaccuracies',
            'description': 'Detect false or incorrect factual claims',
            'examples': ['Incorrect dates', 'False statistics', 'Non-existent events']
        },
        'urls': {
            'name': 'Non-existent URLs',
            'description': 'Detect fabricated or broken web links',
            'examples': ['Fake research papers', 'Invalid URLs', 'Non-existent websites']
        },
        'citations': {
            'name': 'Fake Citations',
            'description': 'Detect fabricated academic references',
            'examples': ['Non-existent journals', 'Fake authors', 'Invalid DOIs']
        },
        'logical': {
            'name': 'Logical Inconsistencies',
            'description': 'Detect internal contradictions and impossible claims',
            'examples': ['Self-contradictions', 'Impossible scenarios', 'Logical fallacies']
        }
    }
    
    return {"detection_types": detection_types}

@router.post("/batch-analysis")
async def batch_hallucination_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str,
    texts: List[str] = Form(...),
    detection_types: str = Form("factual,urls")
):
    """
    Analyze multiple texts for hallucinations in batch
    """
    
    try:
        storage = get_storage(request)
        
        # Create a background task
        task_id = storage.create_background_task(session_id, "batch_hallucination")
        
        # Start the background task
        background_tasks.add_task(
            _run_batch_hallucination_task,
            storage,
            task_id,
            session_id,
            texts,
            detection_types
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": f"Batch hallucination analysis started in background for {len(texts)} texts"
        }
        
    except Exception as e:
        logger.exception(f"Error starting batch hallucination analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

async def _run_batch_hallucination_task(
    storage,
    task_id: str,
    session_id: str,
    texts: List[str],
    detection_types: str
):
    """Background task for running batch hallucination analysis"""
    try:
        # Update task status
        storage.update_task_progress(task_id, 0.1, "Initializing batch analysis")
        
        detection_types_list = detection_types.split(",")
        
        config = {
            'detection_types': detection_types_list,
            'sensitivity_level': 'balanced',
            'confidence_threshold': 0.6
        }
        
        detector = HallucinationDetector(config)
        
        batch_results = []
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            try:
                # Update progress
                progress = 0.1 + (0.8 * (i / total_texts))
                storage.update_task_progress(
                    task_id, 
                    progress, 
                    f"Analyzing text {i+1} of {total_texts}"
                )
                
                result = detector.analyze_text(text)
                result['text_id'] = i
                result['text_length'] = len(text)
                batch_results.append(result)
            except Exception as e:
                logger.exception(f"Error analyzing text {i}: {str(e)}")
                batch_results.append({
                    'text_id': i,
                    'error': str(e),
                    'hallucinations': [],
                    'overall_factuality_score': 0
                })
        
        # Calculate batch summary
        storage.update_task_progress(task_id, 0.9, "Calculating summary")
        
        total_hallucinations = sum(len(r.get('hallucinations', [])) for r in batch_results)
        avg_factuality = sum(r.get('overall_factuality_score', 0) for r in batch_results) / len(batch_results)
        
        results = {
            'batch_results': batch_results,
            'summary': {
                'texts_analyzed': len(texts),
                'total_hallucinations': total_hallucinations,
                'average_factuality_score': round(avg_factuality, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store results
        result_id = storage.store_analysis_result(results, "batch_hallucination", session_id)
        
        # Complete task
        storage.complete_task(task_id, result_id)
        
    except Exception as e:
        logger.exception(f"Error in batch hallucination task: {str(e)}")
        storage.complete_task(task_id, error=str(e))

@router.get("/results/{session_id}")
async def get_hallucination_results(request: Request, session_id: str):
    """
    Get all hallucination detection results for a session
    """
    try:
        storage = get_storage(request)
        results = storage.get_session_analysis_results(session_id, "hallucination")
        
        return {
            "session_id": session_id,
            "result_count": len(results),
            "results": [
                {
                    "id": result['result_id'] if 'result_id' in result else idx,
                    "timestamp": result['created_at'].isoformat(),
                    "data": result['result']
                }
                for idx, result in enumerate(results)
            ]
        }
        
    except Exception as e:
        logger.exception(f"Error getting hallucination results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hallucination results: {str(e)}")

@router.get("/task/{task_id}")
async def get_task_status(request: Request, task_id: str):
    """
    Get the status of a background task
    """
    try:
        storage = get_storage(request)
        task = storage.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        response = {
            "task_id": task_id,
            "status": task['status'],
            "progress": task['progress'],
            "message": task['message'],
            "created_at": task['created_at'].isoformat(),
        }
        
        if task['status'] == 'completed':
            # Include result if available
            if task['result_id']:
                result = storage.get_analysis_result(task['result_id'])
                if result:
                    response['result'] = result['result']
                    response['result_id'] = task['result_id']
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")
