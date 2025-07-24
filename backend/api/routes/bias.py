from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Request
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import asyncio
import time
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.models.requests import BiasAuditRequest
from backend.api.models.responses import BiasAuditResponse
from core.bias.detector import BiasDetector

router = APIRouter()
logger = logging.getLogger("ai_toolkit")

# Helper function to get the storage service from the request
def get_storage(request: Request):
    return request.app.state.storage

@router.post("/upload/model")
async def upload_model(
    request: Request,
    session_id: str,
    model_file: UploadFile = File(...),
):
    """
    Upload and store a model file
    """
    try:
        storage = get_storage(request)
        model_id = await storage.store_model(model_file, session_id)
        
        model_info = storage.get_model(model_id)
        
        return {
            "model_id": model_id,
            "filename": model_file.filename,
            "model_type": model_info['model_type'],
            "upload_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")

@router.post("/upload/dataset")
async def upload_dataset(
    request: Request,
    session_id: str,
    dataset_file: UploadFile = File(...),
):
    """
    Upload and store a dataset file
    """
    try:
        storage = get_storage(request)
        dataset_id = await storage.store_dataset(dataset_file, session_id)
        
        # Get session to access dataset metadata
        session = storage.get_session(session_id)
        
        return {
            "dataset_id": dataset_id,
            "filename": dataset_file.filename,
            "shape": session['dataset_metadata']['shape'],
            "columns": session['dataset_metadata']['columns'],
            "upload_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")

@router.post("/audit", response_model=BiasAuditResponse)
async def run_bias_audit(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str,
    protected_attributes: str = Form(...),
    target_column: str = Form(...),
    bias_threshold: float = Form(0.1),
    fairness_metrics: str = Form("demographic_parity,equalized_odds")
):
    """
    Run comprehensive bias audit on uploaded model and dataset
    """
    
    try:
        storage = get_storage(request)
        
        # Create a background task
        task_id = storage.create_background_task(session_id, "bias_audit")
        
        # Start the background task
        background_tasks.add_task(
            _run_bias_audit_task,
            storage,
            task_id,
            session_id,
            protected_attributes,
            target_column,
            bias_threshold,
            fairness_metrics
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Bias audit started in background"
        }
        
    except Exception as e:
        logger.exception(f"Error starting bias audit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bias audit failed: {str(e)}")

async def _run_bias_audit_task(
    storage,
    task_id: str,
    session_id: str,
    protected_attributes: str,
    target_column: str,
    bias_threshold: float,
    fairness_metrics: str
):
    """Background task for running bias audit"""
    try:
        # Update task status
        storage.update_task_progress(task_id, 0.1, "Loading model and dataset")
        
        # Parse form data
        protected_attrs = json.loads(protected_attributes)
        metrics_list = fairness_metrics.split(",")
        
        # Get model and dataset from storage
        model_data = storage.get_session_model(session_id)
        if not model_data:
            raise ValueError("No model found for this session")
        
        model = model_data['model']
        
        df = storage.get_session_dataset(session_id)
        if df is None:
            raise ValueError("No dataset found for this session")
        
        # Update progress
        storage.update_task_progress(task_id, 0.2, "Validating inputs")
        
        # Validate inputs
        validation_errors = _validate_bias_audit_inputs(
            df, protected_attrs, target_column, model
        )
        if validation_errors:
            raise ValueError(validation_errors)
        
        # Update progress
        storage.update_task_progress(task_id, 0.3, "Preparing data")
        
        # Prepare data
        from core.shared.data_processors import DataPreprocessor
        data_preprocessor = DataPreprocessor()

        # Identify categorical columns that are not protected attributes or target
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in protected_attrs + [target_column]]
        
        # One-hot encode identified categorical columns
        if categorical_cols:
            df_processed = data_preprocessor.encode_categorical_features(df.copy(), categorical_cols, encoding_method='onehot')
        else:
            df_processed = df.copy()

        feature_columns = [col for col in df_processed.columns
                           if col not in protected_attrs + [target_column]]
        
        X = df_processed[feature_columns]
        y_true = df_processed[target_column].values
        
        # Update progress
        storage.update_task_progress(task_id, 0.4, "Generating predictions")
        
        # Generate predictions
        y_pred = model.predict(X)
        
        # Update progress
        storage.update_task_progress(task_id, 0.5, "Initializing bias detector")
        
        # Initialize bias detector
        config = {
            'bias_threshold': bias_threshold,
            'fairness_metrics': metrics_list
        }
        
        bias_detector = BiasDetector(config)
        
        # Update progress
        storage.update_task_progress(task_id, 0.6, "Running bias detection")
        
        # Run bias detection
        bias_results = bias_detector.detect_bias(
            df, y_true, y_pred, protected_attrs
        )
        
        # Update progress
        storage.update_task_progress(task_id, 0.8, "Processing results")
        
        # Add metadata
        bias_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'dataset_shape': df.shape,
            'protected_attributes': protected_attrs,
            'target_column': target_column,
            'bias_threshold': bias_threshold
        }
        
        # Calculate ethics score
        ethics_score = _calculate_ethics_score(bias_results['overall_bias_score'])
        bias_results['ethics_score'] = ethics_score
        
        # Store results
        storage.update_task_progress(task_id, 0.9, "Storing results")
        result_id = storage.store_analysis_result(bias_results, "bias_audit", session_id)
        
        # Complete task
        storage.complete_task(task_id, result_id)
        
    except Exception as e:
        logger.exception(f"Error in bias audit task: {str(e)}")
        storage.complete_task(task_id, error=str(e))

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

@router.get("/results/{session_id}")
async def get_bias_results(request: Request, session_id: str):
    """
    Get all bias audit results for a session
    """
    try:
        storage = get_storage(request)
        results = storage.get_session_analysis_results(session_id, "bias_audit")
        
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
        logger.exception(f"Error getting bias results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bias results: {str(e)}")

@router.post("/quick-check")
async def quick_bias_check(
    request: Request,
    session_id: str,
    protected_attributes: List[str],
    target_column: str
):
    """
    Quick bias check using already uploaded data
    """
    
    try:
        storage = get_storage(request)
        
        # Get model and dataset
        model_data = storage.get_session_model(session_id)
        if not model_data:
            raise HTTPException(status_code=400, detail="No model found for this session")
        
        model = model_data['model']
        
        df = storage.get_session_dataset(session_id)
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found for this session")
        
        # Validate inputs
        validation_errors = _validate_bias_audit_inputs(
            df, protected_attributes, target_column, model
        )
        if validation_errors:
            raise HTTPException(status_code=400, detail=validation_errors)
        
        # Prepare data
        feature_columns = [col for col in df.columns 
                          if col not in protected_attributes + [target_column]]
        
        X = df[feature_columns]
        y_true = df[target_column].values
        
        # Generate predictions for a sample
        sample_size = min(1000, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        
        X_sample = X.iloc[sample_indices]
        y_true_sample = y_true[sample_indices]
        
        y_pred_sample = model.predict(X_sample)
        
        # Initialize bias detector with default config
        config = {
            'bias_threshold': 0.1,
            'fairness_metrics': ['demographic_parity', 'equalized_odds']
        }
        
        bias_detector = BiasDetector(config)
        
        # Run quick bias detection on sample
        df_sample = df.iloc[sample_indices].copy()
        quick_results = bias_detector.detect_bias(
            df_sample, y_true_sample, y_pred_sample, protected_attributes
        )
        
        # Calculate ethics score
        ethics_score = _calculate_ethics_score(quick_results['overall_bias_score'])
        
        # Return simplified results
        return {
            'overall_bias_score': quick_results['overall_bias_score'],
            'ethics_score': ethics_score,
            'bias_detected': quick_results['overall_bias_score'] > 0.1,
            'primary_bias_type': quick_results['bias_summary'].get('primary_bias_types', ['None'])[0],
            'recommendations': quick_results['recommendations'][:3],
            'is_sample': True,
            'sample_size': sample_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in quick bias check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick check failed: {str(e)}")

@router.get("/metrics")
async def get_available_metrics():
    """
    Get list of available fairness metrics
    """
    
    metrics = {
        'demographic_parity': {
            'name': 'Demographic Parity',
            'description': 'Equal positive prediction rates across groups',
            'formula': 'P(Y_hat=1|A=a) should be equal for all a'
        },
        'equalized_odds': {
            'name': 'Equalized Odds', 
            'description': 'Equal TPR and FPR across groups',
            'formula': 'P(Y_hat=1|Y=y,A=a) should be equal for all a, y'
        },
        'equal_opportunity': {
            'name': 'Equal Opportunity',
            'description': 'Equal TPR across groups',
            'formula': 'P(Y_hat=1|Y=1,A=a) should be equal for all a'
        },
        'calibration': {
            'name': 'Calibration',
            'description': 'Equal predicted vs actual positive rates',
            'formula': 'P(Y=1|Y_hat=p,A=a) should be equal for all a'
        }
    }
    
    return {"available_metrics": metrics}

@router.get("/thresholds")
async def get_bias_thresholds():
    """
    Get recommended bias thresholds for different use cases
    """
    
    thresholds = {
        'strict': {
            'bias_threshold': 0.05,
            'description': 'Strict fairness requirements (e.g., hiring, lending)',
            'use_cases': ['hiring', 'lending', 'criminal_justice']
        },
        'moderate': {
            'bias_threshold': 0.1,
            'description': 'Moderate fairness requirements (e.g., marketing)',
            'use_cases': ['marketing', 'recommendations', 'general_ml']
        },
        'lenient': {
            'bias_threshold': 0.2,
            'description': 'Lenient requirements (e.g., research)',
            'use_cases': ['research', 'experimentation', 'prototype']
        }
    }
    
    return {"recommended_thresholds": thresholds}

def _validate_bias_audit_inputs(df: pd.DataFrame, 
                               protected_attrs: List[str],
                               target_column: str,
                               model) -> Optional[str]:
    """Validate bias audit inputs"""
    
    # Check if protected attributes exist
    missing_attrs = [attr for attr in protected_attrs if attr not in df.columns]
    if missing_attrs:
        return f"Protected attributes not found: {missing_attrs}"
    
    # Check if target column exists
    if target_column not in df.columns:
        return f"Target column '{target_column}' not found"
    
    # Check if model has predict method
    if not hasattr(model, 'predict'):
        return "Model must have a 'predict' method"
    
    # Check for sufficient data
    if len(df) < 100:
        return "Dataset too small for reliable bias analysis (minimum 100 rows)"
    
    return None

def _calculate_ethics_score(bias_score: float) -> float:
    """Calculate ethics score from bias score"""
    
    # Convert bias score to ethics score (0-10 scale)
    # Lower bias = higher ethics score
    ethics_score = max(0, 10 - (bias_score * 20))
    return round(ethics_score, 1)
