from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.models.requests import ExplainabilityRequest
from backend.api.models.responses import ExplainabilityResponse
from core.explainability.explainer import ModelExplainer

router = APIRouter()
logger = logging.getLogger("ai_toolkit")

# Helper function to get the storage service from the request
def get_storage(request: Request):
    return request.app.state.storage

@router.post("/explain", response_model=ExplainabilityResponse)
async def generate_explanations(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str,
    method: str = Form("shap"),
    target_column: str = Form(...),
    sample_size: int = Form(200),
    num_features: int = Form(10)
):
    """
    Generate model explanations using specified method
    """
    
    try:
        storage = get_storage(request)
        
        # Create a background task
        task_id = storage.create_background_task(session_id, "explainability")
        
        # Start the background task
        background_tasks.add_task(
            _run_explainability_task,
            storage,
            task_id,
            session_id,
            method,
            target_column,
            sample_size,
            num_features
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": f"Explainability analysis ({method}) started in background"
        }
        
    except Exception as e:
        logger.exception(f"Error starting explainability analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explainability analysis failed: {str(e)}")

async def _run_explainability_task(
    storage,
    task_id: str,
    session_id: str,
    method: str,
    target_column: str,
    sample_size: int,
    num_features: int
):
    """Background task for running explainability analysis"""
    try:
        # Update task status
        storage.update_task_progress(task_id, 0.1, "Loading model and dataset")
        
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
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Update progress
        storage.update_task_progress(task_id, 0.3, "Preparing configuration")
        
        # Prepare configuration
        feature_columns = [col for col in df.columns if col != target_column]
        
        config = {
            'target_column': target_column,
            'selected_features': feature_columns,
            'sample_size': min(sample_size, len(df)),
            'explainer_type': method.title() + 'Explainer',
            'num_features': num_features
        }
        
        # Update progress
        storage.update_task_progress(task_id, 0.4, "Initializing explainer")
        
        # Initialize explainer
        explainer = ModelExplainer(model, df, config)
        
        # Update progress
        storage.update_task_progress(task_id, 0.5, f"Generating {method} explanations")
        
        # Generate explanations based on method
        if method.lower() == 'shap':
            results = explainer.generate_shap_explanations()
        elif method.lower() == 'lime':
            results = explainer.generate_lime_explanations()
        else:
            results = explainer.generate_custom_explanations()
        
        # Update progress
        storage.update_task_progress(task_id, 0.8, "Processing results")
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'model_type': type(model).__name__,
            'dataset_shape': df.shape,
            'configuration': config
        }
        
        # Store results
        storage.update_task_progress(task_id, 0.9, "Storing results")
        result_id = storage.store_analysis_result(results, "explainability", session_id)
        
        # Complete task
        storage.complete_task(task_id, result_id)
        
    except Exception as e:
        logger.exception(f"Error in explainability task: {str(e)}")
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

@router.post("/feature-importance")
async def calculate_feature_importance(
    request: Request,
    session_id: str,
    target_column: str = Form(...),
    method: str = Form("permutation")
):
    """
    Calculate feature importance using specified method
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
        
        # Prepare data
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        if method == "permutation":
            # Use sklearn's permutation importance
            from sklearn.inspection import permutation_importance
            
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=42
            )
            
            importance_data = {
                'features': feature_columns,
                'importance_scores': perm_importance.importances_mean.tolist(),
                'importance_std': perm_importance.importances_std.tolist()
            }
            
        elif method == "model_based" and hasattr(model, 'feature_importances_'):
            # Use model's built-in feature importance
            importance_data = {
                'features': feature_columns,
                'importance_scores': model.feature_importances_.tolist(),
                'importance_std': None
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Method '{method}' not supported or model doesn't have feature_importances_")
        
        # Store the result
        result = {
            'method': method,
            'importance_data': importance_data,
            'timestamp': datetime.now().isoformat()
        }
        
        result_id = storage.store_analysis_result(result, "feature_importance", session_id)
        
        return {
            'result_id': result_id,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error calculating feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance calculation failed: {str(e)}")

@router.get("/methods")
async def get_available_methods():
    """
    Get list of available explanation methods
    """
    
    methods = {
        'shap': {
            'name': 'SHAP',
            'description': 'SHapley Additive exPlanations',
            'types': ['TreeExplainer', 'LinearExplainer', 'KernelExplainer'],
            'supports': ['global', 'local', 'interactions'],
            'best_for': ['tree_models', 'linear_models', 'any_model']
        },
        'lime': {
            'name': 'LIME',
            'description': 'Local Interpretable Model-agnostic Explanations',
            'types': ['Tabular', 'Text', 'Image'],
            'supports': ['local'],
            'best_for': ['complex_models', 'black_box_models']
        },
        'permutation': {
            'name': 'Permutation Importance',
            'description': 'Feature importance via permutation',
            'types': ['Standard'],
            'supports': ['global'],
            'best_for': ['any_model', 'feature_selection']
        }
    }
    
    return {"available_methods": methods}

@router.post("/local-explanation")
async def get_local_explanation(
    request: Request,
    session_id: str,
    instance_index: int = Form(...),
    target_column: str = Form(...),
    method: str = Form("lime")
):
    """
    Get explanation for a specific instance
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
        
        # Validate instance index
        if instance_index >= len(df):
            raise HTTPException(status_code=400, detail="Instance index out of range")
        
        # Get specific instance
        feature_columns = [col for col in df.columns if col != target_column]
        instance = df.iloc[instance_index]
        
        # Generate prediction
        prediction = model.predict([instance[feature_columns].values])[0]
        
        # Generate explanation (simplified LIME-like)
        if method.lower() == "lime":
            # Simplified local explanation
            feature_values = instance[feature_columns].to_dict()
            
            # Mock feature contributions (in production, use actual LIME)
            np.random.seed(42)
            contributions = {
                feature: np.random.normal(0, 0.1) 
                for feature in feature_columns[:10]  # Top 10 features
            }
            
            explanation = {
                'instance_index': instance_index,
                'prediction': float(prediction) if isinstance(prediction, (np.float32, np.float64)) else prediction,
                'feature_values': feature_values,
                'feature_contributions': contributions,
                'explanation_method': method
            }
            
            # Store the result
            result_id = storage.store_analysis_result(explanation, "local_explanation", session_id)
            
            return {
                'result_id': result_id,
                **explanation
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Method '{method}' not supported for local explanations")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating local explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local explanation failed: {str(e)}")

@router.get("/results/{session_id}")
async def get_explainability_results(request: Request, session_id: str):
    """
    Get all explainability results for a session
    """
    try:
        storage = get_storage(request)
        results = storage.get_session_analysis_results(session_id, "explainability")
        
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
        logger.exception(f"Error getting explainability results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get explainability results: {str(e)}")
