"""
API client for communicating with the FastAPI backend.
"""
import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the AI Ethics Toolkit API."""
    
    def __init__(self, base_url: str = "http://localhost:12000"):
        """Initialize the API client."""
        self.base_url = base_url
        self.session_id = self._get_or_create_session_id()
    
    def _get_or_create_session_id(self) -> str:
        """Get or create a session ID."""
        if "api_session_id" not in st.session_state:
            try:
                response = requests.get(f"{self.base_url}/api/v1/session")
                response.raise_for_status()
                st.session_state.api_session_id = response.json()["session_id"]
            except Exception as e:
                logger.error(f"Failed to create session: {str(e)}")
                # Generate a fallback session ID
                import uuid
                st.session_state.api_session_id = str(uuid.uuid4())
        
        return st.session_state.api_session_id
    
    def _send_post_request(self, endpoint: str, data: Dict[str, Any], is_demo: bool = False, is_json: bool = False) -> Dict[str, Any]:
        params = {"session_id": self.session_id}
        if is_demo:
            params["is_demo"] = "true"
        
        if is_json:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data, # Send as JSON
                params=params
            )
        else:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                data=data,
                params=params
            )
        response.raise_for_status()
        return response.json()

    def _send_get_request(self, endpoint: str, is_demo: bool = False) -> Dict[str, Any]:
        params = {"session_id": self.session_id}
        if is_demo:
            params["is_demo"] = "true"
        
        response = requests.get(
            f"{self.base_url}{endpoint}",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def upload_model(self, model_file) -> Dict[str, Any]:
        """
        Upload a model file to the backend.
        
        Args:
            model_file: The model file to upload
            
        Returns:
            Dict containing the upload response
        """
        try:
            files = {"model_file": model_file}
            response = requests.post(
                f"{self.base_url}/api/v1/bias/upload/model",
                files=files,
                params={"session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to upload model: {str(e)}")
            raise ValueError(f"Failed to upload model: {str(e)}")
    
    def upload_dataset(self, dataset_file) -> Dict[str, Any]:
        """
        Upload a dataset file to the backend.
        
        Args:
            dataset_file: The dataset file to upload
            
        Returns:
            Dict containing the upload response
        """
        try:
            files = {"dataset_file": dataset_file}
            response = requests.post(
                f"{self.base_url}/api/v1/bias/upload/dataset",
                files=files,
                params={"session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to upload dataset: {str(e)}")
            raise ValueError(f"Failed to upload dataset: {str(e)}")
    
    def run_bias_audit(self,
                      protected_attributes: List[str],
                      target_column: str,
                      bias_threshold: float = 0.1,
                      fairness_metrics: List[str] = ["demographic_parity", "equalized_odds"],
                      is_demo: bool = False) -> Dict[str, Any]:
        """
        Run a bias audit on the uploaded model and dataset.
        
        Args:
            protected_attributes: List of protected attribute column names
            target_column: Target column name
            bias_threshold: Bias threshold
            fairness_metrics: List of fairness metrics to calculate
            is_demo: Whether to run in demo mode
            
        Returns:
            Dict containing the task information
        """
        try:
            data = {
                "protected_attributes": protected_attributes,
                "target_column": target_column,
                "bias_threshold": bias_threshold,
                "fairness_metrics": fairness_metrics
            }
            return self._send_post_request("/api/v1/bias/audit", data, is_demo, is_json=True)
        except Exception as e:
            logger.error(f"Failed to run bias audit: {str(e)}")
            raise ValueError(f"Failed to run bias audit: {str(e)}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a background task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Dict containing the task status
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/bias/task/{task_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            raise ValueError(f"Failed to get task status: {str(e)}")
    
    def wait_for_task_completion(self,
                                task_id: str,
                                timeout: int = 300,
                                poll_interval: int = 1,
                                progress_callback=None) -> Dict[str, Any]:
        """
        Wait for a background task to complete.
        
        Args:
            task_id: The ID of the task
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict containing the task result
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                status = self.get_task_status(task_id)
                
                # Update progress if callback provided
                if progress_callback and "progress" in status:
                    progress_callback(status["progress"], status.get("message", ""))
                
                if status["status"] == "completed":
                    return status
                elif status["status"] == "failed":
                    raise ValueError(f"Task failed: {status.get('message', 'Unknown error')}")
                
                time.sleep(poll_interval)
            except Exception as e:
                if "Task not found" in str(e):
                    raise ValueError(f"Task not found: {task_id}")
                logger.error(f"Error checking task status: {str(e)}")
                time.sleep(poll_interval)
        
        raise TimeoutError(f"Task did not complete within {timeout} seconds")
    
    def run_explainability_analysis(self,
                                  target_column: str,
                                  method: str = "shap",
                                  sample_size: int = 200,
                                  num_features: int = 10,
                                  is_demo: bool = False) -> Dict[str, Any]:
        """
        Run an explainability analysis on the uploaded model and dataset.
        
        Args:
            target_column: Target column name
            method: Explainability method (shap, lime)
            sample_size: Number of samples to use
            num_features: Number of features to include in explanation
            is_demo: Whether to run in demo mode
            
        Returns:
            Dict containing the task information
        """
        try:
            data = {
                "target_column": target_column,
                "method": method,
                "sample_size": sample_size,
                "num_features": num_features
            }
            return self._send_post_request("/api/v1/explainability/explain", data, is_demo, is_json=True)
        except Exception as e:
            logger.error(f"Failed to run explainability analysis: {str(e)}")
            raise ValueError(f"Failed to run explainability analysis: {str(e)}")
    
    def get_feature_importance(self, target_column: str, method: str = "permutation", is_demo: bool = False) -> Dict[str, Any]:
        """
        Get feature importance for the uploaded model and dataset.
        
        Args:
            target_column: Target column name
            method: Feature importance method (permutation, model_based)
            is_demo: Whether to run in demo mode
            
        Returns:
            Dict containing the feature importance results
        """
        try:
            data = {
                "target_column": target_column,
                "method": method
            }
            return self._send_post_request("/api/v1/explainability/feature-importance", data, is_demo, is_json=True)
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            raise ValueError(f"Failed to get feature importance: {str(e)}")
    
    def detect_hallucinations(self,
                            text_content: str,
                            detection_types: List[str] = ["factual", "urls", "citations"],
                            sensitivity_level: str = "balanced",
                            confidence_threshold: float = 0.6,
                            is_demo: bool = False) -> Dict[str, Any]:
        """
        Detect hallucinations in text.
        
        Args:
            text_content: The text to analyze
            detection_types: Types of hallucinations to detect
            sensitivity_level: Sensitivity level (low, balanced, high)
            confidence_threshold: Confidence threshold
            is_demo: Whether to run in demo mode
            
        Returns:
            Dict containing the hallucination detection results
        """
        try:
            data = {
                "text_content": text_content,
                "detection_types": detection_types,
                "sensitivity_level": sensitivity_level,
                "confidence_threshold": confidence_threshold,
                "verify_links": True
            }
            return self._send_post_request("/api/v1/hallucination/detect", data, is_demo, is_json=True)
        except Exception as e:
            logger.error(f"Failed to detect hallucinations: {str(e)}")
            raise ValueError(f"Failed to detect hallucinations: {str(e)}")
    
    def get_bias_results(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get all bias audit results for the current session.
        
        Returns:
            Dict containing the bias audit results
        """
        try:
            return self._send_get_request(f"/api/v1/bias/results/{self.session_id}", is_demo)
        except Exception as e:
            logger.error(f"Failed to get bias results: {str(e)}")
            raise ValueError(f"Failed to get bias results: {str(e)}")
    
    def get_explainability_results(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get all explainability results for the current session.
        
        Returns:
            Dict containing the explainability results
        """
        try:
            return self._send_get_request(f"/api/v1/explainability/results/{self.session_id}", is_demo)
        except Exception as e:
            logger.error(f"Failed to get explainability results: {str(e)}")
            raise ValueError(f"Failed to get explainability results: {str(e)}")
    
    def get_hallucination_results(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get all hallucination detection results for the current session.
        
        Returns:
            Dict containing the hallucination detection results
        """
        try:
            return self._send_get_request(f"/api/v1/hallucination/results/{self.session_id}", is_demo)
        except Exception as e:
            logger.error(f"Failed to get hallucination results: {str(e)}")
            raise ValueError(f"Failed to get hallucination results: {str(e)}")
    
    def quick_bias_check(self, protected_attributes: List[str], target_column: str, is_demo: bool = False) -> Dict[str, Any]:
        """
        Run a quick bias check on the uploaded model and dataset.
        
        Args:
            protected_attributes: List of protected attribute column names
            target_column: Target column name
            is_demo: Whether to run in demo mode
            
        Returns:
            Dict containing the quick bias check results
        """
        try:
            data = {"protected_attributes": protected_attributes, "target_column": target_column}
            return self._send_post_request("/api/v1/bias/quick-check", data, is_demo, is_json=True)
        except Exception as e:
            logger.error(f"Failed to run quick bias check: {str(e)}")
            raise ValueError(f"Failed to run quick bias check: {str(e)}")
    
    def get_available_metrics(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get available fairness metrics.
        
        Returns:
            Dict containing the available metrics
        """
        try:
            return self._send_get_request("/api/v1/bias/metrics", is_demo)
        except Exception as e:
            logger.error(f"Failed to get available metrics: {str(e)}")
            raise ValueError(f"Failed to get available metrics: {str(e)}")
    
    def get_available_explanation_methods(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get available explanation methods.
        
        Returns:
            Dict containing the available methods
        """
        try:
            return self._send_get_request("/api/v1/explainability/methods", is_demo)
        except Exception as e:
            logger.error(f"Failed to get available explanation methods: {str(e)}")
            raise ValueError(f"Failed to get available explanation methods: {str(e)}")
    
    def get_available_detection_types(self, is_demo: bool = False) -> Dict[str, Any]:
        """
        Get available hallucination detection types.
        
        Returns:
            Dict containing the available detection types
        """
        try:
            return self._send_get_request("/api/v1/hallucination/detection-types", is_demo)
        except Exception as e:
            logger.error(f"Failed to get available detection types: {str(e)}")
            raise ValueError(f"Failed to get available detection types: {str(e)}")