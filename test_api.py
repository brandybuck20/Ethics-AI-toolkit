#!/usr/bin/env python3
"""
Test script for the AI Ethics Toolkit API.
"""
import requests
import json
import time
import os
from pathlib import Path

# API base URL
API_URL = "http://localhost:12000"

def test_health_endpoint():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_session_creation():
    """Test session creation."""
    print("Testing session creation...")
    response = requests.get(f"{API_URL}/api/v1/session")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.json()["session_id"]

def test_model_upload(session_id):
    """Test model upload."""
    print("Testing model upload...")
    model_path = "demo_data/loan_approval_model.joblib"
    
    with open(model_path, "rb") as f:
        files = {"model_file": (os.path.basename(model_path), f, "application/octet-stream")}
        response = requests.post(
            f"{API_URL}/api/v1/bias/upload/model",
            files=files,
            params={"session_id": session_id}
        )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.json()["model_id"]

def test_dataset_upload(session_id):
    """Test dataset upload."""
    print("Testing dataset upload...")
    dataset_path = "demo_data/loan_approval_dataset.csv"
    
    with open(dataset_path, "rb") as f:
        files = {"dataset_file": (os.path.basename(dataset_path), f, "text/csv")}
        response = requests.post(
            f"{API_URL}/api/v1/bias/upload/dataset",
            files=files,
            params={"session_id": session_id}
        )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.json()["dataset_id"]

def test_quick_bias_check(session_id):
    """Test quick bias check."""
    print("Testing quick bias check...")
    
    data = {
        "protected_attributes": ["gender", "race", "age"],
        "target_column": "loan_approved"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bias/quick-check",
        json=data,
        params={"session_id": session_id}
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_bias_audit(session_id):
    """Test bias audit."""
    print("Testing bias audit...")
    
    data = {
        "protected_attributes": json.dumps(["gender", "race", "age"]),
        "target_column": "loan_approved",
        "bias_threshold": "0.1",
        "fairness_metrics": "demographic_parity,equalized_odds"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bias/audit",
        data=data,
        params={"session_id": session_id}
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    task_id = response.json()["task_id"]
    return task_id

def test_task_status(task_id):
    """Test task status."""
    print("Testing task status...")
    
    max_attempts = 10
    for attempt in range(max_attempts):
        response = requests.get(f"{API_URL}/api/v1/bias/task/{task_id}")
        status = response.json()["status"]
        progress = response.json()["progress"]
        
        print(f"Attempt {attempt + 1}/{max_attempts}: Status = {status}, Progress = {progress:.2f}")
        
        if status == "completed":
            print("Task completed!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            break
        elif status == "failed":
            print("Task failed!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            break
        
        time.sleep(1)
    
    print()

def main():
    """Run all tests."""
    print("=== AI Ethics Toolkit API Test ===\n")
    
    # Test health endpoint
    test_health_endpoint()
    
    # Test session creation
    session_id = test_session_creation()
    
    # Test model upload
    model_id = test_model_upload(session_id)
    
    # Test dataset upload
    dataset_id = test_dataset_upload(session_id)
    
    print("=== Basic tests completed ===")
    print(f"Session ID: {session_id}")
    print(f"Model ID: {model_id}")
    print(f"Dataset ID: {dataset_id}")
    print("\nThe backend and frontend are now ready to use!")
    print("- Frontend: http://localhost:12002")
    print("- Backend API: http://localhost:12000/docs")

if __name__ == "__main__":
    main()