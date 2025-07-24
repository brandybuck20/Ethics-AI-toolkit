#!/usr/bin/env python3
"""
Test script to verify the complete AI Ethics Toolkit workflow
"""
import requests
import pandas as pd
import numpy as np
import json
import time
from io import StringIO

# Configuration
API_BASE_URL = "http://localhost:12000"
SESSION_ID = "test_session_123"

def create_sample_dataset():
    """Create a sample dataset with potential bias"""
    np.random.seed(42)
    
    # Create synthetic data with bias
    n_samples = 1000
    
    # Gender bias example
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3])
    age = np.random.normal(35, 10, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # Introduce bias: higher salaries for males
    base_salary = 50000 + age * 1000 + np.random.normal(0, 5000, n_samples)
    gender_bias = np.where(gender == 'Male', 10000, 0)
    salary = base_salary + gender_bias
    
    # Create hiring decision with bias
    hiring_score = (salary / 1000) + np.random.normal(0, 5, n_samples)
    hired = (hiring_score > 75).astype(int)
    
    df = pd.DataFrame({
        'gender': gender,
        'age': age,
        'education': education,
        'salary': salary,
        'hired': hired
    })
    
    return df

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API health...")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_bias_detection():
    """Test bias detection workflow"""
    print("\n🎯 Testing bias detection workflow...")
    
    # Create session first
    print("🔧 Creating session...")
    response = requests.get(f"{API_BASE_URL}/api/v1/session")
    if response.status_code != 200:
        print(f"❌ Session creation failed: {response.status_code}")
        print(response.text)
        return False
    
    session_data = response.json()
    session_id = session_data.get('session_id', SESSION_ID)
    print(f"✅ Session created: {session_id}")
    
    # Create sample dataset
    df = create_sample_dataset()
    csv_data = df.to_csv(index=False)
    
    # Upload dataset
    print("📤 Uploading dataset...")
    files = {'dataset_file': ('test_data.csv', csv_data, 'text/csv')}
    params = {'session_id': session_id}
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/bias/upload/dataset",
        files=files,
        params=params
    )
    
    if response.status_code != 200:
        print(f"❌ Dataset upload failed: {response.status_code}")
        print(response.text)
        return False
    
    print("✅ Dataset uploaded successfully")
    
    # Start bias analysis
    print("🔍 Starting bias analysis...")
    analysis_data = {
        "target_column": "hired",
        "protected_attributes": "gender",
        "bias_threshold": 0.1,
        "fairness_metrics": "demographic_parity,equalized_odds"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/bias/audit",
        data=analysis_data,
        params={'session_id': session_id}
    )
    
    if response.status_code != 200:
        print(f"❌ Bias analysis failed: {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    task_id = result.get('task_id')
    print(f"✅ Analysis started with task ID: {task_id}")
    
    # Poll for results
    print("⏳ Waiting for analysis results...")
    max_attempts = 30
    for attempt in range(max_attempts):
        response = requests.get(
            f"{API_BASE_URL}/api/v1/bias/results/{task_id}",
            params={'session_id': session_id}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                print("✅ Analysis completed!")
                print(f"Overall bias score: {result.get('overall_bias_score', 'N/A')}")
                print(f"Issues found: {len(result.get('issues', []))}")
                return True
            elif result.get('status') == 'failed':
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return False
        
        time.sleep(2)
    
    print("❌ Analysis timed out")
    return False

def test_frontend_connection():
    """Test frontend connectivity"""
    print("\n🌐 Testing frontend connectivity...")
    try:
        response = requests.get("http://localhost:12002", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AI Ethics Toolkit Workflow Test")
    print("=" * 50)
    
    tests = [
        ("API Health", test_api_health),
        ("Bias Detection", test_bias_detection),
        ("Frontend Connection", test_frontend_connection),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! The AI Ethics Toolkit is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()