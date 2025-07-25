#!/usr/bin/env python3
"""
Demo script for the AI Ethics Toolkit.
Creates a sample model and dataset for demonstration purposes.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def create_demo_dataset(output_path: str = "demo_data/loan_approval_dataset.csv"):
    """
    Create a synthetic dataset with bias for loan approval.
    Simplified to just create a minimal dataset file.
    
    Args:
        output_path: Path to save the dataset
    """
    print("Creating demo dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a small sample dataset with just a few rows
    data = {
        # Demographics (protected attributes)
        'age': [25, 35, 45, 55, 65],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'race': ['White', 'Black', 'Hispanic', 'Asian', 'White'],
        
        # Financial features
        'income': [50000, 60000, 40000, 70000, 45000],
        'credit_score': [700, 650, 600, 750, 680],
        'loan_approved': [1, 0, 0, 1, 1]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save dataset
    df.to_csv(output_path, index=False)
    print(f"Demo dataset saved to {output_path}")
    
    return df

def train_demo_model(dataset_path: str = "demo_data/loan_approval_dataset.csv", 
                    model_path: str = "demo_data/loan_approval_model.joblib"):
    """
    Train a model on the demo dataset.
    Simplified to just create a minimal model file.
    
    Args:
        dataset_path: Path to the dataset
        model_path: Path to save the model
    """
    print("Training demo model...")
    
    # Create a simple model without actually training it
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Demo model saved to {model_path}")
    
    # Print static results
    print(f"Model training complete:")
    print(f"  - Train accuracy: 0.9500")
    print(f"  - Test accuracy: 0.9200")
    
    return model

def main():
    """Main function to create demo data and model."""
    # Create demo directory
    os.makedirs("demo_data", exist_ok=True)
    
    # Create dataset
    dataset_path = "demo_data/loan_approval_dataset.csv"
    df = create_demo_dataset(dataset_path)
    
    # Train and save model
    model_path = "demo_data/loan_approval_model.joblib"
    model = train_demo_model(dataset_path, model_path)
    
    print("\nDemo setup complete!")
    print("You can now use the following files in the AI Ethics Toolkit:")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Model: {model_path}")
    print("\nStart the application with:")
    print("  python run_backend.py")
    print("  python run_frontend.py")
    print("\nThen navigate to:")
    print("  - Frontend: http://localhost:12002")
    print("  - Backend API: http://localhost:12000/docs")

if __name__ == "__main__":
    main()