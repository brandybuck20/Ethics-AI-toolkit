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
    
    Args:
        output_path: Path to save the dataset
    """
    print("Creating demo dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = 5000
    
    # Create features
    data = {
        # Demographics (protected attributes)
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
        
        # Financial features
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples),
        'loan_term': np.random.choice([10, 15, 20, 30], n_samples),
        'employment_years': np.random.uniform(0, 30, n_samples),
        'savings_amount': np.random.normal(20000, 15000, n_samples),
        
        # Additional features
        'num_credit_accounts': np.random.randint(0, 10, n_samples),
        'num_late_payments': np.random.randint(0, 5, n_samples),
        'has_mortgage': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_auto_loan': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some derived features
    df['income_to_loan_ratio'] = df['income'] / df['loan_amount']
    df['monthly_payment'] = df['loan_amount'] / (df['loan_term'] * 12)
    df['payment_to_income_ratio'] = df['monthly_payment'] / (df['income'] / 12)
    
    # Create biased target variable
    # Base approval probability on financial factors
    approval_prob = 1 / (1 + np.exp(-(
        0.5 * (df['credit_score'] - 600) / 300 +
        0.3 * (df['income'] - 30000) / 50000 -
        0.4 * df['debt_to_income_ratio'] +
        0.2 * df['employment_years'] / 10 -
        0.3 * df['num_late_payments'] +
        0.2 * df['savings_amount'] / 10000 -
        0.1 * df['payment_to_income_ratio']
    )))
    
    # Add bias based on protected attributes
    # Gender bias
    gender_bias = (df['gender'] == 'Male').astype(float) * 0.15
    
    # Race bias
    race_bias = pd.Series(0, index=df.index)
    race_bias[df['race'] == 'White'] = 0.1
    race_bias[df['race'] == 'Asian'] = 0.05
    race_bias[df['race'] == 'Hispanic'] = -0.05
    race_bias[df['race'] == 'Black'] = -0.1
    
    # Age bias (against younger and older applicants)
    age_bias = -0.1 * ((df['age'] < 25) | (df['age'] > 60)).astype(float)
    
    # Combine biases
    total_bias = gender_bias + race_bias + age_bias
    
    # Apply bias to approval probability
    biased_approval_prob = approval_prob + total_bias
    biased_approval_prob = np.clip(biased_approval_prob, 0, 1)
    
    # Generate final approval decision
    df['loan_approved'] = (np.random.random(n_samples) < biased_approval_prob).astype(int)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    df.to_csv(output_path, index=False)
    print(f"Demo dataset saved to {output_path}")
    
    return df

def train_demo_model(dataset_path: str = "demo_data/loan_approval_dataset.csv", 
                    model_path: str = "demo_data/loan_approval_model.joblib"):
    """
    Train a model on the demo dataset.
    
    Args:
        dataset_path: Path to the dataset
        model_path: Path to save the model
    """
    print("Training demo model...")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Prepare features and target
    protected_attributes = ['gender', 'race', 'age']
    target = 'loan_approved'
    
    # Convert categorical variables to one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['gender', 'race', 'education'])
    
    # Split features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Model training complete:")
    print(f"  - Train accuracy: {train_accuracy:.4f}")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Demo model saved to {model_path}")
    
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