import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import joblib
import pickle

class DataValidator:
    """Data validation utilities for AI Ethics Toolkit"""
    
    @staticmethod
    def validate_uploaded_file(uploaded_file, 
                              allowed_extensions: List[str] = None,
                              max_size_mb: float = 100.0) -> Tuple[bool, str]:
        """Validate uploaded file"""
        
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
        
        # Check file extension
        if allowed_extensions:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in [ext.lower() for ext in allowed_extensions]:
                return False, f"Invalid file type: .{file_extension}. Allowed: {', '.join(allowed_extensions)}"
        
        # Check if file is empty
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        return True, "File is valid"

    @staticmethod
    def validate_model_file(uploaded_file) -> Tuple[bool, str, Optional[Any]]:
        """Validate and load model file"""
        
        # First validate file
        is_valid, message = DataValidator.validate_uploaded_file(
            uploaded_file, 
            allowed_extensions=['pkl', 'joblib'],
            max_size_mb=200.0
        )
        
        if not is_valid:
            return False, message, None
        
        # Try to load model
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['pkl', 'joblib']:
                model = joblib.load(uploaded_file)
            else:
                model = pickle.load(uploaded_file)
            
            # Validate model has required methods
            if not hasattr(model, 'predict'):
                return False, "Model must have a 'predict' method", None
            
            return True, "Model loaded successfully", model
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}", None

    @staticmethod
    def validate_dataset(df: pd.DataFrame, 
                        min_rows: int = 1,
                        max_rows: int = 100000,
                        min_columns: int = 1,
                        max_columns: int = 1000) -> Tuple[bool, str]:
        """Validate dataset"""
        
        if df is None:
            return False, "Dataset is None"
        
        if not isinstance(df, pd.DataFrame):
            return False, "Data must be a pandas DataFrame"
        
        # Check dimensions
        n_rows, n_cols = df.shape
        
        if n_rows < min_rows:
            return False, f"Dataset too small: {n_rows} rows (minimum: {min_rows})"
        
        if n_rows > max_rows:
            return False, f"Dataset too large: {n_rows} rows (maximum: {max_rows})"
        
        if n_cols < min_columns:
            return False, f"Too few columns: {n_cols} (minimum: {min_columns})"
        
        if n_cols > max_columns:
            return False, f"Too many columns: {n_cols} (maximum: {max_columns})"
        
        # Check for completely empty dataset
        if df.empty:
            return False, "Dataset is empty"
        
        # Check if all values are null
        if df.isnull().all().all():
            return False, "Dataset contains only null values"
        
        return True, "Dataset is valid"

    @staticmethod
    def validate_column_selection(df: pd.DataFrame, 
                                 selected_columns: List[str],
                                 column_type: str = "feature") -> Tuple[bool, str]:
        """Validate column selection"""
        
        if not selected_columns:
            return False, f"No {column_type} columns selected"
        
        # Check if columns exist in dataset
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            return False, f"Columns not found in dataset: {', '.join(missing_columns)}"
        
        # Check for duplicate selections
        if len(selected_columns) != len(set(selected_columns)):
            return False, f"Duplicate {column_type} columns selected"
        
        return True, f"{column_type.title()} columns are valid"

    @staticmethod
    def validate_protected_attributes(df: pd.DataFrame, 
                                    protected_attrs: List[str]) -> Tuple[bool, str]:
        """Validate protected attributes for bias analysis"""
        
        is_valid, message = DataValidator.validate_column_selection(
            df, protected_attrs, "protected attribute"
        )
        
        if not is_valid:
            return False, message
        
        # Check each protected attribute
        for attr in protected_attrs:
            # Check if column has reasonable number of unique values
            unique_values = df[attr].nunique()
            
            if unique_values < 2:
                return False, f"Protected attribute '{attr}' has fewer than 2 unique values"
            
            if unique_values > 20:
                st.warning(f"Protected attribute '{attr}' has {unique_values} unique values. Consider grouping categories.")
            
            # Check for missing values
            null_count = df[attr].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                if null_percentage > 10:
                    return False, f"Protected attribute '{attr}' has {null_percentage:.1f}% missing values"
        
        return True, "Protected attributes are valid"

    @staticmethod
    def validate_target_column(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """Validate target column for analysis"""
        
        if not target_column:
            return False, "No target column specified"
        
        if target_column not in df.columns:
            return False, f"Target column '{target_column}' not found in dataset"
        
        target_series = df[target_column]
        
        # Check for missing values
        null_count = target_series.isnull().sum()
        if null_count > 0:
            null_percentage = (null_count / len(df)) * 100
            if null_percentage > 5:
                return False, f"Target column has {null_percentage:.1f}% missing values"
        
        # Check unique values
        unique_values = target_series.nunique()
        
        if unique_values < 2:
            return False, "Target column must have at least 2 unique values"
        
        # Classify as binary or multiclass
        if unique_values == 2:
            classification_type = "binary"
        elif unique_values <= 10:
            classification_type = "multiclass"
        else:
            # Might be regression
            if pd.api.types.is_numeric_dtype(target_series):
                classification_type = "regression"
            else:
                return False, f"Target column has too many unique values ({unique_values}) for classification"
        
        return True, f"Target column is valid for {classification_type} task"

    @staticmethod
    def validate_model_data_compatibility(model, df: pd.DataFrame, 
                                        feature_columns: List[str]) -> Tuple[bool, str]:
        """Validate model-data compatibility"""
        
        # Check if model has feature names
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            
            # Check if selected features match model features
            if set(feature_columns) != set(model_features):
                missing_in_data = set(model_features) - set(feature_columns)
                extra_in_data = set(feature_columns) - set(model_features)
                
                error_msg = "Feature mismatch between model and data:\n"
                if missing_in_data:
                    error_msg += f"Missing in data: {', '.join(missing_in_data)}\n"
                if extra_in_data:
                    error_msg += f"Extra in data: {', '.join(extra_in_data)}"
                
                return False, error_msg
        
        # Check feature count
        elif hasattr(model, 'n_features_in_'):
            if len(feature_columns) != model.n_features_in_:
                return False, f"Feature count mismatch: model expects {model.n_features_in_}, got {len(feature_columns)}"
        
        # Try a test prediction
        try:
            test_data = df[feature_columns].iloc[:1]
            _ = model.predict(test_data)
        except Exception as e:
            return False, f"Model prediction failed: {str(e)}"
        
        return True, "Model and data are compatible"

class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_threshold(value: float, min_val: float = 0.0, max_val: float = 1.0) -> Tuple[bool, str]:
        """Validate threshold values"""
        
        if not isinstance(value, (int, float)):
            return False, "Threshold must be a number"
        
        if value < min_val or value > max_val:
            return False, f"Threshold must be between {min_val} and {max_val}"
        
        return True, "Threshold is valid"

    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email address"""
        
        if not email:
            return False, "Email is required"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        return True, "Email is valid"

    @staticmethod
    def validate_password(password: str, min_length: int = 8) -> Tuple[bool, str]:
        """Validate password strength"""
        
        if not password:
            return False, "Password is required"
        
        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters long"
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        # Check for at least one letter
        if not re.search(r'[a-zA-Z]', password):
            return False, "Password must contain at least one letter"
        
        return True, "Password is strong"

    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username"""
        
        if not username:
            return False, "Username is required"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 50:
            return False, "Username must be less than 50 characters"
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "Username can only contain letters, numbers, underscores, and hyphens"
        
        return True, "Username is valid"

    @staticmethod
    def validate_text_input(text: str, 
                           min_length: int = 0,
                           max_length: int = 10000,
                           allow_empty: bool = False) -> Tuple[bool, str]:
        """Validate text input"""
        
        if not text and not allow_empty:
            return False, "Text input is required"
        
        if text and len(text) < min_length:
            return False, f"Text must be at least {min_length} characters long"
        
        if text and len(text) > max_length:
            return False, f"Text must be less than {max_length} characters long"
        
        return True, "Text input is valid"

    @staticmethod
    def validate_numeric_input(value: Union[int, float, str],
                              min_val: Optional[float] = None,
                              max_val: Optional[float] = None,
                              allow_negative: bool = True) -> Tuple[bool, str, Optional[float]]:
        """Validate numeric input"""
        
        # Try to convert to float
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return False, "Input must be a valid number", None
        
        # Check if negative values are allowed
        if not allow_negative and numeric_value < 0:
            return False, "Negative values are not allowed", None
        
        # Check min/max bounds
        if min_val is not None and numeric_value < min_val:
            return False, f"Value must be at least {min_val}", None
        
        if max_val is not None and numeric_value > max_val:
            return False, f"Value must be at most {max_val}", None
        
        return True, "Numeric input is valid", numeric_value

class AdvancedValidator:
    """Advanced validation utilities"""
    
    @staticmethod
    def validate_json_input(json_string: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate JSON input"""
        
        if not json_string.strip():
            return False, "JSON input is empty", None
        
        try:
            parsed_json = json.loads(json_string)
            return True, "JSON is valid", parsed_json
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None

    @staticmethod
    def validate_regex_pattern(pattern: str) -> Tuple[bool, str]:
        """Validate regex pattern"""
        
        if not pattern:
            return False, "Pattern is empty"
        
        try:
            re.compile(pattern)
            return True, "Regex pattern is valid"
        except re.error as e:
            return False, f"Invalid regex pattern: {str(e)}"

    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Validate URL format"""
        
        if not url:
            return False, "URL is required"
        
        url_pattern = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        
        if not re.match(url_pattern, url):
            return False, "Invalid URL format"
        
        return True, "URL is valid"

    @staticmethod
    def validate_api_key(api_key: str, provider: str = "generic") -> Tuple[bool, str]:
        """Validate API key format"""
        
        if not api_key:
            return False, "API key is required"
        
        # Provider-specific validation
        if provider.lower() == "openai":
            if not api_key.startswith("sk-"):
                return False, "OpenAI API key must start with 'sk-'"
            if len(api_key) < 40:
                return False, "OpenAI API key is too short"
        
        elif provider.lower() == "anthropic":
            if not api_key.startswith("sk-ant-"):
                return False, "Anthropic API key must start with 'sk-ant-'"
        
        # Generic validation
        if len(api_key) < 10:
            return False, "API key is too short"
        
        return True, "API key format is valid"

# Convenience functions for common validations
def validate_file_upload(uploaded_file, file_type: str = "general") -> Tuple[bool, str]:
    """Convenience function for file validation"""
    
    type_configs = {
        "model": {"extensions": ["pkl", "joblib"], "max_size": 200.0},
        "dataset": {"extensions": ["csv", "json", "parquet"], "max_size": 100.0},
        "text": {"extensions": ["txt", "md", "json"], "max_size": 10.0},
        "general": {"extensions": None, "max_size": 50.0}
    }
    
    config = type_configs.get(file_type, type_configs["general"])
    
    return DataValidator.validate_uploaded_file(
        uploaded_file,
        allowed_extensions=config["extensions"],
        max_size_mb=config["max_size"]
    )

def validate_bias_audit_inputs(df: pd.DataFrame, 
                              protected_attrs: List[str],
                              target_column: str) -> List[str]:
    """Validate all inputs for bias audit"""
    
    errors = []
    
    # Validate dataset
    is_valid, message = DataValidator.validate_dataset(df)
    if not is_valid:
        errors.append(f"Dataset: {message}")
    
    # Validate protected attributes
    is_valid, message = DataValidator.validate_protected_attributes(df, protected_attrs)
    if not is_valid:
        errors.append(f"Protected attributes: {message}")
    
    # Validate target column
    is_valid, message = DataValidator.validate_target_column(df, target_column)
    if not is_valid:
        errors.append(f"Target column: {message}")
    
    return errors

