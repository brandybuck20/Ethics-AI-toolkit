import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import joblib
import pickle

class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        pass

class SklearnModelInterface(BaseModelInterface):
    """Interface for scikit-learn models"""
    
    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using sklearn model"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Get prediction probabilities if available"""
        if hasattr(self.model, 'predict_proba'):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.model.predict_proba(X)
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get sklearn model information"""
        info = {
            'model_type': self.model_type,
            'framework': 'scikit-learn',
            'has_predict_proba': hasattr(self.model, 'predict_proba'),
            'has_feature_importance': hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_')
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        # Add feature information
        if hasattr(self.model, 'feature_names_in_'):
            info['feature_names'] = list(self.model.feature_names_in_)
            info['n_features'] = len(self.model.feature_names_in_)
        elif hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        # Add class information for classifiers
        if hasattr(self.model, 'classes_'):
            info['classes'] = list(self.model.classes_)
            info['n_classes'] = len(self.model.classes_)
        
        return info

class XGBoostModelInterface(BaseModelInterface):
    """Interface for XGBoost models"""
    
    def __init__(self, model):
        self.model = model
        self.model_type = 'XGBoost'
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using XGBoost model"""
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Get prediction probabilities for XGBoost"""
        try:
            return self.model.predict_proba(X)
        except:
            return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get XGBoost feature importance"""
        try:
            return self.model.feature_importances_
        except:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get XGBoost model information"""
        return {
            'model_type': self.model_type,
            'framework': 'XGBoost',
            'has_predict_proba': hasattr(self.model, 'predict_proba'),
            'has_feature_importance': hasattr(self.model, 'feature_importances_'),
            'n_estimators': getattr(self.model, 'n_estimators', None),
            'max_depth': getattr(self.model, 'max_depth', None)
        }

class LLMModelInterface(BaseModelInterface):
    """Interface for Large Language Models"""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = 'LLM'
        
    def predict(self, X: Union[List[str], str]) -> List[str]:
        """Generate text predictions"""
        if isinstance(X, str):
            X = [X]
        
        # This is a placeholder - actual implementation depends on specific LLM
        return [f"Generated response for: {text[:50]}..." for text in X]
    
    def generate(self, 
                prompt: str, 
                max_length: int = 100,
                temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        # Placeholder implementation
        return f"Generated text for prompt: {prompt[:50]}..."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LLM model information"""
        return {
            'model_type': self.model_type,
            'framework': 'Transformers/Custom',
            'has_tokenizer': self.tokenizer is not None,
            'supports_generation': True,
            'model_name': getattr(self.model, 'name_or_path', 'Unknown')
        }

class ModelInterfaceFactory:
    """Factory class to create appropriate model interfaces"""
    
    @staticmethod
    def create_interface(model) -> BaseModelInterface:
        """Create appropriate interface for the given model"""
        
        model_type = type(model).__name__.lower()
        model_module = type(model).__module__.lower()
        
        # XGBoost models
        if 'xgb' in model_type or 'xgboost' in model_module:
            return XGBoostModelInterface(model)
        
        # LightGBM models
        elif 'lgb' in model_type or 'lightgbm' in model_module:
            return SklearnModelInterface(model)  # Can use sklearn interface
        
        # Transformers/LLM models
        elif 'transformer' in model_module or hasattr(model, 'generate'):
            return LLMModelInterface(model)
        
        # Default to sklearn interface
        else:
            return SklearnModelInterface(model)
    
    @staticmethod
    def load_model_from_file(file_path: str) -> BaseModelInterface:
        """Load model from file and create appropriate interface"""
        
        try:
            # Try joblib first
            model = joblib.load(file_path)
        except:
            try:
                # Fallback to pickle
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load model from {file_path}: {str(e)}")
        
        return ModelInterfaceFactory.create_interface(model)

class UniversalModelWrapper:
    """Universal wrapper that works with any model type"""
    
    def __init__(self, model, interface: Optional[BaseModelInterface] = None):
        self.model = model
        self.interface = interface or ModelInterfaceFactory.create_interface(model)
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame, List[str]]) -> np.ndarray:
        """Universal predict method"""
        return self.interface.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Universal predict_proba method"""
        if hasattr(self.interface, 'predict_proba'):
            return self.interface.predict_proba(X)
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Universal feature importance method"""
        if hasattr(self.interface, 'get_feature_importance'):
            return self.interface.get_feature_importance()
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return self.interface.get_model_info()
    
    def is_classifier(self) -> bool:
        """Check if model is a classifier"""
        info = self.get_model_info()
        return 'classes' in info or info.get('has_predict_proba', False)
    
    def is_regressor(self) -> bool:
        """Check if model is a regressor"""
        # Simple heuristic - not a classifier and has predict method
        return not self.is_classifier() and hasattr(self.interface, 'predict')
    
    def supports_probabilities(self) -> bool:
        """Check if model supports probability predictions"""
        return hasattr(self.interface, 'predict_proba') and self.interface.predict_proba is not None
    
    def get_prediction_type(self) -> str:
        """Determine the type of predictions this model makes"""
        if self.is_classifier():
            return 'classification'
        elif self.is_regressor():
            return 'regression'
        else:
            return 'unknown'
