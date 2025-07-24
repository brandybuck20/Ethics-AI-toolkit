import joblib
import pickle
from typing import Any, Optional
from fastapi import UploadFile, HTTPException
import tempfile
import os

class ModelService:
    """Service for handling model loading and management"""
    
    def __init__(self):
        self.loaded_models = {}
    
    async def load_model(self, model_file: UploadFile) -> Any:
        """Load model from uploaded file"""
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                content = await model_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Load model based on file extension
            file_extension = model_file.filename.split('.')[-1].lower()
            
            if file_extension in ['pkl', 'joblib']:
                model = joblib.load(temp_file_path)
            else:
                model = pickle.load(open(temp_file_path, 'rb'))
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Validate model
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have a 'predict' method")
            
            return model
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
    
    def cache_model(self, model_id: str, model: Any):
        """Cache loaded model for reuse"""
        self.loaded_models[model_id] = model
    
    def get_cached_model(self, model_id: str) -> Optional[Any]:
        """Retrieve cached model"""
        return self.loaded_models.get(model_id)
