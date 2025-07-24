import pandas as pd
from typing import Optional
from fastapi import UploadFile, HTTPException
import tempfile
import os

class DataService:
    """Service for handling data loading and processing"""
    
    def __init__(self):
        pass
    
    async def load_dataset(self, dataset_file: UploadFile) -> pd.DataFrame:
        """Load dataset from uploaded file"""
        
        try:
            # Read file content
            content = await dataset_file.read()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Load based on file extension
            file_extension = dataset_file.filename.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(temp_file_path)
            elif file_extension == 'json':
                df = pd.read_json(temp_file_path)
            elif file_extension == 'parquet':
                df = pd.read_parquet(temp_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if len(df) > 100000:
                raise ValueError("Dataset too large (max 100,000 rows)")
            
            return df
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate dataset basic requirements"""
        
        if df is None or df.empty:
            return False
        
        if len(df) < 10:
            return False
        
        return True
