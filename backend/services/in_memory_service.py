"""
In-memory storage service for AI Ethics Toolkit prototype.
Provides session-based storage for models, datasets, and analysis results.
"""
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple
import uuid
import io
import pandas as pd
import numpy as np
import joblib
import pickle
from fastapi import UploadFile
import asyncio
import threading
import time

logger = logging.getLogger("ai_toolkit")

class InMemoryService:
    """
    In-memory storage service for the AI Ethics Toolkit prototype.
    Provides session-based storage for models, datasets, and analysis results.
    """
    
    # Class variables for shared state
    _sessions: Dict[str, Dict[str, Any]] = {}
    _models: Dict[str, Any] = {}
    _datasets: Dict[str, pd.DataFrame] = {}
    _analysis_results: Dict[str, Dict[str, Any]] = {}
    _background_tasks: Dict[str, Dict[str, Any]] = {}
    _task_progress: Dict[str, float] = {}
    _task_messages: Dict[str, str] = {}
    _lock = threading.RLock()
    
    def __init__(self):
        """Initialize the in-memory service."""
        self._cleanup_interval = 3600  # 1 hour
        self._session_timeout = 24  # 24 hours
        self._last_cleanup = datetime.now()
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            session = {
                'id': session_id,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'model_id': None,
                'dataset_id': None,
                'analysis_ids': [],
                'task_ids': []
            }
            
            self._sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            
            # Run cleanup if needed
            self._maybe_cleanup()
            
            return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            # Update last accessed time
            self._sessions[session_id]['last_accessed'] = datetime.now()
            return self._sessions[session_id]
    
    async def store_model(self, file: UploadFile, session_id: str) -> str:
        """
        Store a model file in memory.
        
        Args:
            file: The uploaded model file
            session_id: The session ID
            
        Returns:
            model_id: The ID of the stored model
        """
        with self._lock:
            # Check if session exists
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Read file content
            content = await file.read()
            
            # Store model in memory
            try:
                # Try to load with joblib first
                model = joblib.load(io.BytesIO(content))
            except Exception:
                try:
                    # Fall back to pickle
                    model = pickle.load(io.BytesIO(content))
                except Exception as e:
                    raise ValueError(f"Failed to load model: {str(e)}")
            
            # Store model
            self._models[model_id] = {
                'model': model,
                'filename': file.filename,
                'content_type': file.content_type,
                'size': len(content),
                'uploaded_at': datetime.now(),
                'session_id': session_id,
                'model_type': type(model).__name__
            }
            
            # Update session
            session['model_id'] = model_id
            session['last_accessed'] = datetime.now()
            
            logger.info(f"Stored model {model_id} for session {session_id}")
            
            return model_id
    
    async def store_dataset(self, file: UploadFile, session_id: str) -> str:
        """
        Store a dataset file in memory.
        
        Args:
            file: The uploaded dataset file
            session_id: The session ID
            
        Returns:
            dataset_id: The ID of the stored dataset
        """
        with self._lock:
            # Check if session exists
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Generate dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Read file content
            content = await file.read()
            
            # Store dataset in memory
            try:
                # Determine file type and load accordingly
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(content))
                elif file.filename.endswith('.json'):
                    df = pd.read_json(io.BytesIO(content))
                elif file.filename.endswith('.parquet'):
                    df = pd.read_parquet(io.BytesIO(content))
                else:
                    raise ValueError(f"Unsupported file format: {file.filename}")
                
                # Store dataset
                self._datasets[dataset_id] = df
                
                # Store metadata
                metadata = {
                    'dataset_id': dataset_id,
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'size': len(content),
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'uploaded_at': datetime.now(),
                    'session_id': session_id
                }
                
                # Update session
                session['dataset_id'] = dataset_id
                session['dataset_metadata'] = metadata
                session['last_accessed'] = datetime.now()
                
                logger.info(f"Stored dataset {dataset_id} for session {session_id}")
                
                return dataset_id
                
            except Exception as e:
                raise ValueError(f"Failed to load dataset: {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        with self._lock:
            if model_id not in self._models:
                return None
            
            return self._models[model_id]
    
    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Get dataset by ID."""
        with self._lock:
            if dataset_id not in self._datasets:
                return None
            
            return self._datasets[dataset_id]
    
    def get_session_model(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the model associated with a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session or not session.get('model_id'):
                return None
            
            return self.get_model(session['model_id'])
    
    def get_session_dataset(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get the dataset associated with a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session or not session.get('dataset_id'):
                return None
            
            return self.get_dataset(session['dataset_id'])
    
    def store_analysis_result(self, result: Dict[str, Any], analysis_type: str, session_id: str) -> str:
        """Store analysis result."""
        with self._lock:
            # Check if session exists
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            # Store result
            self._analysis_results[analysis_id] = {
                'result': result,
                'type': analysis_type,
                'created_at': datetime.now(),
                'session_id': session_id
            }
            
            # Update session
            session['analysis_ids'].append(analysis_id)
            session['last_accessed'] = datetime.now()
            
            logger.info(f"Stored {analysis_type} analysis result {analysis_id} for session {session_id}")
            
            return analysis_id
    
    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result by ID."""
        with self._lock:
            if analysis_id not in self._analysis_results:
                return None
            
            return self._analysis_results[analysis_id]
    
    def get_session_analysis_results(self, session_id: str, analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all analysis results for a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return []
            
            results = []
            for analysis_id in session.get('analysis_ids', []):
                analysis = self.get_analysis_result(analysis_id)
                if analysis and (analysis_type is None or analysis['type'] == analysis_type):
                    results.append(analysis)
            
            return results
    
    def create_background_task(self, session_id: str, task_type: str) -> str:
        """Create a background task."""
        with self._lock:
            # Check if session exists
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create task
            task = {
                'id': task_id,
                'type': task_type,
                'status': 'pending',
                'created_at': datetime.now(),
                'started_at': None,
                'completed_at': None,
                'session_id': session_id,
                'result_id': None,
                'error': None
            }
            
            # Store task
            self._background_tasks[task_id] = task
            self._task_progress[task_id] = 0.0
            self._task_messages[task_id] = f"Task {task_type} created"
            
            # Update session
            session['task_ids'].append(task_id)
            session['last_accessed'] = datetime.now()
            
            logger.info(f"Created background task {task_id} of type {task_type} for session {session_id}")
            
            return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        with self._lock:
            if task_id not in self._background_tasks:
                return None
            
            task = self._background_tasks[task_id].copy()
            task['progress'] = self._task_progress.get(task_id, 0.0)
            task['message'] = self._task_messages.get(task_id, "")
            
            return task
    
    def update_task_progress(self, task_id: str, progress: float, message: str = None):
        """Update task progress."""
        with self._lock:
            if task_id not in self._background_tasks:
                return
            
            self._task_progress[task_id] = progress
            if message:
                self._task_messages[task_id] = message
    
    def complete_task(self, task_id: str, result_id: str = None, error: str = None):
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self._background_tasks:
                return
            
            task = self._background_tasks[task_id]
            task['status'] = 'completed' if not error else 'failed'
            task['completed_at'] = datetime.now()
            task['result_id'] = result_id
            task['error'] = error
            
            self._task_progress[task_id] = 1.0 if not error else -1.0
            self._task_messages[task_id] = "Task completed successfully" if not error else f"Task failed: {error}"
            
            logger.info(f"Completed task {task_id} with status {task['status']}")
    
    def _maybe_cleanup(self):
        """Run cleanup if needed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        self._cleanup_expired_sessions()
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions and their associated data."""
        now = datetime.now()
        expired_sessions = []
        
        # Find expired sessions
        for session_id, session in self._sessions.items():
            if (now - session['last_accessed']).total_seconds() > self._session_timeout * 3600:
                expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _cleanup_session(self, session_id: str):
        """Clean up a session and its associated data."""
        with self._lock:
            if session_id not in self._sessions:
                return
            
            session = self._sessions[session_id]
            
            # Clean up model
            if session.get('model_id') and session['model_id'] in self._models:
                del self._models[session['model_id']]
            
            # Clean up dataset
            if session.get('dataset_id') and session['dataset_id'] in self._datasets:
                del self._datasets[session['dataset_id']]
            
            # Clean up analysis results
            for analysis_id in session.get('analysis_ids', []):
                if analysis_id in self._analysis_results:
                    del self._analysis_results[analysis_id]
            
            # Clean up tasks
            for task_id in session.get('task_ids', []):
                if task_id in self._background_tasks:
                    del self._background_tasks[task_id]
                if task_id in self._task_progress:
                    del self._task_progress[task_id]
                if task_id in self._task_messages:
                    del self._task_messages[task_id]
            
            # Clean up session
            del self._sessions[session_id]
    
    def clear(self):
        """Clear all data."""
        with self._lock:
            self._sessions.clear()
            self._models.clear()
            self._datasets.clear()
            self._analysis_results.clear()
            self._background_tasks.clear()
            self._task_progress.clear()
            self._task_messages.clear()
            
            logger.info("Cleared all in-memory data")