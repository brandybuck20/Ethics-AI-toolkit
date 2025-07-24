"""
AI Ethics Toolkit - Frontend Utilities

This module provides utility functions and classes for the AI Ethics Toolkit frontend,
including state management, theme handling, data validation, and formatting utilities.
"""

from .state_manager import StateManager
from .theme import ThemeManager, load_custom_css, apply_theme, get_current_theme
from .validators import (
    DataValidator, InputValidator, AdvancedValidator, 
    validate_file_upload, validate_bias_audit_inputs
)
from .formatters import (
    DataFormatter, MetricFormatter, TableFormatter, 
    ReportFormatter, ChartFormatter, format_metric_value, safe_format
)

# Version info
__version__ = "1.0.0"
__author__ = "AI Ethics Toolkit Team"

# Convenience imports for common operations
initialize_session_state = StateManager.initialize_session_state
# update_audit_stats = StateManager.update_audit_stats
get_session_summary = StateManager.get_session_summary
clear_cache = StateManager.clear_cache

# Common validation functions
validate_dataset = DataValidator.validate_dataset
validate_model_file = DataValidator.validate_model_file
validate_protected_attributes = DataValidator.validate_protected_attributes

# Common formatting functions
format_number = DataFormatter.format_number
format_percentage = DataFormatter.format_percentage
format_datetime = DataFormatter.format_datetime
format_file_size = DataFormatter.format_file_size

__all__ = [
    # Classes
    'StateManager',
    'ThemeManager', 
    'DataValidator',
    'InputValidator',
    'AdvancedValidator',
    'DataFormatter',
    'MetricFormatter',
    'TableFormatter',
    'ReportFormatter',
    'ChartFormatter',
    
    # State management functions
    'initialize_session_state',
    'update_audit_stats',
    'get_session_summary',
    'clear_cache',
    
    # Theme functions
    'load_custom_css',
    'apply_theme',
    'get_current_theme',
    
    # Validation functions
    'validate_dataset',
    'validate_model_file',
    'validate_protected_attributes',
    'validate_file_upload',
    'validate_bias_audit_inputs',
    
    # Formatting functions
    'format_number',
    'format_percentage', 
    'format_datetime',
    'format_file_size',
    'format_metric_value',
    'safe_format'
]
