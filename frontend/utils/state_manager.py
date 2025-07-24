import streamlit as st
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import hashlib

class StateManager:
    """Advanced session state management for AI Ethics Toolkit"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables with default values"""
        
        # Core application state
        session_defaults = {
            # Navigation
            'current_page': 'Home',
            'previous_page': None,
            'page_history': [],
            
            # Authentication
            'user_authenticated': False,
            'current_user': None,
            'session_token': None,
            'login_attempts': 0,
            'last_login_attempt': None,
            
            # Model management
            'uploaded_model': None,
            'model_type': None,
            'model_metadata': {},
            'model_features': [],
            'model_hash': None,
            
            # Dataset management
            'uploaded_dataset': None,
            'dataset_shape': (0, 0),
            'dataset_columns': [],
            'dataset_metadata': {},
            'dataset_hash': None,
            
            # Analysis results cache
            'bias_results': None,
            'privacy_results': None,
            'explainability_results': None,
            'hallucination_results': None,
            'analysis_cache': {},
            
            # Configuration settings
            'protected_attributes': [],
            'target_column': None,
            'fairness_metrics': ['demographic_parity', 'equalized_odds'],
            'bias_threshold': 0.1,
            'privacy_threshold': 0.05,
            'explainability_method': 'shap',
            'hallucination_threshold': 0.6,
            
            # API keys and external services
            'api_keys': {},
            'openai_key': None,
            'anthropic_key': None,
            'external_services_enabled': False,
            
            # Audit statistics and history
            'audit_stats': {
                'total_audits': 0,
                'bias_audits': 0,
                'privacy_audits': 0,
                'explainability_audits': 0,
                'hallucination_audits': 0,
                'issues_found': 0,
                'avg_ethics_score': 0.0,
                'last_audit_date': None,
                'audit_history': [],
                'monthly_audits': 0,
                'recent_audits': 0
            },
            
            # UI preferences
            'theme': 'light',
            'sidebar_collapsed': False,
            'show_advanced_options': False,
            'auto_refresh': True,
            'show_debug_info': False,
            
            # Report management
            'generated_reports': [],
            'report_settings': {
                'auto_generate': True,
                'default_format': 'PDF',
                'include_visualizations': True,
                'retention_days': 90
            },
            
            # Performance and caching
            'cache_enabled': True,
            'cache_expiry': 3600,  # 1 hour
            'last_cache_clear': None,
            
            # Error handling and debugging
            'last_error': None,
            'error_count': 0,
            'debug_mode': False,
            'performance_metrics': {},
            
            # Feature flags
            'feature_flags': {
                'advanced_bias_detection': True,
                'experimental_features': False,
                'beta_hallucination_detection': True,
                'enhanced_privacy_analysis': True
            },
            
            # User preferences
            'user_preferences': {
                'default_upload_format': 'csv',
                'auto_save_results': True,
                'notification_preferences': {
                    'audit_completion': True,
                    'error_alerts': True,
                    'weekly_summary': False
                }
            }
        }
        
        # Initialize all defaults
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Mark as initialized
        st.session_state.state_initialized = True

    @staticmethod
    def update_audit_stats(audit_type: str, issues_count: int, ethics_score: float):
        """Update audit statistics after completing an analysis"""
        
        if 'audit_stats' not in st.session_state:
            StateManager.initialize_session_state()
        
        stats = st.session_state.audit_stats
        
        # Update counters
        stats['total_audits'] += 1
        stats[f'{audit_type}_audits'] = stats.get(f'{audit_type}_audits', 0) + 1
        stats['issues_found'] += issues_count
        
        # Update average ethics score
        current_avg = stats['avg_ethics_score']
        current_total = stats['total_audits']
        stats['avg_ethics_score'] = ((current_avg * (current_total - 1)) + ethics_score) / current_total
        
        # Update timestamps
        stats['last_audit_date'] = datetime.now().isoformat()
        
        # Add to history
        audit_record = {
            'id': StateManager.generate_audit_id(),
            'timestamp': datetime.now().isoformat(),
            'type': audit_type,
            'issues_count': issues_count,
            'ethics_score': ethics_score,
            'model_type': st.session_state.get('model_type', 'unknown'),
            'dataset_size': st.session_state.get('dataset_shape', (0, 0))[0]
        }
        stats['audit_history'].append(audit_record)
        
        # Keep only last 100 records
        if len(stats['audit_history']) > 100:
            stats['audit_history'] = stats['audit_history'][-100:]
        
        # Update monthly counters
        StateManager._update_monthly_stats()
        
        st.session_state.audit_stats = stats

    @staticmethod
    def _update_monthly_stats():
        """Update monthly audit statistics"""
        
        current_month = datetime.now().strftime('%Y-%m')
        monthly_key = f'monthly_audits_{current_month}'
        
        if monthly_key not in st.session_state:
            st.session_state[monthly_key] = 0
        
        st.session_state[monthly_key] += 1
        st.session_state.audit_stats['monthly_audits'] = st.session_state[monthly_key]

    @staticmethod
    def cache_analysis_result(analysis_type: str, input_hash: str, result: Dict[str, Any], 
                             expiry_hours: int = 1):
        """Cache analysis results for performance optimization"""
        
        if not st.session_state.get('cache_enabled', True):
            return
        
        cache_key = f"{analysis_type}_{input_hash}"
        expiry_time = datetime.now() + timedelta(hours=expiry_hours)
        
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        st.session_state.analysis_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'expiry': expiry_time.isoformat(),
            'access_count': 0
        }
        
        # Clean old cache entries
        StateManager._clean_expired_cache()

    @staticmethod
    def get_cached_analysis_result(analysis_type: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result if available and not expired"""
        
        if not st.session_state.get('cache_enabled', True):
            return None
        
        cache_key = f"{analysis_type}_{input_hash}"
        cache = st.session_state.get('analysis_cache', {})
        
        if cache_key not in cache:
            return None
        
        cached_item = cache[cache_key]
        expiry_time = datetime.fromisoformat(cached_item['expiry'])
        
        if datetime.now() > expiry_time:
            # Expired, remove from cache
            del st.session_state.analysis_cache[cache_key]
            return None
        
        # Update access count
        cached_item['access_count'] += 1
        
        return cached_item['result']

    @staticmethod
    def _clean_expired_cache():
        """Remove expired cache entries"""
        
        if 'analysis_cache' not in st.session_state:
            return
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, cached_item in st.session_state.analysis_cache.items():
            expiry_time = datetime.fromisoformat(cached_item['expiry'])
            if current_time > expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del st.session_state.analysis_cache[key]

    @staticmethod
    def clear_cache(cache_type: str = 'all'):
        """Clear cached data"""
        
        if cache_type == 'all' or cache_type == 'analysis':
            st.session_state.analysis_cache = {}
        
        if cache_type == 'all' or cache_type == 'results':
            st.session_state.bias_results = None
            st.session_state.privacy_results = None
            st.session_state.explainability_results = None
            st.session_state.hallucination_results = None
        
        st.session_state.last_cache_clear = datetime.now().isoformat()

    @staticmethod
    def get_session_summary() -> Dict[str, Any]:
        """Get comprehensive summary of current session state"""
        
        return {
            'authentication': {
                'authenticated': st.session_state.get('user_authenticated', False),
                'user': st.session_state.get('current_user', {}).get('username', 'Anonymous'),
                'role': st.session_state.get('current_user', {}).get('role', 'guest')
            },
            'data_loaded': {
                'model': st.session_state.get('uploaded_model') is not None,
                'dataset': st.session_state.get('uploaded_dataset') is not None,
                'model_type': st.session_state.get('model_type', 'None'),
                'dataset_size': st.session_state.get('dataset_shape', (0, 0))
            },
            'audit_status': {
                'total_audits': st.session_state.get('audit_stats', {}).get('total_audits', 0),
                'last_audit': st.session_state.get('audit_stats', {}).get('last_audit_date'),
                'avg_score': st.session_state.get('audit_stats', {}).get('avg_ethics_score', 0.0),
                'pending_results': {
                    'bias': st.session_state.get('bias_results') is not None,
                    'privacy': st.session_state.get('privacy_results') is not None,
                    'explainability': st.session_state.get('explainability_results') is not None,
                    'hallucination': st.session_state.get('hallucination_results') is not None
                }
            },
            'cache_status': {
                'enabled': st.session_state.get('cache_enabled', True),
                'entries': len(st.session_state.get('analysis_cache', {})),
                'last_clear': st.session_state.get('last_cache_clear')
            },
            'preferences': {
                'theme': st.session_state.get('theme', 'light'),
                'advanced_options': st.session_state.get('show_advanced_options', False),
                'debug_mode': st.session_state.get('debug_mode', False)
            }
        }

    @staticmethod
    def export_session_state(include_sensitive: bool = False) -> str:
        """Export session state to JSON string"""
        
        # Keys to exclude from export
        exclude_keys = ['uploaded_model', 'uploaded_dataset', 'analysis_cache']
        
        if not include_sensitive:
            exclude_keys.extend(['api_keys', 'session_token', 'current_user'])
        
        export_data = {}
        for key, value in st.session_state.items():
            if key not in exclude_keys:
                try:
                    # Test if value is JSON serializable
                    json.dumps(value, default=str)
                    export_data[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable items
                    continue
        
        return json.dumps(export_data, indent=2, default=str)

    @staticmethod
    def import_session_state(json_data: str) -> bool:
        """Import session state from JSON string"""
        
        try:
            imported_data = json.loads(json_data)
            
            # Validate imported data
            if not isinstance(imported_data, dict):
                return False
            
            # Import non-sensitive data only
            safe_keys = [
                'theme', 'bias_threshold', 'privacy_threshold', 
                'explainability_method', 'fairness_metrics',
                'protected_attributes', 'report_settings',
                'user_preferences', 'feature_flags'
            ]
            
            for key in safe_keys:
                if key in imported_data:
                    st.session_state[key] = imported_data[key]
            
            return True
            
        except json.JSONDecodeError:
            return False

    @staticmethod
    def reset_session_state(preserve_auth: bool = True):
        """Reset session state to defaults"""
        
        # Keys to preserve across resets
        preserve_keys = []
        if preserve_auth:
            preserve_keys.extend([
                'user_authenticated', 'current_user', 'session_token'
            ])
        
        # Store preserved values
        preserved_values = {}
        for key in preserve_keys:
            if key in st.session_state:
                preserved_values[key] = st.session_state[key]
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Restore preserved values
        for key, value in preserved_values.items():
            st.session_state[key] = value
        
        # Reinitialize
        StateManager.initialize_session_state()

    @staticmethod
    def generate_audit_id() -> str:
        """Generate unique audit ID"""
        
        timestamp = datetime.now().isoformat()
        counter = st.session_state.get('audit_counter', 0)
        st.session_state.audit_counter = counter + 1
        
        id_string = f"audit_{timestamp}_{counter}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]

    @staticmethod
    def calculate_data_hash(data: Any) -> str:
        """Calculate hash for data caching"""
        
        try:
            if hasattr(data, 'to_string'):  # DataFrame
                data_string = data.to_string()
            elif hasattr(data, '__dict__'):  # Model object
                data_string = str(sorted(data.__dict__.items()))
            else:
                data_string = str(data)
            
            return hashlib.md5(data_string.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(id(data)).encode()).hexdigest()

    @staticmethod
    def log_performance_metric(operation: str, duration: float, additional_info: Dict = None):
        """Log performance metrics"""
        
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        
        metrics = st.session_state.performance_metrics
        
        if operation not in metrics:
            metrics[operation] = {
                'count': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'min_duration': float('inf'),
                'max_duration': 0,
                'history': []
            }
        
        op_metrics = metrics[operation]
        op_metrics['count'] += 1
        op_metrics['total_duration'] += duration
        op_metrics['avg_duration'] = op_metrics['total_duration'] / op_metrics['count']
        op_metrics['min_duration'] = min(op_metrics['min_duration'], duration)
        op_metrics['max_duration'] = max(op_metrics['max_duration'], duration)
        
        # Keep last 10 measurements
        op_metrics['history'].append({
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'info': additional_info or {}
        })
        
        if len(op_metrics['history']) > 10:
            op_metrics['history'] = op_metrics['history'][-10:]

    @staticmethod
    def get_performance_summary() -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        metrics = st.session_state.get('performance_metrics', {})
        
        summary = {
            'operations_tracked': len(metrics),
            'operations': {}
        }
        
        for operation, data in metrics.items():
            summary['operations'][operation] = {
                'count': data['count'],
                'avg_duration': round(data['avg_duration'], 3),
                'min_duration': round(data['min_duration'], 3),
                'max_duration': round(data['max_duration'], 3)
            }
        
        return summary

# Initialize state manager on import
if 'state_initialized' not in st.session_state:
    StateManager.initialize_session_state()
