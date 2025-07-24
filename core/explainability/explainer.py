import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Main explainability engine for AI models"""
    
    def __init__(self, model, dataset: pd.DataFrame, config: Dict[str, Any]):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.X = None
        self.y = None
        self.feature_names = []
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for explainability analysis"""
        target_col = self.config['target_column']
        selected_features = self.config['selected_features']
        sample_size = self.config['sample_size']
        
        # Sample data if needed
        if len(self.dataset) > sample_size:
            sampled_data = self.dataset.sample(n=sample_size, random_state=42)
        else:
            sampled_data = self.dataset.copy()
        
        # Prepare features and target
        self.X = sampled_data[selected_features]
        self.y = sampled_data[target_col]
        self.feature_names = selected_features
    
    def generate_shap_explanations(self) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        explainer_type = self.config.get('explainer_type', 'KernelExplainer')
        
        try:
            # Initialize SHAP explainer based on model type
            if explainer_type == 'TreeExplainer' and hasattr(self.model, 'estimators_'):
                explainer = shap.TreeExplainer(self.model)
            elif explainer_type == 'LinearExplainer':
                explainer = shap.LinearExplainer(self.model, self.X)
            else:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(self.model.predict, self.X.sample(100))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self.X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Generate results
            results = {
                'explainer_type': explainer_type,
                'global_importance': self._calculate_global_importance(shap_values),
                'local_explanations': self._generate_local_explanations(shap_values),
                'summary_plot_data': self._prepare_summary_plot_data(shap_values),
                'overall_interpretability_score': self._calculate_interpretability_score(shap_values)
            }
            
            return results
            
        except Exception as e:
            # Fallback to simpler method
            return self._fallback_explanations(f"SHAP error: {str(e)}")
    
    def generate_lime_explanations(self) -> Dict[str, Any]:
        """Generate LIME explanations"""
        from lime.lime_tabular import LimeTabularExplainer
        
        try:
            # Initialize LIME explainer
            explainer = LimeTabularExplainer(
                self.X.values,
                feature_names=self.feature_names,
                class_names=['Negative', 'Positive'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Generate explanations for sample instances
            num_instances = min(5, len(self.X))
            local_explanations = []
            
            for i in range(num_instances):
                explanation = explainer.explain_instance(
                    self.X.iloc[i].values,
                    self.model.predict_proba,
                    num_features=self.config.get('num_features', 10)
                )
                
                # Extract explanation data
                feature_importance = explanation.as_list()
                
                local_explanations.append({
                    'instance_id': i,
                    'prediction': self.model.predict([self.X.iloc[i].values])[0],
                    'instance_data': self.X.iloc[i].to_dict(),
                    'feature_importance': {
                        'features': [item[0] for item in feature_importance],
                        'values': [item[1] for item in feature_importance]
                    }
                })
            
            results = {
                'method': 'LIME',
                'local_explanations': local_explanations,
                'overall_interpretability_score': 8.5  # LIME generally provides good local interpretability
            }
            
            return results
            
        except Exception as e:
            return self._fallback_explanations(f"LIME error: {str(e)}")
    
    def generate_custom_explanations(self) -> Dict[str, Any]:
        """Generate custom explainability methods"""
        methods = self.config.get('methods', [])
        results = {
            'methods_used': methods,
            'method_results': {},
            'overall_interpretability_score': 7.0
        }
        
        for method in methods:
            if method == "Permutation Importance":
                results['method_results'][method] = self._permutation_importance()
            elif method == "Partial Dependence Plots":
                results['method_results'][method] = self._partial_dependence()
            # Add other methods as needed
        
        return results
    
    def generate_comparison_explanations(self) -> Dict[str, Any]:
        """Generate comparison between multiple methods"""
        methods = self.config.get('methods', [])
        method_results = {}
        comparison_summary = []
        
        for method in methods:
            if 'SHAP' in method:
                method_results[method] = self.generate_shap_explanations()
            elif 'LIME' in method:
                method_results[method] = self.generate_lime_explanations()
            elif 'Permutation' in method:
                method_results[method] = self.generate_custom_explanations()
        
        # Generate comparison metrics
        for method, result in method_results.items():
            comparison_summary.append({
                'Method': method,
                'Interpretability Score': result.get('overall_interpretability_score', 'N/A'),
                'Explanation Type': 'Global & Local' if 'SHAP' in method else 'Local' if 'LIME' in method else 'Global',
                'Computation Time': 'Fast' if 'Permutation' in method else 'Medium'
            })
        
        return {
            'methods_compared': methods,
            'method_results': method_results,
            'comparison_summary': comparison_summary,
            'overall_interpretability_score': 8.0
        }
    
    def _calculate_global_importance(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Calculate global feature importance from SHAP values"""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort by importance
        importance_indices = np.argsort(mean_abs_shap)[::-1]
        
        return {
            'features': [self.feature_names[i] for i in importance_indices],
            'importance': mean_abs_shap[importance_indices].tolist()
        }
    
    def _generate_local_explanations(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Generate local explanations for individual instances"""
        local_explanations = []
        num_instances = min(5, len(shap_values))
        
        for i in range(num_instances):
            instance_shap = shap_values[i]
            
            # Sort features by absolute SHAP value
            feature_indices = np.argsort(np.abs(instance_shap))[::-1]
            
            local_explanations.append({
                'instance_id': i,
                'prediction': self.model.predict([self.X.iloc[i].values])[0],
                'instance_data': self.X.iloc[i].to_dict(),
                'feature_importance': {
                    'features': [self.feature_names[idx] for idx in feature_indices],
                    'values': instance_shap[feature_indices].tolist()
                }
            })
        
        return local_explanations
    
    def _prepare_summary_plot_data(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Prepare data for SHAP summary plot"""
        summary_data = []
        
        for i, feature in enumerate(self.feature_names):
            for j in range(len(shap_values)):
                summary_data.append({
                    'feature': feature,
                    'shap_value': shap_values[j, i],
                    'feature_value': self.X.iloc[j, i],
                    'abs_shap_value': abs(shap_values[j, i])
                })
        
        return pd.DataFrame(summary_data)
    
    def _permutation_importance(self) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            perm_importance = permutation_importance(
                self.model, self.X, self.y,
                n_repeats=5, random_state=42
            )
            
            return {
                'features': self.feature_names,
                'importance_scores': perm_importance.importances_mean.tolist(),
                'importance_std': perm_importance.importances_std.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _partial_dependence(self) -> Dict[str, Any]:
        """Calculate partial dependence (simplified)"""
        # Simplified implementation - in production use sklearn.inspection.partial_dependence
        return {
            'message': 'Partial dependence plots would be implemented here',
            'features_analyzed': self.feature_names[:5]  # Top 5 features
        }
    
    def _calculate_interpretability_score(self, shap_values: np.ndarray) -> float:
        """Calculate overall interpretability score"""
        # Simple heuristic based on feature importance distribution
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Higher score if importance is well distributed (not dominated by few features)
        importance_gini = self._calculate_gini_coefficient(mean_abs_shap)
        
        # Score between 6-10 based on interpretability factors
        base_score = 8.0
        distribution_bonus = (1 - importance_gini) * 2  # Bonus for well-distributed importance
        
        return min(10.0, base_score + distribution_bonus)
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for importance distribution"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _fallback_explanations(self, error_msg: str) -> Dict[str, Any]:
        """Fallback explanations when main methods fail"""
        # Use feature importance from the model if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            
            # Sort by importance
            importance_indices = np.argsort(feature_importance)[::-1]
            
            return {
                'method': 'Model Feature Importance (Fallback)',
                'error': error_msg,
                'global_importance': {
                    'features': [self.feature_names[i] for i in importance_indices],
                    'importance': feature_importance[importance_indices].tolist()
                },
                'overall_interpretability_score': 6.0
            }
        else:
            return {
                'method': 'Basic Analysis (Fallback)',
                'error': error_msg,
                'message': 'Advanced explainability methods failed. Model does not provide feature importance.',
                'overall_interpretability_score': 4.0
            }
