import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SHAPWrapper:
    """Enhanced SHAP wrapper for comprehensive model explanations"""
    
    def __init__(self, model, data: pd.DataFrame, config: Dict[str, Any]):
        self.model = model
        self.data = data
        self.config = config
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self) -> bool:
        """Initialize appropriate SHAP explainer based on model type"""
        
        try:
            model_type = type(self.model).__name__.lower()
            
            if any(tree_type in model_type for tree_type in ['forest', 'tree', 'xgb', 'lgb']):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = 'TreeExplainer'
                
            elif any(linear_type in model_type for linear_type in ['linear', 'logistic', 'ridge', 'lasso']):
                self.explainer = shap.LinearExplainer(self.model, self.data)
                self.explainer_type = 'LinearExplainer'
                
            elif hasattr(self.model, 'predict_proba'):
                # Use KernelExplainer as fallback for any model with predict_proba
                background_data = shap.sample(self.data, min(100, len(self.data)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
                self.explainer_type = 'KernelExplainer'
                
            else:
                # Final fallback - use predict method
                background_data = shap.sample(self.data, min(100, len(self.data)))
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                self.explainer_type = 'KernelExplainer'
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize SHAP explainer: {str(e)}")
            return False
    
    def calculate_shap_values(self, data: Optional[pd.DataFrame] = None) -> bool:
        """Calculate SHAP values for given data"""
        
        if self.explainer is None:
            if not self.initialize_explainer():
                return False
        
        if data is None:
            data = self.data
        
        try:
            # Calculate SHAP values
            if self.explainer_type == 'TreeExplainer':
                self.shap_values = self.explainer.shap_values(data)
            else:
                self.shap_values = self.explainer.shap_values(data)
            
            # Handle multi-class case
            if isinstance(self.shap_values, list):
                # For binary classification, take positive class
                if len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]
                else:
                    # For multi-class, might need different handling
                    self.shap_values = self.shap_values[0]  # Take first class for now
            
            return True
            
        except Exception as e:
            print(f"Failed to calculate SHAP values: {str(e)}")
            return False
    
    def get_global_importance(self) -> Dict[str, Any]:
        """Get global feature importance from SHAP values"""
        
        if self.shap_values is None:
            if not self.calculate_shap_values():
                return {}
        
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            
            # Create feature importance ranking
            feature_names = list(self.data.columns)
            importance_pairs = list(zip(feature_names, mean_abs_shap))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'features': [pair[0] for pair in importance_pairs],
                'importance': [pair[1] for pair in importance_pairs],
                'feature_rankings': {feat: rank for rank, (feat, _) in enumerate(importance_pairs, 1)},
                'total_importance': np.sum(mean_abs_shap)
            }
            
        except Exception as e:
            print(f"Failed to calculate global importance: {str(e)}")
            return {}
    
    def get_local_explanations(self, instances: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get local explanations for specific instances"""
        
        if self.shap_values is None:
            if not self.calculate_shap_values():
                return []
        
        if instances is None:
            instances = list(range(min(5, len(self.data))))
        
        explanations = []
        
        try:
            for idx in instances:
                if idx >= len(self.shap_values):
                    continue
                
                instance_shap = self.shap_values[idx]
                instance_data = self.data.iloc[idx]
                
                # Get feature contributions
                feature_contributions = list(zip(self.data.columns, instance_shap))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                explanation = {
                    'instance_id': idx,
                    'instance_data': instance_data.to_dict(),
                    'shap_values': instance_shap.tolist(),
                    'feature_contributions': {
                        'features': [contrib[0] for contrib in feature_contributions],
                        'values': [contrib[1] for contrib in feature_contributions]
                    },
                    'prediction_explanation': self._generate_natural_explanation(
                        feature_contributions[:5]  # Top 5 features
                    )
                }
                
                # Add prediction if possible
                try:
                    if hasattr(self.model, 'predict'):
                        prediction = self.model.predict([instance_data])[0]
                        explanation['prediction'] = prediction
                        
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba([instance_data])[0]
                        explanation['prediction_probability'] = probabilities.tolist()
                        explanation['confidence'] = max(probabilities)
                        
                except Exception:
                    pass  # Skip prediction if it fails
                
                explanations.append(explanation)
                
        except Exception as e:
            print(f"Failed to generate local explanations: {str(e)}")
        
        return explanations
    
    def get_feature_interactions(self) -> Dict[str, Any]:
        """Analyze feature interactions using SHAP"""
        
        if not self.config.get('include_interactions', False):
            return {}
        
        try:
            # This is a simplified interaction analysis
            # Full implementation would use SHAP interaction values
            
            if self.shap_values is None:
                if not self.calculate_shap_values():
                    return {}
            
            # Calculate correlation between SHAP values as proxy for interactions
            shap_df = pd.DataFrame(self.shap_values, columns=self.data.columns)
            correlation_matrix = shap_df.corr()
            
            # Find strongest interactions (highest absolute correlations)
            interactions = []
            for i, feature1 in enumerate(self.data.columns):
                for j, feature2 in enumerate(self.data.columns):
                    if i < j:  # Avoid duplicates
                        correlation = correlation_matrix.iloc[i, j]
                        if abs(correlation) > 0.3:  # Threshold for significant interaction
                            interactions.append({
                                'feature1': feature1,
                                'feature2': feature2,
                                'interaction_strength': abs(correlation),
                                'interaction_type': 'positive' if correlation > 0 else 'negative'
                            })
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            return {
                'interactions_found': len(interactions),
                'top_interactions': interactions[:10],  # Top 10 interactions
                'interaction_matrix': correlation_matrix.to_dict()
            }
            
        except Exception as e:
            print(f"Failed to analyze feature interactions: {str(e)}")
            return {}
    
    def get_summary_plot_data(self) -> pd.DataFrame:
        """Prepare data for SHAP summary plot"""
        
        if self.shap_values is None:
            if not self.calculate_shap_values():
                return pd.DataFrame()
        
        try:
            summary_data = []
            
            for i, feature in enumerate(self.data.columns):
                for j in range(len(self.shap_values)):
                    summary_data.append({
                        'feature': feature,
                        'shap_value': self.shap_values[j, i],
                        'feature_value': self.data.iloc[j, i],
                        'abs_shap_value': abs(self.shap_values[j, i]),
                        'instance_id': j
                    })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            print(f"Failed to prepare summary plot data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_natural_explanation(self, feature_contributions: List[Tuple[str, float]]) -> str:
        """Generate natural language explanation from feature contributions"""
        
        if not feature_contributions:
            return "No significant feature contributions found."
        
        explanations = []
        
        for feature, contribution in feature_contributions:
            if contribution > 0:
                explanations.append(f"{feature} strongly supports the prediction")
            elif contribution < 0:
                explanations.append(f"{feature} strongly opposes the prediction")
        
        if explanations:
            return "This prediction is influenced by: " + "; ".join(explanations[:3])
        else:
            return "Feature contributions are balanced with no dominant influences."
    
    def get_dependence_plot_data(self, feature: str) -> Dict[str, Any]:
        """Get data for SHAP dependence plot for a specific feature"""
        
        if self.shap_values is None:
            if not self.calculate_shap_values():
                return {}
        
        try:
            if feature not in self.data.columns:
                return {}
            
            feature_idx = list(self.data.columns).index(feature)
            
            return {
                'feature': feature,
                'feature_values': self.data[feature].tolist(),
                'shap_values': self.shap_values[:, feature_idx].tolist(),
                'interaction_feature': self._find_strongest_interaction_feature(feature_idx),
                'plot_title': f"SHAP Dependence Plot: {feature}"
            }
            
        except Exception as e:
            print(f"Failed to prepare dependence plot data: {str(e)}")
            return {}
    
    def _find_strongest_interaction_feature(self, feature_idx: int) -> Optional[str]:
        """Find the feature that interacts most strongly with the given feature"""
        
        try:
            # Calculate correlation between target feature's SHAP values and other features
            target_shap = self.shap_values[:, feature_idx]
            
            max_correlation = 0
            strongest_feature = None
            
            for i, feature in enumerate(self.data.columns):
                if i != feature_idx:
                    correlation = np.corrcoef(target_shap, self.data.iloc[:, i])[0, 1]
                    if abs(correlation) > max_correlation:
                        max_correlation = abs(correlation)
                        strongest_feature = feature
            
            return strongest_feature
            
        except Exception:
            return None
    
    def get_waterfall_data(self, instance_idx: int) -> Dict[str, Any]:
        """Get data for SHAP waterfall plot for a specific instance"""
        
        if self.shap_values is None:
            if not self.calculate_shap_values():
                return {}
        
        try:
            if instance_idx >= len(self.shap_values):
                return {}
            
            instance_shap = self.shap_values[instance_idx]
            instance_data = self.data.iloc[instance_idx]
            
            # Get base value (expected value)
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0] if len(base_value) > 0 else 0
            else:
                base_value = 0
            
            # Create waterfall data
            contributions = []
            cumulative_value = base_value
            
            # Sort features by absolute contribution
            feature_contributions = list(zip(self.data.columns, instance_shap, instance_data))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, shap_val, feature_val in feature_contributions:
                contributions.append({
                    'feature': feature,
                    'feature_value': feature_val,
                    'shap_value': shap_val,
                    'cumulative_value': cumulative_value + shap_val
                })
                cumulative_value += shap_val
            
            return {
                'instance_id': instance_idx,
                'base_value': base_value,
                'final_value': cumulative_value,
                'contributions': contributions,
                'prediction_change': cumulative_value - base_value
            }
            
        except Exception as e:
            print(f"Failed to prepare waterfall data: {str(e)}")
            return {}
