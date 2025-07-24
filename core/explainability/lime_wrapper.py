import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from typing import Dict, List, Any, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class LIMEWrapper:
    """Enhanced LIME wrapper for local model explanations"""
    
    def __init__(self, model, training_data: pd.DataFrame, config: Dict[str, Any]):
        self.model = model
        self.training_data = training_data
        self.config = config
        self.explainer = None
        self._initialize_explainer()
        
    def _initialize_explainer(self):
        """Initialize LIME explainer"""
        
        try:
            # Determine if it's classification or regression
            mode = self.config.get('mode', 'classification')
            
            # Get categorical features
            categorical_features = self._identify_categorical_features()
            
            # Initialize LIME explainer
            self.explainer = LimeTabularExplainer(
                training_data=self.training_data.values,
                feature_names=list(self.training_data.columns),
                class_names=['Negative', 'Positive'] if mode == 'classification' else None,
                mode=mode,
                categorical_features=categorical_features,
                discretize_continuous=True,
                random_state=42
            )
            
        except Exception as e:
            print(f"Failed to initialize LIME explainer: {str(e)}")
            self.explainer = None
    
    def _identify_categorical_features(self) -> List[int]:
        """Identify categorical features in the dataset"""
        
        categorical_indices = []
        
        for i, column in enumerate(self.training_data.columns):
            # Check if column is categorical or has low cardinality
            if (self.training_data[column].dtype == 'object' or
                self.training_data[column].dtype.name == 'category' or
                self.training_data[column].nunique() <= 10):
                categorical_indices.append(i)
        
        return categorical_indices
    
    def explain_instance(self, 
                        instance: pd.Series,
                        num_features: int = 10,
                        num_samples: int = 5000) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance
        
        Args:
            instance: The instance to explain (pandas Series)
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for local model
            
        Returns:
            Dictionary containing the explanation
        """
        
        if self.explainer is None:
            return {'error': 'LIME explainer not initialized'}
        
        try:
            # Determine prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                data_row=instance.values,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract explanation data
            feature_importance = explanation.as_list()
            
            # Get prediction for this instance
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba([instance.values])[0]
                prediction = np.argmax(prediction_proba)
                confidence = np.max(prediction_proba)
            else:
                prediction = self.model.predict([instance.values])[0]
                prediction_proba = None
                confidence = None
            
            # Create structured explanation
            explanation_dict = {
                'instance_id': hash(str(instance.values)) % 10000,  # Simple ID
                'prediction': prediction,
                'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None,
                'confidence': confidence,
                'local_accuracy': explanation.score,
                'feature_importance': {
                    'features': [item[0] for item in feature_importance],
                    'values': [item[1] for item in feature_importance]
                },
                'instance_data': instance.to_dict(),
                'explanation_quality': self._assess_explanation_quality(explanation),
                'natural_explanation': self._generate_natural_explanation(feature_importance)
            }
            
            return explanation_dict
            
        except Exception as e:
            return {'error': f'Failed to generate explanation: {str(e)}'}
    
    def explain_multiple_instances(self, 
                                  instances: pd.DataFrame,
                                  num_features: int = 10) -> List[Dict[str, Any]]:
        """Generate explanations for multiple instances"""
        
        explanations = []
        
        for idx, (_, instance) in enumerate(instances.iterrows()):
            explanation = self.explain_instance(instance, num_features)
            explanation['batch_index'] = idx
            explanations.append(explanation)
            
            # Limit to prevent excessive computation
            if idx >= 10:
                break
        
        return explanations
    
    def _assess_explanation_quality(self, explanation) -> Dict[str, Any]:
        """Assess the quality of the LIME explanation"""
        
        try:
            # Get local accuracy (R² score for the local model)
            local_accuracy = explanation.score
            
            # Assess feature importance distribution
            feature_values = [abs(item[1]) for item in explanation.as_list()]
            importance_variance = np.var(feature_values) if feature_values else 0
            
            # Quality assessment
            quality = {
                'local_accuracy': local_accuracy,
                'importance_variance': importance_variance,
                'quality_score': self._calculate_quality_score(local_accuracy, importance_variance),
                'reliability': 'High' if local_accuracy > 0.8 else 'Medium' if local_accuracy > 0.6 else 'Low'
            }
            
            return quality
            
        except Exception as e:
            return {'error': f'Failed to assess explanation quality: {str(e)}'}
    
    def _calculate_quality_score(self, accuracy: float, variance: float) -> float:
        """Calculate overall quality score for the explanation"""
        
        # Normalize variance (higher variance = more informative explanation)
        normalized_variance = min(variance / 0.1, 1.0) if variance > 0 else 0
        
        # Combine accuracy and informativeness
        quality_score = (accuracy * 0.7) + (normalized_variance * 0.3)
        
        return round(quality_score, 3)
    
    def _generate_natural_explanation(self, feature_importance: List[tuple]) -> str:
        """Generate natural language explanation"""
        
        if not feature_importance:
            return "No significant features identified for this prediction."
        
        # Sort by absolute importance
        sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 3 features
        top_features = sorted_features[:3]
        
        explanations = []
        
        for feature, importance in top_features:
            if importance > 0:
                explanations.append(f"{feature} increases the likelihood of the prediction")
            else:
                explanations.append(f"{feature} decreases the likelihood of the prediction")
        
        if explanations:
            return "Key factors: " + "; ".join(explanations)
        else:
            return "The prediction is based on a complex combination of features."
    
    def get_feature_selection_insights(self, instances: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which features are most commonly selected by LIME"""
        
        feature_selection_count = {}
        total_explanations = 0
        
        # Generate explanations for sample instances
        sample_size = min(20, len(instances))
        sample_instances = instances.sample(n=sample_size, random_state=42)
        
        for _, instance in sample_instances.iterrows():
            explanation = self.explain_instance(instance, num_features=5)
            
            if 'feature_importance' in explanation:
                total_explanations += 1
                
                for feature in explanation['feature_importance']['features']:
                    feature_selection_count[feature] = feature_selection_count.get(feature, 0) + 1
        
        # Calculate selection frequencies
        feature_frequency = {}
        for feature, count in feature_selection_count.items():
            feature_frequency[feature] = count / total_explanations if total_explanations > 0 else 0
        
        # Sort by frequency
        sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_explanations': total_explanations,
            'feature_selection_frequency': dict(sorted_features),
            'most_important_features': [feat for feat, freq in sorted_features[:5]],
            'consistency_score': self._calculate_consistency_score(feature_frequency)
        }
    
    def _calculate_consistency_score(self, feature_frequency: Dict[str, float]) -> float:
        """Calculate consistency score based on feature selection frequency"""
        
        if not feature_frequency:
            return 0.0
        
        frequencies = list(feature_frequency.values())
        
        # High consistency if a few features are selected very frequently
        # Low consistency if selection is very spread out
        
        # Calculate coefficient of variation
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        if mean_freq == 0:
            return 0.0
        
        cv = std_freq / mean_freq
        
        # Convert to 0-1 scale where higher CV = higher consistency
        consistency_score = min(cv / 2.0, 1.0)
        
        return round(consistency_score, 3)
    
    def compare_explanations(self, 
                           instance1: pd.Series, 
                           instance2: pd.Series) -> Dict[str, Any]:
        """Compare LIME explanations between two instances"""
        
        explanation1 = self.explain_instance(instance1)
        explanation2 = self.explain_instance(instance2)
        
        if 'error' in explanation1 or 'error' in explanation2:
            return {'error': 'Failed to generate one or both explanations'}
        
        # Compare feature importance
        features1 = set(explanation1['feature_importance']['features'])
        features2 = set(explanation2['feature_importance']['features'])
        
        common_features = features1.intersection(features2)
        unique_to_1 = features1 - features2
        unique_to_2 = features2 - features1
        
        # Calculate explanation similarity
        similarity_score = len(common_features) / len(features1.union(features2)) if features1.union(features2) else 0
        
        return {
            'instance1_prediction': explanation1['prediction'],
            'instance2_prediction': explanation2['prediction'],
            'common_important_features': list(common_features),
            'unique_to_instance1': list(unique_to_1),
            'unique_to_instance2': list(unique_to_2),
            'explanation_similarity': similarity_score,
            'prediction_difference': abs(explanation1.get('confidence', 0) - explanation2.get('confidence', 0)),
            'comparison_summary': self._generate_comparison_summary(
                explanation1, explanation2, similarity_score
            )
        }
    
    def _generate_comparison_summary(self, 
                                   exp1: Dict[str, Any], 
                                   exp2: Dict[str, Any], 
                                   similarity: float) -> str:
        """Generate summary of explanation comparison"""
        
        if similarity > 0.7:
            summary = "The two instances have very similar explanations, suggesting consistent model behavior."
        elif similarity > 0.4:
            summary = "The instances show moderately similar explanations with some key differences."
        else:
            summary = "The instances have very different explanations, indicating distinct decision patterns."
        
        # Add prediction comparison
        pred1 = exp1.get('prediction', 'unknown')
        pred2 = exp2.get('prediction', 'unknown')
        
        if pred1 == pred2:
            summary += f" Both instances received the same prediction ({pred1})."
        else:
            summary += f" The instances received different predictions ({pred1} vs {pred2})."
        
        return summary
