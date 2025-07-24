import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculator:
    """Comprehensive metrics calculation for ethics analysis"""
    
    def __init__(self):
        self.calculated_metrics = {}
        
    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Per-class metrics if multiclass
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 2:
                metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
                metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
                metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
            
            # ROC AUC if probabilities provided
            if y_prob is not None:
                try:
                    if len(unique_classes) == 2:
                        # Binary classification
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                    else:
                        # Multiclass
                        metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                except Exception:
                    metrics['roc_auc'] = None
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
        except Exception as e:
            metrics['error'] = f"Error calculating classification metrics: {str(e)}"
        
        return metrics
    
    def calculate_fairness_metrics_by_group(self,
                                          df: pd.DataFrame,
                                          y_true: np.ndarray,
                                          y_pred: np.ndarray,
                                          protected_attribute: str) -> Dict[str, Any]:
        """Calculate fairness metrics broken down by protected groups"""
        
        fairness_metrics = {
            'groups': {},
            'overall_metrics': {},
            'fairness_gaps': {}
        }
        
        try:
            # Get unique groups
            groups = df[protected_attribute].unique()
            
            # Calculate metrics for each group
            for group in groups:
                mask = df[protected_attribute] == group
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                
                if len(group_y_true) > 0:
                    group_metrics = self.calculate_classification_metrics(group_y_true, group_y_pred)
                    group_metrics['group_size'] = len(group_y_true)
                    group_metrics['positive_rate'] = np.mean(group_y_pred)
                    
                    fairness_metrics['groups'][str(group)] = group_metrics
            
            # Calculate fairness gaps
            fairness_metrics['fairness_gaps'] = self._calculate_fairness_gaps(
                fairness_metrics['groups']
            )
            
            # Overall metrics
            fairness_metrics['overall_metrics'] = self.calculate_classification_metrics(y_true, y_pred)
            
        except Exception as e:
            fairness_metrics['error'] = f"Error calculating fairness metrics: {str(e)}"
        
        return fairness_metrics
    
    def _calculate_fairness_gaps(self, group_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate gaps between groups for fairness assessment"""
        
        gaps = {}
        
        if len(group_metrics) < 2:
            return gaps
        
        # Extract metrics for comparison
        accuracy_values = [metrics.get('accuracy', 0) for metrics in group_metrics.values()]
        precision_values = [metrics.get('precision', 0) for metrics in group_metrics.values()]
        recall_values = [metrics.get('recall', 0) for metrics in group_metrics.values()]
        positive_rates = [metrics.get('positive_rate', 0) for metrics in group_metrics.values()]
        
        # Calculate gaps (max - min)
        gaps['accuracy_gap'] = max(accuracy_values) - min(accuracy_values)
        gaps['precision_gap'] = max(precision_values) - min(precision_values)
        gaps['recall_gap'] = max(recall_values) - min(recall_values)
        gaps['positive_rate_gap'] = max(positive_rates) - min(positive_rates)
        
        return gaps
    
    def calculate_privacy_metrics(self, 
                                pii_findings: List[Dict[str, Any]],
                                text_length: int = None) -> Dict[str, Any]:
        """Calculate privacy-related metrics"""
        
        privacy_metrics = {
            'total_pii_instances': len(pii_findings),
            'pii_types_detected': len(set(finding.get('type', '') for finding in pii_findings)),
            'high_confidence_pii': len([f for f in pii_findings if f.get('confidence', 0) > 0.8]),
            'severity_breakdown': {},
            'confidence_distribution': {}
        }
        
        if pii_findings:
            # Severity breakdown
            severity_counts = {}
            for finding in pii_findings:
                severity = finding.get('severity', 'Unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            privacy_metrics['severity_breakdown'] = severity_counts
            
            # Confidence distribution
            confidences = [f.get('confidence', 0) for f in pii_findings]
            privacy_metrics['confidence_distribution'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            
            # PII density (if text length provided)
            if text_length:
                privacy_metrics['pii_density'] = len(pii_findings) / (text_length / 1000)  # per 1000 chars
        
        return privacy_metrics
    
    def calculate_explainability_metrics(self, 
                                       shap_values: Optional[np.ndarray] = None,
                                       feature_importance: Optional[np.ndarray] = None,
                                       explanation_quality_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Calculate explainability-related metrics"""
        
        explainability_metrics = {}
        
        if shap_values is not None:
            # SHAP-based metrics
            explainability_metrics['shap_metrics'] = {
                'mean_absolute_shap': np.mean(np.abs(shap_values)),
                'shap_variance': np.var(shap_values),
                'feature_attribution_consistency': self._calculate_attribution_consistency(shap_values)
            }
        
        if feature_importance is not None:
            # Feature importance metrics
            explainability_metrics['feature_importance_metrics'] = {
                'top_feature_dominance': np.max(feature_importance) / np.sum(feature_importance),
                'feature_concentration': self._calculate_gini_coefficient(feature_importance),
                'effective_features': np.sum(feature_importance > 0.01)  # Features with >1% importance
            }
        
        if explanation_quality_scores is not None:
            # Explanation quality metrics
            explainability_metrics['explanation_quality'] = {
                'mean_quality': np.mean(explanation_quality_scores),
                'quality_consistency': 1 - np.std(explanation_quality_scores),  # Lower std = higher consistency
                'high_quality_explanations': np.sum(np.array(explanation_quality_scores) > 0.8)
            }
        
        return explainability_metrics
    
    def calculate_hallucination_metrics(self, 
                                      hallucination_findings: List[Dict[str, Any]],
                                      fact_check_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate hallucination detection metrics"""
        
        hallucination_metrics = {
            'total_hallucinations': len(hallucination_findings),
            'hallucination_types': {},
            'severity_distribution': {},
            'confidence_stats': {}
        }
        
        if hallucination_findings:
            # Hallucination types
            type_counts = {}
            for finding in hallucination_findings:
                hallucination_type = finding.get('type', 'Unknown')
                type_counts[hallucination_type] = type_counts.get(hallucination_type, 0) + 1
            hallucination_metrics['hallucination_types'] = type_counts
            
            # Severity distribution
            severity_counts = {}
            for finding in hallucination_findings:
                severity = finding.get('severity', 'Unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            hallucination_metrics['severity_distribution'] = severity_counts
            
            # Confidence statistics
            confidences = [f.get('confidence', 0) for f in hallucination_findings]
            hallucination_metrics['confidence_stats'] = {
                'mean_confidence': np.mean(confidences),
                'high_confidence_detections': np.sum(np.array(confidences) > 0.8)
            }
        
        # Fact-checking metrics
        if fact_check_results:
            verified_claims = len([r for r in fact_check_results if r.get('status') == 'Verified'])
            false_claims = len([r for r in fact_check_results if r.get('status') == 'False'])
            total_claims = len(fact_check_results)
            
            hallucination_metrics['fact_checking'] = {
                'total_claims_checked': total_claims,
                'verified_claims': verified_claims,
                'false_claims': false_claims,
                'verification_rate': verified_claims / total_claims if total_claims > 0 else 0,
                'false_claim_rate': false_claims / total_claims if total_claims > 0 else 0
            }
        
        return hallucination_metrics
    
    def _calculate_attribution_consistency(self, shap_values: np.ndarray) -> float:
        """Calculate consistency of feature attributions across instances"""
        
        # Calculate correlation between SHAP values across features
        if shap_values.shape[0] < 2:
            return 1.0
        
        # Calculate pairwise correlations between instances
        correlations = []
        for i in range(min(10, shap_values.shape[0])):  # Limit to 10 instances for efficiency
            for j in range(i + 1, min(10, shap_values.shape[0])):
                corr = np.corrcoef(shap_values[i], shap_values[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for measuring concentration"""
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    def calculate_model_performance_summary(self,
                                          classification_metrics: Dict[str, Any],
                                          fairness_metrics: Dict[str, Any],
                                          privacy_metrics: Dict[str, Any] = None,
                                          explainability_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive model performance summary"""
        
        summary = {
            'overall_score': 0.0,
            'component_scores': {},
            'recommendations': []
        }
        
        # Performance score (accuracy, precision, recall)
        performance_score = classification_metrics.get('accuracy', 0) * 10
        summary['component_scores']['performance'] = performance_score
        
        # Fairness score (inverse of maximum gap)
        fairness_gaps = fairness_metrics.get('fairness_gaps', {})
        max_gap = max(fairness_gaps.values()) if fairness_gaps else 0
        fairness_score = max(0, 10 - (max_gap * 50))  # Scale gap to 0-10
        summary['component_scores']['fairness'] = fairness_score
        
        # Privacy score
        if privacy_metrics:
            pii_density = privacy_metrics.get('pii_density', 0)
            privacy_score = max(0, 10 - min(pii_density * 2, 10))
            summary['component_scores']['privacy'] = privacy_score
        
        # Explainability score
        if explainability_metrics:
            explanation_quality = explainability_metrics.get('explanation_quality', {})
            mean_quality = explanation_quality.get('mean_quality', 0.5)
            explainability_score = mean_quality * 10
            summary['component_scores']['explainability'] = explainability_score
        
        # Calculate overall score (weighted average)
        weights = {
            'performance': 0.3,
            'fairness': 0.4,
            'privacy': 0.15,
            'explainability': 0.15
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for component, score in summary['component_scores'].items():
            weight = weights.get(component, 0)
            weighted_sum += score * weight
            total_weight += weight
        
        summary['overall_score'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Generate recommendations
        summary['recommendations'] = self._generate_performance_recommendations(summary['component_scores'])
        
        return summary
    
    def _generate_performance_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on component scores"""
        
        recommendations = []
        
        for component, score in component_scores.items():
            if score < 6:
                if component == 'performance':
                    recommendations.append("Improve model accuracy through better feature engineering or model selection")
                elif component == 'fairness':
                    recommendations.append("Address fairness issues through bias mitigation techniques")
                elif component == 'privacy':
                    recommendations.append("Implement privacy-preserving techniques to reduce PII exposure")
                elif component == 'explainability':
                    recommendations.append("Enhance model interpretability through better explanation methods")
        
        if not recommendations:
            recommendations.append("Model shows good performance across all ethics dimensions")
        
        return recommendations
    
    def export_metrics_report(self, 
                            metrics_dict: Dict[str, Any],
                            format_type: str = 'dict') -> Union[Dict[str, Any], str]:
        """Export metrics in specified format"""
        
        if format_type == 'dict':
            return metrics_dict
        elif format_type == 'json':
            import json
            return json.dumps(metrics_dict, indent=2, default=str)
        elif format_type == 'summary':
            return self._format_metrics_summary(metrics_dict)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_metrics_summary(self, metrics_dict: Dict[str, Any]) -> str:
        """Format metrics as human-readable summary"""
        
        summary = "METRICS SUMMARY\n"
        summary += "=" * 15 + "\n\n"
        
        for category, metrics in metrics_dict.items():
            summary += f"{category.upper()}:\n"
            summary += "-" * len(category) + "\n"
            
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        summary += f"  {metric}: {value:.3f}\n"
                    else:
                        summary += f"  {metric}: {value}\n"
            else:
                summary += f"  {metrics}\n"
            
            summary += "\n"
        
        return summary
