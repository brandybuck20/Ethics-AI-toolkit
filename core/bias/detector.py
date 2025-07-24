import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
from .metrics import FairnessMetrics
from ..shared.metrics_calculator import MetricsCalculator

class BiasDetector:
    """Main bias detection engine for demographic fairness analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fairness_calculator = FairnessMetrics()
        self.metrics_calculator = MetricsCalculator()
        
    def detect_bias(self, 
                   df: pd.DataFrame,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Main bias detection function
        
        Args:
            df: Dataset with protected attributes
            y_true: True labels
            y_pred: Predicted labels  
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dictionary containing bias analysis results
        """
        
        results = {
            'overall_bias_score': 0.0,
            'protected_attribute_analysis': {},
            'intersectional_analysis': {},
            'bias_summary': {},
            'recommendations': []
        }
        
        # Analyze each protected attribute
        total_bias_score = 0
        for attr in protected_attributes:
            attr_analysis = self._analyze_protected_attribute(
                df, y_true, y_pred, attr
            )
            results['protected_attribute_analysis'][attr] = attr_analysis
            total_bias_score += attr_analysis['bias_score']
        
        # Calculate overall bias score
        results['overall_bias_score'] = total_bias_score / len(protected_attributes)
        
        # Perform intersectional analysis if multiple attributes
        if len(protected_attributes) > 1:
            results['intersectional_analysis'] = self._analyze_intersectional_bias(
                df, y_true, y_pred, protected_attributes
            )
        
        # Generate summary and recommendations
        results['bias_summary'] = self._generate_bias_summary(results)
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_protected_attribute(self,
                                   df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   attribute: str) -> Dict[str, Any]:
        """Analyze bias for a single protected attribute"""
        
        # Get unique groups
        groups = df[attribute].unique()
        group_metrics = {}
        
        # Calculate metrics for each group
        for group in groups:
            mask = df[attribute] == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            group_metrics[group] = {
                'size': mask.sum(),
                'positive_rate': np.mean(group_y_pred),
                'true_positive_rate': self._calculate_tpr(group_y_true, group_y_pred),
                'false_positive_rate': self._calculate_fpr(group_y_true, group_y_pred),
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'precision': self._calculate_precision(group_y_true, group_y_pred),
                'recall': self._calculate_recall(group_y_true, group_y_pred)
            }
        
        # Calculate fairness metrics
        fairness_metrics = self.fairness_calculator.calculate_all_metrics(
            df, y_true, y_pred, attribute
        )
        
        # Determine bias score
        bias_score = self._calculate_bias_score(fairness_metrics)
        
        return {
            'attribute': attribute,
            'groups': list(groups),
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics,
            'bias_score': bias_score,
            'has_bias': bias_score > self.config.get('bias_threshold', 0.1),
            'bias_type': self._identify_bias_type(fairness_metrics)
        }
    
    def _analyze_intersectional_bias(self,
                                   df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   attributes: List[str]) -> Dict[str, Any]:
        """Analyze intersectional bias across multiple protected attributes"""
        
        # Create intersectional groups
        df_copy = df.copy()
        df_copy['intersectional_group'] = df_copy[attributes].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
        
        # Analyze intersectional groups
        intersectional_analysis = self._analyze_protected_attribute(
            df_copy, y_true, y_pred, 'intersectional_group'
        )
        
        # Calculate intersectional bias score
        intersectional_bias_score = self._calculate_intersectional_bias_score(
            intersectional_analysis['group_metrics']
        )
        
        return {
            'intersectional_groups': intersectional_analysis['groups'],
            'group_metrics': intersectional_analysis['group_metrics'],
            'intersectional_bias_score': intersectional_bias_score,
            'most_disadvantaged_group': self._find_most_disadvantaged_group(
                intersectional_analysis['group_metrics']
            ),
            'privilege_gaps': self._calculate_privilege_gaps(
                intersectional_analysis['group_metrics']
            )
        }
    
    def _calculate_tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate True Positive Rate"""
        if np.sum(y_true == 1) == 0:
            return 0.0
        return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    
    def _calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate False Positive Rate"""
        if np.sum(y_true == 0) == 0:
            return 0.0
        return np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Precision"""
        if np.sum(y_pred == 1) == 0:
            return 0.0
        return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Recall (same as TPR)"""
        return self._calculate_tpr(y_true, y_pred)
    
    def _calculate_bias_score(self, fairness_metrics: Dict[str, float]) -> float:
        """Calculate overall bias score from fairness metrics"""
        
        # Weight different fairness metrics
        weights = {
            'demographic_parity_difference': 0.3,
            'equalized_odds_difference': 0.3,
            'equal_opportunity_difference': 0.2,
            'calibration_difference': 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in fairness_metrics:
                weighted_score += abs(fairness_metrics[metric]) * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_bias_type(self, fairness_metrics: Dict[str, float]) -> str:
        """Identify the primary type of bias detected"""
        
        threshold = self.config.get('bias_threshold', 0.1)
        
        if abs(fairness_metrics.get('demographic_parity_difference', 0)) > threshold:
            return "Demographic Parity Violation"
        elif abs(fairness_metrics.get('equalized_odds_difference', 0)) > threshold:
            return "Equalized Odds Violation"
        elif abs(fairness_metrics.get('equal_opportunity_difference', 0)) > threshold:
            return "Equal Opportunity Violation"
        else:
            return "No Significant Bias"
    
    def _calculate_intersectional_bias_score(self, group_metrics: Dict[str, Dict]) -> float:
        """Calculate bias score for intersectional groups"""
        
        if not group_metrics:
            return 0.0
        
        # Calculate variance in positive rates across intersectional groups
        positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
        
        if len(positive_rates) <= 1:
            return 0.0
        
        return np.std(positive_rates)
    
    def _find_most_disadvantaged_group(self, group_metrics: Dict[str, Dict]) -> str:
        """Find the most disadvantaged intersectional group"""
        
        if not group_metrics:
            return "None"
        
        # Find group with lowest positive rate
        min_positive_rate = float('inf')
        most_disadvantaged = None
        
        for group, metrics in group_metrics.items():
            if metrics['positive_rate'] < min_positive_rate:
                min_positive_rate = metrics['positive_rate']
                most_disadvantaged = group
        
        return most_disadvantaged or "None"
    
    def _calculate_privilege_gaps(self, group_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate privilege gaps between intersectional groups"""
        
        if not group_metrics:
            return {}
        
        positive_rates = {group: metrics['positive_rate'] 
                         for group, metrics in group_metrics.items()}
        
        max_rate = max(positive_rates.values())
        
        privilege_gaps = {}
        for group, rate in positive_rates.items():
            privilege_gaps[group] = max_rate - rate
        
        return privilege_gaps
    
    def _generate_bias_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bias analysis summary"""
        
        total_attrs = len(results['protected_attribute_analysis'])
        biased_attrs = sum(1 for analysis in results['protected_attribute_analysis'].values()
                          if analysis['has_bias'])
        
        return {
            'total_protected_attributes': total_attrs,
            'biased_attributes': biased_attrs,
            'overall_bias_detected': biased_attrs > 0,
            'bias_severity': self._classify_bias_severity(results['overall_bias_score']),
            'primary_bias_types': self._get_primary_bias_types(results),
            'most_biased_attribute': self._find_most_biased_attribute(results)
        }
    
    def _classify_bias_severity(self, bias_score: float) -> str:
        """Classify bias severity based on score"""
        
        if bias_score < 0.05:
            return "Low"
        elif bias_score < 0.15:
            return "Moderate"
        elif bias_score < 0.25:
            return "High"
        else:
            return "Critical"
    
    def _get_primary_bias_types(self, results: Dict[str, Any]) -> List[str]:
        """Get primary types of bias detected"""
        
        bias_types = []
        for analysis in results['protected_attribute_analysis'].values():
            if analysis['has_bias']:
                bias_types.append(analysis['bias_type'])
        
        return list(set(bias_types))
    
    def _find_most_biased_attribute(self, results: Dict[str, Any]) -> str:
        """Find the protected attribute with highest bias"""
        
        max_bias_score = 0
        most_biased = None
        
        for attr, analysis in results['protected_attribute_analysis'].items():
            if analysis['bias_score'] > max_bias_score:
                max_bias_score = analysis['bias_score']
                most_biased = attr
        
        return most_biased or "None"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable bias mitigation recommendations"""
        
        recommendations = []
        
        if results['bias_summary']['overall_bias_detected']:
            # General recommendations
            recommendations.extend([
                "Review training data for representation imbalances across protected groups",
                "Consider implementing fairness constraints during model training",
                "Apply bias mitigation techniques such as re-sampling or re-weighting",
                "Regularly monitor model performance across different demographic groups"
            ])
            
            # Specific recommendations based on bias types
            bias_types = results['bias_summary']['primary_bias_types']
            
            if "Demographic Parity Violation" in bias_types:
                recommendations.append(
                    "Implement demographic parity constraints to ensure equal positive prediction rates"
                )
            
            if "Equalized Odds Violation" in bias_types:
                recommendations.append(
                    "Apply equalized odds post-processing to balance true positive and false positive rates"
                )
            
            if "Equal Opportunity Violation" in bias_types:
                recommendations.append(
                    "Focus on equalizing true positive rates across protected groups"
                )
            
            # Intersectional bias recommendations
            if 'intersectional_analysis' in results and results['intersectional_analysis']:
                recommendations.append(
                    "Pay special attention to intersectional groups that may face compounded discrimination"
                )
                
                most_disadvantaged = results['intersectional_analysis']['most_disadvantaged_group']
                if most_disadvantaged != "None":
                    recommendations.append(
                        f"Implement targeted interventions for the most disadvantaged group: {most_disadvantaged}"
                    )
        
        else:
            recommendations.extend([
                "Continue monitoring for bias as new data becomes available",
                "Maintain current fairness levels through regular audits",
                "Document fairness metrics for compliance and transparency"
            ])
        
        return recommendations
