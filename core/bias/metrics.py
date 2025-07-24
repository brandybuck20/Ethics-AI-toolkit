import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """Calculate various fairness and bias metrics"""
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self,
                            df: pd.DataFrame,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            protected_attribute: str) -> Dict[str, float]:
        """Calculate all fairness metrics for a protected attribute"""
        
        metrics = {}
        
        # Demographic Parity
        metrics['demographic_parity_difference'] = self.demographic_parity_difference(
            df, y_pred, protected_attribute
        )
        
        # Equalized Odds
        metrics['equalized_odds_difference'] = self.equalized_odds_difference(
            df, y_true, y_pred, protected_attribute
        )
        
        # Equal Opportunity  
        metrics['equal_opportunity_difference'] = self.equal_opportunity_difference(
            df, y_true, y_pred, protected_attribute
        )
        
        # Calibration
        metrics['calibration_difference'] = self.calibration_difference(
            df, y_true, y_pred, protected_attribute
        )
        
        # Individual Fairness (approximation)
        metrics['individual_fairness_score'] = self.individual_fairness_approximation(
            df, y_pred, protected_attribute
        )
        
        return metrics
    
    def demographic_parity_difference(self,
                                    df: pd.DataFrame,
                                    y_pred: np.ndarray,
                                    protected_attribute: str) -> float:
        """
        Calculate demographic parity difference
        
        Demographic parity requires that the probability of positive prediction
        should be the same across all groups
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 0.0
        
        positive_rates = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() > 0:
                positive_rate = np.mean(y_pred[mask])
                positive_rates.append(positive_rate)
        
        if len(positive_rates) < 2:
            return 0.0
        
        return max(positive_rates) - min(positive_rates)
    
    def equalized_odds_difference(self,
                                df: pd.DataFrame,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                protected_attribute: str) -> float:
        """
        Calculate equalized odds difference
        
        Equalized odds requires that TPR and FPR should be the same across groups
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 0.0
        
        tpr_differences = []
        fpr_differences = []
        
        group_metrics = {}
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # Calculate TPR and FPR for this group
            tpr = self._calculate_tpr(group_y_true, group_y_pred)
            fpr = self._calculate_fpr(group_y_true, group_y_pred)
            
            group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        if len(group_metrics) < 2:
            return 0.0
        
        # Calculate differences
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        
        # Return the maximum of TPR and FPR differences
        return max(tpr_diff, fpr_diff)
    
    def equal_opportunity_difference(self,
                                   df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   protected_attribute: str) -> float:
        """
        Calculate equal opportunity difference
        
        Equal opportunity requires that TPR should be the same across groups
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 0.0
        
        tprs = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            tpr = self._calculate_tpr(group_y_true, group_y_pred)
            tprs.append(tpr)
        
        if len(tprs) < 2:
            return 0.0
        
        return max(tprs) - min(tprs)
    
    def calibration_difference(self,
                             df: pd.DataFrame,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             protected_attribute: str) -> float:
        """
        Calculate calibration difference
        
        Calibration measures if predicted probabilities match actual outcomes
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 0.0
        
        calibration_scores = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # Simple calibration score (predicted vs actual positive rate)
            predicted_positive_rate = np.mean(group_y_pred)
            actual_positive_rate = np.mean(group_y_true)
            
            calibration_error = abs(predicted_positive_rate - actual_positive_rate)
            calibration_scores.append(calibration_error)
        
        if len(calibration_scores) < 2:
            return 0.0
        
        return max(calibration_scores) - min(calibration_scores)
    
    def individual_fairness_approximation(self,
                                        df: pd.DataFrame,
                                        y_pred: np.ndarray,
                                        protected_attribute: str) -> float:
        """
        Approximate individual fairness score
        
        Individual fairness requires similar individuals to receive similar predictions
        """
        
        # This is a simplified approximation
        # In practice, you'd need to define similarity metrics
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 1.0  # Perfect individual fairness if only one group
        
        # Calculate coefficient of variation in predictions within each group
        within_group_variations = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() <= 1:
                continue
                
            group_predictions = y_pred[mask]
            
            if np.mean(group_predictions) == 0:
                cv = 0
            else:
                cv = np.std(group_predictions) / np.mean(group_predictions)
            
            within_group_variations.append(cv)
        
        if not within_group_variations:
            return 1.0
        
        # Lower variation indicates better individual fairness
        avg_variation = np.mean(within_group_variations)
        
        # Convert to score (higher is better)
        return max(0, 1 - avg_variation)
    
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
    
    def statistical_parity_difference(self,
                                    df: pd.DataFrame,
                                    y_pred: np.ndarray,
                                    protected_attribute: str) -> float:
        """Statistical parity difference (same as demographic parity)"""
        return self.demographic_parity_difference(df, y_pred, protected_attribute)
    
    def disparate_impact_ratio(self,
                             df: pd.DataFrame,
                             y_pred: np.ndarray,
                             protected_attribute: str) -> float:
        """
        Calculate disparate impact ratio
        
        Ratio of positive prediction rates between groups
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) != 2:
            return 1.0  # Only meaningful for binary protected attributes
        
        positive_rates = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() > 0:
                positive_rate = np.mean(y_pred[mask])
                positive_rates.append(positive_rate)
        
        if len(positive_rates) != 2 or positive_rates[1] == 0:
            return 1.0
        
        return min(positive_rates) / max(positive_rates)
    
    def predictive_parity_difference(self,
                                   df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   protected_attribute: str) -> float:
        """
        Calculate predictive parity difference
        
        Requires that precision should be the same across groups
        """
        
        groups = df[protected_attribute].unique()
        if len(groups) < 2:
            return 0.0
        
        precisions = []
        
        for group in groups:
            mask = df[protected_attribute] == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # Calculate precision
            if np.sum(group_y_pred == 1) == 0:
                precision = 0.0
            else:
                precision = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_pred == 1)
            
            precisions.append(precision)
        
        if len(precisions) < 2:
            return 0.0
        
        return max(precisions) - min(precisions)
