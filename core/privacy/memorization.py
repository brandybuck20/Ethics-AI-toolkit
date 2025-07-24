import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MemorizationDetector:
    """Detect training data memorization in ML models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
    def detect_memorization(self, 
                          model,
                          test_data: pd.DataFrame,
                          suspected_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect potential memorization of training data
        
        Args:
            model: Trained ML model
            test_data: Test dataset
            suspected_training_data: Suspected training data samples
            
        Returns:
            Dictionary containing memorization analysis results
        """
        
        results = {
            'memorization_detected': False,
            'memorization_score': 0.0,
            'suspicious_samples': [],
            'memorization_evidence': [],
            'risk_level': 'Low'
        }
        
        # Canary token detection
        canary_results = self._detect_canary_tokens(model, test_data)
        results['canary_detection'] = canary_results
        
        # Membership inference analysis
        if suspected_training_data is not None:
            membership_results = self._membership_inference_test(
                model, test_data, suspected_training_data
            )
            results['membership_inference'] = membership_results
        
        # Data extraction analysis
        extraction_results = self._analyze_data_extraction_risk(model, test_data)
        results['extraction_risk'] = extraction_results
        
        # Calculate overall memorization score
        results['memorization_score'] = self._calculate_memorization_score(results)
        results['memorization_detected'] = results['memorization_score'] > self.confidence_threshold
        results['risk_level'] = self._assess_risk_level(results['memorization_score'])
        
        return results
    
    def _detect_canary_tokens(self, model, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect canary tokens that might indicate memorization"""
        
        canary_results = {
            'canary_tokens_found': [],
            'canary_score': 0.0,
            'suspicious_patterns': []
        }
        
        # Look for suspicious patterns in model predictions
        if hasattr(model, 'predict_proba'):
            try:
                # Get prediction probabilities
                probabilities = model.predict_proba(test_data.select_dtypes(include=[np.number]))
                
                # Look for extremely high confidence predictions
                max_probs = np.max(probabilities, axis=1)
                high_confidence_mask = max_probs > 0.99
                
                if np.sum(high_confidence_mask) > 0:
                    suspicious_indices = np.where(high_confidence_mask)[0]
                    
                    for idx in suspicious_indices[:10]:  # Limit to first 10
                        canary_results['suspicious_patterns'].append({
                            'sample_index': int(idx),
                            'confidence': float(max_probs[idx]),
                            'prediction': int(np.argmax(probabilities[idx])),
                            'risk_score': min(1.0, (max_probs[idx] - 0.99) * 100)
                        })
                
                # Calculate canary score based on high confidence predictions  
                canary_results['canary_score'] = min(1.0, np.sum(high_confidence_mask) / len(test_data))
                
            except Exception as e:
                canary_results['error'] = f"Error in canary detection: {str(e)}"
        
        return canary_results
    
    def _membership_inference_test(self,
                                 model,
                                 test_data: pd.DataFrame,
                                 training_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform membership inference attack to detect memorization"""
        
        membership_results = {
            'attack_accuracy': 0.0,
            'vulnerable_samples': [],
            'membership_scores': []
        }
        
        try:
            # Simple membership inference based on prediction confidence
            if hasattr(model, 'predict_proba'):
                # Get numeric columns only
                numeric_cols = test_data.select_dtypes(include=[np.number]).columns
                test_numeric = test_data[numeric_cols]
                train_numeric = training_data[numeric_cols].head(len(test_numeric))  # Match sizes
                
                # Get prediction probabilities
                test_probs = model.predict_proba(test_numeric)
                train_probs = model.predict_proba(train_numeric)
                
                # Calculate confidence scores
                test_confidence = np.max(test_probs, axis=1)
                train_confidence = np.max(train_probs, axis=1)
                
                # Simple threshold-based membership inference
                threshold = np.median(np.concatenate([test_confidence, train_confidence]))
                
                # Classify samples as members or non-members
                test_predicted_members = test_confidence > threshold
                train_predicted_members = train_confidence > threshold
                
                # Calculate attack accuracy (assuming training data are members, test data are not)
                correct_predictions = np.sum(train_predicted_members) + np.sum(~test_predicted_members)
                total_predictions = len(train_predicted_members) + len(test_predicted_members)
                
                membership_results['attack_accuracy'] = correct_predictions / total_predictions
                
                # Identify vulnerable samples
                for i, (conf, is_member) in enumerate(zip(test_confidence, test_predicted_members)):
                    if is_member and conf > 0.95:
                        membership_results['vulnerable_samples'].append({
                            'sample_index': i,
                            'confidence': float(conf),
                            'membership_score': float(conf - threshold)
                        })
                
                membership_results['membership_scores'] = test_confidence.tolist()
        
        except Exception as e:
            membership_results['error'] = f"Error in membership inference: {str(e)}"
        
        return membership_results
    
    def _analyze_data_extraction_risk(self, model, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk of data extraction from model"""
        
        extraction_results = {
            'extraction_risk_score': 0.0,
            'vulnerable_features': [],
            'data_leakage_indicators': []
        }
        
        try:
            # Check if model has accessible parameters that might leak training data
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                
                # Look for extremely high feature importance (potential overfitting)
                high_importance_threshold = 0.8
                high_importance_features = np.where(importance_values > high_importance_threshold)[0]
                
                for feature_idx in high_importance_features:
                    if feature_idx < len(test_data.columns):
                        feature_name = test_data.columns[feature_idx]
                        extraction_results['vulnerable_features'].append({
                            'feature_name': feature_name,
                            'importance': float(importance_values[feature_idx]),
                            'risk_type': 'High Feature Importance'
                        })
            
            # Check for potential data leakage through model structure
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                # Tree-based models might memorize specific data points
                extraction_results['data_leakage_indicators'].append({
                    'indicator': 'Tree-based model structure',
                    'description': 'Tree-based models can potentially memorize training samples',
                    'risk_level': 'Medium'
                })
            
            # Calculate extraction risk score
            feature_risk = len(extraction_results['vulnerable_features']) / len(test_data.columns)
            structure_risk = len(extraction_results['data_leakage_indicators']) * 0.2
            
            extraction_results['extraction_risk_score'] = min(1.0, feature_risk + structure_risk)
        
        except Exception as e:
            extraction_results['error'] = f"Error in extraction risk analysis: {str(e)}"
        
        return extraction_results
    
    def _calculate_memorization_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall memorization score"""
        
        scores = []
        
        # Canary detection score
        if 'canary_detection' in results:
            canary_score = results['canary_detection'].get('canary_score', 0.0)
            scores.append(canary_score * 0.4)
        
        # Membership inference score
        if 'membership_inference' in results:
            membership_accuracy = results['membership_inference'].get('attack_accuracy', 0.5)
            # Convert accuracy to risk score (higher accuracy = higher risk)
            membership_risk = max(0, (membership_accuracy - 0.5) * 2)
            scores.append(membership_risk * 0.4)
        
        # Extraction risk score
        if 'extraction_risk' in results:
            extraction_score = results['extraction_risk'].get('extraction_risk_score', 0.0)
            scores.append(extraction_score * 0.2)
        
        return sum(scores) if scores else 0.0
    
    def _assess_risk_level(self, memorization_score: float) -> str:
        """Assess memorization risk level"""
        
        if memorization_score < 0.3:
            return 'Low'
        elif memorization_score < 0.6:
            return 'Medium'
        elif memorization_score < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def generate_memorization_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable memorization report"""
        
        report = "MEMORIZATION ANALYSIS REPORT\n"
        report += "=" * 30 + "\n\n"
        
        report += f"Overall Memorization Score: {results['memorization_score']:.3f}\n"
        report += f"Risk Level: {results['risk_level']}\n"
        report += f"Memorization Detected: {'Yes' if results['memorization_detected'] else 'No'}\n\n"
        
        # Canary detection results
        if 'canary_detection' in results:
            canary = results['canary_detection']
            report += "CANARY TOKEN ANALYSIS:\n"
            report += f"- Canary Score: {canary.get('canary_score', 0):.3f}\n"
            report += f"- Suspicious Patterns Found: {len(canary.get('suspicious_patterns', []))}\n\n"
        
        # Membership inference results
        if 'membership_inference' in results:
            membership = results['membership_inference']
            report += "MEMBERSHIP INFERENCE ANALYSIS:\n"
            report += f"- Attack Accuracy: {membership.get('attack_accuracy', 0):.3f}\n"
            report += f"- Vulnerable Samples: {len(membership.get('vulnerable_samples', []))}\n\n"
        
        # Extraction risk results
        if 'extraction_risk' in results:
            extraction = results['extraction_risk']
            report += "DATA EXTRACTION RISK ANALYSIS:\n"
            report += f"- Extraction Risk Score: {extraction.get('extraction_risk_score', 0):.3f}\n"
            report += f"- Vulnerable Features: {len(extraction.get('vulnerable_features', []))}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        if results['memorization_detected']:
            report += "- Consider implementing differential privacy techniques\n"
            report += "- Review training data for sensitive information\n"
            report += "- Implement data deduplication in training pipeline\n"
            report += "- Consider model regularization techniques\n"
        else:
            report += "- Continue monitoring for memorization in future model updates\n"
            report += "- Maintain current privacy protection measures\n"
        
        return report
