import pandas as pd
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

class GDPRComplianceChecker:
    """Check GDPR compliance for AI systems and data processing"""
    
    def __init__(self):
        self.gdpr_articles = self._load_gdpr_articles()
        
    def assess_compliance(self, 
                         dataset: Optional[pd.DataFrame] = None,
                         processing_activities: Optional[List[str]] = None,
                         data_subject_rights: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Assess GDPR compliance across multiple dimensions
        
        Args:
            dataset: Dataset being processed
            processing_activities: List of data processing activities
            data_subject_rights: Implementation status of data subject rights
            
        Returns:
            Dictionary with compliance assessment results
        """
        
        compliance_results = {
            'overall_compliance_score': 0.0,
            'article_compliance': {},
            'risk_assessment': {},
            'violations_detected': [],
            'recommendations': []
        }
        
        # Assess each relevant GDPR article
        for article_num, article_info in self.gdpr_articles.items():
            compliance_score = self._assess_article_compliance(
                article_num, dataset, processing_activities, data_subject_rights
            )
            compliance_results['article_compliance'][article_num] = compliance_score
        
        # Calculate overall compliance score
        article_scores = list(compliance_results['article_compliance'].values())
        compliance_results['overall_compliance_score'] = sum(score['score'] for score in article_scores) / len(article_scores)
        
        # Identify violations
        compliance_results['violations_detected'] = [
            article for article, assessment in compliance_results['article_compliance'].items()
            if assessment['score'] < 0.7
        ]
        
        # Generate risk assessment
        compliance_results['risk_assessment'] = self._generate_risk_assessment(compliance_results)
        
        # Generate recommendations
        compliance_results['recommendations'] = self._generate_gdpr_recommendations(compliance_results)
        
        return compliance_results
    
    def _load_gdpr_articles(self) -> Dict[str, Dict[str, Any]]:
        """Load GDPR articles relevant to AI systems"""
        
        return {
            'Article_5': {
                'title': 'Principles relating to processing of personal data',
                'key_requirements': [
                    'Lawfulness, fairness and transparency',
                    'Purpose limitation',
                    'Data minimisation',
                    'Accuracy',
                    'Storage limitation',
                    'Integrity and confidentiality'
                ]
            },
            'Article_6': {
                'title': 'Lawfulness of processing',
                'key_requirements': [
                    'Consent',
                    'Contract',
                    'Legal obligation',
                    'Vital interests',
                    'Public task',
                    'Legitimate interests'
                ]
            },
            'Article_7': {
                'title': 'Conditions for consent',
                'key_requirements': [
                    'Clear and affirmative action',
                    'Freely given',
                    'Specific',
                    'Informed',
                    'Withdrawable'
                ]
            },
            'Article_9': {
                'title': 'Processing of special categories of personal data',
                'key_requirements': [
                    'Explicit consent',
                    'Special category protections',
                    'Additional safeguards'
                ]
            },
            'Article_13': {
                'title': 'Information to be provided where personal data are collected',
                'key_requirements': [
                    'Identity of controller',
                    'Purposes of processing',
                    'Legal basis',
                    'Data subject rights'
                ]
            },
            'Article_15': {
                'title': 'Right of access by the data subject',
                'key_requirements': [
                    'Confirmation of processing',
                    'Access to personal data',
                    'Information about processing'
                ]
            },
            'Article_17': {
                'title': 'Right to erasure (right to be forgotten)',
                'key_requirements': [
                    'Right to erasure',
                    'Obligation to erase',
                    'Exceptions to erasure'
                ]
            },
            'Article_22': {
                'title': 'Automated individual decision-making',
                'key_requirements': [
                    'Right not to be subject to automated decision-making',
                    'Exceptions for automated decision-making',
                    'Safeguards for automated decision-making'
                ]
            },
            'Article_25': {
                'title': 'Data protection by design and by default',
                'key_requirements': [
                    'Data protection by design',
                    'Data protection by default',
                    'Privacy-enhancing technologies'
                ]
            },
            'Article_35': {
                'title': 'Data protection impact assessment',
                'key_requirements': [
                    'DPIA requirement',
                    'Risk assessment',
                    'Mitigation measures'
                ]
            }
        }
    
    def _assess_article_compliance(self,
                                 article_num: str,
                                 dataset: Optional[pd.DataFrame],
                                 processing_activities: Optional[List[str]],
                                 data_subject_rights: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Assess compliance with a specific GDPR article"""
        
        if article_num == 'Article_5':
            return self._assess_article_5(dataset, processing_activities)
        elif article_num == 'Article_6':
            return self._assess_article_6(processing_activities)
        elif article_num == 'Article_7':
            return self._assess_article_7(data_subject_rights)
        elif article_num == 'Article_9':
            return self._assess_article_9(dataset)
        elif article_num == 'Article_13':
            return self._assess_article_13()
        elif article_num == 'Article_15':
            return self._assess_article_15(data_subject_rights)
        elif article_num == 'Article_17':
            return self._assess_article_17(data_subject_rights)
        elif article_num == 'Article_22':
            return self._assess_article_22(processing_activities)
        elif article_num == 'Article_25':
            return self._assess_article_25()
        elif article_num == 'Article_35':
            return self._assess_article_35(processing_activities)
        else:
            return {'score': 0.5, 'issues': ['Assessment not implemented'], 'compliant': False}
    
    def _assess_article_5(self, dataset: Optional[pd.DataFrame], 
                         processing_activities: Optional[List[str]]) -> Dict[str, Any]:
        """Assess Article 5 - Principles relating to processing"""
        
        issues = []
        score = 1.0
        
        if dataset is not None:
            # Check data minimization
            if len(dataset.columns) > 50:
                issues.append("Large number of data fields may violate data minimization")
                score -= 0.2
            
            # Check for potential special category data
            sensitive_patterns = ['race', 'ethnic', 'religion', 'health', 'sexual', 'political']
            for col in dataset.columns:
                if any(pattern in col.lower() for pattern in sensitive_patterns):
                    issues.append(f"Column '{col}' may contain special category data")
                    score -= 0.1
        
        # Check processing activities
        if processing_activities:
            automated_processing = any('automated' in activity.lower() for activity in processing_activities)
            if automated_processing:
                issues.append("Automated processing requires additional safeguards")
                score -= 0.1
        
        return {
            'score': max(0, score),
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_5']['title']
        }
    
    def _assess_article_6(self, processing_activities: Optional[List[str]]) -> Dict[str, Any]:
        """Assess Article 6 - Lawfulness of processing"""
        
        # This is a simplified assessment - in practice would require more context
        issues = []
        score = 0.6  # Assume moderate compliance without explicit lawful basis documentation
        
        issues.append("Lawful basis for processing should be explicitly documented")
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_6']['title']
        }
    
    def _assess_article_7(self, data_subject_rights: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Assess Article 7 - Conditions for consent"""
        
        issues = []
        score = 0.5
        
        if data_subject_rights:
            if data_subject_rights.get('consent_withdrawal', False):
                score += 0.3
            else:
                issues.append("Consent withdrawal mechanism not implemented")
            
            if data_subject_rights.get('clear_consent', False):
                score += 0.2
            else:
                issues.append("Clear consent mechanism not documented")
        else:
            issues.append("Data subject rights implementation not specified")
        
        return {
            'score': min(1.0, score),
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_7']['title']
        }
    
    def _assess_article_9(self, dataset: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Assess Article 9 - Processing of special categories"""
        
        issues = []
        score = 1.0
        
        if dataset is not None:
            # Check for special category data indicators
            special_category_patterns = {
                'race': ['race', 'ethnic', 'nationality'],
                'health': ['health', 'medical', 'diagnosis', 'symptom'],
                'religion': ['religion', 'belief', 'faith'],
                'sexual': ['sexual', 'orientation', 'preference'],
                'political': ['political', 'party', 'vote'],
                'biometric': ['biometric', 'fingerprint', 'facial'],
                'genetic': ['genetic', 'dna', 'genome']
            }
            
            detected_categories = []
            for category, patterns in special_category_patterns.items():
                for col in dataset.columns:
                    if any(pattern in col.lower() for pattern in patterns):
                        detected_categories.append(category)
                        break
            
            if detected_categories:
                score = 0.3  # Assume non-compliance without explicit consent
                issues.append(f"Special category data detected: {', '.join(set(detected_categories))}")
                issues.append("Explicit consent or other Article 9 exemption required")
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_9']['title']
        }
    
    def _assess_article_15(self, data_subject_rights: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Assess Article 15 - Right of access"""
        
        issues = []
        score = 0.5
        
        if data_subject_rights and data_subject_rights.get('data_access', False):
            score = 0.8
        else:
            issues.append("Data subject access right not implemented")
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_15']['title']
        }
    
    def _assess_article_17(self, data_subject_rights: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Assess Article 17 - Right to erasure"""
        
        issues = []
        score = 0.5
        
        if data_subject_rights and data_subject_rights.get('data_erasure', False):
            score = 0.8
        else:
            issues.append("Right to erasure not implemented")
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_17']['title']
        }
    
    def _assess_article_22(self, processing_activities: Optional[List[str]]) -> Dict[str, Any]:
        """Assess Article 22 - Automated decision-making"""
        
        issues = []
        score = 0.8
        
        if processing_activities:
            automated_decision_keywords = ['automated decision', 'profiling', 'algorithmic', 'ai decision']
            has_automated_decisions = any(
                any(keyword in activity.lower() for keyword in automated_decision_keywords)
                for activity in processing_activities
            )
            
            if has_automated_decisions:
                score = 0.4
                issues.append("Automated decision-making detected - requires human intervention safeguards")
                issues.append("Data subjects must be informed of automated decision-making")
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_22']['title']
        }
    
    def _assess_article_25(self) -> Dict[str, Any]:
        """Assess Article 25 - Data protection by design and default"""
        
        # This would require architectural assessment
        issues = ["Data protection by design assessment requires system architecture review"]
        score = 0.6  # Neutral score without detailed assessment
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_25']['title']
        }
    
    def _assess_article_35(self, processing_activities: Optional[List[str]]) -> Dict[str, Any]:
        """Assess Article 35 - Data protection impact assessment"""
        
        issues = []
        score = 0.5
        
        # Check if DPIA is required
        dpia_triggers = [
            'systematic monitoring', 'special category', 'large scale',
            'automated decision', 'profiling', 'biometric', 'genetic'
        ]
        
        requires_dpia = False
        if processing_activities:
            for activity in processing_activities:
                if any(trigger in activity.lower() for trigger in dpia_triggers):
                    requires_dpia = True
                    break
        
        if requires_dpia:
            issues.append("Data Protection Impact Assessment (DPIA) required")
            score = 0.3
        else:
            score = 0.8
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_35']['title']
        }
    
    def _assess_article_13(self) -> Dict[str, Any]:
        """Assess Article 13 - Information to be provided"""
        
        # This would require privacy notice assessment
        issues = ["Privacy notice compliance requires manual review"]
        score = 0.6
        
        return {
            'score': score,
            'issues': issues,
            'compliant': score >= 0.7,
            'article_title': self.gdpr_articles['Article_13']['title']
        }
    
    def _generate_risk_assessment(self, compliance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall GDPR compliance risk assessment"""
        
        overall_score = compliance_results['overall_compliance_score']
        violations = compliance_results['violations_detected']
        
        if overall_score < 0.5:
            risk_level = 'High'
            risk_description = 'Multiple GDPR compliance issues detected'
        elif overall_score < 0.7:
            risk_level = 'Medium'
            risk_description = 'Some GDPR compliance concerns identified'
        else:
            risk_level = 'Low'
            risk_description = 'Generally GDPR compliant with minor issues'
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'critical_violations': [v for v in violations if v in ['Article_9', 'Article_22']],
            'compliance_percentage': overall_score * 100
        }
    
    def _generate_gdpr_recommendations(self, compliance_results: Dict[str, Any]) -> List[str]:
        """Generate GDPR compliance recommendations"""
        
        recommendations = []
        violations = compliance_results['violations_detected']
        
        if 'Article_5' in violations:
            recommendations.append("Implement data minimization and purpose limitation measures")
        
        if 'Article_6' in violations:
            recommendations.append("Document lawful basis for all data processing activities")
        
        if 'Article_7' in violations:
            recommendations.append("Implement clear consent mechanisms with withdrawal options")
        
        if 'Article_9' in violations:
            recommendations.append("Obtain explicit consent for special category data processing")
        
        if 'Article_15' in violations:
            recommendations.append("Implement data subject access request procedures")
        
        if 'Article_17' in violations:
            recommendations.append("Implement right to erasure (right to be forgotten) procedures")
        
        if 'Article_22' in violations:
            recommendations.append("Implement human intervention safeguards for automated decisions")
        
        if 'Article_35' in violations:
            recommendations.append("Conduct Data Protection Impact Assessment (DPIA)")
        
        # General recommendations
        recommendations.extend([
            "Regular GDPR compliance audits recommended",
            "Staff training on data protection principles",
            "Implement privacy by design in system architecture"
        ])
        
        return recommendations
