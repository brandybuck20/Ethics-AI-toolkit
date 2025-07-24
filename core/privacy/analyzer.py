import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from .pii_detector import PIIDetector
from .gdpr_compliance import GDPRComplianceChecker as GDPRChecker

class PrivacyAnalyzer:
    """Main privacy analysis engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pii_detector = PIIDetector(config)
        self.gdpr_checker = GDPRChecker() if 'GDPR' in config.get('compliance_frameworks', []) else None
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze single text for privacy risks"""
        
        findings = []
        
        # PII Detection
        pii_findings = self.pii_detector.detect_pii(text)
        findings.extend(pii_findings)
        
        # Generate overall results
        results = self._generate_results(findings, 'text')
        
        # Add compliance assessment
        if self.gdpr_checker:
            compliance_results = self.gdpr_checker.assess_compliance(findings, text)
            results['compliance'] = compliance_results
        
        return results
    
    def analyze_batch_text(self, text_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze multiple text samples"""
        
        all_findings = []
        
        for item in text_data:
            text = item['content']
            filename = item['filename']
            
            # Detect PII in this text
            pii_findings = self.pii_detector.detect_pii(text)
            
            # Add source information
            for finding in pii_findings:
                finding['source_file'] = filename
            
            all_findings.extend(pii_findings)
        
        # Generate batch results
        results = self._generate_results(all_findings, 'batch_text')
        results['summary']['files_analyzed'] = len(text_data)
        
        return results
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset for privacy compliance"""
        
        findings = []
        
        # Analyze each column for PII
        for column in df.columns:
            column_findings = self._analyze_column(df[column], column)
            findings.extend(column_findings)
        
        # Check for sensitive data patterns
        sensitive_findings = self._detect_sensitive_patterns(df)
        findings.extend(sensitive_findings)
        
        results = self._generate_results(findings, 'dataset')
        results['summary']['columns_analyzed'] = len(df.columns)
        results['summary']['rows_analyzed'] = len(df)
        
        return results
    
    def analyze_model(self, model) -> Dict[str, Any]:
        """Analyze model for privacy risks"""
        
        findings = []
        
        # Model introspection for memorization
        memorization_findings = self._detect_memorization(model)
        findings.extend(memorization_findings)
        
        # Feature analysis
        if hasattr(model, 'feature_names_in_'):
            feature_findings = self._analyze_feature_names(model.feature_names_in_)
            findings.extend(feature_findings)
        
        results = self._generate_results(findings, 'model')
        
        return results
    
    def analyze_llm_output(self, llm_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze LLM prompt and response for privacy risks"""
        
        findings = []
        
        # Analyze prompt
        prompt_findings = self.pii_detector.detect_pii(llm_data['prompt'])
        for finding in prompt_findings:
            finding['location'] = 'User Prompt'
        findings.extend(prompt_findings)
        
        # Analyze response
        response_findings = self.pii_detector.detect_pii(llm_data['response'])
        for finding in response_findings:
            finding['location'] = 'LLM Response'
            # Higher severity for PII in model output
            if finding['severity'] == 'Medium':
                finding['severity'] = 'High'
        findings.extend(response_findings)
        
        results = self._generate_results(findings, 'llm_output')
        
        return results
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> List[Dict[str, Any]]:
        """Analyze a pandas series for PII"""
        
        findings = []
        
        # Convert series to text for analysis
        text_data = ' '.join(series.astype(str).values[:100])  # Sample first 100 values
        
        # Detect PII in the column data
        column_findings = self.pii_detector.detect_pii(text_data)
        
        # Update findings with column information
        for finding in column_findings:
            finding['location'] = f'Column: {column_name}'
            finding['column_name'] = column_name
            
            # Estimate prevalence in the column
            pii_pattern = finding.get('pattern', '')
            if pii_pattern:
                matches = series.astype(str).str.contains(pii_pattern, regex=True, na=False)
                prevalence = matches.sum() / len(series)
                finding['prevalence'] = prevalence
                
                # Adjust severity based on prevalence
                if prevalence > 0.5:
                    finding['severity'] = 'High'
                elif prevalence > 0.1:
                    finding['severity'] = 'Medium'
        
        findings.extend(column_findings)
        
        return findings
    
    def _detect_sensitive_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect sensitive data patterns in dataset"""
        
        findings = []
        
        # Check for potentially sensitive column names
        sensitive_column_patterns = [
            (r'(?i).*ssn.*|.*social.*security.*', 'Social Security Number'),
            (r'(?i).*password.*|.*pwd.*|.*pass.*', 'Password'),
            (r'(?i).*credit.*card.*|.*cc.*num.*', 'Credit Card'),
            (r'(?i).*salary.*|.*income.*|.*wage.*', 'Financial Information'),
            (r'(?i).*medical.*|.*health.*|.*diagnosis.*', 'Medical Information'),
            (r'(?i).*birth.*date.*|.*dob.*|.*birthday.*', 'Date of Birth')
        ]
        
        for pattern, pii_type in sensitive_column_patterns:
            matching_columns = [col for col in df.columns if re.match(pattern, col)]
            
            for col in matching_columns:
                findings.append({
                    'type': pii_type,
                    'category': 'Sensitive Column',
                    'severity': 'High',
                    'confidence': 0.9,
                    'description': f'Column "{col}" appears to contain {pii_type.lower()}',
                    'location': f'Column: {col}',
                    'column_name': col,
                    'remediation': f'Consider encrypting, masking, or removing the {col} column'
                })
        
        return findings
    
    def _detect_memorization(self, model) -> List[Dict[str, Any]]:
        """Detect potential training data memorization"""
        
        findings = []
        
        # This is a simplified implementation
        # In practice, you would use techniques like:
        # - Canary token injection
        # - Membership inference attacks
        # - Data extraction attacks
        
        # Placeholder finding for demonstration
        findings.append({
            'type': 'Potential Memorization',
            'category': 'Model Privacy',
            'severity': 'Medium',
            'confidence': 0.6,
            'description': 'Model may have memorized training data patterns',
            'location': 'Model Architecture',
            'remediation': 'Consider differential privacy training or data deduplication'
        })
        
        return findings
    
    def _analyze_feature_names(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Analyze feature names for privacy concerns"""
        
        findings = []
        
        # Check for potentially identifying feature names
        identifying_patterns = [
            (r'(?i).*id$|.*_id$|.*identifier.*', 'Identifier'),
            (r'(?i).*name.*|.*fname.*|.*lname.*', 'Name'),
            (r'(?i).*email.*|.*mail.*', 'Email'),
            (r'(?i).*phone.*|.*tel.*|.*mobile.*', 'Phone'),
            (r'(?i).*address.*|.*addr.*|.*zip.*', 'Address')
        ]
        
        for feature in feature_names:
            for pattern, pii_type in identifying_patterns:
                if re.match(pattern, feature):
                    findings.append({
                        'type': f'{pii_type} Feature',
                        'category': 'Feature Privacy',
                        'severity': 'Medium',
                        'confidence': 0.7,
                        'description': f'Feature "{feature}" may contain {pii_type.lower()} information',
                        'location': f'Feature: {feature}',
                        'remediation': f'Consider removing or encoding the {feature} feature'
                    })
        
        return findings
    
    def _generate_results(self, findings: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """Generate comprehensive results summary"""
        
        # Calculate summary statistics
        total_issues = len(findings)
        high_risk_findings = len([f for f in findings if f['severity'] == 'High'])
        medium_risk_findings = len([f for f in findings if f['severity'] == 'Medium'])
        low_risk_findings = len([f for f in findings if f['severity'] == 'Low'])
        
        # Count PII instances
        pii_instances = sum(1 for f in findings if f['category'] in ['PII', 'Sensitive Data'])
        
        # Determine overall risk level
        if high_risk_findings > 0:
            risk_level = 'High'
            overall_score = max(0, 10 - (high_risk_findings * 2 + medium_risk_findings * 1))
        elif medium_risk_findings > 2:
            risk_level = 'Medium'
            overall_score = max(4, 10 - (medium_risk_findings * 1.5 + low_risk_findings * 0.5))
        else:
            risk_level = 'Low'
            overall_score = max(7, 10 - (medium_risk_findings * 0.5 + low_risk_findings * 0.2))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'overall_score': round(overall_score, 1),
            'summary': {
                'total_issues': total_issues,
                'high_risk_findings': high_risk_findings,
                'medium_risk_findings': medium_risk_findings,
                'low_risk_findings': low_risk_findings,
                'pii_instances': pii_instances,
                'risk_level': risk_level
            },
            'findings': findings,
            'recommendations': self._generate_recommendations(findings, risk_level)
        }
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]], risk_level: str) -> List[str]:
        """Generate recommendations based on findings"""
        
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("🚨 Immediate action required: Remove or mask detected PII")
            recommendations.append("🔒 Implement data encryption for sensitive information")
            recommendations.append("📋 Conduct thorough privacy impact assessment")
        
        if any(f['category'] == 'PII' for f in findings):
            recommendations.append("🛡️ Implement PII detection in your data pipeline")
            recommendations.append("📝 Update privacy policies and consent mechanisms")
        
        recommendations.append("🔄 Set up regular privacy audits")
        recommendations.append("👥 Train team on privacy-preserving techniques")
        
        return recommendations
