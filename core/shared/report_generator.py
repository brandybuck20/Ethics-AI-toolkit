import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import base64
from io import BytesIO

class ReportGenerator:
    """Centralized report generation for all ethics modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.report_templates = self._load_report_templates()
        
    def generate_comprehensive_report(self,
                                    bias_results: Optional[Dict[str, Any]] = None,
                                    privacy_results: Optional[Dict[str, Any]] = None,
                                    explainability_results: Optional[Dict[str, Any]] = None,
                                    hallucination_results: Optional[Dict[str, Any]] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive ethics audit report"""
        
        report = {
            'metadata': self._generate_report_metadata(metadata),
            'executive_summary': {},
            'detailed_findings': {},
            'recommendations': [],
            'compliance_summary': {},
            'appendices': {}
        }
        
        # Process each module's results
        if bias_results:
            report['detailed_findings']['bias_analysis'] = self._process_bias_results(bias_results)
            
        if privacy_results:
            report['detailed_findings']['privacy_analysis'] = self._process_privacy_results(privacy_results)
            
        if explainability_results:
            report['detailed_findings']['explainability_analysis'] = self._process_explainability_results(explainability_results)
            
        if hallucination_results:
            report['detailed_findings']['hallucination_analysis'] = self._process_hallucination_results(hallucination_results)
        
        # Generate executive summary
        report['executive_summary'] = self._generate_executive_summary(report['detailed_findings'])
        
        # Compile recommendations
        report['recommendations'] = self._compile_recommendations(report['detailed_findings'])
        
        # Generate compliance summary
        report['compliance_summary'] = self._generate_compliance_summary(report['detailed_findings'])
        
        return report
    
    def _generate_report_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report metadata"""
        
        default_metadata = {
            'report_id': f"ethics_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'toolkit_version': '1.0.0',
            'report_type': 'Comprehensive Ethics Audit',
            'generated_by': 'AI Ethics Toolkit'
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        return default_metadata
    
    def _process_bias_results(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process bias analysis results for reporting"""
        
        processed = {
            'summary': {
                'overall_bias_score': bias_results.get('overall_bias_score', 0),
                'bias_detected': bias_results.get('bias_summary', {}).get('overall_bias_detected', False),
                'protected_attributes_analyzed': len(bias_results.get('protected_attribute_analysis', {})),
                'severity': bias_results.get('bias_summary', {}).get('bias_severity', 'Unknown')
            },
            'findings': [],
            'recommendations': bias_results.get('recommendations', [])
        }
        
        # Process protected attribute findings
        for attr, analysis in bias_results.get('protected_attribute_analysis', {}).items():
            if analysis.get('has_bias', False):
                processed['findings'].append({
                    'type': 'Demographic Bias',
                    'attribute': attr,
                    'bias_score': analysis.get('bias_score', 0),
                    'bias_type': analysis.get('bias_type', 'Unknown'),
                    'affected_groups': analysis.get('groups', []),
                    'severity': 'High' if analysis.get('bias_score', 0) > 0.1 else 'Medium'
                })
        
        # Process intersectional findings
        if 'intersectional_analysis' in bias_results:
            intersectional = bias_results['intersectional_analysis']
            if intersectional.get('intersectional_bias_score', 0) > 0.1:
                processed['findings'].append({
                    'type': 'Intersectional Bias',
                    'bias_score': intersectional.get('intersectional_bias_score', 0),
                    'most_disadvantaged_group': intersectional.get('most_disadvantaged_group', 'Unknown'),
                    'severity': 'High'
                })
        
        return processed
    
    def _process_privacy_results(self, privacy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process privacy analysis results for reporting"""
        
        processed = {
            'summary': {
                'overall_factuality_score': privacy_results.get('overall_factuality_score', 10),
                'pii_instances_detected': len(privacy_results.get('findings', [])),
                'high_risk_findings': privacy_results.get('summary', {}).get('high_risk_findings', 0),
                'compliance_score': privacy_results.get('compliance', {}).get('GDPR', {}).get('score', 10)
            },
            'findings': [],
            'recommendations': privacy_results.get('recommendations', [])
        }
        
        # Process PII findings
        for finding in privacy_results.get('findings', []):
            processed['findings'].append({
                'type': 'PII Detection',
                'pii_type': finding.get('type', 'Unknown'),
                'severity': finding.get('severity', 'Medium'),
                'confidence': finding.get('confidence', 0),
                'location': finding.get('location', 'Unknown'),
                'remediation': finding.get('remediation', 'Review and mask PII')
            })
        
        # Process compliance findings
        compliance_results = privacy_results.get('compliance', {})
        for framework, assessment in compliance_results.items():
            if not assessment.get('compliant', True):
                processed['findings'].append({
                    'type': 'Compliance Violation',
                    'framework': framework,
                    'severity': 'High',
                    'issues_count': assessment.get('issues_count', 0),
                    'critical_issues': assessment.get('critical_issues', 0)
                })
        
        return processed
    
    def _process_explainability_results(self, explainability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process explainability analysis results for reporting"""
        
        processed = {
            'summary': {
                'interpretability_score': explainability_results.get('overall_interpretability_score', 8),
                'explanations_generated': len(explainability_results.get('explanations', [])),
                'method_used': explainability_results.get('method', 'Unknown'),
                'global_importance_features': len(explainability_results.get('global_importance', {}).get('features', []))
            },
            'findings': [],
            'recommendations': []
        }
        
        # Assess interpretability level
        interpretability_score = processed['summary']['interpretability_score']
        if interpretability_score < 6:
            processed['findings'].append({
                'type': 'Low Interpretability',
                'severity': 'High',
                'score': interpretability_score,
                'description': 'Model shows low interpretability which may impact trust and compliance'
            })
        elif interpretability_score < 8:
            processed['findings'].append({
                'type': 'Moderate Interpretability',
                'severity': 'Medium',
                'score': interpretability_score,
                'description': 'Model interpretability could be improved for better transparency'
            })
        
        # Generate recommendations
        if interpretability_score < 8:
            processed['recommendations'].extend([
                'Consider using more interpretable model architectures',
                'Implement additional explanation methods for critical decisions',
                'Provide model documentation and decision rationale'
            ])
        
        return processed
    
    def _process_hallucination_results(self, hallucination_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process hallucination detection results for reporting"""
        
        processed = {
            'summary': {
                'factuality_score': hallucination_results.get('overall_factuality_score', 10),
                'hallucinations_detected': len(hallucination_results.get('hallucinations', [])),
                'high_severity_hallucinations': len([h for h in hallucination_results.get('hallucinations', []) 
                                                   if h.get('severity') == 'High']),
                'fact_checks_performed': len(hallucination_results.get('fact_checks', []))
            },
            'findings': [],
            'recommendations': hallucination_results.get('recommendations', [])
        }
        
        # Process hallucination findings
        for hallucination in hallucination_results.get('hallucinations', []):
            processed['findings'].append({
                'type': 'Hallucination',
                'hallucination_type': hallucination.get('type', 'Unknown'),
                'severity': hallucination.get('severity', 'Medium'),
                'confidence': hallucination.get('confidence', 0),
                'claim': hallucination.get('claim', 'Unknown'),
                'evidence': hallucination.get('evidence', 'No evidence provided')
            })
        
        # Process fact-checking results
        fact_checks = hallucination_results.get('fact_checks', [])
        false_claims = [fc for fc in fact_checks if fc.get('status') == 'False']
        
        if false_claims:
            processed['findings'].append({
                'type': 'False Claims',
                'severity': 'High',
                'false_claims_count': len(false_claims),
                'description': f'{len(false_claims)} claims were verified as false'
            })
        
        return processed
    
    def _generate_executive_summary(self, detailed_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from detailed findings"""
        
        summary = {
            'overall_ethics_score': 0.0,
            'risk_level': 'Low',
            'total_issues_found': 0,
            'critical_issues': 0,
            'key_recommendations_count': 0,
            'compliance_status': 'Compliant'
        }
        
        # Collect scores and issues
        module_scores = []
        total_issues = 0
        critical_issues = 0
        
        for module, findings in detailed_findings.items():
            # Extract score based on module type
            if module == 'bias_analysis':
                score = 10 - (findings['summary'].get('overall_bias_score', 0) * 10)
                module_scores.append(max(0, score))
            elif module == 'privacy_analysis':
                score = findings['summary'].get('overall_factuality_score', 10)
                module_scores.append(score)
            elif module == 'explainability_analysis':
                score = findings['summary'].get('interpretability_score', 8)
                module_scores.append(score)
            elif module == 'hallucination_analysis':
                score = findings['summary'].get('factuality_score', 10)
                module_scores.append(score)
            
            # Count issues
            module_issues = len(findings.get('findings', []))
            total_issues += module_issues
            
            # Count critical issues
            critical_in_module = len([f for f in findings.get('findings', []) 
                                    if f.get('severity') == 'High'])
            critical_issues += critical_in_module
        
        # Calculate overall score
        if module_scores:
            summary['overall_ethics_score'] = np.mean(module_scores)
        
        # Determine risk level
        if critical_issues > 2 or summary['overall_ethics_score'] < 5:
            summary['risk_level'] = 'High'
        elif critical_issues > 0 or summary['overall_ethics_score'] < 7:
            summary['risk_level'] = 'Medium'
        else:
            summary['risk_level'] = 'Low'
        
        summary['total_issues_found'] = total_issues
        summary['critical_issues'] = critical_issues
        
        # Compliance status
        if critical_issues > 0:
            summary['compliance_status'] = 'Non-Compliant'
        elif total_issues > 3:
            summary['compliance_status'] = 'Partially Compliant'
        
        return summary
    
    def _compile_recommendations(self, detailed_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile and prioritize recommendations from all modules"""
        
        all_recommendations = []
        
        for module, findings in detailed_findings.items():
            module_recommendations = findings.get('recommendations', [])
            
            for rec in module_recommendations:
                # Convert string recommendations to structured format
                if isinstance(rec, str):
                    priority = self._assess_recommendation_priority(rec, findings)
                    all_recommendations.append({
                        'module': module.replace('_analysis', '').title(),
                        'recommendation': rec,
                        'priority': priority,
                        'impact': 'High' if priority == 'Critical' else 'Medium'
                    })
                elif isinstance(rec, dict):
                    rec['module'] = module.replace('_analysis', '').title()
                    all_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'Low'), 3))
        
        return all_recommendations[:15]  # Limit to top 15 recommendations
    
    def _assess_recommendation_priority(self, recommendation: str, findings: Dict[str, Any]) -> str:
        """Assess priority of a recommendation based on content and findings"""
        
        critical_keywords = ['critical', 'immediate', 'urgent', 'compliance', 'violation']
        high_keywords = ['bias', 'privacy', 'security', 'risk']
        
        rec_lower = recommendation.lower()
        
        # Check for critical issues in findings
        critical_findings = [f for f in findings.get('findings', []) if f.get('severity') == 'High']
        
        if critical_findings or any(keyword in rec_lower for keyword in critical_keywords):
            return 'Critical'
        elif any(keyword in rec_lower for keyword in high_keywords):
            return 'High'
        else:
            return 'Medium'
    
    def _generate_compliance_summary(self, detailed_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance summary across all modules"""
        
        compliance_summary = {
            'overall_compliance': 'Compliant',
            'frameworks_assessed': [],
            'violations_found': 0,
            'compliance_scores': {}
        }
        
        # Check privacy compliance
        if 'privacy_analysis' in detailed_findings:
            privacy_findings = detailed_findings['privacy_analysis']
            
            # Look for compliance violations
            compliance_violations = [f for f in privacy_findings.get('findings', []) 
                                   if f.get('type') == 'Compliance Violation']
            
            if compliance_violations:
                compliance_summary['violations_found'] += len(compliance_violations)
                compliance_summary['overall_compliance'] = 'Non-Compliant'
                
                for violation in compliance_violations:
                    framework = violation.get('framework', 'Unknown')
                    compliance_summary['frameworks_assessed'].append(framework)
        
        # Check bias compliance (fairness standards)
        if 'bias_analysis' in detailed_findings:
            bias_findings = detailed_findings['bias_analysis']
            bias_score = bias_findings['summary'].get('overall_bias_score', 0)
            
            compliance_summary['compliance_scores']['Fairness'] = max(0, 10 - (bias_score * 10))
            
            if bias_score > 0.1:  # Significant bias threshold
                compliance_summary['violations_found'] += 1
                if compliance_summary['overall_compliance'] == 'Compliant':
                    compliance_summary['overall_compliance'] = 'Partially Compliant'
        
        return compliance_summary
    
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates for different formats"""
        
        return {
            'executive_summary': """
# Executive Summary

## Overall Assessment
- **Ethics Score:** {overall_ethics_score:.1f}/10
- **Risk Level:** {risk_level}
- **Compliance Status:** {compliance_status}

## Key Findings
- Total Issues: {total_issues_found}
- Critical Issues: {critical_issues}

## Immediate Actions Required
{critical_recommendations}
            """,
            
            'detailed_section': """
## {module_name} Analysis

### Summary
{module_summary}

### Findings
{module_findings}

### Recommendations
{module_recommendations}
            """
        }
    
    def export_report(self, 
                     report_data: Dict[str, Any],
                     format_type: str = 'markdown',
                     include_charts: bool = True) -> str:
        """Export report in specified format"""
        
        if format_type == 'markdown':
            return self._export_markdown_report(report_data, include_charts)
        elif format_type == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif format_type == 'summary':
            return self._export_summary_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_markdown_report(self, report_data: Dict[str, Any], include_charts: bool) -> str:
        """Export report as markdown"""
        
        markdown = f"""# AI Ethics Audit Report

**Report ID:** {report_data['metadata']['report_id']}
**Generated:** {report_data['metadata']['generated_at']}
**Toolkit Version:** {report_data['metadata']['toolkit_version']}

---

{self._format_executive_summary_markdown(report_data['executive_summary'])}

---

{self._format_detailed_findings_markdown(report_data['detailed_findings'])}

---

{self._format_recommendations_markdown(report_data['recommendations'])}

---

{self._format_compliance_summary_markdown(report_data['compliance_summary'])}

---

*Report generated by AI Ethics Toolkit*
        """
        
        return markdown
    
    def _format_executive_summary_markdown(self, executive_summary: Dict[str, Any]) -> str:
        """Format executive summary as markdown"""
        
        return f"""## Executive Summary

### Overall Assessment
- **Overall Ethics Score:** {executive_summary.get('overall_ethics_score', 0):.1f}/10
- **Risk Level:** {executive_summary.get('risk_level', 'Unknown')}
- **Total Issues Found:** {executive_summary.get('total_issues_found', 0)}
- **Critical Issues:** {executive_summary.get('critical_issues', 0)}
- **Compliance Status:** {executive_summary.get('compliance_status', 'Unknown')}
        """
    
    def _format_detailed_findings_markdown(self, detailed_findings: Dict[str, Any]) -> str:
        """Format detailed findings as markdown"""
        
        markdown = "## Detailed Analysis Results\n\n"
        
        for module, findings in detailed_findings.items():
            module_name = module.replace('_analysis', '').replace('_', ' ').title()
            markdown += f"### {module_name}\n\n"
            
            # Summary
            summary = findings.get('summary', {})
            markdown += "#### Summary\n"
            for key, value in summary.items():
                formatted_key = key.replace('_', ' ').title()
                markdown += f"- **{formatted_key}:** {value}\n"
            markdown += "\n"
            
            # Findings
            module_findings = findings.get('findings', [])
            if module_findings:
                markdown += "#### Key Findings\n"
                for i, finding in enumerate(module_findings, 1):
                    markdown += f"{i}. **{finding.get('type', 'Unknown')}** ({finding.get('severity', 'Medium')} severity)\n"
                    if 'description' in finding:
                        markdown += f"   - {finding['description']}\n"
                markdown += "\n"
        
        return markdown
    
    def _format_recommendations_markdown(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations as markdown"""
        
        markdown = "## Recommendations\n\n"
        
        # Group by priority
        priority_groups = {}
        for rec in recommendations:
            priority = rec.get('priority', 'Medium')
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(rec)
        
        for priority in ['Critical', 'High', 'Medium', 'Low']:
            if priority in priority_groups:
                markdown += f"### {priority} Priority\n\n"
                for i, rec in enumerate(priority_groups[priority], 1):
                    markdown += f"{i}. **{rec.get('module', 'General')}:** {rec.get('recommendation', 'No recommendation')}\n"
                markdown += "\n"
        
        return markdown
    
    def _format_compliance_summary_markdown(self, compliance_summary: Dict[str, Any]) -> str:
        """Format compliance summary as markdown"""
        
        return f"""## Compliance Summary

- **Overall Compliance:** {compliance_summary.get('overall_compliance', 'Unknown')}
- **Violations Found:** {compliance_summary.get('violations_found', 0)}
- **Frameworks Assessed:** {', '.join(compliance_summary.get('frameworks_assessed', ['None']))}

### Compliance Scores
{self._format_compliance_scores(compliance_summary.get('compliance_scores', {}))}
        """
    
    def _format_compliance_scores(self, scores: Dict[str, float]) -> str:
        """Format compliance scores"""
        
        if not scores:
            return "- No compliance scores available"
        
        formatted = ""
        for framework, score in scores.items():
            formatted += f"- **{framework}:** {score:.1f}/10\n"
        
        return formatted
    
    def _export_summary_report(self, report_data: Dict[str, Any]) -> str:
        """Export condensed summary report"""
        
        executive_summary = report_data.get('executive_summary', {})
        
        summary = f"""AI ETHICS AUDIT SUMMARY
========================

Report ID: {report_data['metadata']['report_id']}
Generated: {report_data['metadata']['generated_at']}

OVERALL ASSESSMENT:
- Ethics Score: {executive_summary.get('overall_ethics_score', 0):.1f}/10
- Risk Level: {executive_summary.get('risk_level', 'Unknown')}
- Issues Found: {executive_summary.get('total_issues_found', 0)}
- Critical Issues: {executive_summary.get('critical_issues', 0)}
- Compliance: {executive_summary.get('compliance_status', 'Unknown')}

TOP RECOMMENDATIONS:
"""
        
        # Add top 5 recommendations
        recommendations = report_data.get('recommendations', [])[:5]
        for i, rec in enumerate(recommendations, 1):
            summary += f"{i}. [{rec.get('priority', 'Medium')}] {rec.get('recommendation', 'No recommendation')}\n"
        
        return summary
