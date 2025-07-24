import re
from typing import List, Dict, Any, Tuple
import numpy as np

class PIIDetector:
    """Detect personally identifiable information in text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.mask_findings = config.get('mask_findings', True)
        
        # Define PII patterns
        self.pii_patterns = {
            'Email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'confidence': 0.95,
                'severity': 'Medium'
            },
            'Phone': {
                'pattern': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                'confidence': 0.85,
                'severity': 'Medium'
            },
            'SSN': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                'confidence': 0.9,
                'severity': 'High'
            },
            'Credit Card': {
                'pattern': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'confidence': 0.8,
                'severity': 'High'
            },
            'Names': {
                'pattern': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'confidence': 0.6,
                'severity': 'Low'
            },
            'IP Address': {
                'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                'confidence': 0.85,
                'severity': 'Low'
            },
            'Address': {
                'pattern': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)',
                'confidence': 0.7,
                'severity': 'Medium'
            }
        }
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in given text"""
        
        findings = []
        selected_types = self.config.get('pii_types', [])
        
        for pii_type, pattern_info in self.pii_patterns.items():
            # Check if this PII type is selected for detection
            if not any(pii_type.lower() in selected.lower() for selected in selected_types):
                continue
            
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                confidence = pattern_info['confidence']
                
                # Apply context analysis if enabled
                if self.config.get('context_analysis', True):
                    confidence = self._adjust_confidence_with_context(
                        text, match, pii_type, confidence
                    )
                
                # Skip if below confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Create finding
                finding = {
                    'type': pii_type,
                    'category': 'PII',
                    'severity': pattern_info['severity'],
                    'confidence': confidence,
                    'description': f'{pii_type} detected in text',
                    'location': f'Position {match.start()}-{match.end()}',
                    'pattern': pattern,
                    'original_text': match.group() if not self.mask_findings else self._mask_text(match.group(), pii_type),
                    'context': self._extract_context(text, match) if not self.mask_findings else '[MASKED]',
                    'remediation': f'Remove or mask the detected {pii_type.lower()}'
                }
                
                findings.append(finding)
        
        return findings
    
    def _adjust_confidence_with_context(self, text: str, match: re.Match, pii_type: str, base_confidence: float) -> float:
        """Adjust confidence based on surrounding context"""
        
        # Extract context around the match
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].lower()
        
        # Context keywords that increase confidence
        positive_keywords = {
            'Email': ['email', 'e-mail', 'contact', 'address', 'send', '@'],
            'Phone': ['phone', 'tel', 'call', 'number', 'contact', 'mobile'],
            'SSN': ['ssn', 'social security', 'tax id', 'identification'],
            'Credit Card': ['credit', 'card', 'payment', 'visa', 'mastercard'],
            'Names': ['name', 'mr.', 'mrs.', 'dr.', 'prof.', 'dear'],
            'Address': ['address', 'street', 'live', 'located', 'residence']
        }
        
        # Context keywords that decrease confidence
        negative_keywords = {
            'Names': ['example', 'sample', 'test', 'dummy', 'fake'],
            'Phone': ['example', 'xxx-xxx-xxxx', '000', '111'],
            'Email': ['example.com', 'test.com', 'sample@']
        }
        
        confidence = base_confidence
        
        # Check for positive context
        if pii_type in positive_keywords:
            for keyword in positive_keywords[pii_type]:
                if keyword in context:
                    confidence = min(1.0, confidence + 0.1)
        
        # Check for negative context
        if pii_type in negative_keywords:
            for keyword in negative_keywords[pii_type]:
                if keyword in context:
                    confidence = max(0.1, confidence - 0.2)
        
        return confidence
    
    def _mask_text(self, text: str, pii_type: str) -> str:
        """Mask detected PII text"""
        
        masking_patterns = {
            'Email': lambda t: re.sub(r'[^@.]', '*', t),
            'Phone': lambda t: re.sub(r'\d', '*', t),
            'SSN': lambda t: '***-**-****',
            'Credit Card': lambda t: '**** **** **** ' + t[-4:] if len(t) >= 4 else '****',
            'Names': lambda t: ' '.join(['*' * len(word) for word in t.split()]),
            'Address': lambda t: re.sub(r'\d+', '***', t)
        }
        
        if pii_type in masking_patterns:
            return masking_patterns[pii_type](text)
        else:
            return '*' * len(text)
    
    def _extract_context(self, text: str, match: re.Match, context_size: int = 30) -> str:
        """Extract context around a match"""
        
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        
        context = text[start:end]
        
        # Highlight the match in the context
        match_text = match.group()
        highlighted_context = context.replace(match_text, f"[{match_text}]")
        
        return highlighted_context
