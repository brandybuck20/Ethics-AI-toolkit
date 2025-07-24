import re
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from urllib.parse import urlparse
import time

class HallucinationDetector:
    """Main hallucination detection engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.detection_types = config.get('detection_types', [])
        
        # Initialize detection patterns
        self.patterns = self._initialize_patterns()
        
        # Initialize fact-checking sources
        self.fact_checkers = self._initialize_fact_checkers()
    
    def analyze_text(self, text: str, is_demo: bool = False) -> Dict[str, Any]:
        """Analyze text for hallucinations"""
        
        hallucinations = []
        fact_checks = []
        
        # Detect different types of hallucinations
        for detection_type in self.detection_types:
            if "Factual Inaccuracies" in detection_type:
                hallucinations.extend(self._detect_factual_inaccuracies(text))
            elif "Non-existent URLs" in detection_type:
                hallucinations.extend(self._detect_fake_urls(text))
            elif "Fake Citations" in detection_type:
                hallucinations.extend(self._detect_fake_citations(text))
            elif "Impossible Claims" in detection_type:
                hallucinations.extend(self._detect_impossible_claims(text))
            elif "Internal Contradictions" in detection_type:
                hallucinations.extend(self._detect_contradictions(text))
        
        # Perform fact-checking if enabled
        if self.config.get('fact_check_sources'):
            fact_checks = self._perform_fact_checking(text)
        
        # Generate results
        results = self._generate_results(hallucinations, fact_checks, text, is_demo)
        
        return results
    
    def analyze_llm_interaction(self, prompt: str, response: str) -> Dict[str, Any]:
        """Analyze LLM prompt-response interaction"""
        
        # Analyze the response for hallucinations
        response_results = self.analyze_text(response)
        
        # Add context about the prompt
        response_results['interaction_context'] = {
            'prompt': prompt,
            'response_length': len(response),
            'prompt_complexity': self._assess_prompt_complexity(prompt)
        }
        
        # Adjust scores based on prompt-response relationship
        response_results = self._adjust_for_prompt_context(response_results, prompt)
        
        return response_results
    
    def _initialize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize detection patterns"""
        
        patterns = {
            'fake_citations': [
                {
                    'pattern': r'Journal of [A-Za-z\s]+ \(Vol\. \d+, Issue \d+\)',
                    'description': 'Potentially fabricated journal citation',
                    'confidence': 0.7
                },
                {
                    'pattern': r'published in the [A-Za-z\s]+ Journal',
                    'description': 'Generic journal reference that may not exist',
                    'confidence': 0.6
                },
                {
                    'pattern': r'according to a study by Dr\. [A-Z][a-z]+ [A-Z][a-z]+',
                    'description': 'Potentially fabricated researcher citation',
                    'confidence': 0.8
                }
            ],
            'impossible_claims': [
                {
                    'pattern': r'(\d+)% effective|(\d+)% accurate',
                    'description': 'Potentially fabricated percentage claim',
                    'confidence': 0.5
                },
                {
                    'pattern': r'won.*Nobel Prize.*in (\d{4})',
                    'description': 'Potentially false Nobel Prize claim',
                    'confidence': 0.9
                }
            ],
            'fake_urls': [
                {
                    'pattern': r'https?://[^\s]+',
                    'description': 'URL that may not exist',
                    'confidence': 0.8
                }
            ]
        }
        
        return patterns
    
    def _detect_factual_inaccuracies(self, text: str) -> List[Dict[str, Any]]:
        """Detect factual inaccuracies in text"""
        
        hallucinations = []
        
        # Common factual error patterns
        factual_patterns = [
            (r'Python (\d+\.\d+) was released.*(\d{4})', 'Python version release date'),
            (r'(\d+) million.*billion', 'Number magnitude inconsistency'),
            (r'impossible.*solve.*O\(1\)', 'Computational complexity impossibility'),
            (r'(\w+) can completely prevent all forms of (\w+)', 'Medical absolutism')
        ]
        
        for pattern, description in factual_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                hallucinations.append({
                    'type': 'Factual Inaccuracy',
                    'claim': match.group(),
                    'description': description,
                    'severity': 'High',
                    'confidence': 0.8,
                    'location': f'Position {match.start()}-{match.end()}',
                    'evidence': f'Pattern matched: {pattern}'
                })
        
        return hallucinations
    
    def _detect_fake_urls(self, text: str) -> List[Dict[str, Any]]:
        """Detect potentially fake URLs"""
        
        hallucinations = []
        url_pattern = r'https?://[^\s]+'
        
        urls = re.finditer(url_pattern, text)
        for url_match in urls:
            url = url_match.group()
            
            # Skip trusted domains
            trusted_domains = self.config.get('trusted_domains', [])
            domain = urlparse(url).netloc
            
            if any(trusted in domain for trusted in trusted_domains):
                continue
            
            # Check if URL exists (simplified check)
            if self.config.get('verify_links', False):
                is_valid = self._validate_url_existence(url)
                if not is_valid:
                    hallucinations.append({
                        'type': 'Non-existent URL',
                        'claim': url,
                        'description': 'URL appears to be non-existent or inaccessible',
                        'severity': 'High',
                        'confidence': 0.9,
                        'location': f'Position {url_match.start()}-{url_match.end()}',
                        'evidence': 'URL validation failed'
                    })
        
        return hallucinations
    
    def _detect_fake_citations(self, text: str) -> List[Dict[str, Any]]:
        """Detect fake academic citations"""
        
        hallucinations = []
        
        for pattern_info in self.patterns['fake_citations']:
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Additional validation for academic citations
                citation = match.group()
                confidence = pattern_info['confidence']
                
                # Increase confidence if citation looks suspicious
                if 'Temporal Physics' in citation or 'Hydration' in citation:
                    confidence = 0.95
                
                if confidence >= self.confidence_threshold:
                    hallucinations.append({
                        'type': 'Fake Citation',
                        'claim': citation,
                        'description': pattern_info['description'],
                        'severity': 'Medium',
                        'confidence': confidence,
                        'location': f'Position {match.start()}-{match.end()}',
                        'evidence': 'Suspicious citation pattern detected'
                    })
        
        return hallucinations
    
    def _detect_impossible_claims(self, text: str) -> List[Dict[str, Any]]:
        """Detect logically impossible claims"""
        
        hallucinations = []
        
        impossible_patterns = [
            (r'time machine.*transport.*future', 'Time travel claim'),
            (r'quantum.*solve.*NP-complete.*O\(1\)', 'Computational impossibility'),
            (r'telepathic.*variable.*assignment', 'Impossible programming feature'),
            (r'drinking.*(\d+\.\d+).*glasses.*prevent all.*cancer', 'Medical impossibility')
        ]
        
        for pattern, description in impossible_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                hallucinations.append({
                    'type': 'Impossible Claim',
                    'claim': match.group(),
                    'description': description,
                    'severity': 'High',
                    'confidence': 0.9,
                    'location': f'Position {match.start()}-{match.end()}',
                    'evidence': 'Claim violates known physical/logical laws'
                })
        
        return hallucinations
    
    def _detect_contradictions(self, text: str) -> List[Dict[str, Any]]:
        """Detect internal contradictions"""
        
        hallucinations = []
        
        # Simple contradiction detection (could be expanded)
        sentences = text.split('.')
        
        # Look for contradictory statements
        contradiction_pairs = [
            (r'prevent.*cancer', r'may cause.*cancer'),
            (r'completely safe', r'side effects'),
            (r'never fails', r'sometimes fails')
        ]
        
        for pos_pattern, neg_pattern in contradiction_pairs:
            pos_matches = [s for s in sentences if re.search(pos_pattern, s, re.IGNORECASE)]
            neg_matches = [s for s in sentences if re.search(neg_pattern, s, re.IGNORECASE)]
            
            if pos_matches and neg_matches:
                hallucinations.append({
                    'type': 'Internal Contradiction',
                    'claim': f"{pos_matches[0].strip()} vs {neg_matches[0].strip()}",
                    'description': 'Contradictory statements found in text',
                    'severity': 'Medium',
                    'confidence': 0.7,
                    'location': 'Multiple locations',
                    'evidence': 'Contradictory patterns detected'
                })
        
        return hallucinations
    
    def _perform_fact_checking(self, text: str) -> List[Dict[str, Any]]:
        """Perform fact-checking against external sources"""
        
        fact_checks = []
        
        # Extract factual claims for checking
        factual_claims = self._extract_factual_claims(text)
        
        for claim in factual_claims:
            # Simplified fact-checking (in production, use real APIs)
            check_result = {
                'claim': claim,
                'status': 'Unverified',
                'source': 'Manual review required',
                'confidence': 0.5
            }
            
            # Simple Wikipedia-style checking
            if 'Wikipedia' in self.config.get('fact_check_sources', []):
                check_result = self._check_against_wikipedia(claim)
            
            fact_checks.append(check_result)
        
        return fact_checks
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text"""
        
        # Simple claim extraction based on patterns
        claim_patterns = [
            r'[A-Z][^.]*(?:study|research|found|discovered|proved)[^.]*\.',
            r'[A-Z][^.]*(?:according to|based on)[^.]*\.',
            r'[A-Z][^.]*(?:\d+%|\d+ percent)[^.]*\.'
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, text)
            claims.extend(matches)
        
        return claims[:5]  # Limit to prevent overload
    
    def _check_against_wikipedia(self, claim: str) -> Dict[str, Any]:
        """Simplified Wikipedia fact-checking"""
        
        # In production, this would use Wikipedia API
        # For now, return mock results
        suspicious_terms = ['time machine', 'ChronoShift', 'Journal of Temporal Physics', 'telepathic']
        
        if any(term in claim.lower() for term in suspicious_terms):
            return {
                'claim': claim,
                'status': 'False',
                'source': 'No Wikipedia entry found',
                'confidence': 0.9
            }
        else:
            return {
                'claim': claim,
                'status': 'Unverified',
                'source': 'Wikipedia search inconclusive',
                'confidence': 0.5
            }
    
    def _validate_url_existence(self, url: str) -> bool:
        """Check if URL exists and is accessible"""
        
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except (requests.RequestException, Exception):
            return False
    
    def _assess_prompt_complexity(self, prompt: str) -> str:
        """Assess the complexity of the input prompt"""
        
        word_count = len(prompt.split())
        
        if word_count > 50:
            return "Complex"
        elif word_count > 20:
            return "Medium"
        else:
            return "Simple"
    
    def _adjust_for_prompt_context(self, results: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Adjust results based on prompt context"""
        
        # Increase severity if hallucinations appear in response to factual questions
        factual_keywords = ['what is', 'who is', 'when did', 'where is', 'how many']
        
        if any(keyword in prompt.lower() for keyword in factual_keywords):
            for hallucination in results.get('hallucinations', []):
                if hallucination['severity'] == 'Medium':
                    hallucination['severity'] = 'High'
        
        return results
    
    def _generate_results(self, hallucinations: List[Dict[str, Any]], 
                         fact_checks: List[Dict[str, Any]], 
                         text: str, is_demo: bool = False) -> Dict[str, Any]:
        """Generate comprehensive results summary"""
        
        # Calculate summary statistics
        total_hallucinations = len(hallucinations)
        high_severity = len([h for h in hallucinations if h['severity'] == 'High'])
        verified_false = len([fc for fc in fact_checks if fc['status'] == 'False'])
        
        # Calculate factuality score
        text_length = len(text.split())
        hallucination_density = total_hallucinations / max(text_length / 100, 1)
        
        factuality_score = max(0, 10 - (high_severity * 2 + hallucination_density))
        
        # Determine risk level
        if high_severity > 2 or verified_false > 1:
            risk_level = 'High'
        elif total_hallucinations > 3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_factuality_score': round(factuality_score, 1),
            'summary': {
                'total_issues': total_hallucinations,
                'high_risk_findings': high_severity,
                'verified_claims': len(fact_checks),
                'avg_confidence': np.mean([h['confidence'] for h in hallucinations]) if hallucinations else 0.8,
                'factuality_level': 'High' if factuality_score > 7 else 'Medium' if factuality_score > 4 else 'Low',
                'risk_level': risk_level
            },
            'hallucinations': hallucinations,
            'fact_checks': fact_checks,
            'is_demo': is_demo
        }
