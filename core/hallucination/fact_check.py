import requests
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import time

class FactChecker:
    """Fact verification and checking utilities without external APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fact_database = self._initialize_fact_database()
        
    def verify_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract and verify factual claims from text"""
        
        # Extract potential factual claims
        claims = self._extract_claims(text)
        
        verified_claims = []
        for claim in claims:
            verification_result = self._verify_single_claim(claim)
            verified_claims.append(verification_result)
        
        return verified_claims
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text"""
        
        # Patterns that often indicate factual claims
        claim_patterns = [
            r'[A-Z][^.]*(?:study|research|found|discovered|showed|revealed|proved)[^.]*\.',
            r'[A-Z][^.]*(?:according to|based on|research shows|studies indicate)[^.]*\.',
            r'[A-Z][^.]*(?:\d+%|\d+ percent|statistics|data shows)[^.]*\.',
            r'[A-Z][^.]*(?:in \d{4}|since \d{4}|by \d{4})[^.]*\.',
            r'[A-Z][^.]*(?:scientists|researchers|experts|doctors)[^.]*\.',
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        # Also extract sentences with specific claim indicators
        sentences = text.split('.')
        for sentence in sentences:
            if self._contains_factual_indicators(sentence):
                claims.append(sentence.strip())
        
        # Remove duplicates and empty claims
        unique_claims = list(set([claim for claim in claims if len(claim.strip()) > 10]))
        
        return unique_claims[:10]  # Limit to prevent overload
    
    def _contains_factual_indicators(self, sentence: str) -> bool:
        """Check if sentence contains indicators of factual claims"""
        
        factual_indicators = [
            'published in', 'journal of', 'university of', 'institute of',
            'dr.', 'professor', 'research', 'study', 'clinical trial',
            'according to', 'scientists', 'researchers', 'experts',
            'statistics', 'data shows', 'evidence suggests',
            'nobel prize', 'fda approved', 'peer reviewed'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
    
    def _verify_single_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a single factual claim"""
        
        verification_result = {
            'claim': claim,
            'verification_status': 'unverified',
            'confidence': 0.5,
            'evidence': [],
            'contradictions': [],
            'fact_check_source': 'local_analysis'
        }
        
        # Check against local fact database
        local_check = self._check_local_facts(claim)
        if local_check:
            verification_result.update(local_check)
            return verification_result
        
        # Pattern-based verification
        pattern_check = self._pattern_based_verification(claim)
        verification_result.update(pattern_check)
        
        # URL verification if claim contains URLs
        urls = re.findall(r'https?://[^\s]+', claim)
        if urls:
            url_check = self._verify_urls(urls)
            verification_result['url_verification'] = url_check
            
            # Adjust verification status based on URL validity
            valid_urls = sum(1 for result in url_check if result['valid'])
            if len(urls) > 0:
                url_validity_ratio = valid_urls / len(urls)
                if url_validity_ratio == 0:
                    verification_result['verification_status'] = 'likely_false'
                    verification_result['confidence'] = 0.8
                elif url_validity_ratio < 0.5:
                    verification_result['confidence'] = 0.3
        
        return verification_result
    
    def _initialize_fact_database(self) -> Dict[str, Any]:
        """Initialize local fact database with known facts"""
        
        return {
            'known_false_claims': [
                'time machine',
                'teleportation device',
                'perpetual motion machine',
                'cure for all diseases',
                'telepathic communication',
                'precognitive abilities'
            ],
            'suspicious_patterns': [
                r'journal of .* that doesn\'t exist',
                r'university of .* (fake|nonexistent)',
                r'dr\. .* who doesn\'t exist',
                r'study with \d+\.\d+ million participants',  # Unrealistic sample sizes
                r'\d+% effective against all .*',  # Too good to be true claims
            ],
            'temporal_impossibilities': [
                r'invented in \d{4}.*before \d{4}',  # Timeline contradictions
                r'published in \d{4}.*future',
            ]
        }
    
    def _check_local_facts(self, claim: str) -> Optional[Dict[str, Any]]:
        """Check claim against local fact database"""
        
        claim_lower = claim.lower()
        
        # Check known false claims
        for false_claim in self.fact_database['known_false_claims']:
            if false_claim in claim_lower:
                return {
                    'verification_status': 'false',
                    'confidence': 0.9,
                    'evidence': [f"Contains known false claim: {false_claim}"],
                    'fact_check_source': 'local_database'
                }
        
        # Check suspicious patterns
        for pattern in self.fact_database['suspicious_patterns']:
            if re.search(pattern, claim_lower):
                return {
                    'verification_status': 'suspicious',
                    'confidence': 0.2,
                    'evidence': [f"Matches suspicious pattern: {pattern}"],
                    'fact_check_source': 'pattern_analysis'
                }
        
        return None
    
    def _pattern_based_verification(self, claim: str) -> Dict[str, Any]:
        """Verify claim using pattern-based analysis"""
        
        verification_updates = {
            'verification_status': 'unverified',
            'confidence': 0.5,
            'evidence': []
        }
        
        # Check for unrealistic numbers
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', claim)
        for percentage in percentage_matches:
            pct_value = float(percentage)
            if pct_value > 100:
                verification_updates['verification_status'] = 'likely_false'
                verification_updates['confidence'] = 0.8
                verification_updates['evidence'].append(f"Impossible percentage: {pct_value}%")
        
        # Check for temporal inconsistencies
        years = re.findall(r'\b(19|20)\d{2}\b', claim)
        if len(years) > 1:
            year_values = [int(year) for year in years]
            if max(year_values) > 2024:
                verification_updates['verification_status'] = 'likely_false'
                verification_updates['confidence'] = 0.9
                verification_updates['evidence'].append("References future dates")
        
        # Check for impossible scientific claims
        impossible_science = [
            r'faster than light',
            r'perpetual motion',
            r'energy from nothing',
            r'time travel',
            r'telepathy',
            r'precognition'
        ]
        
        for pattern in impossible_science:
            if re.search(pattern, claim, re.IGNORECASE):
                verification_updates['verification_status'] = 'likely_false'
                verification_updates['confidence'] = 0.85
                verification_updates['evidence'].append(f"Contains scientifically impossible claim: {pattern}")
        
        return verification_updates
    
    def _verify_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Verify if URLs are valid and accessible"""
        
        url_results = []
        
        for url in urls:
            result = {
                'url': url,
                'valid': False,
                'accessible': False,
                'error': None
            }
            
            # Basic URL format validation
            try:
                parsed = urlparse(url)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    result['valid'] = True
                else:
                    result['error'] = 'Invalid URL format'
            except Exception as e:
                result['error'] = f'URL parsing error: {str(e)}'
            
            # Check accessibility (with timeout)
            if result['valid']:
                try:
                    response = requests.head(url, timeout=5, allow_redirects=True)
                    if response.status_code == 200:
                        result['accessible'] = True
                    else:
                        result['error'] = f'HTTP {response.status_code}'
                except requests.RequestException as e:
                    result['error'] = f'Connection error: {str(e)}'
                except Exception as e:
                    result['error'] = f'Unknown error: {str(e)}'
            
            url_results.append(result)
        
        return url_results
    
    def check_citation_validity(self, text: str) -> List[Dict[str, Any]]:
        """Check validity of academic citations in text"""
        
        # Extract citation patterns
        citation_patterns = [
            r'(?:Journal of|International Journal of|Proceedings of) [A-Za-z\s]+ \((?:Vol\. \d+, )?(?:Issue \d+, )?\d{4}\)',
            r'[A-Z][a-z]+ et al\. \(\d{4}\)',
            r'[A-Z][a-z]+, [A-Z]\. [A-Z]\. \(\d{4}\)',
            r'doi:\s*10\.\d+/[^\s]+',
            r'PMID:\s*\d+',
            r'arXiv:\d{4}\.\d+',
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append({
                    'citation': match.group(),
                    'type': self._identify_citation_type(match.group()),
                    'position': (match.start(), match.end()),
                    'validity_check': self._check_citation_validity(match.group())
                })
        
        return citations
    
    def _identify_citation_type(self, citation: str) -> str:
        """Identify the type of citation"""
        
        if 'doi:' in citation.lower():
            return 'DOI'
        elif 'pmid:' in citation.lower():
            return 'PubMed'
        elif 'arxiv:' in citation.lower():
            return 'arXiv'
        elif 'journal of' in citation.lower():
            return 'Journal'
        elif 'et al.' in citation:
            return 'Author Citation'
        else:
            return 'Unknown'
    
    def _check_citation_validity(self, citation: str) -> Dict[str, Any]:
        """Check validity of a specific citation"""
        
        validity_result = {
            'likely_valid': False,
            'confidence': 0.5,
            'issues': []
        }
        
        # Check for suspicious journal names
        suspicious_journals = [
            'journal of temporal physics',
            'international journal of time travel',
            'proceedings of impossible science',
            'journal of hydration',
            'quantum consciousness quarterly'
        ]
        
        citation_lower = citation.lower()
        for suspicious in suspicious_journals:
            if suspicious in citation_lower:
                validity_result['likely_valid'] = False
                validity_result['confidence'] = 0.1
                validity_result['issues'].append(f"Suspicious journal name: {suspicious}")
                return validity_result
        
        # Check year reasonableness
        years = re.findall(r'\b(19|20)\d{2}\b', citation)
        for year in years:
            year_int = int(year)
            if year_int > 2024:
                validity_result['issues'].append(f"Future publication year: {year}")
                validity_result['confidence'] *= 0.5
            elif year_int < 1900:
                validity_result['issues'].append(f"Suspiciously old publication year: {year}")
                validity_result['confidence'] *= 0.7
        
        # If no issues found, assume reasonably valid
        if not validity_result['issues']:
            validity_result['likely_valid'] = True
            validity_result['confidence'] = 0.7
        
        return validity_result
