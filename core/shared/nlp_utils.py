import re
import string
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

class NLPUtils:
    """Natural Language Processing utilities for ethics analysis"""
    
    @staticmethod
    def extract_sentences(text: str, min_length: int = 10) -> List[str]:
        """Extract sentences from text with improved accuracy"""
        
        # Improved sentence boundary detection
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= min_length and not sentence.isdigit():
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    @staticmethod
    def extract_claims(text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text for verification"""
        
        # Patterns that typically indicate factual claims
        claim_patterns = [
            r'(?:according to|based on|studies show|research indicates|data shows|statistics reveal)\s+(.+?)(?:\.|$)',
            r'(?:it is|this is|that is)\s+(proven|confirmed|established|verified|documented)\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'(?:scientists|researchers|experts|studies|data)\s+(?:have|has)\s+(?:found|discovered|proven|shown|revealed)\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'(\d+(?:\.\d+)?%?)\s+(?:of|percent|percentage)\s+(.+?)(?:\.|$)',
            r'(?:in|on|during)\s+(\d{4}|\w+\s+\d{4})\s*,?\s*(.+?)(?:\.|$)'
        ]
        
        claims = []
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(0).strip()
                
                # Extract the actual claim (usually the last group)
                groups = match.groups()
                actual_claim = groups[-1] if groups else claim_text
                
                claims.append({
                    'claim': actual_claim.strip(),
                    'full_context': claim_text,
                    'start_position': match.start(),
                    'end_position': match.end(),
                    'pattern_type': 'factual_claim'
                })
        
        return claims
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities using regex patterns"""
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'numbers': [],
            'urls': [],
            'emails': []
        }
        
        # Person names (simplified pattern)
        person_pattern = r'\b(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|Miss)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:Jr\.?|Sr\.?|III?|IV?))?(?=\s+(?:said|stated|reported|claimed|argued|wrote|published|founded|discovered))'
        persons = re.findall(person_pattern, text)
        entities['persons'] = list(set(persons))
        
        # Organizations (simplified)
        org_pattern = r'\b(?:University of|Institute of|Corporation|Company|Inc\.?|Ltd\.?|LLC|Foundation|Association)\s+[A-Z][a-zA-Z\s]*\b|\b[A-Z][a-zA-Z\s]*(?:\s+(?:University|Institute|Corporation|Company|Inc\.?|Ltd\.?|LLC|Foundation|Association))\b'
        orgs = re.findall(org_pattern, text)
        entities['organizations'] = list(set(orgs))
        
        # Dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(date_pattern, text)
        entities['dates'] = list(set(dates))
        
        # Numbers (percentages, statistics)
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        numbers = re.findall(number_pattern, text)
        entities['numbers'] = list(set(numbers))
        
        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        entities['urls'] = list(set(urls))
        
        # Emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities['emails'] = list(set(emails))
        
        return entities
    
    @staticmethod
    def detect_contradictions(sentences: List[str]) -> List[Dict[str, Any]]:
        """Detect potential contradictions between sentences"""
        
        contradictions = []
        
        # Simple contradiction patterns
        contradiction_pairs = [
            (r'\b(?:always|never|all|none|every|no)\b', r'\b(?:sometimes|occasionally|some|few|many|most)\b'),
            (r'\bincreases?\b', r'\bdecreases?\b'),
            (r'\bprevents?\b', r'\bcauses?\b'),
            (r'\bsafe\b', r'\bunsafe|dangerous|harmful\b'),
            (r'\beffective\b', r'\bineffective|useless\b'),
            (r'\btrue\b', r'\bfalse|incorrect|wrong\b')
        ]
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                
                # Check for direct contradictions
                for pattern1, pattern2 in contradiction_pairs:
                    if (re.search(pattern1, sentence1, re.IGNORECASE) and 
                        re.search(pattern2, sentence2, re.IGNORECASE)) or \
                       (re.search(pattern2, sentence1, re.IGNORECASE) and 
                        re.search(pattern1, sentence2, re.IGNORECASE)):
                        
                        contradictions.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'sentence1_index': i,
                            'sentence2_index': j,
                            'contradiction_type': 'semantic_opposition',
                            'confidence': 0.7
                        })
        
        return contradictions
    
    @staticmethod
    def extract_citations(text: str) -> List[Dict[str, Any]]:
        """Extract academic citations and references"""
        
        citations = []
        
        # Academic citation patterns
        citation_patterns = [
            # Journal citations: Author(s) (Year). Title. Journal, Volume(Issue), pages.
            r'([A-Z][a-zA-Z\s,]+)\s*\((\d{4})\)\.\s*(.+?)\.\s*([A-Z][a-zA-Z\s&]+),\s*(\d+)(?:\((\d+)\))?,?\s*(\d+(?:-\d+)?)',
            
            # Book citations: Author (Year). Title. Publisher.
            r'([A-Z][a-zA-Z\s,]+)\s*\((\d{4})\)\.\s*(.+?)\.\s*([A-Z][a-zA-Z\s&]+)',
            
            # DOI patterns
            r'(?:doi:|DOI:)\s*(10\.\d+/[^\s]+)',
            
            # URL citations
            r'(?:Retrieved from|Available at|URL:)\s*(https?://[^\s]+)',
            
            # In-text citations
            r'\(([A-Z][a-zA-Z\s,&]+),?\s*(\d{4})\)',
            
            # Reference to studies/research
            r'(?:according to|citing|referencing)\s+([A-Z][a-zA-Z\s,&]+(?:\s*\(\d{4}\))?)'
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = {
                    'full_citation': match.group(0),
                    'start_position': match.start(),
                    'end_position': match.end(),
                    'groups': match.groups()
                }
                
                # Try to parse components
                groups = match.groups()
                if len(groups) >= 2:
                    citation['author'] = groups[0].strip() if groups[0] else None
                    citation['year'] = groups[1].strip() if groups[1] else None
                    
                    if len(groups) >= 3:
                        citation['title'] = groups[2].strip() if groups[2] else None
                
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def assess_text_quality(text: str) -> Dict[str, Any]:
        """Assess overall text quality for various metrics"""
        
        quality_metrics = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(NLPUtils.extract_sentences(text)),
            'avg_sentence_length': 0,
            'readability_score': 0,
            'complexity_indicators': {},
            'quality_issues': []
        }
        
        sentences = NLPUtils.extract_sentences(text)
        
        if sentences:
            # Average sentence length
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            quality_metrics['avg_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
            
            # Simple readability approximation (Flesch-like)
            avg_sentence_length = quality_metrics['avg_sentence_length']
            avg_syllables_per_word = NLPUtils._estimate_syllables_per_word(text)
            
            # Simplified Flesch Reading Ease approximation
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            quality_metrics['readability_score'] = max(0, min(100, readability_score))
        
        # Complexity indicators
        quality_metrics['complexity_indicators'] = {
            'long_sentences': len([s for s in sentences if len(s.split()) > 30]),
            'complex_words': len(re.findall(r'\b\w{10,}\b', text)),
            'passive_voice_indicators': len(re.findall(r'\b(?:was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE)),
            'jargon_indicators': len(re.findall(r'\b(?:methodology|paradigm|framework|utilize|facilitate|demonstrate)\b', text, re.IGNORECASE))
        }
        
        # Quality issues
        if quality_metrics['avg_sentence_length'] > 25:
            quality_metrics['quality_issues'].append('Sentences are too long on average')
        
        if quality_metrics['readability_score'] < 30:
            quality_metrics['quality_issues'].append('Text may be difficult to read')
        
        if quality_metrics['complexity_indicators']['complex_words'] > quality_metrics['word_count'] * 0.2:
            quality_metrics['quality_issues'].append('High proportion of complex words')
        
        return quality_metrics
    
    @staticmethod
    def _estimate_syllables_per_word(text: str) -> float:
        """Estimate average syllables per word (simplified)"""
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 1.0
        
        total_syllables = 0
        for word in words:
            # Simple syllable counting heuristic
            syllables = max(1, len(re.findall(r'[aeiouy]+', word)))
            if word.endswith('e'):
                syllables -= 1
            if syllables <= 0:
                syllables = 1
            total_syllables += syllables
        
        return total_syllables / len(words)
    
    @staticmethod
    def clean_text_for_analysis(text: str, 
                               remove_punctuation: bool = False,
                               remove_numbers: bool = False,
                               lowercase: bool = False) -> str:
        """Clean text for various analysis purposes"""
        
        cleaned_text = text
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove punctuation if requested
        if remove_punctuation:
            cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers if requested
        if remove_numbers:
            cleaned_text = re.sub(r'\d+', '', cleaned_text)
        
        # Convert to lowercase if requested
        if lowercase:
            cleaned_text = cleaned_text.lower()
        
        # Remove extra spaces again after cleaning
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """Extract keywords using simple frequency analysis"""
        
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_k]
