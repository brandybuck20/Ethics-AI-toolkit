import re
from typing import Dict, List, Any, Optional
import numpy as np

class ToxicityChecker:
    """Detect toxic, harmful, or inappropriate content without external APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.toxicity_patterns = self._initialize_toxicity_patterns()
        
    def analyze_toxicity(self, text: str) -> Dict[str, Any]:
        """Analyze text for toxic content"""
        
        results = {
            'overall_toxicity_score': 0.0,
            'toxicity_categories': {},
            'flagged_content': [],
            'severity_level': 'low',
            'recommendations': []
        }
        
        # Analyze different toxicity categories
        for category, patterns in self.toxicity_patterns.items():
            category_result = self._analyze_category(text, category, patterns)
            results['toxicity_categories'][category] = category_result
        
        # Calculate overall toxicity score
        results['overall_toxicity_score'] = self._calculate_overall_score(results['toxicity_categories'])
        
        # Determine severity level
        results['severity_level'] = self._determine_severity(results['overall_toxicity_score'])
        
        # Extract flagged content
        results['flagged_content'] = self._extract_flagged_content(text, results['toxicity_categories'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_toxicity_recommendations(results)
        
        return results
    
    def _initialize_toxicity_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize patterns for detecting different types of toxic content"""
        
        return {
            'hate_speech': [
                {
                    'pattern': r'\b(hate|despise|loathe)\s+(all\s+)?(jews|muslims|christians|blacks|whites|asians|hispanics|latinos|lgbtq|gay|lesbian|trans)',
                    'severity': 0.9,
                    'description': 'Direct hate speech targeting groups'
                },
                {
                    'pattern': r'\b(kill|murder|destroy|eliminate)\s+(all\s+)?(jews|muslims|christians|blacks|whites|asians|hispanics|latinos|lgbtq)',
                    'severity': 1.0,
                    'description': 'Violence targeting specific groups'
                }
            ],
            'harassment': [
                {
                    'pattern': r'\b(you\s+are\s+)?(stupid|idiot|moron|dumb|retard|pathetic|worthless|loser)',
                    'severity': 0.6,
                    'description': 'Personal insults and harassment'
                },
                {
                    'pattern': r'\b(shut\s+up|go\s+away|nobody\s+cares|kill\s+yourself)',
                    'severity': 0.8,
                    'description': 'Aggressive dismissal and harassment'
                }
            ],
            'threats': [
                {
                    'pattern': r'\b(i\s+will\s+)?(kill|murder|hurt|harm|attack|beat|destroy)\s+(you|your|them)',
                    'severity': 1.0,
                    'description': 'Direct threats of violence'
                },
                {
                    'pattern': r'\b(you\s+better\s+watch\s+out|i\s+know\s+where\s+you\s+live|you\'re\s+dead)',
                    'severity': 0.9,
                    'description': 'Implied threats and intimidation'
                }
            ],
            'profanity': [
                {
                    'pattern': r'\b(f[u\*]ck|sh[i\*]t|b[i\*]tch|d[a\*]mn|h[e\*]ll|cr[a\*]p)\b',
                    'severity': 0.3,
                    'description': 'Mild profanity'
                },
                {
                    'pattern': r'\bf\*{2,}k|s\*{2,}t|b\*{2,}h',
                    'severity': 0.3,
                    'description': 'Censored profanity'
                }
            ],
            'discrimination': [
                {
                    'pattern': r'\b(women|men|blacks|whites|asians|jews|muslims)\s+(are\s+)?(inferior|superior|better|worse|dumber|smarter)\s+than',
                    'severity': 0.8,
                    'description': 'Discriminatory comparisons'
                },
                {
                    'pattern': r'\b(all\s+)?(women|men|blacks|whites|asians|jews|muslims)\s+(are\s+)?(lazy|violent|criminals|terrorists|evil)',
                    'severity': 0.9,
                    'description': 'Stereotyping and discrimination'
                }
            ],
            'self_harm': [
                {
                    'pattern': r'\b(want\s+to\s+die|kill\s+myself|end\s+it\s+all|suicide|self\s+harm)',
                    'severity': 0.9,
                    'description': 'Self-harm or suicidal content'
                },
                {
                    'pattern': r'\b(cutting|self\s+injury|hurt\s+myself)',
                    'severity': 0.8,
                    'description': 'Self-injury references'
                }
            ],
            'misinformation': [
                {
                    'pattern': r'\b(vaccines\s+cause\s+autism|covid\s+is\s+fake|earth\s+is\s+flat|holocaust\s+never\s+happened)',
                    'severity': 0.7,
                    'description': 'Dangerous misinformation'
                },
                {
                    'pattern': r'\b(5g\s+causes\s+cancer|chemtrails|lizard\s+people|qanon)',
                    'severity': 0.6,
                    'description': 'Conspiracy theories and misinformation'
                }
            ]
        }
    
    def _analyze_category(self, text: str, category: str, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text for a specific toxicity category"""
        
        matches = []
        max_severity = 0.0
        total_score = 0.0
        
        text_lower = text.lower()
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            severity = pattern_info['severity']
            description = pattern_info['description']
            
            # Find all matches
            pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            
            for match in pattern_matches:
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'severity': severity,
                    'description': description
                })
                
                max_severity = max(max_severity, severity)
                total_score += severity
        
        # Calculate category score
        if matches:
            # Score based on number and severity of matches
            category_score = min(1.0, total_score / len(matches) * (len(matches) ** 0.5) / 10)
        else:
            category_score = 0.0
        
        return {
            'score': category_score,
            'max_severity': max_severity,
            'matches_found': len(matches),
            'matches': matches,
            'flagged': category_score > 0.3
        }
    
    def _calculate_overall_score(self, category_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall toxicity score from category results"""
        
        if not category_results:
            return 0.0
        
        # Weight different categories
        category_weights = {
            'threats': 1.0,
            'hate_speech': 0.9,
            'self_harm': 0.9,
            'harassment': 0.7,
            'discrimination': 0.8,
            'misinformation': 0.6,
            'profanity': 0.3
        }
        
        weighted_scores = []
        for category, result in category_results.items():
            weight = category_weights.get(category, 0.5)
            weighted_score = result['score'] * weight
            weighted_scores.append(weighted_score)
        
        # Overall score is the maximum weighted score (worst category dominates)
        overall_score = max(weighted_scores) if weighted_scores else 0.0
        
        return min(1.0, overall_score)
    
    def _determine_severity(self, toxicity_score: float) -> str:
        """Determine severity level based on toxicity score"""
        
        if toxicity_score >= 0.8:
            return 'critical'
        elif toxicity_score >= 0.6:
            return 'high'
        elif toxicity_score >= 0.4:
            return 'medium'
        elif toxicity_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _extract_flagged_content(self, text: str, category_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract specific flagged content pieces"""
        
        flagged_items = []
        
        for category, result in category_results.items():
            if result['flagged']:
                for match in result['matches']:
                    # Get context around the match
                    context_start = max(0, match['start'] - 50)
                    context_end = min(len(text), match['end'] + 50)
                    context = text[context_start:context_end]
                    
                    flagged_items.append({
                        'category': category,
                        'flagged_text': match['text'],
                        'context': context,
                        'severity': match['severity'],
                        'description': match['description'],
                        'position': (match['start'], match['end'])
                    })
        
        # Sort by severity (highest first)
        flagged_items.sort(key=lambda x: x['severity'], reverse=True)
        
        return flagged_items
    
    def _generate_toxicity_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on toxicity analysis"""
        
        recommendations = []
        severity = results['severity_level']
        flagged_categories = [cat for cat, result in results['toxicity_categories'].items() if result['flagged']]
        
        if severity in ['critical', 'high']:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Remove or revise toxic content before publication")
            
            if 'threats' in flagged_categories:
                recommendations.append("⚠️ Direct threats detected - consider reporting to appropriate authorities")
            
            if 'hate_speech' in flagged_categories:
                recommendations.append("⚠️ Hate speech detected - violates most platform community guidelines")
            
            if 'self_harm' in flagged_categories:
                recommendations.append("⚠️ Self-harm content detected - provide mental health resources")
        
        elif severity == 'medium':
            recommendations.append("⚠️ Moderate toxicity detected - review and consider revisions")
            
            if 'harassment' in flagged_categories:
                recommendations.append("• Remove personal attacks and harassing language")
            
            if 'discrimination' in flagged_categories:
                recommendations.append("• Eliminate discriminatory language and stereotypes")
        
        elif severity == 'low':
            recommendations.append("ℹ️ Minor issues detected - consider cleanup for professional tone")
            
            if 'profanity' in flagged_categories:
                recommendations.append("• Consider replacing profanity with more professional language")
        
        # General recommendations
        if flagged_categories:
            recommendations.extend([
                "• Review community guidelines and content policies",
                "• Consider implementing content moderation workflows",
                "• Train content creators on appropriate language use"
            ])
        else:
            recommendations.append("✅ No significant toxicity detected - content appears appropriate")
        
        return recommendations
    
    def check_cultural_sensitivity(self, text: str) -> Dict[str, Any]:
        """Check for cultural insensitivity and offensive content"""
        
        # Patterns for culturally insensitive content
        sensitivity_patterns = [
            {
                'pattern': r'\b(primitive|backwards|uncivilized|savage)\s+(culture|people|society)',
                'severity': 0.8,
                'description': 'Culturally insensitive language'
            },
            {
                'pattern': r'\b(third\s+world|developing)\s+(mentality|mindset)',
                'severity': 0.6,
                'description': 'Potentially offensive cultural references'
            },
            {
                'pattern': r'\b(exotic|oriental|tribal)\s+(people|culture)',
                'severity': 0.5,
                'description': 'Outdated or potentially offensive cultural terms'
            }
        ]
        
        matches = []
        for pattern_info in sensitivity_patterns:
            pattern_matches = list(re.finditer(pattern_info['pattern'], text, re.IGNORECASE))
            for match in pattern_matches:
                matches.append({
                    'text': match.group(),
                    'severity': pattern_info['severity'],
                    'description': pattern_info['description'],
                    'position': (match.start(), match.end())
                })
        
        overall_score = max([match['severity'] for match in matches]) if matches else 0.0
        
        return {
            'sensitivity_score': overall_score,
            'issues_found': len(matches),
            'flagged_content': matches,
            'recommendations': self._generate_sensitivity_recommendations(matches)
        }
    
    def _generate_sensitivity_recommendations(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Generate cultural sensitivity recommendations"""
        
        if not matches:
            return ["✅ No cultural sensitivity issues detected"]
        
        recommendations = [
            "⚠️ Cultural sensitivity issues detected",
            "• Review language for potential cultural bias or insensitivity",
            "• Consider consulting with cultural sensitivity experts",
            "• Use more inclusive and respectful terminology",
            "• Avoid generalizations about cultures or peoples"
        ]
        
        return recommendations
