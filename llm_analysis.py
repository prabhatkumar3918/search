"""
LLM Analysis Module using Cohere
Provides sentiment analysis and relevance scoring for mentions
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass
import json

# Cohere imports
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logging.warning("Cohere not available. Install with: pip install cohere")

from mention_monitor import Mention

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of LLM analysis"""
    sentiment: str  # positive, negative, neutral
    sentiment_confidence: float
    relevance_score: float
    key_topics: List[str]
    summary: str

class CohereAnalyzer:
    """LLM analyzer using Cohere API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.rate_limit_delay = 1  # seconds between API calls
        
        if api_key and COHERE_AVAILABLE:
            try:
                self.client = cohere.Client(api_key)
                logger.info("Cohere client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client: {e}")
                self.client = None
        elif not COHERE_AVAILABLE:
            logger.warning("Cohere not available")
        else:
            logger.warning("Cohere API key not provided")
    
    async def analyze_mentions(self, mentions: List[Mention], search_term: str) -> List[Mention]:
        """Analyze a batch of mentions for sentiment and relevance"""
        if not self.client:
            logger.warning("Cohere client not available, skipping analysis")
            return mentions
        
        analyzed_mentions = []
        
        for mention in mentions:
            try:
                analysis = await self._analyze_single_mention(mention, search_term)
                
                # Update mention with analysis results
                mention.sentiment = analysis.sentiment
                mention.relevance_score = analysis.relevance_score
                
                analyzed_mentions.append(mention)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error analyzing mention {mention.url}: {e}")
                # Keep original mention if analysis fails
                analyzed_mentions.append(mention)
        
        return analyzed_mentions
    
    async def _analyze_single_mention(self, mention: Mention, search_term: str) -> AnalysisResult:
        """Analyze a single mention"""
        # Combine title and snippet for analysis
        text_to_analyze = f"Title: {mention.title}\nContent: {mention.snippet}"
        
        # Perform sentiment analysis
        sentiment_result = await self._get_sentiment(text_to_analyze)
        
        # Calculate relevance score
        relevance_score = await self._calculate_relevance(text_to_analyze, search_term)
        
        # Extract key topics
        key_topics = await self._extract_topics(text_to_analyze)
        
        # Generate summary
        summary = await self._generate_summary(text_to_analyze)
        
        return AnalysisResult(
            sentiment=sentiment_result['sentiment'],
            sentiment_confidence=sentiment_result['confidence'],
            relevance_score=relevance_score,
            key_topics=key_topics,
            summary=summary
        )
    
    async def _get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment analysis from Cohere"""
        try:
            prompt = f"""
Analyze the sentiment of the following text. Respond with only a JSON object containing 'sentiment' (positive/negative/neutral) and 'confidence' (0.0-1.0):
Try going through the text and identifying the overall sentiment, considering the tone, context, and any explicit sentiment indicators. and decide the confidence
based on the source and context of the text. Don't just randomly assign a sentiment, but analyze the text deeply.

Text: {text[:200]}  # Limit text length

JSON Response:"""

            response = self.client.generate(
                model='command-r-plus',
                prompt=prompt,
                max_tokens=100,
                temperature=0.1,
                stop_sequences=['\n\n']
            )
            
            # Try to parse JSON response
            try:
                result = json.loads(response.generations[0].text.strip())
                return {
                    'sentiment': result.get('sentiment', 'neutral'),
                    'confidence': float(result.get('confidence', 0.5))
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                text_response = response.generations[0].text.lower()
                if 'positive' in text_response:
                    sentiment = 'positive'
                elif 'negative' in text_response:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {'sentiment': sentiment, 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    async def _calculate_relevance(self, text: str, search_term: str) -> float:
        """Calculate relevance score using Cohere"""
        try:
            prompt = f"""
Rate how relevant this text is to the search term "{search_term}" on a scale of 0.0 to 1.0, where 1.0 is highly relevant and 0.0 is not relevant at all.

Consider:
- Direct mentions of the search term
- Context and topic relevance
- Quality of the mention

Text: {text[:1000]}

Respond with only a number between 0.0 and 1.0:"""

            response = self.client.generate(
                model='command',
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract numeric score
            score_text = response.generations[0].text.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                return 0.5  # Default score if parsing fails
                
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using Cohere"""
        try:
            prompt = f"""
Extract the top 3-5 key topics or themes from this text. Return only a comma-separated list of topics.

Text: {text[:1000]}

Topics:"""

            response = self.client.generate(
                model='command',
                prompt=prompt,
                max_tokens=50,
                temperature=0.2
            )
            
            topics_text = response.generations[0].text.strip()
            topics = [topic.strip() for topic in topics_text.split(',')]
            return topics[:5]  # Limit to 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _generate_summary(self, text: str) -> str:
        """Generate a brief summary using Cohere"""
        try:
            prompt = f"""
Summarize this text in 1-2 sentences, focusing on the main point:

Text: {text[:1000]}

Summary:"""

            response = self.client.generate(
                model='command',
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

class MentionEnricher:
    """Enriches mentions with additional analysis"""
    
    def __init__(self, analyzer: CohereAnalyzer):
        self.analyzer = analyzer
    
    async def enrich_mentions(self, mentions: List[Mention], search_term: str) -> List[Mention]:
        """Enrich mentions with LLM analysis"""
        if not mentions:
            return mentions
        
        logger.info(f"Enriching {len(mentions)} mentions with LLM analysis")
        
        # Batch process mentions
        enriched_mentions = await self.analyzer.analyze_mentions(mentions, search_term)
        
        # Additional enrichments can be added here
        enriched_mentions = self._add_linkedin_metrics(enriched_mentions)
        
        return enriched_mentions
    
    def _add_linkedin_metrics(self, mentions: List[Mention]) -> List[Mention]:
        """Add LinkedIn-specific metrics (placeholder for now)"""
        # This would integrate with LinkedIn API or scraping
        # For now, we'll simulate some metrics
        
        for mention in mentions:
            # Simulate LinkedIn engagement metrics
            if 'linkedin.com' in mention.url.lower():
                # These would be real metrics from LinkedIn API
                mention.linkedin_likes = 0
                mention.linkedin_comments = 0
                mention.linkedin_shares = 0
                mention.linkedin_views = 0
        
        return mentions

# Factory function
def create_analyzer(cohere_api_key: Optional[str] = None) -> Optional[CohereAnalyzer]:
    """Create Cohere analyzer if available"""
    if cohere_api_key and COHERE_AVAILABLE:
        return CohereAnalyzer(cohere_api_key)
    else:
        logger.warning("Cohere analyzer not available")
        return None