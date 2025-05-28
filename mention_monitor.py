"""
Mention Monitor - Web Scraping Application for Public Mentions
A modular application that scrapes public mentions of companies/individuals
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Mention:
    """Data model for a single mention"""
    source: str
    title: str
    url: str
    snippet: str
    date_found: datetime
    search_term: str
    relevance_score: float = 0.0
    sentiment: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['date_found'] = self.date_found.isoformat()
        return data

@dataclass
class MentionStats:
    """Statistics for mentions over time"""
    total_mentions: int
    mentions_last_7_days: int
    mentions_by_day: Dict[str, int]
    sources_breakdown: Dict[str, int]
    sentiment_breakdown: Dict[str, int]
    avg_relevance_score: float

class MentionStorage:
    """Simple file-based storage for mentions"""
    
    def __init__(self, storage_path: str = "data/mentions.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        
    def save_mentions(self, mentions: List[Mention]) -> None:
        """Save mentions to storage"""
        try:
            existing_mentions = self.load_mentions()
            
            # Merge with existing mentions (avoid duplicates by URL)
            existing_urls = {m['url'] for m in existing_mentions}
            new_mentions = [m for m in mentions if m.url not in existing_urls]
            
            all_mentions = existing_mentions + [m.to_dict() for m in new_mentions]
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(all_mentions, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(new_mentions)} new mentions to storage")
            
        except Exception as e:
            logger.error(f"Error saving mentions: {e}")
            
    def load_mentions(self) -> List[Dict[str, Any]]:
        """Load mentions from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading mentions: {e}")
            return []
            
    def get_mentions_for_term(self, search_term: str) -> List[Dict[str, Any]]:
        """Get mentions for a specific search term"""
        all_mentions = self.load_mentions()
        return [m for m in all_mentions if m.get('search_term', '').lower() == search_term.lower()]

class MentionAnalytics:
    """Analytics engine for mention data"""
    
    def __init__(self, storage: MentionStorage):
        self.storage = storage
        
    def calculate_stats(self, search_term: str) -> MentionStats:
        """Calculate comprehensive statistics for mentions"""
        mentions = self.storage.get_mentions_for_term(search_term)
        
        if not mentions:
            return MentionStats(
                total_mentions=0,
                mentions_last_7_days=0,
                mentions_by_day={},
                sources_breakdown={},
                sentiment_breakdown={},
                avg_relevance_score=0.0
            )
        
        # Convert date strings back to datetime objects
        for mention in mentions:
            if isinstance(mention['date_found'], str):
                mention['date_found'] = datetime.fromisoformat(mention['date_found'])
        
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_mentions = [m for m in mentions if m['date_found'] >= seven_days_ago]
        
        # Mentions by day (last 7 days)
        mentions_by_day = {}
        for i in range(7):
            day = datetime.now() - timedelta(days=i)
            day_str = day.strftime('%Y-%m-%d')
            day_mentions = [m for m in recent_mentions 
                          if m['date_found'].strftime('%Y-%m-%d') == day_str]
            mentions_by_day[day_str] = len(day_mentions)
        
        # Sources breakdown
        sources_breakdown = {}
        for mention in mentions:
            source = mention.get('source', 'Unknown')
            sources_breakdown[source] = sources_breakdown.get(source, 0) + 1
        
        # Sentiment breakdown
        sentiment_breakdown = {}
        for mention in mentions:
            sentiment = mention.get('sentiment', 'neutral')
            sentiment_breakdown[sentiment] = sentiment_breakdown.get(sentiment, 0) + 1
        
        # Average relevance score
        relevance_scores = [m.get('relevance_score', 0.0) for m in mentions]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return MentionStats(
            total_mentions=len(mentions),
            mentions_last_7_days=len(recent_mentions),
            mentions_by_day=mentions_by_day,
            sources_breakdown=sources_breakdown,
            sentiment_breakdown=sentiment_breakdown,
            avg_relevance_score=avg_relevance
        )

class MentionMonitorApp:
    """Main application class"""
    
    def __init__(self):
        self.storage = MentionStorage()
        self.analytics = MentionAnalytics(self.storage)
        self.scrapers = []
        
    def register_scraper(self, scraper):
        """Register a scraper module"""
        self.scrapers.append(scraper)
        
    async def search_mentions(self, search_term: str) -> List[Mention]:
        """Search for mentions using all available scrapers"""
        all_mentions = []
        
        for scraper in self.scrapers:
            try:
                logger.info(f"Searching with {scraper.__class__.__name__}")
                mentions = await scraper.search(search_term)
                all_mentions.extend(mentions)
                logger.info(f"Found {len(mentions)} mentions from {scraper.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error with {scraper.__class__.__name__}: {e}")
                continue
                
        # Remove duplicates based on URL
        unique_mentions = {}
        for mention in all_mentions:
            unique_mentions[mention.url] = mention
            
        final_mentions = list(unique_mentions.values())
        
        # Save to storage
        if final_mentions:
            self.storage.save_mentions(final_mentions)
            
        return final_mentions
        
    def get_stats(self, search_term: str) -> MentionStats:
        """Get analytics for a search term"""
        return self.analytics.calculate_stats(search_term)
        
    def get_recent_mentions(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent mentions for display"""
        mentions = self.storage.get_mentions_for_term(search_term)
        
        # Sort by date (most recent first)
        mentions.sort(key=lambda x: x['date_found'], reverse=True)
        
        return mentions[:limit]
