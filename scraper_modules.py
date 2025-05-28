import asyncio
import aiohttp
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from urllib.parse import quote_plus, urljoin
import json
import time
import random
from dataclasses import dataclass

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not available. Install with: pip install selenium webdriver-manager")

# BeautifulSoup imports
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

# Import our main data models
from mention_monitor import Mention

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Abstract base class for all scrapers"""
    
    def __init__(self):
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_results = 20
        
    @abstractmethod
    async def search(self, query: str) -> List[Mention]:
        """Search for mentions of the given query"""
        pass
        
    def _create_mention(self, source: str, title: str, url: str, snippet: str, search_term: str) -> Mention:
        """Helper to create a Mention object"""
        return Mention(
            source=source,
            title=title,
            url=url,
            snippet=snippet,
            date_found=datetime.now(),
            search_term=search_term,
            relevance_score=0.5  # Default score
        )
        
    async def _rate_limit(self):
        """Apply rate limiting"""
        await asyncio.sleep(self.rate_limit_delay + random.uniform(0, 0.5))

class DuckDuckGoScraper(BaseScraper):
    """Scraper using DuckDuckGo Instant Answer API"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.duckduckgo.com/"
        self.search_url = "https://html.duckduckgo.com/html/"
        
    async def search(self, query: str) -> List[Mention]:
        """Search DuckDuckGo for mentions"""
        mentions = []
        
        try:
            # Use the HTML interface for more reliable results
            params = {
                'q': query,
                'b': '',  # Start from beginning
                'kl': 'us-en',
                'df': 'w'  # Last week
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(self.search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        mentions = self._parse_duckduckgo_html(html_content, query)
                    else:
                        logger.warning(f"DuckDuckGo returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            
        await self._rate_limit()
        return mentions
        
    def _parse_duckduckgo_html(self, html: str, query: str) -> List[Mention]:
        """Parse DuckDuckGo HTML results"""
        mentions = []
        
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available for parsing")
            return mentions
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = soup.find_all('div', class_='result')
            
            for result in results[:self.max_results]:
                try:
                    # Extract title and URL
                    title_elem = result.find('a', class_='result__a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = result.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url:
                        mention = self._create_mention("DuckDuckGo", title, url, snippet, query)
                        mentions.append(mention)
                        
                except Exception as e:
                    logger.debug(f"Error parsing DuckDuckGo result: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo HTML: {e}")
            
        return mentions

class TravilySearchScraper(BaseScraper):
    """Scraper using Travily Search API (if available)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = "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"
        self.base_url = "https://api.tavily.com/search"
        
    async def search(self, query: str) -> List[Mention]:
        """Search using Travily API"""
        mentions = []
        
        if not self.api_key:
            logger.warning("Travily API key not provided, skipping Travily search")
            return mentions
            
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": self.max_results,
                "include_domains": ["linkedin.com"],
                "exclude_domains": []
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        mentions = self._parse_travily_results(data, query)
                    else:
                        logger.warning(f"Travily returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Travily search error: {e}")
            
        await self._rate_limit()
        return mentions
        
    def _parse_travily_results(self, data: Dict[str, Any], query: str) -> List[Mention]:
        """Parse Travily search results"""
        mentions = []
        
        try:
            results = data.get('results', [])
            
            for result in results:
                title = result.get('title', '')
                url = result.get('url', '')
                snippet = result.get('content', '')
                
                if title and url:
                    mention = self._create_mention("Travily", title, url, snippet, query)
                    mentions.append(mention)
                    
        except Exception as e:
            logger.error(f"Error parsing Travily results: {e}")
            
        return mentions

class SeleniumScraper(BaseScraper):
    """Fallback scraper using Selenium WebDriver"""
    
    def __init__(self):
        super().__init__()
        self.driver = None
        
    def _get_driver(self):
        """Initialize Chrome WebDriver with options"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available")
            
        if self.driver is None:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            try:
                self.driver = webdriver.Chrome(options=options)
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}")
                raise
                
        return self.driver
        
    async def search(self, query: str) -> List[Mention]:
        """Search using Selenium WebDriver"""
        mentions = []
        
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, skipping Selenium search")
            return mentions
            
        try:
            driver = self._get_driver()
            
            # Search on Google (as fallback)
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=nws&tbs=qdr:w"
            driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-sokoban-container]"))
            )
            
            # Parse results
            mentions = self._parse_google_results(driver, query)
            
        except Exception as e:
            logger.error(f"Selenium search error: {e}")
            
        await self._rate_limit()
        return mentions
        
    def _parse_google_results(self, driver, query: str) -> List[Mention]:
        """Parse Google search results using Selenium"""
        mentions = []
        
        try:
            # Find result containers
            results = driver.find_elements(By.CSS_SELECTOR, "div[data-sokoban-container] div.g")
            
            for result in results[:self.max_results]:
                try:
                    # Extract title and URL
                    title_elem = result.find_element(By.CSS_SELECTOR, "h3")
                    link_elem = result.find_element(By.CSS_SELECTOR, "a")
                    
                    title = title_elem.text if title_elem else ""
                    url = link_elem.get_attribute('href') if link_elem else ""
                    
                    # Extract snippet
                    snippet_elem = result.find_element(By.CSS_SELECTOR, ".VwiC3b")
                    snippet = snippet_elem.text if snippet_elem else ""
                    
                    if title and url:
                        mention = self._create_mention("Google News", title, url, snippet, query)
                        mentions.append(mention)
                        
                except Exception as e:
                    logger.debug(f"Error parsing Google result: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing Google results: {e}")
            
        return mentions
        
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

class BeautifulSoupScraper(BaseScraper):
    """Generic scraper using BeautifulSoup for various sites"""
    
    def __init__(self):
        super().__init__()
        self.session = None
        
    async def search(self, query: str) -> List[Mention]:
        """Search using BeautifulSoup for web scraping"""
        mentions = []
        
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available, skipping BS4 search")
            return mentions
            
        try:
            # Use requests for simpler sites
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Search on Bing as an alternative
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&qft=interval%3d%227%22"
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                mentions = self._parse_bing_results(response.text, query)
                
        except Exception as e:
            logger.error(f"BeautifulSoup search error: {e}")
            
        await self._rate_limit()
        return mentions
        
    def _parse_bing_results(self, html: str, query: str) -> List[Mention]:
        """Parse Bing search results"""
        mentions = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = soup.find_all('li', class_='b_algo')
            
            for result in results[:self.max_results]:
                try:
                    # Extract title and URL
                    title_elem = result.find('h2')
                    if not title_elem:
                        continue
                        
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    url = link_elem.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = result.find('p')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url:
                        mention = self._create_mention("Bing", title, url, snippet, query)
                        mentions.append(mention)
                        
                except Exception as e:
                    logger.debug(f"Error parsing Bing result: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing Bing HTML: {e}")
            
        return mentions


class LinkedInScraper(BaseScraper):
    """Scraper for LinkedIn content using Selenium."""

    def __init__(self):
        super().__init__()
        self.driver = None

    def _get_driver(self):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available")

        if self.driver is None:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

            try:
                self.driver = webdriver.Chrome(options=options)
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver for LinkedInScraper: {e}")
                raise

        return self.driver

    async def search(self, query: str) -> List[Mention]:
        mentions = []

        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, skipping LinkedIn search")
            return mentions

        try:
            driver = self._get_driver()
            search_url = f"https://www.linkedin.com/search/results/content/?keywords={quote_plus(query)}"
            driver.get(search_url)

            # Let the page load
            time.sleep(5)

            post_containers = driver.find_elements(By.CLASS_NAME, 'reusable-search__result-container')

            for post in post_containers[:self.max_results]:
                try:
                    text = post.text.strip()
                    url_elem = post.find_element(By.TAG_NAME, 'a')
                    url = url_elem.get_attribute('href') if url_elem else ''

                    if text and url:
                        mention = self._create_mention("LinkedIn", text[:80], url, text, query)
                        mentions.append(mention)

                except Exception as e:
                    logger.debug(f"Error parsing LinkedIn result: {e}")

        except Exception as e:
            logger.error(f"LinkedIn search error: {e}")

        await self._rate_limit()
        return mentions

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

# Factory function to create scrapers
def create_scrapers(travily_api_key: Optional[str] = None) -> List[BaseScraper]:
    """Create and return available scrapers"""
    scrapers = []
    
    # Always try DuckDuckGo first
    scrapers.append(DuckDuckGoScraper())
    
    # Add Travily if API key is provided
    if travily_api_key:
        scrapers.append(TravilySearchScraper(travily_api_key))
    
    # Add BeautifulSoup scraper
    if BS4_AVAILABLE:
        scrapers.append(BeautifulSoupScraper())
    
    # Add Selenium as fallback
    if SELENIUM_AVAILABLE:
        scrapers.append(SeleniumScraper())
        scrapers.append(LinkedInScraper())
    
    return scrapers

# Cleanup function
def cleanup_scrapers(scrapers: List[BaseScraper]):
    """Clean up scraper resources"""
    for scraper in scrapers:
        if isinstance(scraper, (SeleniumScraper, LinkedInScraper)):
            scraper.close()