"""Newsletter processing and scraping functionality."""
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from pathlib import Path
import google.generativeai as genai
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass
from services.config_service import ConfigService
from services.cache_service import CacheService
from services.rate_limiting_service import RateLimitingService
from services.storage_service import StorageService
from contextlib import asynccontextmanager
import aiofiles
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import ssl
import certifi
import signal
import sys
import hashlib
from urllib.parse import urlparse
import threading
import atexit
from contextlib import contextmanager
import calendar
from services.constants import TOPICS

# Import the WebCrawler implementation
from web_crawler import WebCrawler

# Import crawler integration
from crawler_integration import scrape_news_with_crawler

# Configure services
config = ConfigService()
cache_service = CacheService()
rate_limiter = RateLimitingService()
storage_service = StorageService()
config_service = config  # Add this line to make config accessible as config_service

# Configure rate limits for different operations
rate_limiter.configure_limit("feed_fetch", calls=5, period=1)  # 5 feeds per second
rate_limiter.configure_limit("article_process", calls=10, period=1)  # 10 articles per second
rate_limiter.configure_limit("summary_generation", calls=2, period=1)  # 2 summaries per second

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add cache configuration
CACHE_FILE = config.cache_file_path
CRAWLER_CACHE_FILE = "crawler_cache.json"

# Add these constants near the top after other constants
DEFAULT_SCRAPE_INTERVAL = 3600  # Default to running every hour (in seconds)
MINIMUM_SCRAPE_INTERVAL = 300   # Minimum 5 minutes between scrapes

class RateLimitException(Exception):
    """Exception raised when rate limits are exceeded."""
    pass

class FeedError(Exception):
    """Custom exception for feed processing errors."""
    pass

class CacheError(Exception):
    """Custom exception for cache operations."""
    pass

@dataclass
class ScrapingMetrics:
    """Track scraping performance metrics."""
    start_time: float
    total_feeds: int = 0
    successful_feeds: int = 0
    failed_feeds: int = 0
    total_articles: int = 0
    matched_articles: int = 0
    summary_generated: int = 0
    summary_failed: int = 0
    cache_hits: int = 0
    rate_limits: int = 0

    def get_stats(self) -> Dict:
        """Get scraping statistics."""
        duration = time.time() - self.start_time
        return {
            "duration_seconds": round(duration, 2),
            "total_feeds": self.total_feeds,
            "successful_feeds": self.successful_feeds,
            "failed_feeds": self.failed_feeds,
            "total_articles": self.total_articles,
            "matched_articles": self.matched_articles,
            "summary_success_rate": f"{(self.summary_generated/self.matched_articles)*100:.1f}%" if self.matched_articles else "0%",
            "cache_hit_rate": f"{(self.cache_hits/self.matched_articles)*100:.1f}%" if self.matched_articles else "0%",
            "rate_limits": self.rate_limits
        }

class ResourceManager:
    """Manages cleanup of application resources."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._sessions: List[requests.Session] = []
        self._async_sessions: List[aiohttp.ClientSession] = []
        self._cleanup_handlers: List[callable] = []
        self._initialized = True
        
        # Register cleanup on program exit
        atexit.register(self.cleanup)
    
    def register_session(self, session: requests.Session) -> None:
        """Register a requests session for cleanup."""
        self._sessions.append(session)
    
    def register_async_session(self, session: aiohttp.ClientSession) -> None:
        """Register an async session for cleanup."""
        self._async_sessions.append(session)
    
    def register_cleanup_handler(self, handler: callable) -> None:
        """Register a cleanup handler."""
        self._cleanup_handlers.append(handler)
    
    def cleanup(self) -> None:
        """Clean up all registered resources."""
        # Clean up sessions
        for session in self._sessions:
            try:
                session.close()
            except Exception as e:
                logging.error(f"Error closing session: {e}")
        
        # Clean up async sessions
        if self._async_sessions:
            loop = asyncio.get_event_loop()
            for session in self._async_sessions:
                try:
                    if not loop.is_closed():
                        loop.run_until_complete(session.close())
                except Exception as e:
                    logging.error(f"Error closing async session: {e}")
        
        # Run cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logging.error(f"Error in cleanup handler: {e}")

@contextmanager
def managed_session():
    """Context manager for creating and cleaning up a requests session."""
    session = requests.Session()
    
    # Configure retries
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        yield session
    finally:
        session.close()

async def cleanup_async_resources():
    """Clean up async resources."""
    # This will be called by the ResourceManager
    pass

@cache_service.cache_decorator(namespace="feed", ttl=3600)
async def fetch_feed(session: aiohttp.ClientSession, source: str, url: str) -> List[Dict]:
    """Fetch and parse an RSS feed."""
    await rate_limiter.acquire("feed_fetch")
    
    try:
        async with session.get(url, ssl=ssl.create_default_context(cafile=certifi.where())) as response:
            response.raise_for_status()
            content = await response.text()
            
        soup = BeautifulSoup(content, 'xml')
        items = soup.find_all(['item', 'entry'])
        
        articles = []
        for item in items:
            try:
                article = {
                    'title': item.find(['title', 'headline']).text.strip(),
                    'link': item.find(['link', 'url']).text.strip(),
                    'description': item.find(['description', 'summary', 'content']).text.strip(),
                    'pub_date': item.find(['pubDate', 'published', 'updated']).text.strip(),
                    'source': source
                }
                articles.append(article)
            except (AttributeError, TypeError) as e:
                logging.warning(f"Error parsing feed item from {source}: {e}")
                continue
                
        return articles
        
    except Exception as e:
        logging.error(f"Error fetching feed {url}: {e}")
        raise FeedError(f"Failed to fetch feed {url}: {str(e)}")

@rate_limiter.rate_limit_decorator(key="article_process")
async def process_article(session: aiohttp.ClientSession, article: Dict, source: str) -> Optional[Dict]:
    """Process a single article."""
    try:
        # Generate article ID
        article_id = generate_article_id(article)
        
        # Check if article already exists
        if await storage_service.article_exists(article_id):
            return None
            
        # Parse publication date
        pub_date = parse_date(article['pub_date'])
        
        # Check if article is recent (within last 48 hours)
        is_recent = False
        try:
            pub_datetime = datetime.fromisoformat(pub_date)
            is_recent = (datetime.now() - pub_datetime) < timedelta(hours=48)
        except (ValueError, TypeError):
            # If date parsing fails, don't mark as recent
            pass
            
        # Clean and validate article data
        cleaned_article = {
            'id': article_id,
            'metadata': {
                'title': article['title'],
                'link': article['link'],
                'description': clean_html(article['description']),
                'pub_date': pub_date,
                'source': source,
                'topic': classify_topic(article['title'], article['description']),
                'has_full_content': False,
                'is_recent': is_recent  # Add is_recent flag
            }
        }
        
        # Try to fetch full content
        try:
            full_content = await fetch_full_article_content(session, article['link'])
            if full_content:
                cleaned_article['metadata']['description'] = full_content
                cleaned_article['metadata']['has_full_content'] = True
                
                # Calculate reading time (average 200 words per minute)
                word_count = len(full_content.split())
                cleaned_article['metadata']['reading_time'] = max(1, round(word_count / 200))
                cleaned_article['metadata']['word_count'] = word_count
        except Exception as e:
            logging.warning(f"Could not fetch full content for {article['link']}: {e}")
        
        # Generate summary if needed
        if cleaned_article['metadata']['has_full_content']:
            try:
                summary = await summarize_text(cleaned_article['metadata']['description'])
                if summary:
                    cleaned_article['metadata']['summary'] = summary
            except Exception as e:
                logging.warning(f"Could not generate summary for {article_id}: {e}")
        
        return cleaned_article
        
    except Exception as e:
        logging.error(f"Error processing article: {e}")
        return None

# Add the missing scrape_news function
async def scrape_news(status_callback: Optional[Callable] = None) -> bool:
    """
    Main function to scrape news from feeds.
    
    Args:
        status_callback: Optional function to report progress
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Pass to the crawler integration implementation
        return await scrape_news_with_crawler(status_callback)
    except Exception as e:
        logging.error(f"Error in scrape_news: {e}")
        return False

async def generate_missing_summaries() -> bool:
    """
    Generate summaries for articles that don't have them.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logging.info("Generating missing summaries...")
        
        # Get all articles that have full content but no summary
        articles = await storage_service.get_articles_without_summaries()
        
        if not articles:
            logging.info("No articles need summaries")
            return True
            
        success_count = 0
        failure_count = 0
        
        for article in articles:
            try:
                # Add delay to avoid rate limiting
                await asyncio.sleep(config_service.rate_limit_delay)
                
                # Generate summary using AI service
                summary = await summarize_text(article['metadata']['description'])
                
                if summary:
                    # Update article with new summary
                    await storage_service.update_article_summary(article['id'], summary)
                    success_count += 1
                else:
                    logging.warning(f"Empty summary generated for article {article['id']}")
                    failure_count += 1
                    
            except RateLimitException as e:
                logging.warning(f"Rate limit hit while generating summary for article {article['id']}: {e}")
                # Add longer delay before retrying
                await asyncio.sleep(config_service.rate_limit_delay * 3)
                try:
                    summary = await summarize_text(article['metadata']['description'])
                    if summary:
                        await storage_service.update_article_summary(article['id'], summary)
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as retry_e:
                    logging.error(f"Retry failed for article {article['id']}: {retry_e}")
                    failure_count += 1
            except Exception as e:
                logging.error(f"Error generating summary for article {article['id']}: {e}")
                failure_count += 1
                continue
                
        logging.info(f"Summary generation complete. Success: {success_count}, Failures: {failure_count}")
        return failure_count == 0
        
    except Exception as e:
        logging.error(f"Error generating missing summaries: {e}")
        return False

async def verify_database_consistency() -> tuple[bool, dict]:
    """
    Verify database consistency and repair if needed.
    
    Returns:
        tuple: (success, stats)
    """
    try:
        logging.info("Verifying database consistency...")
        stats = {
            "checked": 0,
            "repaired": 0,
            "errors": 0
        }
        
        # Get all articles from storage
        articles = await storage_service.get_all_articles()
        stats["checked"] = len(articles)
        
        for article in articles:
            try:
                # Check for required fields
                required_fields = ['id', 'metadata', 'document']
                missing_fields = [field for field in required_fields if field not in article]
                
                if missing_fields:
                    logging.warning(f"Article {article.get('id', 'unknown')} missing fields: {missing_fields}")
                    stats["errors"] += 1
                    continue
                
                # Check metadata consistency
                if not article['metadata'] or not isinstance(article['metadata'], dict):
                    logging.warning(f"Article {article['id']} has invalid metadata")
                    # Attempt to repair by initializing empty metadata
                    await storage_service.update_article_metadata(article['id'], {})
                    stats["repaired"] += 1
                    continue
                
                # Check for duplicate articles by URL
                if 'link' in article['metadata']:
                    duplicates = await storage_service.find_articles_by_url(article['metadata']['link'])
                    if len(duplicates) > 1:
                        logging.warning(f"Found {len(duplicates)} duplicate articles for URL: {article['metadata']['link']}")
                        # Keep the most recent one and delete others
                        sorted_dupes = sorted(duplicates, key=lambda x: x.get('metadata', {}).get('published_date', ''), reverse=True)
                        for dupe in sorted_dupes[1:]:
                            await storage_service.delete_article(dupe['id'])
                            stats["repaired"] += 1
            
            except RateLimitException:
                # Handle rate limiting by pausing
                logging.warning("Rate limit hit during database verification, pausing...")
                await asyncio.sleep(1)  # Use the rate limit delay from config
                stats["errors"] += 1
                
            except Exception as article_error:
                logging.error(f"Error processing article {article.get('id', 'unknown')}: {article_error}")
                stats["errors"] += 1
        
        logging.info(f"Database verification complete: {stats}")
        return True, stats
    except Exception as e:
        logging.error(f"Error verifying database consistency: {e}")
        return False, {"error": str(e)}

def parse_date(date_str: str) -> str:
    """
    Parse dates in various formats and return in a standard format.
    Handles special formats including the problematic 'Tue, 03/04/2025 - 20:34' format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Standardized date string in ISO format, or the current time in ISO format if parsing fails
    """
    if not date_str:
        return datetime.now().isoformat()
        
    try:
        # Handle problematic format: 'Tue, 03/04/2025 - 20:34' or 'Thu, 01/30/2025 - 19:50'
        drupal_match = re.search(r'(\w+), (\d{2})/(\d{2})/(\d{4}) - (\d{2}):(\d{2})', date_str)
        if drupal_match:
            _, month, day, year, hour, minute = drupal_match.groups()
            try:
                # Create datetime object - properly handle month/day order
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
                return dt.isoformat()
            except ValueError:
                # If month > 12, then it's likely day/month format instead of month/day
                if int(month) > 12 and int(day) <= 12:
                    dt = datetime(int(year), int(day), int(month), int(hour), int(minute))
                    return dt.isoformat()
                # Otherwise re-raise the exception to be caught by the outer try/except
                raise
            
        # Handle common RSS/RFC formats with timezones
        rfc_patterns = [
            # GMT format: "Sun, 02 Mar 2025 23:56:00 GMT"
            r'(\w+), (\d{2}) (\w+) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT',
            # UTC offset format: "Tue, 01 Apr 2025 15:07:40 +0000"
            r'(\w+), (\d{2}) (\w+) (\d{4}) (\d{2}):(\d{2}):(\d{2}) [+-]\d{4}',
            # EST/EDT format: "Mon, 3 Mar 2025 00:00:00 EST"
            r'(\w+), (\d+) (\w+) (\d{4}) (\d{2}):(\d{2}):(\d{2}) (EST|EDT)'
        ]
        
        for pattern in rfc_patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                weekday, day, month, year, hour, minute, second = groups[:7]
                # Convert month name to number
                month_num = list(calendar.month_abbr).index(month[:3].title())
                # Create datetime object
                dt = datetime(int(year), month_num, int(day), int(hour), int(minute), int(second))
                
                # Handle timezone adjustments
                if pattern.endswith('GMT'):
                    # Already in UTC/GMT, just return as is
                    return dt.isoformat()
                elif 'EST' in pattern or 'EDT' in pattern:
                    # Convert EST/EDT to UTC by adding 4/5 hours
                    tz_offset = 5 if groups[-1] == 'EST' else 4
                    dt = dt + timedelta(hours=tz_offset)
                    return dt.isoformat()
                else:
                    # For +0000 format, already in UTC
                    return dt.isoformat()
                
        # Try ISO format if it contains a T
        if 'T' in date_str:
            # Remove Z and replace with +00:00 for consistent parsing
            clean_date = date_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(clean_date)
            # Convert to UTC and remove timezone info
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt.isoformat()
        
        # Try common formats
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%b %d, %Y',
            '%B %d, %Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%a, %d %b %Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%m/%d/%Y %H:%M',  # Additional format for MM/DD/YYYY HH:MM
            '%a, %m/%d/%Y - %H:%M'  # Additional format for Thu, 01/30/2025 - 19:50
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
    
    except Exception as e:
        logging.warning(f"Error parsing date '{date_str}': {e}")
    
    # Return the current time in ISO format if all parsing attempts fail
    logging.warning(f"Failed to parse date string '{date_str}'. Using current time.")
    return datetime.now().isoformat()

def clean_html(raw_html):
    """Clean HTML content for readability."""
    try:
        if not raw_html:
            return ""
            
        # Use BeautifulSoup to clean HTML
        soup = BeautifulSoup(raw_html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
            
        # Remove unwanted elements like ads, nav, etc.
        for unwanted in soup.select('.ad, .advertisement, .sidebar, nav, .nav, .menu, .comments, .social, .share'):
            if unwanted:
                unwanted.decompose()
        
        # Get text but preserve some structure
        if soup.body:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            if paragraphs:
                # Join paragraphs with space
                clean_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            else:
                # Fallback to all text
                clean_text = soup.get_text(separator=' ', strip=True)
        else:
            clean_text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    except Exception as e:
        logging.warning(f"Error cleaning HTML: {e}")
        return raw_html if raw_html else ""

# Add the classify_topic function
async def classify_topic(title: str, description: str, source: str = "") -> str:
    """
    Classify article into a topic based on title and description.
    
    Args:
        title: Article title
        description: Article description or content
        source: Source name (optional)
        
    Returns:
        Topic name, never returns None or undefined
    """
    # Combine title and description for better matching
    content = f"{title} {description}".lower()
    
    # Use crawler_integration's match_topic function
    from crawler_integration import match_topic
    
    # Try to get a match
    topic = await match_topic(content, source)
    
    # Never return None or undefined - use "Uncategorized" as fallback
    if topic is None or topic == 'undefined':
        # Try some additional broader matches for common medical/pharma topics
        if any(term in content for term in ['drug', 'medicine', 'pharma', 'therapeutic', 'treatment']):
            return 'Drug Development'
        elif any(term in content for term in ['trial', 'study', 'patient', 'clinical']):
            return 'Clinical Trials'
        elif any(term in content for term in ['fda', 'ema', 'regulation', 'approval', 'agency']):
            return 'Regulatory'
        elif any(term in content for term in ['market', 'sales', 'revenue', 'launch', 'commercial']):
            return 'Market'
        elif any(term in content for term in ['research', 'discovery', 'science', 'technology']):
            return 'R&D'
        elif any(term in content for term in ['merger', 'acquisition', 'partnership', 'deal']):
            return 'Business'
        else:
            return 'Uncategorized'
            
    return topic

# Synchronous version for backward compatibility
def classify_topic(title: str, description: str, source: str = "") -> str:
    """
    Synchronous wrapper for the async classify_topic function.
    
    Args:
        title: Article title
        description: Article description or content
        source: Source name (optional)
        
    Returns:
        Topic name, never returns None or undefined
    """
    # Create a simple pattern matcher for common topics
    content = f"{title} {description}".lower()
    
    # Special handling for specific sources
    if "fda" in source.lower():
        if any(term in content for term in ["approval", "approved", "label", "labeling", "nda", "anda", "bla"]):
            return "Regulatory"
        if any(term in content for term in ["recall", "safety alert", "warning letter", "adverse event"]):
            return "Safety"
        if any(term in content for term in ["clinical trial", "phase", "study results"]):
            return "Clinical Trials"
        if any(term in content for term in ["guidance", "guideline", "draft guidance"]):
            return "Regulatory"
        # Default for FDA sources
        return "Regulatory"
    
    # Special handling for EMA feeds
    elif "ema" in source.lower() or "europe" in source.lower():
        if any(term in content for term in ["approval", "authorisation", "authorization", "chmp", "committee"]):
            return "Regulatory"
        if any(term in content for term in ["pharmacovigilance", "safety", "risk", "prac"]):
            return "Safety"
        if any(term in content for term in ["orphan", "rare disease", "designation"]):
            return "R&D"
        if any(term in content for term in ["paediatric", "pediatric", "pdco"]):
            return "Clinical Trials"
        # Default for EMA sources
        return "Regulatory"
    
    # Import TOPICS from constants
    from services.constants import TOPICS
    
    # Check for topics based on keywords
    for topic, keywords in TOPICS.items():
        for keyword in keywords:
            if f" {keyword.lower()} " in f" {content} ":
                return topic
    
    # Broader fallback matching for common categories
    if any(term in content for term in ['drug', 'medicine', 'pharma', 'therapeutic', 'treatment']):
        return 'Drug Development'
    elif any(term in content for term in ['trial', 'study', 'patient', 'clinical']):
        return 'Clinical Trials'
    elif any(term in content for term in ['fda', 'ema', 'regulation', 'approval', 'agency']):
        return 'Regulatory'
    elif any(term in content for term in ['market', 'sales', 'revenue', 'launch', 'commercial']):
        return 'Market Access'
    elif any(term in content for term in ['research', 'discovery', 'science', 'technology']):
        return 'Research'
    elif any(term in content for term in ['merger', 'acquisition', 'partnership', 'deal']):
        return 'Business'
    
    # Last resort
    return 'Uncategorized'

async def fetch_full_article_content(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetch the full content of an article using WebCrawler.
    
    Args:
        session: aiohttp session
        url: The URL to fetch content from
        
    Returns:
        String containing the article content, or empty string if fetch fails
    """
    try:
        # Use the WebCrawler for better content extraction if possible
        from web_crawler import WebCrawler
        
        async with WebCrawler(
            cache_file=CRAWLER_CACHE_FILE,
            calls_per_second=1.0,
            max_urls_per_domain=100
        ) as crawler:
            # Override crawler's session with the provided session for consistency
            old_session = crawler.session
            try:
                crawler.session = session
                result = await crawler.fetch_url(url)
                
                if result.get('success'):
                    if result.get('has_full_content') and result.get('content'):
                        return result['content']
                    elif result.get('content'):
                        # Fallback to partial content
                        return result['content']
            finally:
                crawler.session = old_session
        
        # Fallback to simple fetch if WebCrawler failed
        async with session.get(url, ssl=ssl.create_default_context(cafile=certifi.where())) as response:
            if response.status != 200:
                return ""
                
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try to find the main content
            main_content = None
            for selector in [
                'article', '.article-content', '.article-body', '.post-content', 
                '.entry-content', '.content-body', 'main', '[itemprop="articleBody"]'
            ]:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                # Remove non-content elements
                for unwanted in main_content.select('aside, nav, .nav, .menu, .comments, .social, .share, script, style'):
                    if unwanted:
                        unwanted.decompose()
                
                # Extract text from paragraphs
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    return ' '.join(p.get_text(strip=True) for p in paragraphs)
                else:
                    return main_content.get_text(separator=' ', strip=True)
            else:
                # Just get all paragraphs if we can't find the main content
                paragraphs = soup.find_all('p')
                return ' '.join(p.get_text(strip=True) for p in paragraphs)
    
    except Exception as e:
        logging.error(f"Error fetching full article content from {url}: {e}")
        return ""

def generate_article_id(article: Dict) -> str:
    """
    Generate a unique ID for an article based on its URL and title.
    
    Args:
        article: Article dictionary containing at least 'link' and 'title'
        
    Returns:
        String containing a unique hash ID for the article
    """
    # Combine URL and title to create a unique identifier
    unique_string = f"{article['link']}_{article['title']}"
    
    # Generate hash
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

@cache_service.cache_decorator(namespace="summary", ttl=86400*30)  # Cache for 30 days
@rate_limiter.rate_limit_decorator(key="summary_generation")
async def summarize_text(text: str, max_length: int = 280) -> Optional[str]:
    """
    Generate a concise summary of the provided text.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary in characters
        
    Returns:
        A summary string or None if summary generation fails
    """
    try:
        if not text or len(text) < 200:
            return None
            
        # If Google's Generative AI key is available, use it for summarization
        try:
            from context.api_config import GOOGLE_API_KEY
            if GOOGLE_API_KEY:
                genai.configure(api_key=GOOGLE_API_KEY)
                
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"""
                Summarize the following article in 1-2 concise, informative sentences:
                
                {text[:10000]}
                
                SUMMARY:
                """
                
                response = model.generate_content(prompt)
                
                if response and response.text:
                    summary = response.text.strip()
                    if len(summary) > max_length:
                        # Truncate to max_length and add ellipsis
                        summary = summary[:max_length-3] + "..."
                    return summary
        except (ImportError, Exception) as e:
            logging.warning(f"Google GenerativeAI summarization failed: {e}")
            
        # Simple extraction-based fallback summary if generative AI fails
        # Take the first 2 sentences or the first paragraph, whichever is shorter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 2:
            simple_summary = ' '.join(sentences[:2])
            if len(simple_summary) > max_length:
                simple_summary = simple_summary[:max_length-3] + "..."
            return simple_summary
        else:
            # Just take the first part of the text if we can't find sentences
            simple_summary = text[:max_length-3] + "..."
            return simple_summary
        
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return None

# ... rest of the file remains unchanged ...