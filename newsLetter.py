import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import json
from pathlib import Path
import google.generativeai as genai
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
from services.config_service import ConfigService
from contextlib import asynccontextmanager
import aiofiles
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import ssl
import certifi
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add cache configuration
CACHE_FILE = "summary_cache.json"
RATE_LIMIT_DELAY = 1  # seconds between API calls

# Add these constants near the top after other constants
DEFAULT_SCRAPE_INTERVAL = 3600  # Default to running every hour (in seconds)
MINIMUM_SCRAPE_INTERVAL = 300   # Minimum 5 minutes between scrapes

class RateLimitException(Exception):
    pass

class FeedError(Exception):
    """Custom exception for feed processing errors"""
    pass

class CacheError(Exception):
    """Custom exception for cache operations"""
    pass

class SummaryCache:
    def __init__(self, cache_file: str, max_age_days: int = 30):
        self.cache_file = cache_file
        self.max_age = timedelta(days=max_age_days)
        self.cache = self._load_cache()
        self._lock = asyncio.Lock()

    def _load_cache(self) -> Dict:
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                # Filter out old entries
                current_time = datetime.now()
                return {
                    k: v for k, v in data.items() 
                    if isinstance(v, dict) and 'timestamp' in v and 
                    current_time - datetime.fromisoformat(v['timestamp']) <= self.max_age
                }
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
        return {}

    def get(self, key: str) -> Optional[str]:
        """Get summary from cache if it exists"""
        if key in self.cache:
            return self.cache[key]['summary']
        return None

    async def set(self, key: str, summary: str):
        """Add summary to cache with timestamp"""
        async with self._lock:
            self.cache[key] = {
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            await self._save_cache()

    async def _save_cache(self):
        """Save cache to file with async lock"""
        try:
            async with aiofiles.open(self.cache_file, 'w') as f:
                await f.write(json.dumps(self.cache))
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
            raise CacheError(f"Failed to save cache: {str(e)}")

# Define the topics and keywords
TOPICS = {
    "483 Notifications": ["FDA Form 483", "483 observations", "FDA inspection", "warning letter"],
    "Adulterated Drugs": ["adulterated drugs", "drug recall", "adulteration", "counterfeit drugs", "drug safety"],
    "AI in Medicine": ["AI in medicine", "artificial intelligence in healthcare", "machine learning in healthcare", "deep learning in medicine", "AI drug discovery"],
    "Contamination Control in Pharma": ["contamination control", "pharma contamination", "cleanroom practices", "sterile manufacturing", "microbial control"],
    "Regulatory Affairs": ["FDA approval", "EMA decision", "regulatory update", "regulatory compliance", "FDA guidance"],
    "Clinical Trials": ["clinical trial results", "phase III study", "trial enrollment", "clinical research", "drug trial"],
    "Mergers and Acquisitions": ["merger", "acquisition", "partnership", "buyout", "deal", "M&A"],
    "Drug Development": ["drug discovery", "new therapy", "pharmaceutical innovation", "drug pipeline", "preclinical studies"],
    "Market Trends": ["market analysis", "financial report", "investment news", "pharma market", "biotech industry"],
    "Biotechnology Innovations": ["biotech breakthrough", "biotechnology advancement", "biotech research", "synthetic biology", "genetic engineering"],
    "Personalized Medicine": ["personalized medicine", "precision therapy", "genomic medicine", "targeted therapy", "pharmacogenomics"],
    "Gene and Cell Therapies": ["gene editing", "CAR-T therapy", "cell therapy", "gene therapy", "CRISPR"],
    "Vaccine Development": ["vaccine research", "vaccine approval", "vaccine distribution", "immunization", "vaccine efficacy"],
    "Intellectual Property and Legal Affairs": ["patent dispute", "legal case", "policy change", "intellectual property", "patent law"],
    "Drug Pricing and Access": ["drug pricing", "drug costs", "affordable medicine", "price gouging", "healthcare access"],
    "Supply Chain Issues": ["drug shortage", "supply chain disruption", "pharmaceutical supply chain", "manufacturing delays"],
    "Digital Health": ["digital health", "telemedicine", "wearable technology", "mobile health", "eHealth"],
    "Healthcare Policy": ["healthcare reform", "health policy", "public health", "healthcare legislation"],
    "Emerging Infectious Diseases": ["infectious disease outbreak", "pandemic preparedness", "antimicrobial resistance", "viral infections"],
    "Rare Diseases": ["orphan drugs", "rare disease research", "rare disease treatment"],
    "Sustainability in Pharma": ["green chemistry", "sustainable manufacturing", "environmental impact", "pharma waste"],
    "Patient Advocacy": ["patient rights", "patient support groups", "patient advocacy"],
    "Pharmacovigilance": ["adverse events", "drug safety monitoring", "post-market surveillance"],
    "Manufacturing and Quality Control": ["GMP", "Good Manufacturing Practices", "quality control", "pharmaceutical manufacturing"],
    "Drug Delivery Systems": ["drug delivery", "nanotechnology in drug delivery", "controlled release"],
}

# Define news sources and RSS feeds
RSS_FEEDS = {
    "CNN": "http://rss.cnn.com/rss/cnn_latest.rss",
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "Sky News": "https://feeds.skynews.com/feeds/rss/home.xml",
    "European Pharmaceutical Review": "https://www.europeanpharmaceuticalreview.com/feed/",
    "PDA": "https://journal.pda.org/rss/current.xml",
    "FDA": "http://www.fda.gov/AboutFDA/ContactFDA/StayInformed/RSSFeeds/MedWatch/rss.xml",
    #"BioPharmaDive": "https://www.biopharmadive.com/feeds/news/",
    "FDA Drug Updates": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
    "FDA Recalls": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml",
    "Nature Biotechnology": "https://www.nature.com/nbt.rss",
    "BioDrugs": "https://link.springer.com/journal/40259/rss.xml",
    "Contract Pharma": "https://www.contractpharma.com/rss/news",
    "ClinicalTrials.gov": "https://clinicaltrials.gov/ct2/results/rss.xml?rcv_d=14",
    "PharmaTimes": "https://www.pharmatimes.com/rss",
    "EPR": "https://www.europeanpharmaceuticalreview.com/feed/",
    #"Reuters Health": "http://feeds.reuters.com/reuters/USHealthNews",
    "Pharmaceutical Technology": "https://www.pharmaceutical-technology.com/feed/",
    "In-Pharma Technologist": "https://www.in-pharmatechnologist.com/rss",
    "Outsourcing-Pharma.com": "https://www.outsourcing-pharma.com/rss",
    "PM Live": "https://www.pmlive.com/rss/all",
    #"The Pharma Journal": "https://www.pharmj.com/rss/latest/rss.xml",
    "Drug Topics": "https://www.drugtopics.com/rss",
    "STAT News": "https://www.statnews.com/feed/",
    #"The Lancet": "https://www.thelancet.com/rss/general",
    #"New England Journal of Medicine": "https://www.nejm.org/rss/most-recent",
    #"JAMA": "https://jamanetwork.com/rss/most-recent",
    "Nature": "https://www.nature.com/nature.rss",
    #"Science": "https://www.science.org/rss/news",
}

# Function to fetch and parse RSS feeds
def fetch_rss_feed(feed_url):
    try:
        response = requests.get(feed_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        articles = soup.find_all('item')
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching RSS feed from {feed_url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error parsing RSS feed from {feed_url}: {e}")
        return []

# Function to process and filter articles
def process_articles(articles, source):
    filtered_articles = []
    for article in articles:
        try:
            # Extract article content
            title = article.title.text if article.title else ""
            description = article.description.text if article.description else ""
            link = article.link.text if article.link else ""
            pub_date = article.pubDate.text if article.pubDate else ""

            # Format the publication date
            try:
                pub_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pub_date = "Unknown date"

            # Check for matches with keywords
            matched_topic = None
            for topic, keywords in TOPICS.items():
                for keyword in keywords:
                    if re.search(rf"\b{re.escape(keyword)}\b", (title + " " + description), re.IGNORECASE):
                        matched_topic = topic
                        break
                if matched_topic:
                    break

            if matched_topic:
                # Add delay between API calls to prevent rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
                # Summarize the description
                try:
                    summary = summarize_text(description)
                except RateLimitException:
                    logging.warning(f"Rate limit reached while processing article: {title}")
                    summary = "Summary generation delayed due to rate limiting"
                
                filtered_articles.append({
                    "source": source,
                    "title": title,
                    "description": description,
                    "link": link,
                    "pub_date": pub_date,
                    "topic": matched_topic,
                    "summary": summary
                })

        except Exception as e:
            logging.error(f"Error processing article from {source}: {str(e)}")
            continue

    return filtered_articles

# Update the summarize_text function
async def summarize_text(text: str) -> Optional[str]:
    """Generate a summary of the text using AI"""
    try:
        config = ConfigService()
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Check cache first
        cache = SummaryCache(CACHE_FILE)
        cached_summary = cache.get(text[:200])  # Use first 200 chars as key
        if cached_summary:
            return cached_summary

        prompt = f"""
        Please provide a concise summary of this pharmaceutical industry news:
        {text}
        
        Focus on:
        1. Key findings or announcements
        2. Industry impact
        3. Regulatory implications (if any)
        
        Keep the summary under 200 words.
        """

        # Generate summary with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                summary = response.text.strip()
                
                if summary:
                    # Clean summary of unwanted characters or HTML tags
                    summary = BeautifulSoup(summary, "html.parser").get_text()
                    # Cache the successful summary
                    await cache.set(text[:200], summary)
                    return summary
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                continue

        return None

    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return None

def fetch_and_process_feed(source: str, feed_url: str) -> List[Dict]:
    """Fetch and process a single RSS feed."""
    try:
        logging.info(f"Fetching from {source}: {feed_url}")
        response = requests.get(feed_url, timeout=10)
        soup = BeautifulSoup(response.content, features='xml')
        articles = soup.find_all('item')
        return [process_article(article, source) for article in articles]
    except Exception as e:
        logging.error(f"Error processing feed {feed_url}: {str(e)}")
        return []

def scrape_news():
    """Main function to scrape and process news."""
    news_data = []
    logging.info("Starting news scraping process...")
    
    try:
        # Fetch and process each feed
        for source, feed_url in RSS_FEEDS.items():
            articles = fetch_and_process_feed(source, feed_url)
            news_data.extend(articles)

        # Save articles to CSV
        save_articles_to_csv(news_data)
        
        return True
    except Exception as e:
        logging.error(f"Error in scrape_news: {str(e)}")
        return False

def save_articles_to_csv(articles: List[Dict]):
    """Save articles to a CSV file."""
    if articles:
        new_df = pd.DataFrame(articles)
        
        # If file exists, merge with existing data
        if os.path.exists('news_alerts.csv'):
            existing_df = pd.read_csv('news_alerts.csv')
            
            # Ensure all columns are strings
            for col in new_df.columns:
                new_df[col] = new_df[col].fillna('').astype(str)
                existing_df[col] = existing_df[col].fillna('').astype(str)
            
            # Combine new and existing data
            combined_df = pd.concat([new_df, existing_df])
            # Remove duplicates, keeping the first occurrence (new articles)
            combined_df.drop_duplicates(subset=['title', 'link'], keep='first', inplace=True)
            
            # Archive old data
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            existing_df.to_csv(f'news_alerts_{timestamp}.csv', index=False)
            
            # Save combined data
            combined_df.to_csv('news_alerts.csv', index=False)
            logging.info(f"Saved {len(new_df)} new articles, total: {len(combined_df)}")
        else:
            # Save new data
            new_df.to_csv('news_alerts.csv', index=False)
            logging.info(f"Saved {len(new_df)} new articles")
        
        return True
    else:
        logging.info("No new articles found")
        return True

async def fetch_rss_feed_async(session, source, feed_url):
    """Asynchronously fetch RSS feed"""
    try:
        async with session.get(feed_url, timeout=10) as response:
            content = await response.text()
            soup = BeautifulSoup(content, 'xml')
            articles = soup.find_all('item')
            return source, articles
    except Exception as e:
        logging.error(f"Error fetching RSS feed from {source} ({feed_url}): {e}")
        return source, []

async def fetch_all_feeds():
    """Fetch all RSS feeds concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for source, url in RSS_FEEDS.items():
            task = fetch_rss_feed_async(session, source, url)
            tasks.append(task)
        return await asyncio.gather(*tasks)

def retry_with_backoff(func):
    """Decorator for retry logic with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
    return wrapper

@retry_with_backoff
def process_article(article, source):
    """Process a single article with retry logic"""
    # Extract article content
    title = article.title.text if article.title else ""
    description = article.description.text if article.description else ""
    link = article.link.text if article.link else ""
    pub_date = article.pubDate.text if article.pubDate else ""

    # Format the publication date
    try:
        pub_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        pub_date = "Unknown date"

    # Check for matches with keywords
    matched_topic = None
    for topic, keywords in TOPICS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", (title + " " + description), re.IGNORECASE):
                matched_topic = topic
                break
        if matched_topic:
            break

    if matched_topic:
        # Generate summary immediately for new articles
        try:
            logging.info(f"Generating summary for: {title}")
            time.sleep(RATE_LIMIT_DELAY)  # Prevent rate limiting
            summary = summarize_text(description)
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            summary = None

        return {
            "source": source,
            "title": title,
            "description": description,
            "link": link,
            "pub_date": pub_date,
            "topic": matched_topic,
            "summary": summary
        }
    return None

@dataclass
class Article:
    source: str
    title: str
    description: str
    link: str
    pub_date: str
    topic: Optional[str] = None
    summary: Optional[str] = None
    
    @classmethod
    def from_rss_item(cls, item, source: str) -> Optional['Article']:
        """Create an Article from an RSS item with validation"""
        try:
            title = getattr(item.title, 'text', '').strip()
            description = getattr(item.description, 'text', '').strip()
            link = getattr(item.link, 'text', '').strip()
            pub_date = getattr(item.pubDate, 'text', '')

            if not all([title, description, link]):
                return None

            # Clean description
            clean_description = BeautifulSoup(description, "html.parser").get_text()

            # Parse date with multiple format support
            formatted_date = cls._parse_date(pub_date)

            return cls(
                source=source,
                title=title,
                description=clean_description,
                link=link,
                pub_date=formatted_date
            )
        except Exception as e:
            logging.error(f"Error creating article from RSS item: {e}")
            return None

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse date string with multiple format support"""
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        return "Unknown date"

    def match_topic(self, topics: Dict[str, List[str]]) -> Optional[str]:
        """Match article against topics and keywords"""
        content = f"{self.title} {self.description}".lower()
        for topic, keywords in topics.items():
            if any(re.search(rf"\b{re.escape(keyword.lower())}\b", content) for keyword in keywords):
                return topic
        return None

@dataclass
class ScrapingMetrics:
    """Track scraping performance metrics"""
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
        """Get scraping statistics"""
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

def generate_missing_summaries():
    """Generate summaries for articles that don't have them"""
    try:
        if not os.path.exists('news_alerts.csv'):
            logging.info("No articles file found")
            return False

        df = pd.read_csv('news_alerts.csv')
        total_articles = len(df)
        updated_count = 0

        # Convert NaN to empty string
        df['summary'] = df['summary'].fillna('')
        
        # Process articles without summaries
        for index, row in df.iterrows():
            if not row['summary'] or row['summary'] in ['nan', 'None', '']:
                try:
                    logging.info(f"Generating summary for article {index + 1}/{total_articles}")
                    time.sleep(RATE_LIMIT_DELAY)  # Prevent rate limiting
                    
                    summary = summarize_text(row['description'])
                    if summary and summary not in ['Error generating summary', 'Failed to generate summary after multiple attempts']:
                        df.at[index, 'summary'] = summary
                        updated_count += 1
                        
                        # Save progress periodically
                        if updated_count % 5 == 0:
                            df.to_csv('news_alerts.csv', index=False)
                            logging.info(f"Progress saved: {updated_count} summaries generated")
                            
                except Exception as e:
                    logging.error(f"Error generating summary for article {index + 1}: {e}")
                    continue

        # Final save
        df.to_csv('news_alerts.csv', index=False)
        logging.info(f"Completed: Generated {updated_count} new summaries")
        return True

    except Exception as e:
        logging.error(f"Error in generate_missing_summaries: {e}")
        return False

# Add this function to handle async feed fetching
async def fetch_feed(session: aiohttp.ClientSession, source: str, url: str) -> List[Dict]:
    """Asynchronously fetch and process a single feed"""
    try:
        # Configure SSL context for aiohttp
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Special handling for FDA domains
        if 'fda.gov' in url:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        async with session.get(url, timeout=30, ssl=ssl_context) as response:
            if response.status == 404:
                logging.error(f"Feed not found for {source} ({url})")
                return []
            
            if response.status != 200:
                logging.error(f"Error status {response.status} from {source} ({url})")
                return []
                
            content = await response.text()
            soup = BeautifulSoup(content, 'xml')
            
            if not soup.find('item'):  # Check if feed is valid
                logging.error(f"No items found in feed from {source} ({url})")
                return []
                
            articles = soup.find_all('item')
            
            processed_articles = []
            for article in articles:
                try:
                    # Extract article data with error checking
                    title = getattr(article.title, 'text', '') if article.title else ""
                    description = getattr(article.description, 'text', '') if article.description else ""
                    link = getattr(article.link, 'text', '') if article.link else ""
                    pub_date = getattr(article.pubDate, 'text', '') if article.pubDate else ""

                    if not (title and (description or link)):  # Skip articles without minimum required data
                        continue

                    # Clean description of HTML tags
                    clean_description = BeautifulSoup(description, "html.parser").get_text()

                    # Format the publication date with better error handling
                    try:
                        pub_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            # Try alternative date format
                            pub_date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pub_date = "Unknown date"

                    # Check for matches with keywords
                    matched_topic = None
                    for topic, keywords in TOPICS.items():
                        if any(re.search(rf"\b{re.escape(keyword)}\b", 
                               (title + " " + clean_description), 
                               re.IGNORECASE) for keyword in keywords):
                            matched_topic = topic
                            break

                    if matched_topic:
                        processed_articles.append({
                            "source": source,
                            "title": title.strip(),
                            "description": clean_description.strip(),
                            "link": link.strip(),
                            "pub_date": pub_date,
                            "topic": matched_topic
                        })

                except Exception as e:
                    logging.error(f"Error processing article from {source}: {str(e)}")
                    continue

            return processed_articles

    except aiohttp.ClientError as e:
        logging.error(f"Network error fetching feed from {source} ({url}): {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error fetching feed from {source} ({url}): {str(e)}")
        return []

async def scrape_news() -> bool:
    """Main function to scrape and process news"""
    metrics = ScrapingMetrics(start_time=time.time())
    
    try:
        async with FeedManager(RSS_FEEDS) as feed_manager:
            tasks = []
            for source, url in RSS_FEEDS.items():
                tasks.append(feed_manager.fetch_feed(source, url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_articles = []
            for articles in results:
                if isinstance(articles, list):
                    all_articles.extend(articles)
                    
            if all_articles:
                # Process summaries with rate limiting
                rate_limiter = RateLimiter(calls_per_second=0.5)  # 2 seconds between API calls
                for article in all_articles:
                    await rate_limiter.acquire()
                    try:
                        article.summary = await summarize_text(article.description)
                    except Exception as e:
                        logging.error(f"Error generating summary: {e}")
                        continue

                # Save to CSV
                success = await save_articles_to_csv(all_articles)
                if not success:
                    logging.error("Failed to save articles")
                    return False

                logging.info(f"Successfully processed {len(all_articles)} articles")
                return True
            
            logging.info("No new articles found")
            return True

    except Exception as e:
        logging.error(f"Error in scrape_news: {e}")
        return False

def create_session_with_retries():
    """Create a requests session with retry logic and SSL configuration"""
    session = requests.Session()
    
    # Configure retries
    retries = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504],  # retry on these status codes
        allowed_methods=["GET"]  # only retry on GET requests
    )
    
    # Add retry adapter
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Configure SSL verification
    session.verify = certifi.where()
    
    return session

class NewsletterScheduler:
    def __init__(self, interval_seconds=DEFAULT_SCRAPE_INTERVAL):
        self.interval = max(interval_seconds, MINIMUM_SCRAPE_INTERVAL)
        self.is_running = False
        self._task = None
        
    async def start(self):
        """Start the periodic scraping"""
        self.is_running = True
        logging.info(f"Starting periodic news scraping every {self.interval} seconds")
        
        while self.is_running:
            try:
                success = await scrape_news()
                if success:
                    logging.info("Periodic scrape completed successfully")
                else:
                    logging.error("Periodic scrape completed with errors")
            except Exception as e:
                logging.error(f"Error during periodic scrape: {e}")
            
            # Wait for next interval
            await asyncio.sleep(self.interval)
    
    def stop(self):
        """Stop the periodic scraping"""
        self.is_running = False
        if self._task:
            self._task.cancel()
        logging.info("Stopping periodic news scraping")
    
    async def run(self):
        """Run the scheduler"""
        self._task = asyncio.create_task(self.start())
        try:
            await self._task
        except asyncio.CancelledError:
            logging.info("Newsletter scheduler was cancelled")

def handle_shutdown(scheduler: NewsletterScheduler, sig=None):
    """Handle graceful shutdown"""
    if sig:
        logging.info(f'Received shutdown signal: {sig.name}')
    
    scheduler.stop()

async def run_newsletter_service(interval_seconds=DEFAULT_SCRAPE_INTERVAL):
    """Run the newsletter service with periodic scraping"""
    scheduler = NewsletterScheduler(interval_seconds)
    
    # Set up signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _: handle_shutdown(scheduler, s))
    
    try:
        # Do initial scrape immediately
        logging.info("Performing initial scrape...")
        await scrape_news()
        
        # Start periodic scraping
        await scheduler.run()
    except Exception as e:
        logging.error(f"Error in newsletter service: {e}")
        scheduler.stop()
    finally:
        logging.info("Newsletter service shutdown complete")

# Update the main block to use the scheduler
if __name__ == "__main__":
    try:
        # Get interval from environment variable or use default
        interval = int(os.getenv('SCRAPE_INTERVAL_SECONDS', DEFAULT_SCRAPE_INTERVAL))
        
        # Run the service
        asyncio.run(run_newsletter_service(interval))
    except KeyboardInterrupt:
        logging.info("Newsletter service stopped by user")
    except Exception as e:
        logging.error(f"Newsletter service error: {e}")
        sys.exit(1)

class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call_time: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str = "default") -> None:
        """Wait if necessary to maintain rate limits"""
        async with self._lock:
            if key in self.last_call_time:
                elapsed = time.time() - self.last_call_time[key]
                if elapsed < self.minimum_interval:
                    await asyncio.sleep(self.minimum_interval - elapsed)
            
            self.last_call_time[key] = time.time()

# Add this new class for feed management
class FeedManager:
    def __init__(self, feeds: Dict[str, str], timeout: int = 30):
        self.feeds = feeds
        self.timeout = timeout
        self.rate_limiter = RateLimiter(calls_per_second=2.0)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up the aiohttp session"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the session"""
        if self.session:
            await self.session.close()

    async def fetch_feed(self, source: str, url: str) -> List[Article]:
        """Fetch and process a single feed"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        await self.rate_limiter.acquire(url)
        
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            if 'fda.gov' in url:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            async with self.session.get(url, ssl=ssl_context) as response:
                if response.status != 200:
                    logging.error(f"Error fetching {source}: {response.status}")
                    return []

                content = await response.text()
                soup = BeautifulSoup(content, 'xml')
                items = soup.find_all('item')
                
                articles = []
                for item in items:
                    if article := Article.from_rss_item(item, source):
                        if topic := article.match_topic(TOPICS):
                            article.topic = topic
                            articles.append(article)

                return articles

        except Exception as e:
            logging.error(f"Error processing feed {source}: {e}")
            return []