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
import hashlib
from urllib.parse import urlparse

# Import the WebCrawler implementation
from web_crawler import WebCrawler, RobustCache

# Import the NewsAPI configuration
try:
    from context.api_config import (
        NEWSAPI_KEY, NEWSAPI_ENABLED, NEWSAPI_BASE_URL,
        NEWSAPI_QUERIES, NEWSAPI_DAYS_BACK, NEWSAPI_MAX_RESULTS
    )
    HAS_NEWSAPI_CONFIG = True
except ImportError:
    logging.warning("NewsAPI configuration not found. NewsAPI features will be disabled.")
    HAS_NEWSAPI_CONFIG = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add cache configuration
CACHE_FILE = "summary_cache.json"
CRAWLER_CACHE_FILE = "crawler_cache.json"
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

# SummaryCache is now replaced by RobustCache from web_crawler.py
# For backward compatibility, we'll create an alias
SummaryCache = RobustCache

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
    "FDA": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
    "FDA Drug Updates": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
    "FDA Recalls": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml",
    "Nature Biotechnology": "https://www.nature.com/nbt.rss",
    "Contract Pharma": "https://www.contractpharma.com/contents/rss",
    "ClinicalTrials.gov": "https://clinicaltrials.gov/ct2/results/rss.xml?rcv_d=14",
    "EPR": "https://www.europeanpharmaceuticalreview.com/feed/",
    "Pharmaceutical Technology": "https://www.pharmaceutical-technology.com/feed/",
    "Drug Topics": "https://www.drugtopics.com/rss",
    "STAT News": "https://www.statnews.com/feed/",
    "Nature": "https://www.nature.com/nature.rss",
    "The Lancet": "https://www.thelancet.com/rssfeed/lancet_current.xml",
    "Science": "https://www.science.org/rss/news_current.xml",
    "Annals of Internal Medicine": "https://www.acpjournals.org/action/showFeed?type=etoc&feed=rss&jc=aim",
    "Cell": "https://www.cell.com/cell/current.rss",
    "Nature Medicine": "https://www.nature.com/nm.rss",
    "PNAS": "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=pnas",
    "Journal of Clinical Investigation": "https://www.jci.org/rss/current.xml",
    "Molecular Cell": "https://www.cell.com/molecular-cell/current.rss",
    "Immunity": "https://www.cell.com/immunity/current.rss",
    "Genome Biology": "https://genomebiology.biomedcentral.com/articles/most-recent/rss.xml",
}

# Function to fetch and parse RSS feeds
def fetch_rss_feed(feed_url):
    try:
        # Create a session with retry logic
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Set a longer timeout and add user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = session.get(feed_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Try different parsers if the default one fails
        try:
            soup = BeautifulSoup(response.content, 'xml')
            articles = soup.find_all('item')
            if not articles:
                # Try alternate parsing for Atom feeds
                articles = soup.find_all('entry')
        except Exception:
            # Fallback to html parser
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('item') or soup.find_all('entry')
            
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
        # Use a shorter key for caching to avoid issues with very long texts
        cache_key = text[:200] + str(len(text))
        cached_summary = cache.get(cache_key)
        if cached_summary:
            return cached_summary

        # For very long texts, split into chunks and summarize each chunk
        if len(text) > 10000:
            chunks = split_text_into_chunks(text, 8000)
            chunk_summaries = []
            
            for chunk in chunks:
                chunk_prompt = f"""
                Please provide a concise summary of this pharmaceutical industry news excerpt:
                {chunk}
                
                Focus on:
                1. Key findings or announcements
                2. Industry impact
                3. Regulatory implications (if any)
                
                Keep the summary under 200 words.
                """
                
                response = model.generate_content(chunk_prompt)
                if response.text:
                    chunk_summaries.append(response.text.strip())
            
            # If we have multiple chunk summaries, combine them
            if len(chunk_summaries) > 1:
                combined_text = " ".join(chunk_summaries)
                final_prompt = f"""
                Please provide a unified, concise summary of these pharmaceutical industry news excerpts:
                {combined_text}
                
                Focus on:
                1. Key findings or announcements
                2. Industry impact
                3. Regulatory implications (if any)
                
                Keep the summary under 200 words.
                """
                
                final_response = model.generate_content(final_prompt)
                summary = final_response.text.strip()
            else:
                summary = chunk_summaries[0] if chunk_summaries else None
        else:
            # For shorter texts, summarize directly
            prompt = f"""
            Please provide a concise summary of this pharmaceutical industry news:
            {text}
            
            Focus on:
            1. Key findings or announcements
            2. Industry impact
            3. Regulatory implications (if any)
            
            Keep the summary under 200 words.
            """
            
            response = model.generate_content(prompt)
            summary = response.text.strip()
        
        if summary:
            # Clean summary of unwanted characters or HTML tags
            summary = BeautifulSoup(summary, "html.parser").get_text()
            # Cache the successful summary
            await cache.set(cache_key, summary)
            return summary
            
        return None

    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return None

# Helper function to split text into chunks
def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for the space
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

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

# Add new function to fetch from NewsAPI
async def fetch_from_newsapi(session: aiohttp.ClientSession) -> List[Dict]:
    """
    Fetch articles from NewsAPI focused on biopharmaceutical news.
    
    Args:
        session: aiohttp ClientSession for making requests
        
    Returns:
        List of article dictionaries
    """
    if not HAS_NEWSAPI_CONFIG or not NEWSAPI_ENABLED:
        logging.info("NewsAPI disabled or not configured. Skipping...")
        return []
    
    all_articles = []
    from_date = (datetime.now() - timedelta(days=NEWSAPI_DAYS_BACK)).strftime('%Y-%m-%d')
    
    try:
        for query in NEWSAPI_QUERIES:
            logging.info(f"Fetching articles from NewsAPI for query: {query}")
            
            # Construct the API URL
            url = f"{NEWSAPI_BASE_URL}/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": NEWSAPI_MAX_RESULTS,
                "apiKey": NEWSAPI_KEY
            }
            
            # Make the API request
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"Error from NewsAPI: {error_text}")
                    continue
                
                data = await response.json()
                
                if data.get('status') != 'ok':
                    logging.error(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                    continue
                
                articles = data.get('articles', [])
                logging.info(f"Retrieved {len(articles)} articles from NewsAPI for query '{query}'")
                
                # Process each article
                for article in articles:
                    # Check for duplicates by URL
                    article_url = article.get('url')
                    if any(a.get('link') == article_url for a in all_articles):
                        continue
                    
                    # Extract relevant fields
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    content = article.get('content', '').strip()
                    source_name = article.get('source', {}).get('name', 'NewsAPI')
                    published_at = article.get('publishedAt', '')
                    url = article.get('url', '')
                    
                    if not all([title, url]):
                        continue
                    
                    # Format publication date
                    try:
                        pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pub_date = "Unknown date"
                    
                    # Match topic
                    matched_topic = None
                    for topic, keywords in TOPICS.items():
                        if any(re.search(rf"\b{re.escape(keyword.lower())}\b", 
                              (title + " " + description).lower()) for keyword in keywords):
                            matched_topic = topic
                            break
                    
                    # Only keep articles with a matched topic
                    if matched_topic:
                        # Use full content if available, otherwise use description
                        full_content = content if content else description
                        
                        # Generate summary for relevant articles
                        summary = await summarize_text(full_content)
                        
                        all_articles.append({
                            "source": f"NewsAPI: {source_name}",
                            "title": title,
                            "description": full_content,
                            "link": url,
                            "pub_date": pub_date,
                            "topic": matched_topic,
                            "summary": summary,
                            "has_full_content": bool(content)
                        })
            
            # Add a short delay between queries to avoid rate limits
            await asyncio.sleep(1)
    
    except Exception as e:
        logging.error(f"Error fetching from NewsAPI: {e}")
    
    logging.info(f"Total articles from NewsAPI after filtering: {len(all_articles)}")
    return all_articles

async def scrape_news(status_callback=None) -> bool:
    """Main function to scrape and process news using the enhanced WebCrawler
    
    Args:
        status_callback: Optional callback function to report progress
                        Function signature: callback(progress, message, sources_processed, total_sources, articles_found)
    
    Returns:
        bool: True if successful, False otherwise
    """
    metrics = ScrapingMetrics(start_time=time.time())
    logging.info("Starting news scraping process with enhanced WebCrawler...")
    
    try:
        # Report initial status
        if status_callback:
            status_callback(5, "Initializing scraping process...", 0, len(RSS_FEEDS), 0)
        
        # Initialize WebCrawler with appropriate settings
        async with WebCrawler(
            cache_file=CRAWLER_CACHE_FILE,
            max_age_days=30,
            calls_per_second=2.0,
            timeout_seconds=30,
            max_redirects=5,
            max_retries=3,
            user_agent="NewsAggregator/1.0",
            max_urls_per_domain=100,
            respect_robots_txt=True
        ) as crawler:
            # Process all RSS feeds
            all_articles = []
            for idx, (source, url) in enumerate(RSS_FEEDS.items()):
                # Update progress periodically
                if status_callback:
                    progress = 5 + (idx / len(RSS_FEEDS) * 30)  # Progress from 5% to 35%
                    status_callback(
                        progress, 
                        f"Processing feed {idx+1} of {len(RSS_FEEDS)}...",
                        idx,
                        len(RSS_FEEDS),
                        len(all_articles)
                    )
                
                try:
                    logging.info(f"Fetching feed: {source} from {url}")
                    # Use the enhanced WebCrawler to fetch RSS feed
                    articles = await crawler.fetch_rss_feed(url)
                    
                    if articles:
                        metrics.successful_feeds += 1
                        
                        # Process each article
                        for article in articles:
                            # Add source information
                            article['source'] = source
                            
                            # Match topics based on title and description
                            content_for_matching = f"{article.get('title', '')} {article.get('description', '')}"
                            
                            # Check for matches with keywords
                            matched_topic = None
                            for topic, keywords in TOPICS.items():
                                if any(re.search(rf"\b{re.escape(keyword.lower())}\b", 
                                      content_for_matching.lower()) for keyword in keywords):
                                    matched_topic = topic
                                    break
                            
                            if matched_topic:
                                article['topic'] = matched_topic
                                metrics.matched_articles += 1
                                
                                # Fetch full content for matched articles
                                if article.get('link'):
                                    try:
                                        # Update status for individual article
                                        if status_callback:
                                            status_callback(
                                                progress,
                                                f"Fetching content for article: {article['title'][:30]}...",
                                                idx,
                                                len(RSS_FEEDS),
                                                len(all_articles)
                                            )
                                        
                                        # Fetch and extract content
                                        result = await crawler.fetch_url(article['link'])
                                        
                                        if result.get('success'):
                                            # Add extracted content and metadata to article
                                            article['content'] = result.get('content', '')
                                            article['description'] = article.get('description', '') or result.get('excerpt', '')
                                            article['image_url'] = result.get('image_url', '')
                                            
                                            # Use more accurate publication date if available
                                            if result.get('pub_date'):
                                                article['pub_date'] = result.get('pub_date')
                                                
                                            # Generate summary for the article if needed
                                            # This maintains compatibility with the existing code
                                            if not article.get('summary') and (article.get('content') or article.get('description')):
                                                text_to_summarize = article.get('content') or article.get('description')
                                                try:
                                                    article['summary'] = await summarize_text(text_to_summarize)
                                                    if article['summary']:
                                                        metrics.summary_generated += 1
                                                    else:
                                                        metrics.summary_failed += 1
                                                except Exception as e:
                                                    logging.error(f"Error generating summary: {e}")
                                                    metrics.summary_failed += 1
                                    except Exception as e:
                                        logging.error(f"Error fetching content for {article['link']}: {e}")
                                
                                # Add to final list
                                all_articles.append(article)
                            
                            metrics.total_articles += 1
                    else:
                        metrics.failed_feeds += 1
                        logging.warning(f"No articles found in feed: {source}")
                        
                except Exception as e:
                    metrics.failed_feeds += 1
                    logging.error(f"Error processing feed {source}: {e}")
            
            # Process NewsAPI if available
            if HAS_NEWSAPI_CONFIG and NEWSAPI_ENABLED:
                if status_callback:
                    status_callback(
                        50,
                        "Fetching articles from NewsAPI...",
                        len(RSS_FEEDS),
                        len(RSS_FEEDS) + 1,
                        len(all_articles)
                    )
                    
                try:
                    # Use existing NewsAPI function
                    newsapi_articles = await fetch_from_newsapi(crawler.session)
                    all_articles.extend(newsapi_articles)
                    metrics.total_articles += len(newsapi_articles)
                    metrics.matched_articles += len(newsapi_articles)
                except Exception as e:
                    logging.error(f"Error fetching from NewsAPI: {e}")
            
            # Final status report
            if status_callback:
                status_callback(
                    90,
                    f"Scraping complete. Saving {len(all_articles)} articles...",
                    len(RSS_FEEDS),
                    len(RSS_FEEDS),
                    len(all_articles)
                )
            
            # Save articles directly to Firestore
            if all_articles:
                success = await save_articles_to_firestore(all_articles)
                if not success:
                    logging.error("Failed to save articles to Firestore")
            
            # Report completion
            stats = metrics.get_stats()
            logging.info(f"Scraping statistics: {json.dumps(stats, indent=2)}")
            
            if status_callback:
                status_callback(
                    100,
                    "Scraping process completed successfully!",
                    len(RSS_FEEDS),
                    len(RSS_FEEDS),
                    len(all_articles)
                )
            
            return True
            
    except Exception as e:
        logging.error(f"Error during scraping process: {e}")
        
        if status_callback:
            status_callback(
                100,
                f"Error during scraping: {str(e)}",
                0,
                len(RSS_FEEDS),
                0
            )
            
        return False

# This fetch_feed function is maintained for backward compatibility
# but we'll now use the WebCrawler implementation internally
async def fetch_feed(session: aiohttp.ClientSession, source: str, url: str) -> List[Dict]:
    """Asynchronously fetch and process a single feed"""
    try:
        # Instead of implementing here, we'll use the WebCrawler
        async with WebCrawler(
            cache_file=CRAWLER_CACHE_FILE, 
            max_age_days=30,
            calls_per_second=1.0
        ) as crawler:
            articles = await crawler.fetch_rss_feed(url)
            
            # Add source and match topics
            processed_articles = []
            for article in articles:
                # Add source information
                article['source'] = source
                
                # Match topics
                content_for_matching = f"{article.get('title', '')} {article.get('description', '')}"
                matched_topic = None
                for topic, keywords in TOPICS.items():
                    if any(re.search(rf"\b{re.escape(keyword.lower())}\b", 
                          content_for_matching.lower()) for keyword in keywords):
                        matched_topic = topic
                        break
                
                if matched_topic:
                    article['topic'] = matched_topic
                    processed_articles.append(article)
            
            return processed_articles
    except Exception as e:
        logging.error(f"Error fetching feed {source} ({url}): {e}")
        return []

async def save_articles_to_firestore(articles: List[Dict]) -> bool:
    """Save articles directly to Firestore database."""
    try:
        if not articles:
            logging.info("No new articles to save")
            return True
            
        # Initialize the storage service
        from services.storage_service import StorageService
        storage = StorageService()
        
        # Ensure consistent fields across all articles
        required_fields = ['title', 'link', 'source', 'pub_date', 'topic', 'description', 'summary']
        standardized_articles = []
        
        for article in articles:
            # Create a standardized article with all required fields 
            # and ensure any missing fields are set to empty string
            standardized = {field: article.get(field, '') for field in required_fields}
            
            # Convert any None values to empty strings
            for key, value in standardized.items():
                if value is None:
                    standardized[key] = ''
            
            # Add additional fields
            for key, value in article.items():
                if key not in required_fields:
                    standardized[key] = value if value is not None else ''
                    
            standardized_articles.append(standardized)
        
        # Batch store all articles directly to Firestore
        success = await storage.batch_store_articles(standardized_articles)
        
        if success:
            logging.info(f"Successfully saved {len(standardized_articles)} articles to Firestore")
        else:
            logging.error("Error saving articles to Firestore")
            
        return success
            
    except Exception as e:
        logging.error(f"Error saving articles to Firestore: {e}")
        return False

# Keep the old function for backward compatibility but make it call the new one
async def save_articles_to_csv(articles: List[Dict]) -> bool:
    """Legacy function maintained for backward compatibility. Now just calls save_articles_to_firestore."""
    logging.warning("save_articles_to_csv is deprecated. Use save_articles_to_firestore instead.")
    return await save_articles_to_firestore(articles)

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

    async def fetch_feed(self, source: str, url: str) -> List[Dict]:
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
                
                if not soup.find('item'):  # Check if feed is valid
                    logging.error(f"No items found in feed from {source} ({url})")
                    return []
                    
                items = soup.find_all('item')
                
                # Process found items
                articles = []
                for item in items:
                    title_tag = item.find('title')
                    title = title_tag.text.strip() if title_tag else ''
                    
                    description_tag = item.find('description')
                    description = description_tag.text.strip() if description_tag else ''
                    
                    link_tag = item.find('link')
                    link = link_tag.text.strip() if link_tag else ''
                    
                    date_tag = item.find('pubDate')
                    pub_date = date_tag.text.strip() if date_tag else ''
                    
                    guid_tag = item.find('guid')
                    guid = guid_tag.text.strip() if guid_tag else link
                    
                    # Skip items without title or link
                    if not title or not link:
                        continue
                        
                    # Create article dict
                    article = {
                        'title': title,
                        'description': description,
                        'link': link,
                        'pub_date': pub_date,
                        'guid': guid,
                        'source': source
                    }
                    
                    # Match topic
                    content_for_matching = f"{title} {description}"
                    matched_topic = None
                    
                    for topic, keywords in TOPICS.items():
                        for keyword in keywords:
                            if keyword.lower() in content_for_matching.lower():
                                matched_topic = topic
                                break
                        if matched_topic:
                            break
                    
                    if matched_topic:
                        article['topic'] = matched_topic
                        articles.append(article)

                return articles

        except Exception as e:
            logging.error(f"Error processing feed {source}: {e}")
            return []

# Add this new function to fetch full article content
async def fetch_full_article_content(session, url):
    """Fetch the full content of an article from its URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Configure SSL context for aiohttp
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with session.get(url, headers=headers, ssl=ssl_context, timeout=30) as response:
            if response.status != 200:
                logging.warning(f"Failed to fetch article content: {url}, status: {response.status}")
                return None
                
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to extract the main article content
            # This is a simplified approach - different sites have different structures
            article_content = ""
            
            # Try common article content selectors
            content_selectors = [
                'article', '.article-content', '.article-body', '.content-body', 
                '.post-content', '.entry-content', '.story-body', 'main',
                '[itemprop="articleBody"]', '.article__body', '.article-text'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    # Remove unwanted elements like ads, related articles, etc.
                    for unwanted in content_element.select('.ad, .advertisement, .related, .sidebar, nav, .nav, .menu, .comments, .social, .share'):
                        unwanted.decompose()
                    
                    # Extract text
                    article_content = content_element.get_text(separator=' ', strip=True)
                    break
            
            # If no content found with selectors, try paragraphs
            if not article_content:
                paragraphs = soup.find_all('p')
                article_content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100])
            
            # Clean up the content
            article_content = re.sub(r'\s+', ' ', article_content).strip()
            
            # If content is too short, it might not be the actual article
            if len(article_content) < 200:
                logging.warning(f"Article content too short, might not be the actual content: {url}")
                return None
                
            return article_content
            
    except Exception as e:
        logging.error(f"Error fetching full article content from {url}: {e}")
        return None

# Update the process_article function to include full content fetching
async def process_article(session, article, source):
    """Process a single article with full content fetching"""
    try:
        # Extract article content
        title = getattr(article.title, 'text', '').strip() if article.title else ""
        description = getattr(article.description, 'text', '').strip() if article.description else ""
        link = getattr(article.link, 'text', '').strip() if article.link else ""
        pub_date = getattr(article.pubDate, 'text', '') if article.pubDate else ""

        if not all([title, link]):
            return None

        # Format the publication date
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
            if any(re.search(rf"\b{re.escape(keyword.lower())}\b", 
                   (title + " " + description).lower()) for keyword in keywords):
                matched_topic = topic
                break

        if matched_topic:
            # Try to fetch full article content
            full_content = await fetch_full_article_content(session, link)
            
            # Use full content if available, otherwise use description
            content_for_summary = full_content if full_content else description
            
            # Generate summary
            summary = await summarize_text(content_for_summary)
            
            return {
                "source": source,
                "title": title,
                "description": full_content if full_content else description,  # Use full content if available
                "link": link,
                "pub_date": pub_date,
                "topic": matched_topic,
                "summary": summary,
                "has_full_content": bool(full_content)  # Flag to indicate if we have full content
            }
        return None
        
    except Exception as e:
        logging.error(f"Error processing article from {source}: {str(e)}")
        return None

# Update the fetch_feed function to use the new process_article function
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
            
            # Process articles concurrently with rate limiting
            tasks = []
            rate_limiter = RateLimiter(calls_per_second=0.5)  # 2 seconds between API calls
            
            for article in articles:
                await rate_limiter.acquire()
                tasks.append(process_article(session, article, source))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None values and exceptions
            processed_articles = []
            for result in results:
                if isinstance(result, dict):
                    processed_articles.append(result)
                elif isinstance(result, Exception):
                    logging.error(f"Error processing article: {result}")
            
            return processed_articles

    except aiohttp.ClientError as e:
        logging.error(f"Network error fetching feed from {source} ({url}): {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error fetching feed from {source} ({url}): {str(e)}")
        return []

async def verify_database_consistency():
    """
    Verify the consistency of the articles database with the storage backend.
    
    This function compares the articles in the CSV file with those in the storage backend,
    and adds any missing articles to the storage backend.
    
    Returns:
        tuple: (bool, dict) - Success status and stats about the verification/repair
    """
    try:
        # Load configuration and services
        from services.storage_service import StorageService
        from services.config_service import ConfigService
        import pandas as pd
        
        config = ConfigService()
        storage = StorageService()
        
        # Stats to return
        stats = {
            "total_csv_articles": 0,
            "articles_checked": 0,
            "already_in_db": 0,
            "articles_to_add": 0,
            "added": 0,
            "failed_to_add": 0,
            "repaired": 0
        }
        
        # Load articles from CSV
        if not os.path.exists(config.articles_file_path):
            return True, {"error": "No articles CSV file found"}
            
        df = pd.read_csv(config.articles_file_path)
        articles = df.to_dict(orient='records')
        stats["total_csv_articles"] = len(articles)
        
        # Get a batch of recent articles to compare against
        # This avoids having to make individual DB queries for each article
        logging.info(f"Loading recent articles from database for comparison...")
        recent_articles = await storage.get_recent_articles(limit=500)
        
        # Create a lookup dictionary by article ID for faster checking
        db_article_ids = {}
        db_article_urls = {}
        
        for article in recent_articles:
            article_id = article.get('id')
            if article_id:
                db_article_ids[article_id] = True
            
            # Also track article URLs to catch duplicates with different IDs
            article_url = article.get('metadata', {}).get('link')
            if article_url:
                db_article_urls[article_url] = True
        
        logging.info(f"Found {len(db_article_ids)} articles in database")
        
        # Process each article from CSV
        for article in articles:
            stats["articles_checked"] += 1
            
            # Make sure required fields are present
            if not all(k in article for k in ['title', 'link', 'pub_date', 'description', 'source']):
                logging.warning(f"Skipping article with missing fields: {article.get('title', 'Unknown')}")
                continue
                
            # Generate article ID the same way storage service would
            article_id = generate_article_id(article)
            
            # Check if article already exists by ID
            if article_id in db_article_ids:
                stats["already_in_db"] += 1
                continue
                
            # Also check if article URL already exists (to catch duplicates with different IDs)
            if article.get('link') in db_article_urls:
                stats["already_in_db"] += 1
                continue
                
            # If we get here, article is not in the database
            stats["articles_to_add"] += 1
            
            try:
                # Store article in the backend
                success = await storage.store_article(article)
                
                if success:
                    stats["added"] += 1
                    stats["repaired"] += 1  # Count as repaired since we added a missing article
                else:
                    stats["failed_to_add"] += 1
                    
            except Exception as e:
                logging.error(f"Error adding article to storage: {e}")
                stats["failed_to_add"] += 1
        
        logging.info(f"Verification complete: {stats['repaired']} articles repaired")
        return True, stats
        
    except Exception as e:
        logging.error(f"Error verifying database consistency: {e}")
        return False, {"error": str(e)}

def generate_article_id(article):
    """Generate an article ID using the same algorithm as the storage service"""
    # Generate the same ID that would be used in storage_service
    from datetime import datetime
    import hashlib
    import time
    
    try:
        # Create a unique ID from title and date
        title_part = article['title'][:50].strip().lower()  # First 50 chars of title
        date_part = article.get('pub_date', datetime.now().isoformat())
        source_part = article.get('source', 'unknown')[:10]
        
        # Clean the ID components
        title_part = "".join(c for c in title_part if c.isalnum() or c.isspace())
        title_part = title_part.replace(" ", "_")
        source_part = source_part.replace(" ", "_")
        
        # Create a more reliable date part
        try:
            if isinstance(date_part, str):
                parsed_date = datetime.fromisoformat(date_part.replace('Z', '+00:00'))
            else:
                parsed_date = date_part
            date_part = parsed_date.strftime('%Y-%m-%d_%H-%M-%S')
        except Exception:
            date_part = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        return f"{source_part}_{title_part}_{date_part}"
        
    except Exception as e:
        # Fallback to a timestamp-based ID with MD5 hash
        fallback_id = f"{article['title']}_{time.time()}"
        hash_id = hashlib.md5(fallback_id.encode()).hexdigest()[:10]
        return f"article_{hash_id}"