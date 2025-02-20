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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add cache configuration
CACHE_FILE = "summary_cache.json"
RATE_LIMIT_DELAY = 1  # seconds between API calls

class RateLimitException(Exception):
    pass

class FeedError(Exception):
    """Custom exception for feed processing errors"""
    pass

class SummaryCache:
    def __init__(self, cache_file: str, max_age_days: int = 30):
        self.cache_file = cache_file
        self.max_age = timedelta(days=max_age_days)
        self.cache = self._load_cache()

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

    def set(self, key: str, summary: str):
        """Add summary to cache with timestamp"""
        self.cache[key] = {
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

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
    "BioPharmaDive": "https://www.biopharmadive.com/feeds/news/",
    "FDA Drug Updates": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
    "FDA Recalls": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml",
    "Nature Biotechnology": "https://www.nature.com/nbt.rss",
    "BioDrugs": "https://link.springer.com/journal/40259/rss.xml",
    "Contract Pharma": "https://www.contractpharma.com/rss/news",
    "ClinicalTrials.gov": "https://clinicaltrials.gov/ct2/results/rss.xml?rcv_d=14",
    "PharmaTimes": "https://www.pharmatimes.com/rss",
    "EPR": "https://www.europeanpharmaceuticalreview.com/feed/",
    "Reuters Health": "http://feeds.reuters.com/reuters/USHealthNews",
    "Pharmaceutical Technology": "https://www.pharmaceutical-technology.com/feed/",
    "In-Pharma Technologist": "https://www.in-pharmatechnologist.com/rss",
    "Outsourcing-Pharma.com": "https://www.outsourcing-pharma.com/rss",
    "PM Live": "https://www.pmlive.com/rss/all",
    "The Pharma Journal": "https://www.pharmj.com/rss/latest/rss.xml",
    "Drug Topics": "https://www.drugtopics.com/rss",
    "STAT News": "https://www.statnews.com/feed/",
    "The Lancet": "https://www.thelancet.com/rss/general",
    "New England Journal of Medicine": "https://www.nejm.org/rss/most-recent",
    "JAMA": "https://jamanetwork.com/rss/most-recent",
    "Nature": "https://www.nature.com/nature.rss",
    "Science": "https://www.science.org/rss/news",
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
def summarize_text(text):
    """Summarize text with caching and rate limit handling"""
    if not text or len(str(text).strip()) == 0:
        return "No text to summarize"
    
    # Convert text to string if it isn't already
    text = str(text)
    
    # Initialize cache
    cache = SummaryCache(CACHE_FILE)
    cache_key = text[:100]  # Use first 100 chars as key
    
    # Check cache first
    cached_summary = cache.get(cache_key)
    if cached_summary:
        logging.info("Using cached summary")
        return cached_summary
    
    # Initialize Google AI client with configuration
    genai.configure(api_key="AIzaSyBpaF2LwIC8Il4ojQWJ9-8ysGrSeV1YrzU")
    
    # Configure the model
    generation_config = {
        "temperature": 0.4,  # More focused/deterministic output
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 150,  # Limit summary length
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    prompt = f"""
    Summarize the following text in a concise, professional manner. Focus on key points and maintain factual accuracy:
    
    {text}
    
    Provide a 2-3 sentence summary that captures the main points.
    """
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Generate summary using Gemini model
            model = genai.GenerativeModel(
                'gemini-pro',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response = model.generate_content(prompt)
            
            if response.text:
                summary = response.text.strip()
                
                # Save to cache if successful
                if summary:
                    cache.set(cache_key, summary)
                
                return summary
            
            raise Exception("Empty response from AI model")
            
        except Exception as e:
            if "RATE_LIMIT" in str(e):
                if retry_count < max_retries - 1:
                    retry_after = RATE_LIMIT_DELAY * (retry_count + 1)
                    logging.warning(f"Rate limit hit, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                raise RateLimitException("Rate limit exceeded")
            
            logging.error(f"Error summarizing text: {e}")
            if retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue
            return "Error generating summary"
            
    return "Failed to generate summary after multiple attempts"

# Main function to scrape and process news
def scrape_news():
    news_data = []
    logging.info("Starting news scraping process...")
    
    try:
        # First, collect all articles
        for source, feed_url in RSS_FEEDS.items():
            try:
                logging.info(f"Fetching from {source}: {feed_url}")
                response = requests.get(feed_url, timeout=10)
                soup = BeautifulSoup(response.content, features='xml')
                
                items = soup.find_all('item')
                for article in items:
                    try:
                        title = article.title.text if article.title else ""
                        description = article.description.text if article.description else ""
                        link = article.link.text if article.link else ""
                        pub_date = article.pubDate.text if article.pubDate else ""

                        # Clean description of HTML tags
                        clean_description = BeautifulSoup(description, "html.parser").get_text()

                        # Format the publication date
                        try:
                            pub_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pub_date = "Unknown date"

                        # Check for matches with keywords
                        matched_topic = None
                        for topic, keywords in TOPICS.items():
                            for keyword in keywords:
                                if re.search(rf"\b{re.escape(keyword)}\b", (title + " " + clean_description), re.IGNORECASE):
                                    matched_topic = topic
                                    break
                            if matched_topic:
                                break

                        if matched_topic:
                            # Generate summary immediately for new articles
                            try:
                                logging.info(f"Generating summary for: {title}")
                                time.sleep(RATE_LIMIT_DELAY)  # Prevent rate limiting
                                summary = summarize_text(clean_description)
                            except Exception as e:
                                logging.error(f"Error generating summary: {e}")
                                summary = None

                            news_data.append({
                                "source": source,
                                "title": title,
                                "description": clean_description,
                                "link": link,
                                "pub_date": pub_date,
                                "topic": matched_topic,
                                "summary": summary
                            })
                            logging.info(f"Added article: {title} with summary")

                    except Exception as e:
                        logging.error(f"Error processing article: {str(e)}")
                        continue
                    
            except Exception as e:
                logging.error(f"Error processing feed {feed_url}: {str(e)}")
                continue

        # Create DataFrame with new articles
        if news_data:
            new_df = pd.DataFrame(news_data)
            
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
            
    except Exception as e:
        logging.error(f"Error in scrape_news: {str(e)}")
        return False

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
    """Data class for article validation"""
    source: str
    title: str
    description: str
    link: str
    pub_date: str
    topic: Optional[str] = None
    summary: Optional[str] = None

    def clean(self) -> 'Article':
        """Clean and validate article data"""
        # Clean HTML from description
        clean_desc = BeautifulSoup(self.description, "html.parser").get_text().strip()
        
        # Validate and format date
        try:
            parsed_date = datetime.strptime(self.pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            formatted_date = "Unknown date"
        
        return Article(
            source=self.source.strip(),
            title=self.title.strip(),
            description=clean_desc,
            link=self.link.strip(),
            pub_date=formatted_date,
            topic=self.topic.strip() if self.topic else None,
            summary=self.summary.strip() if self.summary else None
        )

    def is_valid(self) -> bool:
        """Check if article has required fields"""
        return bool(
            self.title.strip() and 
            self.description.strip() and 
            self.link.strip()
        )

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

if __name__ == "__main__":
    news_data = scrape_news()
    logging.info("News scraping and processing complete!")