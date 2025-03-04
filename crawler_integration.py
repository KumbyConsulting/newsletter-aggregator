#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any

from web_crawler import WebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example RSS feeds (replace with your actual feeds from newsLetter.py)
RSS_FEEDS = {
    'The Verge': 'https://www.theverge.com/rss/index.xml',
    'TechCrunch': 'https://techcrunch.com/feed/',
    'Wired': 'https://www.wired.com/feed/rss'
}

# Example topics (replace with your actual topics)
TOPICS = {
    'Technology': ['ai', 'artificial intelligence', 'machine learning', 'tech', 'technology', 'software'],
    'Science': ['science', 'research', 'discovery', 'study', 'scientific'],
    'Health': ['health', 'medical', 'medicine', 'disease', 'treatment', 'vaccine'],
    'Business': ['business', 'economy', 'finance', 'market', 'stock', 'investment']
}

# Configuration
CACHE_FILE = "crawler_cache.json" 
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "articles.csv")
MAX_AGE_DAYS = 7
RATE_LIMIT = 2.0  # requests per second per domain


async def match_topic(content: str) -> Optional[str]:
    """Match content against topics and keywords"""
    content = content.lower()
    for topic, keywords in TOPICS.items():
        for keyword in keywords:
            if f" {keyword.lower()} " in f" {content} ":
                return topic
    return None


async def scrape_news_with_crawler(status_callback: Optional[Callable] = None) -> bool:
    """
    Replacement for the scrape_news function using the new WebCrawler
    
    Args:
        status_callback: Optional callback for progress reporting
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    logging.info("Starting news scraping with enhanced web crawler...")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # Track metrics for reporting
        total_feeds = len(RSS_FEEDS)
        successful_feeds = 0
        failed_feeds = 0
        total_articles = 0
        matched_articles = 0
        all_articles = []
        
        # Initial status report
        if status_callback:
            status_callback(
                5, 
                "Initializing scraping process...", 
                0, 
                total_feeds, 
                0
            )
        
        # Initialize WebCrawler with appropriate settings using async with
        async with WebCrawler(
            cache_file=CACHE_FILE,
            max_age_days=MAX_AGE_DAYS,
            calls_per_second=RATE_LIMIT,
            timeout_seconds=30,
            max_redirects=5,
            max_retries=3,
            user_agent="NewsletterAggregator/1.0",
            max_urls_per_domain=100,
            respect_robots_txt=True
        ) as crawler:
            # Process all RSS feeds
            for idx, (source, url) in enumerate(RSS_FEEDS.items()):
                # Update status
                if status_callback:
                    progress = 5 + (idx / total_feeds * 40)  # Progress from 5% to 45%
                    status_callback(
                        progress,
                        f"Processing feed {idx+1} of {total_feeds}...",
                        idx,
                        total_feeds,
                        len(all_articles)
                    )
                
                try:
                    # Fetch articles from RSS feed
                    logging.info(f"Fetching feed: {source} from {url}")
                    articles = await crawler.fetch_rss_feed(url)
                    
                    if articles:
                        successful_feeds += 1
                        
                        # Process each article
                        for article in articles:
                            # Add source information
                            article['source'] = source
                            
                            # Match topics based on title and description
                            content_for_matching = f"{article.get('title', '')} {article.get('description', '')}"
                            topic = await match_topic(content_for_matching)
                            
                            if topic:
                                article['topic'] = topic
                                matched_articles += 1
                                
                                # Fetch full content for matched articles
                                if article.get('link'):
                                    try:
                                        # Update status for individual article
                                        if status_callback:
                                            status_callback(
                                                progress,
                                                f"Fetching content for article: {article['title'][:30]}...",
                                                idx,
                                                total_feeds,
                                                len(all_articles)
                                            )
                                        
                                        # Fetch and extract content
                                        result = await crawler.fetch_url(article['link'])
                                        
                                        if result.get('success'):
                                            # Add extracted content and metadata to article
                                            article['content'] = result.get('content', '')
                                            article['excerpt'] = result.get('excerpt', '')
                                            article['image_url'] = result.get('image_url', '')
                                            
                                            # Use more accurate publication date if available
                                            if result.get('pub_date'):
                                                article['pub_date'] = result.get('pub_date')
                                    except Exception as e:
                                        logging.error(f"Error fetching content for {article['link']}: {e}")
                                
                                # Add to final list
                                all_articles.append(article)
                            
                            total_articles += 1
                    else:
                        failed_feeds += 1
                        logging.warning(f"No articles found in feed: {source}")
                        
                except Exception as e:
                    failed_feeds += 1
                    logging.error(f"Error processing feed {source}: {e}")
        
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
            await save_articles_to_firestore(all_articles)
            
        # Report completion
        logging.info(f"Scraping process completed. Found {len(all_articles)} relevant articles.")
        
        # Report completion
        duration = time.time() - start_time
        stats = {
            "duration_seconds": round(duration, 2),
            "total_feeds": total_feeds,
            "successful_feeds": successful_feeds,
            "failed_feeds": failed_feeds,
            "total_articles": total_articles,
            "matched_articles": matched_articles,
            "match_rate": f"{(matched_articles/total_articles)*100:.1f}%" if total_articles else "0%",
            "processing_speed": f"{total_articles/duration:.1f} articles/sec" if duration > 0 else "0 articles/sec"
        }
        
        logging.info(f"Scraping statistics: {json.dumps(stats, indent=2)}")
        
        if status_callback:
            status_callback(
                100,
                "Scraping process completed successfully!",
                total_feeds,
                total_feeds,
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
                total_feeds,
                0
            )
            
        return False


async def save_articles_to_firestore(articles: List[Dict]) -> bool:
    """Save articles directly to Firestore database."""
    try:
        if not articles:
            logging.info("No new articles to save")
            return True
            
        # Initialize the storage service
        from services.config_service import ConfigService
        from services.storage_service import StorageService
        
        config = ConfigService()
        storage = StorageService()
        
        # Ensure consistent fields across all articles
        required_fields = ['title', 'link', 'source', 'pub_date', 'topic', 'description', 'summary']
        standardized_articles = []
        
        for article in articles:
            # Create a standardized article with all required fields
            standardized = {field: article.get(field, '') for field in required_fields}
            
            # Convert any None values to empty strings
            for key, value in standardized.items():
                if value is None:
                    standardized[key] = ''
            
            # Add any additional fields
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


# Example status callback function
def print_status(progress, message, sources_processed, total_sources, articles_found):
    """Example status callback that prints to console"""
    print(f"Progress: {progress:.1f}% - {message}")
    print(f"Sources: {sources_processed}/{total_sources}, Articles: {articles_found}")
    print("-" * 50)


async def main():
    """Main function to demonstrate the crawler integration"""
    print("Starting newsletter aggregator with enhanced web crawler...")
    
    # Example of using the status callback
    success = await scrape_news_with_crawler(status_callback=print_status)
    
    if success:
        print("News scraping completed successfully!")
    else:
        print("News scraping failed.")


if __name__ == "__main__":
    asyncio.run(main()) 