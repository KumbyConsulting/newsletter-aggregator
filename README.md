# Newsletter Aggregator

A comprehensive application for aggregating, summarizing, and querying pharmaceutical industry newsletters using Google Cloud services and Retrieval-Augmented Generation (RAG).

## Features

- **Newsletter Scraping**: Automatically scrapes pharmaceutical industry news from various sources
- **AI-Powered Summaries**: Generates concise summaries of articles using Google's Gemini AI
- **RAG-Based Querying**: Ask questions about pharmaceutical topics and get answers based on the aggregated content
- **Google Cloud Integration**: Seamlessly integrates with Google Cloud services:
  - Cloud Storage for backups
  - Vertex AI for enhanced AI capabilities
  - Cloud Logging for centralized logging
  - Cloud Monitoring for performance tracking
  - Secret Manager for secure credential management
- **Streaming Responses**: Real-time streaming of AI responses for better user experience

## Prerequisites

- Python 3.11+
- Google Cloud account with billing enabled
- Gemini API key

## Local Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd newsletter-aggregator
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file with your configuration:
   - Add your Gemini API key
   - Configure Google Cloud settings if needed
   - Adjust other settings as necessary

6. Run the application:
   ```
   python app.py
   ```

## Deployment

The deployment files for this application have been removed. A new deployment process will be implemented according to your specific requirements.

## Google Cloud Configuration

### Secret Manager

To use Secret Manager for storing sensitive information:

1. Set `USE_SECRET_MANAGER=true` in your `.env` file
2. Configure the following secrets in Google Cloud Secret Manager:
   - `gemini-api-key`
   - `flask-secret-key`

### Cloud Storage Backups

To enable automatic backups to Cloud Storage:

1. Set the following in your `.env` file:
   ```
   USE_GCS_BACKUP=true
   GCS_BUCKET_NAME=your-backup-bucket-name
   ```

2. Create a Cloud Storage bucket with an appropriate lifecycle policy.

## API Endpoints

The application provides several API endpoints:

- `/api/rag`: Submit questions for RAG-based answers
- `/api/rag/stream`: Stream RAG-based answers in real-time
- `/api/rag/history`: Manage conversation history
- `/api/update`: Trigger news scraping
- `/api/backup`: Create backups to Cloud Storage
- `/api/backups`: List available backups
- `/api/restore`: Restore from a backup
- `/api/articles`: Get paginated, filtered articles
- `/api/topics`: Get topic distribution statistics

## Web Interfaces

- `/`: Main article listing page
- `/rag`: RAG interface for asking questions
- `/test_summary`: Test interface for article summarization

## Architecture

The application follows a modular architecture with several key services:

- **Storage Service**: Manages article storage and retrieval using ChromaDB or Firestore
- **AI Service**: Handles interactions with Gemini AI or Vertex AI
- **Configuration Service**: Manages application settings and secrets
- **Logging Service**: Configures logging with Cloud Logging integration
- **Monitoring Service**: Tracks application performance metrics

## Troubleshooting

- **Environment Issues**: Make sure all required environment variables are set correctly in your `.env` file
- **API Key Issues**: Verify that your Gemini API key is valid and correctly formatted
- **Connection Issues**: Check your internet connection and firewall settings
- **Google Cloud Service Issues**: Verify your Google Cloud authentication is correctly set up for any enabled services

## License

[MIT License](LICENSE)

# Newsletter Aggregator Web Crawler

This module enhances the existing newsletter aggregator with a more robust and reliable web crawler. It builds on the foundation of the original `newsLetter.py` implementation but adds additional features for more reliable content extraction, cache management, and error handling.

## Features

- **Robust Cache Management**: Atomic writes and better error handling to prevent JSON parse errors
- **Domain-aware Rate Limiting**: Prevents overloading websites while allowing parallel crawling of different domains
- **Respect for robots.txt**: Follows crawling etiquette by respecting websites' robots.txt directives
- **Intelligent Content Extraction**: Uses multiple strategies to extract article content, titles, authors, and dates
- **Link Discovery and Following**: Ability to discover and follow links with depth control
- **Comprehensive Error Handling**: Retries with exponential backoff, SSL handling, and robust error reporting
- **Metrics Collection**: Detailed statistics on crawl performance and results
- **RSS Feed Support**: Integrated fetching and parsing of RSS feeds
- **Progress Reporting**: Callback mechanism to report on crawl progress

## Installation

```bash
# Install required dependencies
pip install aiohttp beautifulsoup4 feedparser certifi
```

## Usage

### Basic Usage

```python
import asyncio
from web_crawler import WebCrawler

async def main():
    # Initialize crawler as a context manager
    async with WebCrawler(cache_file="crawler_cache.json") as crawler:
        # Fetch a single URL
        result = await crawler.fetch_url("https://example.com")
        print(f"Title: {result.get('title')}")
        print(f"Excerpt: {result.get('excerpt')}")
        
        # Fetch an RSS feed
        articles = await crawler.fetch_rss_feed("https://example.com/feed.xml")
        print(f"Found {len(articles)} articles")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Crawling

```python
import asyncio
from web_crawler import WebCrawler

async def main():
    # Progress callback function
    def status_callback(progress, message, processed, total, found):
        print(f"Progress: {progress}%, {message} ({processed}/{total}, found: {found})")
    
    # Initialize with custom settings
    async with WebCrawler(
        cache_file="crawler_cache.json",
        max_age_days=7,  # Cache TTL
        calls_per_second=2.0,  # Rate limit
        timeout_seconds=30,
        max_redirects=5,
        max_retries=3,
        user_agent="MyCustomBot/1.0",
        max_urls_per_domain=50,
        respect_robots_txt=True
    ) as crawler:
        # Crawl multiple seed URLs with link following
        results = await crawler.crawl(
            seed_urls=["https://example.com", "https://another-site.com"],
            follow_links=True,
            max_depth=2,
            max_urls=100,
            status_callback=status_callback
        )
        
        # Print metrics
        print("\nCrawl Metrics:")
        for key, value in results['metrics'].items():
            print(f"{key}: {value}")
        
        # Process results
        for result in results['results']:
            if result.get('success'):
                print(f"URL: {result['url']}")
                print(f"Title: {result['title']}")
                print(f"Content length: {len(result['content'])}")
                print("---")

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration with Newsletter Aggregator

To use the web crawler with the existing newsletter aggregator system:

1. Import the `WebCrawler` class in your main application:

```python
from web_crawler import WebCrawler
```

2. Replace the existing article content fetching with the new crawler:

```python
async with WebCrawler() as crawler:
    # Fetch and process RSS feeds
    articles = []
    for source, url in RSS_FEEDS.items():
        feed_articles = await crawler.fetch_rss_feed(url)
        for article in feed_articles:
            # Process articles as needed
            articles.append(article)
            
    # Optionally fetch full content for each article
    for article in articles:
        if article.get('link'):
            result = await crawler.fetch_url(article['link'])
            if result.get('success'):
                article['full_content'] = result.get('content', '')
                article['image_url'] = result.get('image_url', '')
```

## Handling Cache Errors

The web crawler includes a `RobustCache` class that handles cache loading errors by:

1. Creating a backup of corrupted cache files
2. Gracefully falling back to an empty cache when JSON parsing fails
3. Using atomic writes to prevent cache corruption during saving
4. Cleaning up old entries to maintain performance

This addresses the cache loading errors seen in the logs such as:
```
ERROR - Error loading cache: Expecting value: line 1 column 1 (char 0)
```

## Customization

The web crawler is highly customizable with parameters for:

- Rate limiting
- Timeout and retry behavior
- Cache management
- Content extraction strategies
- Domain limits

Refer to the class documentation for details on all available options.

## Requirements

- Python 3.7+
- aiohttp
- BeautifulSoup4
- feedparser
- certifi


