#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebCrawler module for NewsAggregator
A more robust implementation for crawling web content that builds on the
existing newsLetter.py foundation with improved error handling,
cache management, and content extraction.
"""

import os
import re
import time
import json
import asyncio
import logging
import ssl
import certifi
import hashlib
from typing import Dict, List, Optional, Set, Union, Callable, Any
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
import feedparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class RobustCache:
    """
    More robust implementation of cache with better error handling
    and atomic write operations to prevent JSON parsing errors
    """
    def __init__(self, cache_file: str, max_age_days: int = 30):
        self.cache_file = cache_file
        self.max_age_days = max_age_days
        self.cache = self._load_cache()
        self._lock = asyncio.Lock()

    def _load_cache(self) -> Dict:
        """Load cache from file with robust error handling"""
        if not os.path.exists(self.cache_file):
            return {}
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Validate and clean cache data
            now = datetime.now()
            cutoff = now - timedelta(days=self.max_age_days)
            cutoff_timestamp = cutoff.timestamp()
            
            # Filter out old entries
            cleaned_cache = {
                k: v for k, v in cache_data.items() 
                if isinstance(v, dict) and 
                   v.get('timestamp', 0) > cutoff_timestamp
            }
            
            return cleaned_cache
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error loading cache: {e}")
            # Create a backup of the corrupted cache file
            if os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.corrupted.{int(time.time())}"
                try:
                    os.rename(self.cache_file, backup_file)
                    logging.info(f"Corrupted cache file backed up to {backup_file}")
                except Exception as rename_err:
                    logging.error(f"Failed to back up corrupted cache: {rename_err}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading cache: {e}")
            return {}

    def get(self, key: str) -> Optional[Dict]:
        """Get a value from the cache"""
        if not key:
            return None
            
        cache_entry = self.cache.get(key)
        if not cache_entry:
            return None
            
        # Check if entry is expired
        now = datetime.now().timestamp()
        if now - cache_entry.get('timestamp', 0) > self.max_age_days * 86400:
            # Remove expired entry
            self.cache.pop(key, None)
            return None
            
        return cache_entry.get('data')

    async def set(self, key: str, data: Any):
        """Set a value in the cache with current timestamp"""
        if not key:
            return
            
        # Update the cache in memory with lock protection
        async with self._lock:
            self.cache[key] = {
                'data': data,
                'timestamp': datetime.now().timestamp()
            }
        
        # Save cache to disk (has its own lock protection)
        await self._save_cache()

    async def _save_cache(self):
        """Save cache to file with atomic write to prevent corruption"""
        try:
            async with self._lock:
                # Write to a temporary file first
                temp_file = f"{self.cache_file}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                    
                    # Make sure file is fully written - flush before fsync
                    f.flush()
                    os.fsync(f.fileno())
                
                # Replace the original file (atomic operation on most file systems)
                os.replace(temp_file, self.cache_file)
                
        except Exception as e:
            logging.error(f"Error saving cache: {e}")


class RateLimiter:
    """Rate limiter to prevent overloading websites"""
    def __init__(self, calls_per_second: float = 1.0, per_domain: bool = True):
        self.calls_per_second = calls_per_second
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call_time: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.per_domain = per_domain
        
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL if per_domain is enabled"""
        if not self.per_domain:
            return "default"
            
        try:
            return urlparse(url).netloc
        except Exception:
            return url
            
    async def acquire(self, url: str = "default") -> None:
        """Wait if necessary to maintain rate limits"""
        key = self._get_domain(url) if self.per_domain else url
        
        async with self._lock:
            if key in self.last_call_time:
                elapsed = time.time() - self.last_call_time[key]
                if elapsed < self.minimum_interval:
                    await asyncio.sleep(self.minimum_interval - elapsed)
            
            self.last_call_time[key] = time.time()


@dataclass
class CrawlMetrics:
    """Track crawling performance metrics"""
    start_time: float
    total_urls: int = 0
    successful_urls: int = 0
    failed_urls: int = 0
    redirects: int = 0
    cache_hits: int = 0
    new_urls_found: int = 0
    rate_limits: int = 0
    total_content_size: int = 0
    
    def get_stats(self) -> Dict:
        """Get crawling statistics"""
        duration = time.time() - self.start_time
        return {
            "duration_seconds": round(duration, 2),
            "total_urls": self.total_urls,
            "successful_urls": self.successful_urls,
            "failed_urls": self.failed_urls,
            "success_rate": f"{(self.successful_urls/self.total_urls)*100:.1f}%" if self.total_urls else "0%",
            "redirects": self.redirects,
            "cache_hits": self.cache_hits,
            "new_urls_found": self.new_urls_found,
            "rate_limits": self.rate_limits,
            "avg_content_size": f"{self.total_content_size/self.successful_urls:.1f} bytes" if self.successful_urls else "0 bytes",
            "processing_speed": f"{self.total_urls/duration:.1f} URLs/sec" if duration > 0 else "0 URLs/sec"
        }


class WebCrawler:
    """
    Advanced web crawler with robust error handling, rate limiting,
    and content extraction capabilities.
    """
    
    def __init__(
        self,
        cache_file: str = "crawler_cache.json",
        max_age_days: int = 7,
        calls_per_second: float = 1.0,
        timeout_seconds: int = 30,
        max_redirects: int = 5,
        max_retries: int = 3,
        user_agent: str = "NewsAggregator WebCrawler/1.0",
        max_urls_per_domain: int = 100,
        respect_robots_txt: bool = True
    ):
        self.cache = RobustCache(cache_file, max_age_days)
        self.rate_limiter = RateLimiter(calls_per_second, per_domain=True)
        self.timeout_seconds = timeout_seconds
        self.max_redirects = max_redirects
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.max_urls_per_domain = max_urls_per_domain
        self.respect_robots_txt = respect_robots_txt
        
        # Track domains and their disallowed paths from robots.txt
        self.robots_txt_cache: Dict[str, Set[str]] = {}
        self.domain_counters: Dict[str, int] = {}
        
        # Session will be initialized in crawl method
        self.session = None
        
    async def __aenter__(self):
        """Context manager entry"""
        # Initialize session when entering the context
        connector = TCPConnector(limit=10, ssl=False)  # Limit concurrent connections
        self.session = aiohttp.ClientSession(connector=connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests"""
        return {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots_txt:
            return True
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            
            # Check if we've already fetched robots.txt for this domain
            if domain not in self.robots_txt_cache:
                # Fetch robots.txt
                robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
                
                # Don't use rate limiter for robots.txt to avoid deadlocks
                async with self.session.get(
                    robots_url,
                    headers=self._get_headers(),
                    timeout=ClientTimeout(total=10),
                    ssl=ssl.create_default_context(cafile=certifi.where())
                ) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        disallowed = set()
                        
                        # Very basic robots.txt parsing
                        user_agent_match = False
                        for line in robots_content.splitlines():
                            line = line.strip().lower()
                            
                            # Check if line applies to our user agent
                            if line.startswith('user-agent:'):
                                agent = line[11:].strip()
                                user_agent_match = agent == '*' or self.user_agent.lower().startswith(agent)
                                
                            # If line applies to our user agent and disallows a path, add to set
                            elif user_agent_match and line.startswith('disallow:'):
                                path_pattern = line[9:].strip()
                                if path_pattern:
                                    disallowed.add(path_pattern)
                                    
                        self.robots_txt_cache[domain] = disallowed
                    else:
                        # If robots.txt not found or error, assume everything is allowed
                        self.robots_txt_cache[domain] = set()
                        
            # Check if URL path is disallowed
            for pattern in self.robots_txt_cache[domain]:
                if path.startswith(pattern):
                    return False
                    
            return True
            
        except Exception as e:
            logging.warning(f"Error checking robots.txt for {url}: {e}")
            # On error, allow the URL
            return True
            
    def _extract_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize URLs from HTML"""
        urls = []
        seen = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            # Skip empty links and javascript
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
                
            # Normalize URL
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            fragment_pos = full_url.find('#')
            if fragment_pos > 0:
                full_url = full_url[:fragment_pos]
                
            # Skip if already seen
            if full_url in seen:
                continue
                
            seen.add(full_url)
            urls.append(full_url)
            
        return urls
        
    def _extract_article_content(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract article content and metadata from HTML"""
        result = {
            'title': '',
            'content': '',
            'excerpt': '',
            'pub_date': '',
            'author': '',
            'image_url': ''
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            result['title'] = title_tag.get_text(strip=True)
            
        # Try to get a better title from og:title or article title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            result['title'] = og_title['content']
            
        h1 = soup.find('h1')
        if h1:
            result['title'] = h1.get_text(strip=True)
            
        # Extract main image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            result['image_url'] = og_image['content']
            
        # Extract publication date
        pub_date = None
        date_selectors = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'itemprop': 'datePublished'}),
            ('time', {})
        ]
        
        for tag, attrs in date_selectors:
            date_elem = soup.find(tag, attrs)
            if date_elem:
                if tag == 'meta':
                    pub_date = date_elem.get('content')
                else:
                    pub_date = date_elem.get('datetime') or date_elem.get_text(strip=True)
                break
                
        if pub_date:
            result['pub_date'] = pub_date
            
        # Extract author
        author_selectors = [
            ('meta', {'property': 'article:author'}),
            ('meta', {'name': 'author'}),
            ('a', {'rel': 'author'}),
            ('span', {'class': 'author'}),
            ('div', {'class': 'author'})
        ]
        
        for tag, attrs in author_selectors:
            author_elem = soup.find(tag, attrs)
            if author_elem:
                if tag == 'meta':
                    result['author'] = author_elem.get('content', '')
                else:
                    result['author'] = author_elem.get_text(strip=True)
                break
                
        # Extract main content
        content_selectors = [
            'article', '.article-content', '.article-body', '.content-body', 
            '.post-content', '.entry-content', '.story-body', 'main',
            '[itemprop="articleBody"]', '.article__body', '.article-text'
        ]
        
        main_content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                # Remove unwanted elements
                for unwanted in content_element.select('.ad, .advertisement, .related, nav, .nav, .menu, .comments, .social, .share'):
                    unwanted.decompose()
                    
                main_content = content_element
                break
                
        # If no content found with selectors, try paragraphs
        if main_content:
            # Get all paragraphs in the main content
            paragraphs = main_content.find_all('p')
            result['content'] = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
            # Get excerpt from first substantial paragraph
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 140:
                    result['excerpt'] = text[:280] + ('...' if len(text) > 280 else '')
                    break
        else:
            # Fallback: use all paragraphs in the document
            paragraphs = soup.find_all('p')
            paragraph_texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
            
            if paragraph_texts:
                result['content'] = ' '.join(paragraph_texts)
                result['excerpt'] = paragraph_texts[0][:280] + ('...' if len(paragraph_texts[0]) > 280 else '')
                
        # Clean up content
        result['content'] = re.sub(r'\s+', ' ', result['content']).strip()
        
        return result
        
    async def fetch_url(self, url: str, follow_links: bool = False, depth: int = 0, max_depth: int = 1) -> Dict[str, Any]:
        """
        Fetch a URL with robust error handling and content extraction
        
        Args:
            url: The URL to fetch
            follow_links: Whether to follow links on the page
            depth: Current crawl depth
            max_depth: Maximum crawl depth
            
        Returns:
            Dict with fetched content and metadata
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with WebCrawler() as crawler:'")
            
        # Generate cache key from URL
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            # Return cached data but don't follow links from cache
            return {**cached_data, 'from_cache': True}
            
        # Check domain limit
        domain = urlparse(url).netloc
        if domain in self.domain_counters and self.domain_counters[domain] >= self.max_urls_per_domain:
            return {
                'url': url,
                'success': False,
                'error': f"Domain limit reached for {domain}",
                'status_code': 0
            }
            
        # Check robots.txt
        allowed = await self._check_robots_txt(url)
        if not allowed:
            return {
                'url': url,
                'success': False,
                'error': "Blocked by robots.txt",
                'status_code': 0
            }
            
        # Apply rate limiting
        await self.rate_limiter.acquire(url)
        
        # Track domain counter
        if domain not in self.domain_counters:
            self.domain_counters[domain] = 0
        self.domain_counters[domain] += 1
        
        # Configure SSL context and headers
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        headers = self._get_headers()
        
        # Attempt to fetch the URL with retries
        retries = 0
        while retries <= self.max_retries:
            try:
                async with self.session.get(
                    url,
                    headers=headers,
                    timeout=ClientTimeout(total=self.timeout_seconds),
                    ssl=ssl_context,
                    allow_redirects=True,
                    max_redirects=self.max_redirects
                ) as response:
                    # Check for redirect
                    is_redirect = response.history and len(response.history) > 0
                    final_url = str(response.url)
                    
                    # Handle unsuccessful responses
                    if response.status != 200:
                        return {
                            'url': url,
                            'final_url': final_url if is_redirect else url,
                            'success': False,
                            'is_redirect': is_redirect,
                            'error': f"HTTP error {response.status}",
                            'status_code': response.status
                        }
                        
                    # Get content type
                    content_type = response.headers.get('Content-Type', '')
                    is_html = 'text/html' in content_type.lower()
                    
                    if not is_html:
                        return {
                            'url': url,
                            'final_url': final_url if is_redirect else url,
                            'success': True,
                            'is_redirect': is_redirect,
                            'content_type': content_type,
                            'status_code': response.status,
                            'is_html': False
                        }
                        
                    # Parse HTML content
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract article content
                    article_data = self._extract_article_content(soup)
                    
                    result = {
                        'url': url,
                        'final_url': final_url if is_redirect else url,
                        'success': True,
                        'is_redirect': is_redirect,
                        'content_type': content_type,
                        'status_code': response.status,
                        'is_html': True,
                        'html_content': html_content,
                        'title': article_data['title'],
                        'content': article_data['content'],
                        'excerpt': article_data['excerpt'],
                        'pub_date': article_data['pub_date'],
                        'author': article_data['author'],
                        'image_url': article_data['image_url'],
                        'links': [],
                        'timestamp': datetime.now().timestamp()
                    }
                    
                    # Extract and follow links if requested
                    if follow_links and depth < max_depth:
                        extracted_urls = self._extract_urls(soup, final_url if is_redirect else url)
                        result['links'] = extracted_urls
                        
                    # Cache successful results
                    await self.cache.set(cache_key, {k: v for k, v in result.items() if k != 'html_content'})
                    
                    return result
                    
            except asyncio.TimeoutError:
                retries += 1
                if retries <= self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retries
                    logging.warning(f"Timeout fetching {url}, retrying in {wait_time}s (attempt {retries}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        'url': url,
                        'success': False,
                        'error': "Timeout after retries",
                        'status_code': 0
                    }
            except Exception as e:
                return {
                    'url': url,
                    'success': False,
                    'error': f"Error: {str(e)}",
                    'status_code': 0
                }
                
    async def crawl(
        self,
        seed_urls: List[str],
        follow_links: bool = False,
        max_depth: int = 1,
        max_urls: int = 100,
        status_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Crawl starting from seed URLs
        
        Args:
            seed_urls: List of URLs to start crawling from
            follow_links: Whether to follow links on pages
            max_depth: Maximum depth for following links
            max_urls: Maximum number of URLs to crawl
            status_callback: Optional callback for reporting progress
            
        Returns:
            Dict with crawl results and metrics
        """
        metrics = CrawlMetrics(start_time=time.time())
        results = []
        
        # Make sure we have a session
        if not self.session:
            connector = TCPConnector(limit=10, ssl=False)  # Limit concurrent connections
            self.session = aiohttp.ClientSession(connector=connector)
            
        # Queue of URLs to crawl: (url, depth)
        queue = [(url, 0) for url in seed_urls]
        crawled_urls = set()
        
        # Report initial status
        if status_callback:
            status_callback(0, "Starting crawl...", 0, len(seed_urls), 0)
            
        # Process queue
        while queue and len(crawled_urls) < max_urls:
            # Get next batch of URLs to process concurrently
            batch_size = min(10, max_urls - len(crawled_urls), len(queue))
            batch = [queue.pop(0) for _ in range(batch_size)]
            
            # Report status
            if status_callback:
                progress = int((len(crawled_urls) / max_urls) * 100)
                status_callback(
                    progress,
                    f"Crawling {len(crawled_urls)}/{max_urls} URLs...",
                    len(crawled_urls),
                    max_urls,
                    len(results)
                )
                
            # Process batch concurrently
            tasks = [self.fetch_url(url, follow_links, depth, max_depth) for url, depth in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                url = batch[batch_results.index(result)][0]
                crawled_urls.add(url)
                
                if isinstance(result, Exception):
                    metrics.failed_urls += 1
                    logging.error(f"Error crawling {url}: {result}")
                    continue
                    
                metrics.total_urls += 1
                
                if result.get('from_cache'):
                    metrics.cache_hits += 1
                    
                if result.get('success'):
                    metrics.successful_urls += 1
                    if result.get('content'):
                        metrics.total_content_size += len(result['content'])
                        
                    if result.get('is_redirect'):
                        metrics.redirects += 1
                        
                    # Append successful result
                    results.append(result)
                    
                    # Add new links to queue if following links
                    if follow_links and 'links' in result:
                        for link in result['links']:
                            # Skip already crawled or queued URLs
                            if link in crawled_urls or any(link == url for url, _ in queue):
                                continue
                                
                            # Add to queue with incremented depth
                            queue.append((link, batch[batch_results.index(result)][1] + 1))
                            metrics.new_urls_found += 1
                else:
                    metrics.failed_urls += 1
                    
        # Final status report
        if status_callback:
            status_callback(
                100,
                f"Crawl complete. Processed {len(crawled_urls)} URLs.",
                len(crawled_urls),
                max_urls,
                len(results)
            )
            
        return {
            'metrics': metrics.get_stats(),
            'results': results
        }
        
    async def fetch_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse an RSS feed
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of articles from the feed
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with WebCrawler() as crawler:'")
            
        # Apply rate limiting
        await self.rate_limiter.acquire(feed_url)
        
        try:
            # Configure SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Fetch feed
            async with self.session.get(
                feed_url,
                headers=self._get_headers(),
                timeout=ClientTimeout(total=self.timeout_seconds),
                ssl=ssl_context
            ) as response:
                if response.status != 200:
                    logging.error(f"Error fetching feed {feed_url}: status {response.status}")
                    return []
                    
                content = await response.text()
                
                # Parse feed with feedparser
                feed = feedparser.parse(content)
                
                articles = []
                for entry in feed.entries:
                    article = {
                        'source': feed.feed.get('title', urlparse(feed_url).netloc),
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'pub_date': entry.get('published', ''),
                        'author': entry.get('author', ''),
                        'id': entry.get('id', ''),
                        'feed_url': feed_url
                    }
                    
                    # Only include articles with required fields
                    if all([article['title'], article['link']]):
                        articles.append(article)
                        
                return articles
                
        except Exception as e:
            logging.error(f"Error fetching RSS feed {feed_url}: {e}")
            return []


# Example usage
async def main():
    # Example seed URLs
    seed_urls = [
        'https://news.ycombinator.com/',
        'https://www.theverge.com/',
        'https://techcrunch.com/'
    ]
    
    # Example RSS feeds
    rss_feeds = {
        'The Verge': 'https://www.theverge.com/rss/index.xml',
        'TechCrunch': 'https://techcrunch.com/feed/',
        'Wired': 'https://www.wired.com/feed/rss'
    }
    
    # Define a status callback
    def status_callback(progress, message, processed, total, found):
        print(f"Progress: {progress}%, {message} ({processed}/{total}, found: {found})")
    
    async with WebCrawler(
        cache_file="crawler_cache.json",
        calls_per_second=1.0,
        max_urls_per_domain=20
    ) as crawler:
        # Fetch articles from RSS feeds
        print("Fetching RSS feeds...")
        all_articles = []
        for source, url in rss_feeds.items():
            print(f"Fetching {source} feed...")
            articles = await crawler.fetch_rss_feed(url)
            print(f"Found {len(articles)} articles in {source}")
            all_articles.extend(articles)
            
        print(f"Total articles from RSS feeds: {len(all_articles)}")
        
        # Crawl seed URLs
        print("\nCrawling seed URLs...")
        crawl_results = await crawler.crawl(
            seed_urls=seed_urls,
            follow_links=True,
            max_depth=1,
            max_urls=50,
            status_callback=status_callback
        )
        
        print("\nCrawl Metrics:")
        for key, value in crawl_results['metrics'].items():
            print(f"{key}: {value}")
            
        print(f"\nFound {len(crawl_results['results'])} pages")
        
        # Example: print titles of first 5 crawled pages
        for i, result in enumerate(crawl_results['results'][:5]):
            print(f"{i+1}. {result.get('title', 'No title')} - {result.get('url')}")


if __name__ == "__main__":
    asyncio.run(main()) 