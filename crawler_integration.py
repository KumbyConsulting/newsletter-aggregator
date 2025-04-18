#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Callable, Any
import aiohttp
import urllib.parse
from datetime import datetime, timedelta
import re

from web_crawler import WebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load NewsAPI configuration
try:
    from context.api_config import NEWSAPI_KEY, NEWSAPI_BASE_URL, NEWSAPI_DAYS_BACK, NEWSAPI_MAX_RESULTS
    NEWSAPI_ENABLED = True
except ImportError:
    logging.warning("NewsAPI configuration not found, using defaults")
    NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
    NEWSAPI_BASE_URL = "https://newsapi.org/v2"
    NEWSAPI_DAYS_BACK = 3
    NEWSAPI_MAX_RESULTS = 20
    NEWSAPI_ENABLED = bool(NEWSAPI_KEY)

# Example RSS feeds (replace with your actual feeds from newsLetter.py)
RSS_FEEDS = {
    # Pharmaceutical Industry News
    'Pharmaceutical Technology': 'https://www.pharmaceutical-technology.com/feed/',
    'FiercePharma': 'https://www.fiercepharma.com/rss/xml',
    'BioPharmaDive': 'https://www.biopharmadive.com/feeds/news/',
    'DrugDiscoveryToday': 'http://www.drugdiscoverytoday.com/rss/news/',
    'PharmaTimes': 'https://www.pharmatimes.com/feed',
    'PharmaTimes-news': 'https://pharmatimes.com/news/feed/',
    'European Pharmaceutical Review': 'https://www.europeanpharmaceuticalreview.com/feed/',
    
    # Medical Journals
    'The Lancet': 'https://www.thelancet.com/rssfeed/lancet_current.xml',
    'New England Journal of Medicine': 'https://www.nejm.org/action/showFeed?type=etoc&feed=rss&jc=nejm',
    'JAMA': 'https://jamanetwork.com/rss/site_3/67.xml',
    'Nature Biotechnology': 'https://www.nature.com/nbt.rss',
    'British Medical Journal': 'http://feeds.bmj.com/bmj/recent',
    'The Lancet Oncology': 'https://www.thelancet.com/rssfeed/lanonc_current.xml',
    'Science Translational Medicine': 'https://stm.sciencemag.org/rss/current.xml',
    'Nature Medicine': 'https://www.nature.com/nm.rss',
    'Cell': 'https://www.cell.com/cell/current.rss',
    
    # Regulatory News
    'EMA News': 'https://www.ema.europa.eu/en/news-and-events/rss-feeds',
    'WHO News': 'https://www.who.int/rss-feeds/news-english.xml',
    'CDC': 'https://tools.cdc.gov/api/v2/resources/media/132608.rss',
    
    # European Medicines Agency (EMA) RSS Feeds
    'EMA - Agendas and minutes': 'https://www.ema.europa.eu/en/rss-feed/committee/feed',
    'EMA - Public consultations': 'https://www.ema.europa.eu/en/rss-feed/consultation/feed',
    'EMA - EURD list': 'https://www.ema.europa.eu/en/rss-feed/eurd/feed',
    'EMA - Events': 'https://www.ema.europa.eu/en/rss-feed/event/feed',
    'EMA - Maximum residue limits': 'https://www.ema.europa.eu/en/rss-feed/mrl/feed',
    'EMA - Paediatric investigation plans': 'https://www.ema.europa.eu/en/rss-feed/pip/feed',
    'EMA - Press releases': 'https://www.ema.europa.eu/en/rss-feed/press_release/feed',
    'EMA - Orphan designations': 'https://www.ema.europa.eu/en/rss-feed/orphan_designation/feed',
    'EMA - Regulatory guidelines': 'https://www.ema.europa.eu/en/rss-feed/regulatory_procedural_guideline/feed',
    'EMA - Scientific guidelines': 'https://www.ema.europa.eu/en/rss-feed/scientific_guideline/feed',
    'EMA - Withdrawn applications': 'https://www.ema.europa.eu/en/rss-feed/withdrawn_applications/feed',
    'EMA - Whats new': 'https://www.ema.europa.eu/en/rss-feed/whatsnew/feed',
    'EMA - Herbal medicines': 'https://www.ema.europa.eu/en/rss-feed/herbal/feed',
    'EMA - New human medicines': 'https://www.ema.europa.eu/en/rss-feed/medicine_human_new/feed',
    'EMA - New veterinary medicines': 'https://www.ema.europa.eu/en/rss-feed/medicine_vet_new/feed',
    'EMA - Human EPARs': 'https://www.ema.europa.eu/en/rss-feed/medicine_human_epar/feed',
    'EMA - Veterinary EPARs': 'https://www.ema.europa.eu/en/rss-feed/medicine_vet_epar/feed',
    'EMA - Fees': 'https://www.ema.europa.eu/en/rss-feed/fee/feed',
    'EMA - Inspections': 'https://www.ema.europa.eu/en/rss-feed/inspection/feed',
    'EMA - Human Medicines Highlights': 'https://www.ema.europa.eu/en/news-events/publications/newsletters',
    'EMA - Procurement': 'https://www.ema.europa.eu/en/rss-feed/procurement/feed',
    
    # FDA RSS Feeds
    'FDA - Agency Updates': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-news-items/rss.xml',
    'FDA - Consumer Health Info': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/consumer-health-information/rss.xml',
    'FDA - Criminal Investigations': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/criminal-investigations/rss.xml',
    'FDA - DISCO Oncology': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-information-soundcast-clinical-oncology-d-i-s-c-o/rss.xml',
    'FDA - Drug Safety Podcasts': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-safety-podcasts/rss.xml',
    'FDA - Outbreaks': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-outbreaks/rss.xml',
    'FDA - Food Allergies': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/food-allergies/rss.xml',
    'FDA - Food Safety Recalls': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/food-safety-recalls/rss.xml',
    'FDA - Health Fraud': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/health-fraud/rss.xml',
    'FDA - MedWatch Alerts': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch-safety-alerts/rss.xml',
    'FDA - ORA FOIA': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/ora-foia-electronic-reading-room/rss.xml',
    'FDA - Press Releases': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml',
    'FDA - Q&A': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/questions-and-answers/rss.xml',
    'FDA - Recalls': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml',
    'FDA - Tainted Supplements': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/tainted-products-marketed-dietary-supplements-potentially-hazardous-products/rss.xml',
    'FDA - Drugs Whats New': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs-news-events/rss.xml',
    'FDA - Vaccines Whats New': 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/biologics-blood-vaccines-news-events/rss.xml',
    
    # NewsAPI Integration (processed separately)
    'NewsAPI_Pharma': 'newsapi:pharmaceutical',
    'NewsAPI_BioTech': 'newsapi:biotechnology',
    'NewsAPI_Clinical': 'newsapi:clinical trials',
    'NewsAPI_Drug': 'newsapi:drug development'
}

# Example topics (replace with your actual topics)
TOPICS = {
    'Clinical Trials': ['clinical trial', 'patient enrollment', 'phase 1', 'phase 2', 'phase 3', 'phase 4', 
                       'clinical study', 'study results', 'patient recruitment', 'cohort', 'double-blind',
                       'placebo-controlled', 'randomized', 'clinical investigation', 'study protocol',
                       'principal investigator', 'human subjects', 'investigator', 'trial design'],
    
    'Drug Development': ['drug development', 'pipeline', 'new drug', 'candidate', 'preclinical', 'discovery',
                         'ind', 'investigational new drug', 'nct', 'lead compound', 'nda', 'biologics',
                         'drug design', 'compound', 'therapeutic', 'formulation', 'dosage form',
                         'pharmacology', 'toxicology', 'absorption', 'distribution', 'metabolism', 'excretion',
                         'pharmacokinetics', 'pharmacodynamics', 'drug delivery', 'biopharmaceutics'],
    
    'Regulatory': ['fda', 'ema', 'approval', 'regulation', 'compliance', 'regulatory', 'label', 'submission',
                  'chmp', 'prac', 'guidance', 'guideline', 'inspection', 'gmp', 'cgmp', 'pdco', 'chmp positive opinion',
                  'scientific advice', 'marketing authorization', 'maa', 'nda', 'bla', 'anda', 'supplemental',
                  'labeling', 'orphan designation', 'breakthrough therapy', 'accelerated approval', 'priority review',
                  'fast track', 'expedited', 'special protocol assessment', 'reference product', 'post-approval',
                  'post-marketing', 'pdufa', 'gdufa', 'rems', 'risk evaluation', 'dossier', 'briefing document'],
    
    'Manufacturing': ['manufacturing', 'production', 'supply chain', 'gmp', 'quality control', 'facility',
                     'batch', 'scale-up', 'process development', 'validation', 'qualification', 'raw materials',
                     'excipient', 'continuous manufacturing', 'quality by design', 'qbd', 'tech transfer',
                     'cmc', 'chemistry manufacturing controls', 'api', 'active pharmaceutical ingredient',
                     'drug product', 'drug substance', 'fill finish', 'lyophilization', 'packaging', 'stability',
                     'shelf life', 'expiration', 'release testing', 'specification', 'deviation', 'cleanroom'],
    
    'Market': ['market', 'revenue', 'sales', 'commercial', 'launch', 'profit', 'forecast', 
               'market share', 'competition', 'pricing', 'reimbursement', 'payer', 'formulary',
               'health economics', 'market access', 'growth', 'opportunity', 'expansion',
               'commercial strategy', 'positioning', 'branding', 'market research', 'marketing',
               'promotion', 'competitive landscape', 'market analysis', 'market trend', 'demand'],
    
    'R&D': ['research', 'innovation', 'technology', 'breakthrough', 'science', 'novel', 'mechanism',
           'target', 'receptor', 'pathway', 'molecular', 'cellular', 'in vitro', 'in vivo', 'assay',
           'biomarker', 'genetic', 'genomic', 'proteomic', 'bioinformatics', 'computational',
           'platform technology', 'screening', 'lead optimization', 'structure-activity relationship',
           'moa', 'mechanism of action', 'translational research', 'bench to bedside'],
    
    'Business': ['merger', 'acquisition', 'partnership', 'collaboration', 'deal', 'investment', 'startup',
                'joint venture', 'license', 'alliance', 'equity', 'financing', 'venture capital', 'ipo',
                'funding', 'series a', 'series b', 'transaction', 'valuation', 'milestone payment',
                'royalty', 'strategy', 'corporate development', 'business development', 'divestiture',
                'spin-off', 'restructuring', 'reorganization', 'board of directors', 'ceo', 'executive'],
    
    'Patents': ['patent', 'intellectual property', 'exclusivity', 'litigation', 'generic', 'biosimilar',
               'patent expiry', 'patent cliff', 'patent term', 'extension', 'paragraph iv', 'challenge',
               'invalidation', 'obviousness', 'novelty', 'prior art', 'claims', 'prosecution', 'patentability',
               'ip strategy', 'freedom to operate', 'patent portfolio', 'trade secret', 'licensing'],
    
    'Medical': ['treatment', 'disease', 'patient', 'efficacy', 'safety', 'side effect', 'therapeutic',
               'indication', 'diagnosis', 'prognosis', 'symptom', 'disorder', 'syndrome', 'condition',
               'chronic', 'acute', 'comorbidity', 'standard of care', 'guideline', 'protocol',
               'intervention', 'therapy', 'medication', 'adverse event', 'benefit-risk', 'pharmacotherapy'],
               
    'Safety': ['safety', 'adverse event', 'adverse reaction', 'side effect', 'toxicity', 'risk',
              'contraindication', 'black box warning', 'boxed warning', 'post-marketing surveillance',
              'pharmacovigilance', 'safety signal', 'adr', 'serious adverse event', 'sae', 'susar',
              'periodic safety update', 'psur', 'recall', 'withdrawal', 'safety alert', 'risk management',
              'drug-drug interaction', 'overdose', 'safety profile', 'warning', 'precaution', 'pregnancy category']
}

# Configuration
CACHE_FILE = "crawler_cache.json" 
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "articles.csv")
MAX_AGE_DAYS = 7
RATE_LIMIT = 2.0  # requests per second per domain


async def match_topic(content: str, source: str = "") -> Optional[str]:
    """
    Match content against topics and keywords with special handling for FDA/EMA sources
    
    Args:
        content: The content to match against topics
        source: Optional source name to help with matching
        
    Returns:
        Matched topic name or None
    """
    content = content.lower()
    
    # Special handling for FDA feeds
    if "fda" in source.lower():
        # Default to Regulatory for FDA sources if no specific match
        default_topic = "Regulatory"
        
        # Check for specific FDA-related topics first
        if any(term in content for term in ["approval", "approved", "label", "labeling", "nda", "anda", "bla"]):
            return "Regulatory"
        if any(term in content for term in ["recall", "safety alert", "warning letter", "adverse event"]):
            return "Safety"
        if any(term in content for term in ["clinical trial", "phase", "study results"]):
            return "Clinical Trials"
        if any(term in content for term in ["guidance", "guideline", "draft guidance"]):
            return "Regulatory"
    
    # Special handling for EMA feeds
    elif "ema" in source.lower() or "europe" in source.lower():
        # Default to Regulatory for EMA sources if no specific match
        default_topic = "Regulatory"
        
        # Check for specific EMA-related topics first
        if any(term in content for term in ["approval", "authorisation", "authorization", "chmp", "committee"]):
            return "Regulatory"
        if any(term in content for term in ["pharmacovigilance", "safety", "risk", "prac"]):
            return "Safety"
        if any(term in content for term in ["orphan", "rare disease", "designation"]):
            return "R&D"
        if any(term in content for term in ["paediatric", "pediatric", "pdco"]):
            return "Clinical Trials"
    else:
        default_topic = None
    
    # General matching for all sources
    for topic, keywords in TOPICS.items():
        for keyword in keywords:
            if f" {keyword.lower()} " in f" {content} ":
                return topic
    
    # If no match but source is FDA/EMA, return the default topic
    return default_topic


async def fetch_from_newsapi(query: str, session: Optional[aiohttp.ClientSession] = None) -> List[Dict]:
    """
    Fetch articles from NewsAPI with the given query
    
    Args:
        query: The search query for NewsAPI
        session: Optional aiohttp session to reuse
        
    Returns:
        List of articles in a format compatible with the RSS feed processing
    """
    if not NEWSAPI_ENABLED or not NEWSAPI_KEY:
        logging.warning("NewsAPI is not enabled or missing API key")
        return []
    
    try:
        # Calculate date range for the query
        to_date = datetime.now()
        from_date = to_date - timedelta(days=NEWSAPI_DAYS_BACK)
        
        # Format dates for NewsAPI
        to_date_str = to_date.strftime("%Y-%m-%d")
        from_date_str = from_date.strftime("%Y-%m-%d")
        
        # Construct the URL with parameters
        params = {
            "q": query,
            "from": from_date_str,
            "to": to_date_str,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": NEWSAPI_MAX_RESULTS,
            "apiKey": NEWSAPI_KEY
        }
        
        # Construct the complete URL
        url = f"{NEWSAPI_BASE_URL}/everything?{urllib.parse.urlencode(params)}"
        
        # Use provided session or create a new one if necessary
        session_provided = session is not None
        if not session_provided:
            session = aiohttp.ClientSession()
        
        try:
            logging.info(f"Requesting NewsAPI with query: {query}")
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"NewsAPI error {response.status}: {error_text}")
                    return []
                
                # Parse the JSON response
                data = await response.json()
                
                if data.get("status") != "ok":
                    logging.error(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                    return []
                
                # Extract and format the articles
                articles = []
                for article in data.get("articles", []):
                    processed_article = {
                        "title": article.get("title", "No title"),
                        "link": article.get("url"),
                        "description": article.get("description", "") or article.get("content", "No description available"),
                        "pub_date": article.get("publishedAt"),
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "image_url": article.get("urlToImage"),
                        "content": article.get("content", ""),
                        "newsapi_data": True  # Mark this as coming from NewsAPI
                    }
                    articles.append(processed_article)
                
                logging.info(f"Retrieved {len(articles)} articles from NewsAPI for query: {query}")
                return articles
        finally:
            # Close the session if we created it
            if not session_provided:
                await session.close()
    
    except Exception as e:
        logging.error(f"Error fetching from NewsAPI: {e}")
        return []


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
    
    # Create a shared aiohttp session that will be passed to both the crawler and NewsAPI
    connector = aiohttp.TCPConnector(limit=10, ssl=False)  # Limit concurrent connections
    shared_session = aiohttp.ClientSession(connector=connector)
    
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
            # Override the crawler's session with our shared session
            if crawler.session:
                await crawler.session.close()
            crawler.session = shared_session
            
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
                    # Check if this is a NewsAPI special feed
                    if url.startswith('newsapi:'):
                        logging.info(f"Processing NewsAPI feed: {source} with query: {url[8:]}")
                        query = url[8:]  # Extract query part after 'newsapi:'
                        news_api_articles = await fetch_from_newsapi(query, shared_session)
                        
                        if news_api_articles:
                            successful_feeds += 1
                            
                            # Process NewsAPI articles
                            for article in news_api_articles:
                                article['source'] = f"{source} ({article.get('source', 'Unknown')})"
                                
                                # Match topics based on title and description
                                content_for_matching = f"{article.get('title', '')} {article.get('description', '')}"
                                topic = await match_topic(content_for_matching, source)
                                
                                if topic:
                                    article['topic'] = topic
                                    matched_articles += 1
                                    all_articles.append(article)
                                
                                total_articles += 1
                        else:
                            failed_feeds += 1
                            logging.warning(f"No articles found in NewsAPI feed: {source}")
                    else:
                        # Normal RSS feed processing
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
                                topic = await match_topic(content_for_matching, source)
                                
                                if topic:
                                    article['topic'] = topic
                                    matched_articles += 1
                                    
                                    # Add specific metadata for FDA and EMA feeds
                                    if "fda" in source.lower():
                                        article['regulatory_agency'] = "FDA"
                                        # Try to extract document number (e.g., FDA-2023-N-0123)
                                        doc_match = re.search(r'FDA-\d{4}-[A-Z]-\d+', article.get('description', ''))
                                        if doc_match:
                                            article['document_id'] = doc_match.group(0)
                                    
                                    elif "ema" in source.lower() or "europe" in source.lower():
                                        article['regulatory_agency'] = "EMA"
                                        # Try to extract EMA document numbers (e.g., EMEA/H/C/123456)
                                        doc_match = re.search(r'EMEA/H/C/\d+|EMA/\d+/\d+', article.get('description', ''))
                                        if doc_match:
                                            article['document_id'] = doc_match.group(0)
                                    
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
                                            
                                            # Set fetch priority higher for specific sources
                                            fetch_priority = "high" if any(agency in source.lower() for agency in ["fda", "ema"]) else "normal"
                                            
                                            # Fetch and extract content
                                            result = await crawler.fetch_url(article['link'])
                                            
                                            if result.get('success'):
                                                # Add extracted content and metadata to article
                                                # Use the potentially longer content if available
                                                if result.get('content'):
                                                    article['content'] = result.get('content')
                                                    # Set the flag based on crawler result
                                                    article['has_full_content'] = result.get('has_full_content', False)
                                                else:
                                                    # If crawler didn't get content, use original description and mark as not full
                                                    article['content'] = article.get('description', '') # Fallback
                                                    article['has_full_content'] = False
                                                    # Set summary from description
                                                    article['summary'] = article.get('description', '')[:280] + '...'

                                                article['excerpt'] = result.get('excerpt', article.get('description', '')[:280] + '...') # Use excerpt or generate from original desc
                                                article['image_url'] = result.get('image_url', '')
                                                
                                                # Use excerpt for summary field
                                                article['summary'] = article['excerpt']

                                                # Use more accurate publication date if available
                                                if result.get('pub_date'):
                                                    article['pub_date'] = result.get('pub_date')

                                                # Calculate reading time if content is available
                                                content_to_measure = article.get('content', '')
                                                if content_to_measure:
                                                    # Estimate reading time: average 200 words per minute
                                                    word_count = len(content_to_measure.split())
                                                    article['reading_time'] = max(1, round(word_count / 200))
                                                    article['word_count'] = word_count
                                            else:
                                                 # If fetch failed, ensure flag is false
                                                 article['has_full_content'] = False
                                                 # Keep original description as content
                                                 article['content'] = article.get('description', '')
                                                 # Set summary from description
                                                 article['summary'] = article.get('description', '')[:280] + '...'

                                        except Exception as e:
                                            logging.error(f"Error fetching content for {article['link']}: {e}")
                                            article['has_full_content'] = False # Ensure flag is false on error
                                            # Keep original description as content
                                            article['content'] = article.get('description', '')
                                    
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
    finally:
        # Always close the shared session
        await shared_session.close()


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
        required_fields = ['title', 'link', 'source', 'pub_date', 'topic', 'description', 'content', 'summary']
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
    
    try:
        # Example of using the status callback
        success = await scrape_news_with_crawler(status_callback=print_status)
        
        if success:
            print("News scraping completed successfully!")
        else:
            print("News scraping failed.")
    except Exception as e:
        print(f"Error running scraper: {e}")


if __name__ == "__main__":
    # Use the asyncio.run pattern which properly creates and destroys the event loop
    asyncio.run(main()) 