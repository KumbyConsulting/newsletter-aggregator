from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta, timezone
import hashlib
import time
import json
import os
import asyncio
import re
import calendar
import pandas as pd
from unittest.mock import MagicMock

# Google Cloud imports
from google.cloud import storage
from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError

from .config_service import ConfigService
from utils.date_utils import normalize_datetime

class StorageException(Exception):
    """Custom exception for storage operations"""
    pass

class StorageService:
    """Simplified storage service using Firestore as the primary backend"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(StorageService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize storage service with proper error handling"""
        # Skip initialization if already initialized
        if self._initialized:
            return
            
        try:
            # Use the singleton pattern directly since ConfigService is already a singleton
            self.config = ConfigService()
            
            # Check if storage_backend attribute exists, otherwise use default
            storage_backend = 'chromadb'  # Default value
            if hasattr(self.config, 'storage_backend'):
                storage_backend = self.config.storage_backend
            else:
                logging.warning("ConfigService has no 'storage_backend' attribute, using 'chromadb' as default")
            
            # Initialize Firestore client only if it's configured as the storage backend
            if storage_backend == 'firestore':
                try:
                    # Attempt to initialize Firestore client
                    self.db = firestore.Client()
                    self.articles_collection = self.db.collection('articles')
                    logging.info("Firestore client initialized successfully.")
                    # Verify connection by attempting a simple read (optional but recommended)
                    # self.db.collection('__health_check__').limit(1).get() 
                    # logging.info("Firestore connection verified.")
                except (DefaultCredentialsError, Exception) as e:
                    logging.warning(f"Failed to initialize Firestore client: {e}. Falling back to '{self.config.storage_backend}' backend.")
                    # Fallback: Change backend setting for this instance and use mock Firestore
                    storage_backend = 'chromadb' # Change the local variable for subsequent logic
                    self.config.storage_backend = 'chromadb' # Change the config instance's setting
                    
                    from unittest.mock import MagicMock
                    self.db = MagicMock()
                    self.articles_collection = self.db.collection('articles')
                    logging.warning(f"Using '{storage_backend}' as storage backend due to Firestore initialization failure.")
                    # If in production and fallback occurs, maybe raise a specific alert/error?
                    # if self.config.is_production:
                    #     logging.error("Firestore fallback occurred in production environment!")
            
            # If storage_backend is not firestore initially or after fallback
            if storage_backend != 'firestore':
                # Use mock for Firestore when using another storage backend
                from unittest.mock import MagicMock
                if not hasattr(self, 'db'): # Ensure db is initialized if fallback didn't run
                    self.db = MagicMock()
                if not hasattr(self, 'articles_collection'): # Ensure collection is initialized
                     self.articles_collection = self.db.collection('articles')
                logging.info(f"Using {storage_backend} as primary storage backend. Mocking Firestore client.")
            
            # Initialize GCS if enabled
            self.backup_bucket = None
            self.storage_client = None
            
            use_gcs_backup = False
            gcs_bucket_name = None
            
            # Check if GCS backup attributes exist
            if hasattr(self.config, 'use_gcs_backup'):
                use_gcs_backup = self.config.use_gcs_backup
            else:
                logging.warning("ConfigService has no 'use_gcs_backup' attribute, using False as default")
                
            if hasattr(self.config, 'gcs_bucket_name'):
                gcs_bucket_name = self.config.gcs_bucket_name
            
            if use_gcs_backup and gcs_bucket_name:
                try:
                    self.storage_client = storage.Client()
                    self.backup_bucket = self.storage_client.bucket(gcs_bucket_name)
                    logging.info(f"Connected to GCS bucket: {gcs_bucket_name}")
                except Exception as e:
                    logging.warning(f"Failed to initialize GCS: {e}. Continuing without GCS backup.")
                    # Only raise in production
                    if hasattr(self.config, 'is_production') and self.config.is_production:
                        raise
                
            logging.info("StorageService initialized successfully")
            self._initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize StorageService: {e}")
            raise StorageException(f"Storage initialization failed: {str(e)}")
            
    def _generate_article_id(self, article: Dict) -> str:
        """Generate a unique ID for an article"""
        try:
            title_part = article['title'][:50].strip().lower()
            date_part = article.get('pub_date', datetime.now().isoformat())
            link_part = article.get('link', '')[:30]
            title_part = "".join(c for c in title_part if c.isalnum() or c.isspace())
            title_part = title_part.replace(" ", "_")
            link_part = "".join(c for c in link_part if c.isalnum() or c in ['/', '.', '-', '_'])
            try:
                parsed_date = normalize_datetime(date_part)
                date_part = parsed_date.strftime('%Y-%m-%d_%H-%M-%S') if parsed_date else datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            except Exception:
                date_part = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            content_hash = hashlib.md5(f"{title_part}{date_part}{link_part}".encode()).hexdigest()[:8]
            return f"{content_hash}_{date_part}"
        except Exception as e:
            logging.error(f"Error generating article ID: {e}")
            return f"invalid_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            
    def _categorize_segment(self, topic: str) -> str:
        """Categorize article into industry segments"""
        segments = {
            'regulatory': ['483 Notifications', 'Regulatory Affairs', 'FDA approval'],
            'research': ['Clinical Trials', 'Drug Development', 'AI in Medicine'],
            'manufacturing': ['Contamination Control', 'GMP', 'Manufacturing'],
            'business': ['Mergers and Acquisitions', 'Market Trends'],
            'innovation': ['Biotechnology Innovations', 'Digital Health']
        }
        
        for segment, topics in segments.items():
            if any(t.lower() in topic.lower() for t in topics):
                return segment
        return 'general'
        
    def store_article(self, article: Dict) -> bool:
        """Store a single article with enhanced error handling and validation"""
        try:
            # Validate required fields
            required_fields = ['title', 'link', 'pub_date']
            missing_fields = [field for field in required_fields if not article.get(field)]
            if missing_fields:
                raise StorageException(f"Missing required fields: {', '.join(missing_fields)}")

            # Generate unique ID
            article_id = self._generate_article_id(article)
            
            # Check for existing article
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            
            if doc.exists:
                logging.info(f"Article already exists: {article_id}")
                return False
                
            # Prepare article data with validation
            article_data = {
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'pub_date': article.get('pub_date', ''),
                'topic': article.get('topic', ''),
                'source': article.get('source', ''),
                'description': article.get('description', ''),
                'summary': article.get('summary', ''),
                'document': article.get('document', ''),
                'last_updated': datetime.now().isoformat()
            }
            
            # Store article with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    doc_ref.set(article_data)
                    logging.info(f"Successfully stored article: {article_id}")
                    return True
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Retry {attempt + 1}/{max_retries} for storing article {article_id}")
                    time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error storing article: {e}")
            raise StorageException(f"Failed to store article: {str(e)}")
            
    def batch_store_articles(self, articles: List[Dict]) -> bool:
        """Store multiple articles with enhanced error handling and duplicate detection"""
        try:
            if not articles:
                logging.warning("No articles to store")
                return False
                
            # Track unique links and content hashes to prevent duplicates
            unique_links = set()
            content_hashes = set()
            
            # First, collect all articles and check for duplicates outside transaction
            articles_to_store = []
            skipped_count = 0
            
            # Pre-check existence of all articles
            existing_docs = {}
            article_ids = []
            
            for article in articles:
                try:
                    # Validate article
                    if not article.get('link'):
                        logging.warning(f"Skipping article without link: {article.get('title', 'Unknown')}")
                        continue
                        
                    # Generate content hash for similarity checking
                    content = f"{article.get('title', '')} {article.get('description', '')} {article.get('summary', '')}"
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # Check for duplicate link
                    if article['link'] in unique_links:
                        logging.info(f"Skipping duplicate article by link: {article['link']}")
                        skipped_count += 1
                        continue
                        
                    # Check for similar content
                    is_similar = False
                    topic = article.get('topic', '').lower()
                    similarity_threshold = 0.85 if topic in ['general', 'market trends'] else 0.92
                    
                    for existing_hash in content_hashes:
                        similarity = self._calculate_similarity(content_hash, existing_hash)
                        if similarity > similarity_threshold:
                            logging.info(f"Skipping similar article: {article.get('title', 'Unknown')}")
                            is_similar = True
                            skipped_count += 1
                            break
                    
                    if is_similar:
                        continue
                        
                    # Generate unique ID
                    article_id = self._generate_article_id(article)
                    article_ids.append(article_id)
                        
                    # Prepare article data
                    article_data = {
                        'title': article.get('title', ''),
                        'link': article.get('link', ''),
                        'pub_date': article.get('pub_date', ''),
                        'topic': article.get('topic', ''),
                        'source': article.get('source', ''),
                        'description': article.get('description', ''),
                        'summary': article.get('summary', ''),
                        'document': article.get('document', ''),
                        'last_updated': datetime.now().isoformat(),
                        'content_hash': content_hash,
                        'created_at': firestore.SERVER_TIMESTAMP,
                    }
                    
                    articles_to_store.append((article_id, article_data))
                    unique_links.add(article['link'])
                    content_hashes.add(content_hash)
                    
                except Exception as e:
                    logging.error(f"Error processing article: {e}")
                    continue
            
            # Batch check existence of all articles
            batch_size = 20
            for i in range(0, len(article_ids), batch_size):
                batch_ids = article_ids[i:i + batch_size]
                docs = [self.articles_collection.document(id_).get() for id_ in batch_ids]
                for doc in docs:
                    if doc.exists:
                        existing_docs[doc.id] = True
            
            # Filter out existing articles
            filtered_articles = [(id_, data) for id_, data in articles_to_store if id_ not in existing_docs]
            
            # Store articles in batches
            max_batch_size = 20
            stored_count = 0
            
            for i in range(0, len(filtered_articles), max_batch_size):
                batch = self.db.batch()
                batch_slice = filtered_articles[i:i + max_batch_size]
                
                # Add writes to batch
                for article_id, article_data in batch_slice:
                    doc_ref = self.articles_collection.document(article_id)
                    batch.set(doc_ref, article_data)
                
                # Commit batch with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch.commit()
                        stored_count += len(batch_slice)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to commit batch after {max_retries} attempts: {e}")
                            raise
                        # Use a transaction for this batch
                        transaction = self.db.transaction()
                        
                        @firestore.transactional
                        def commit_batch(transaction, batch_data):
                            for doc_ref, article_data in batch_data:
                                # Check again within transaction to ensure consistency
                                snapshot = doc_ref.get(transaction=transaction)
                                if not snapshot.exists:
                                    transaction.set(doc_ref, article_data)
                                    
                        # Execute the transaction
                        commit_batch(transaction, batch_slice)
                        stored_count += len(batch_slice)
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to commit transaction after {max_retries} attempts: {e}")
                            raise
                        logging.warning(f"Retry {attempt + 1}/{max_retries} for transaction")
                        time.sleep(1)
                        
            # Report results
            logging.info(f"Successfully stored {stored_count} articles, skipped {skipped_count} duplicates")
            
            # Run cleanup periodically rather than after every batch
            # Only run cleanup if we stored a significant number of articles
            if stored_count > 10:
                try:
                    asyncio.get_event_loop().run_until_complete(self.cleanup_duplicate_articles())
                except Exception as cleanup_error:
                    logging.error(f"Error during cleanup: {cleanup_error}")
                    # Don't fail the overall operation if cleanup fails
                    
            return stored_count > 0
            
        except Exception as e:
            logging.error(f"Error in batch store: {e}")
            raise StorageException(f"Batch store failed: {str(e)}")

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two content hashes with improved algorithm"""
        try:
            # Enhanced similarity calculation combining Hamming distance and Jaccard index
            # Convert hex hashes to binary strings
            bin1 = bin(int(hash1, 16))[2:].zfill(128)
            bin2 = bin(int(hash2, 16))[2:].zfill(128)
            
            # Calculate Hamming distance (bit-level differences)
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            hamming_similarity = 1 - (hamming_distance / 128)
            
            # Calculate Jaccard similarity (set intersection/union)
            # Convert binary strings to sets of 4-bit chunks for more nuanced comparison
            chunks1 = {bin1[i:i+4] for i in range(0, len(bin1), 4)}
            chunks2 = {bin2[i:i+4] for i in range(0, len(bin2), 4)}
            
            intersection = len(chunks1.intersection(chunks2))
            union = len(chunks1.union(chunks2))
            jaccard_similarity = intersection / union if union > 0 else 0
            
            # Weighted combination (emphasize Jaccard for better content comparison)
            return (0.4 * hamming_similarity) + (0.6 * jaccard_similarity)
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def query_articles(self, query: str, n_results: int = 5, exclude_ids: List[str] = None) -> List[Dict]:
        """Query articles using Firestore with improved matching for vague queries
        
        Args:
            query: The search query string
            n_results: Maximum number of results to return
            exclude_ids: List of article IDs to exclude from results
            
        Returns:
            List of matching articles
        """
        try:
            # Simple keyword search implementation
            query_terms = query.lower().split()
            
            # For empty or very short queries, return recent articles instead
            if not query_terms or (len(query_terms) == 1 and len(query_terms[0]) <= 3):
                logging.info(f"Query '{query}' too vague, returning recent articles instead")
                return self.get_recent_articles(limit=n_results)
                
            # Search by keywords in title, topic, and document
            docs = self.articles_collection.limit(100).stream()  # Increase limit for better coverage
            
            results = []
            for doc in docs:
                # Skip if article ID is in exclude list
                if exclude_ids and doc.id in exclude_ids:
                    continue
                    
                data = doc.to_dict()
                title = data.get('title', '').lower()
                topic = data.get('topic', '').lower()
                document = data.get('document', '').lower()
                description = data.get('description', '').lower()
                
                # Calculate relevance score
                score = 0
                
                # Check for exact phrase match first (higher priority)
                if query.lower() in title:
                    score += 10
                if query.lower() in topic:
                    score += 8
                if query.lower() in document:
                    score += 5
                if query.lower() in description:
                    score += 6
                
                # Then check for individual term matches
                for term in query_terms:
                    # Skip very common words and short terms
                    if term in ['the', 'and', 'or', 'in', 'of', 'to', 'a', 'is', 'for'] or len(term) <= 2:
                        continue
                        
                    if term in title:
                        score += 3
                    if term in topic:
                        score += 2
                    if term in document:
                        score += 1
                    if term in description:
                        score += 1.5
                        
                # Only include if there's some relevance
                if score > 0:
                    results.append({
                        'id': doc.id,
                        'document': data.get('document', ''),
                        'metadata': {
                            'title': data.get('title', ''),
                            'link': data.get('link', ''),
                            'pub_date': data.get('pub_date', ''),
                            'topic': data.get('topic', ''),
                            'source': data.get('source', ''),
                            'description': data.get('description', ''),
                            'summary': data.get('summary', '')
                        },
                        'score': score
                    })
            
            # If no results with the standard approach, try fuzzy matching
            if not results:
                logging.info(f"No exact matches for '{query}', trying fuzzy matching")
                for doc in docs:
                    data = doc.to_dict()
                    title = data.get('title', '')
                    
                    # Simple character-level fuzzy matching
                    max_fuzzy_score = 0
                    for term in query_terms:
                        if len(term) <= 3:
                            continue
                            
                        # Check if term is a substring of any word in title
                        title_words = title.lower().split()
                        for word in title_words:
                            if term in word:
                                max_fuzzy_score = max(max_fuzzy_score, 2)
                                break
                    
                    if max_fuzzy_score > 0:
                        results.append({
                            'id': doc.id,
                            'document': data.get('document', ''),
                            'metadata': {
                                'title': data.get('title', ''),
                                'link': data.get('link', ''),
                                'pub_date': data.get('pub_date', ''),
                                'topic': data.get('topic', ''),
                                'source': data.get('source', ''),
                                'description': data.get('description', ''),
                                'summary': data.get('summary', '')
                            },
                            'score': max_fuzzy_score
                        })
            
            # If still no results, return most recent articles as fallback
            if not results:
                logging.warning(f"No matches found for '{query}', falling back to recent articles")
                return self.get_recent_articles(limit=n_results)
            
            # Sort by score (higher is better) and limit results
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Log the results for debugging
            logging.info(f"Found {len(results)} matches for query '{query}', returning top {min(n_results, len(results))}")
            
            return results[:n_results]
            
        except Exception as e:
            logging.error(f"Error querying articles from Firestore: {e}")
            # Return recent articles on error as a fallback strategy
            logging.info(f"Falling back to recent articles due to search error")
            try:
                return self.get_recent_articles(limit=n_results)
            except Exception as fallback_error:
                logging.error(f"Error getting recent articles as fallback: {fallback_error}")
                return []
            
    async def get_recent_articles(self, limit: int = 10, topic: str = None) -> List[Dict]:
        """Get recent articles, optionally filtered by topic"""
        try:
            loop = asyncio.get_event_loop()
            query = self.articles_collection.order_by('pub_date', direction=firestore.Query.DESCENDING)
            if topic:
                query = query.where('topic', '==', topic)
            query = query.limit(limit)
            docs = await loop.run_in_executor(None, lambda: list(query.stream()))
            results = []
            for doc in docs:
                data = doc.to_dict()
                results.append({
                    'id': doc.id,
                    'document': data.get('document', ''),
                    'metadata': {
                        'title': data.get('title', ''),
                        'link': data.get('link', ''),
                        'pub_date': data.get('pub_date', ''),
                        'topic': data.get('topic', ''),
                        'source': data.get('source', ''),
                        'description': data.get('description', ''),
                        'summary': data.get('summary', '')
                    }
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Error getting recent articles: {e}")
            return []
            
    def get_similar_articles(self, article_id: str, n_results: int = 3) -> List[Dict]:
        """Find articles similar to a given article"""
        try:
            # Get the original article
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            if not doc.exists:
                return []
                
            article_data = doc.to_dict()
            
            # Get topic and keywords from the article
            topic = article_data.get('topic', '')
            title_words = set(article_data.get('title', '').lower().split())
            
            # Find articles with the same topic
            similar_articles = []
            
            # First, try to find articles with the same topic
            topic_docs = self.articles_collection.where('topic', '==', topic).limit(20).stream()
            
            for similar_doc in topic_docs:
                if similar_doc.id != article_id:
                    data = similar_doc.to_dict()
                    similar_title_words = set(data.get('title', '').lower().split())
                    
                    # Calculate simple similarity score based on common words
                    common_words = len(title_words.intersection(similar_title_words))
                    similarity_score = common_words / max(1, len(title_words.union(similar_title_words)))
                    
                    similar_articles.append({
                        'id': similar_doc.id,
                        'document': data.get('document', ''),
                        'metadata': {
                            'title': data.get('title', ''),
                            'link': data.get('link', ''),
                            'pub_date': data.get('pub_date', ''),
                            'topic': data.get('topic', ''),
                            'source': data.get('source', ''),
                            'description': data.get('description', ''),
                            'summary': data.get('summary', '')
                        },
                        'similarity_score': similarity_score
                    })
            
            # Sort by similarity and return top results
            similar_articles.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return similar_articles[:n_results]
            
        except Exception as e:
            logging.error(f"Error finding similar articles in Firestore: {e}")
            return []
            
    async def delete_article(self, article_id: str) -> bool:
        """Delete an article from Firestore by ID
        
        Args:
            article_id: The ID of the article to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            doc_ref = self.articles_collection.document(article_id)
            doc = await loop.run_in_executor(None, doc_ref.get)
            
            if not doc.exists:
                logging.warning(f"Article {article_id} not found for deletion")
                return False
                
            # Delete the document
            await loop.run_in_executor(None, doc_ref.delete)
            logging.info(f"Successfully deleted article {article_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting article {article_id}: {e}")
            return False

    async def get_article(self, article_id: str) -> Optional[Dict]:
        """Get a single article by ID
        
        Args:
            article_id: The ID of the article to retrieve
            
        Returns:
            Dict containing the article data if found, None otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            doc_ref = self.articles_collection.document(article_id)
            doc = await loop.run_in_executor(None, doc_ref.get)
            
            if not doc.exists:
                logging.warning(f"Article {article_id} not found")
                return None
                
            # Get the data and add the ID
            data = doc.to_dict()
            
            # Format the response to match the structure used elsewhere
            return {
                'id': doc.id,
                'document': data.get('document', ''),
                'metadata': {
                    'title': data.get('title', ''),
                    'link': data.get('link', ''),
                    'pub_date': data.get('pub_date', ''),
                    'topic': data.get('topic', ''),
                    'source': data.get('source', ''),
                    'description': data.get('description', ''),
                    'summary': data.get('summary', '')
                }
            }
            
        except Exception as e:
            logging.error(f"Error retrieving article {article_id}: {e}")
            return None

    def update_article_metadata(self, article_id: str, metadata: Dict) -> bool:
        """Update article metadata fields
        
        Args:
            article_id: ID of the article to update
            metadata: Dictionary of metadata fields to update
            
        Returns:
            bool: Success status
        """
        try:
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logging.warning(f"Cannot update metadata for non-existent article {article_id}")
                return False
                
            # Update metadata fields
            doc_ref.update(metadata)
            logging.info(f"Updated metadata for article {article_id}")
            return True
        except Exception as e:
            logging.error(f"Error updating article metadata: {e}")
            return False
    
    async def find_articles_by_url(self, url: str) -> List[Dict]:
        """Find articles by URL
        
        Args:
            url: URL to search for
            
        Returns:
            List[Dict]: List of matching articles
        """
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(self.articles_collection.where('link', '==', url).stream()))
            articles = []
            
            for doc in docs:
                data = doc.to_dict()
                articles.append({
                    'id': doc.id,
                    'metadata': {
                        'title': data.get('title', ''),
                        'link': data.get('link', ''),
                        'pub_date': data.get('pub_date', ''),
                        'published_date': data.get('pub_date', '')
                    }
                })
            
            return articles
        except Exception as e:
            logging.error(f"Error finding articles by URL: {e}")
            return []

    async def get_all_articles(self) -> List[Dict]:
        """Get all articles from the database without pagination (async, robust date parsing)"""
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(self.articles_collection.stream()))
            articles = []
            for doc in docs:
                data = doc.to_dict()
                pub_date_raw = data.get('pub_date')
                try:
                    pub_date = self._parse_date(pub_date_raw)
                    formatted_pub_date = self._format_date(pub_date)
                except Exception as e:
                    logging.warning(f"Failed to parse pub_date '{pub_date_raw}' for article {doc.id}: {e}")
                    formatted_pub_date = pub_date_raw or "Unknown"
                articles.append({
                    'id': doc.id,
                    'document': data.get('document', ''),
                    'metadata': {
                        'title': data.get('title', 'Unknown Title'),
                        'description': data.get('description', 'No description available'),
                        'link': data.get('link', '#'),
                        'pub_date': formatted_pub_date,
                        'topic': data.get('topic', 'Uncategorized'),
                        'source': data.get('source', 'Unknown source'),
                        'summary': data.get('summary', None),
                        'image_url': data.get('image_url', ''),
                        'has_full_content': bool(data.get('document', '')),
                        'reading_time': data.get('reading_time', 0)
                    }
                })
            return articles
        except Exception as e:
            logging.error(f"Error getting all articles: {e}")
            return []

    async def enhanced_search(self, query: str, search_type: str = 'auto', **kwargs) -> List[Dict]:
        """Enhanced search with multiple strategies based on query type (async)."""
        # Analyze query to determine best search strategy if auto
        if search_type == 'auto':
            search_type = self._determine_search_strategy(query)

        if search_type == 'exact':
            return await self._exact_field_search(query, **kwargs)
        elif search_type == 'fuzzy':
            return await self._enhanced_fuzzy_search(query, **kwargs)
        else:
            return await self._semantic_search(query, **kwargs)

    def _determine_search_strategy(self, query: str) -> str:
        """Determine the best search strategy based on query characteristics"""
        # Check for exact phrase queries (quoted strings)
        if '"' in query or "'" in query:
            return 'exact'
            
        # Check for special operators
        if any(op in query for op in ['AND', 'OR', 'NOT']):
            return 'exact'
            
        # Check query length and complexity
        words = query.split()
        if len(words) == 1 and len(query) <= 3:
            return 'exact'  # Short, single-word queries
        elif len(words) > 3:
            return 'semantic'  # Longer, natural language queries
        else:
            return 'fuzzy'  # Default to fuzzy for medium complexity

    async def _exact_field_search(self, query: str, fields: List[str] = None, **kwargs) -> List[Dict]:
        """Perform exact field matching with index optimization (async)."""
        try:
            if not fields:
                fields = ['title', 'topic', 'source']  # Default searchable fields
            clean_query = query.strip('"\'').lower()
            base_query = self.articles_collection
            loop = asyncio.get_event_loop()
            results = []
            for field in fields:
                docs = await loop.run_in_executor(None, lambda: list(base_query.where(field, '==', clean_query).stream()))
                for doc in docs:
                    data = doc.to_dict()
                    results.append({
                        'id': doc.id,
                        'document': data.get('document', ''),
                        'metadata': self._format_article_metadata(data),
                        'match_type': 'exact',
                        'matched_field': field,
                        'score': 1.0
                    })
            return results
        except Exception as e:
            logging.error(f"Error in exact field search: {e}")
            return []

    async def _enhanced_fuzzy_search(self, query: str, threshold: float = 0.6, **kwargs) -> List[Dict]:
        """Improved fuzzy search with better scoring and matching (async)."""
        try:
            query_terms = self._normalize_text(query)
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(self.articles_collection.stream()))
            results = []
            for doc in docs:
                data = doc.to_dict()
                title_score = self._fuzzy_match_score(query_terms, self._normalize_text(data.get('title', '')))
                topic_score = self._fuzzy_match_score(query_terms, self._normalize_text(data.get('topic', '')))
                desc_score = self._fuzzy_match_score(query_terms, self._normalize_text(data.get('description', '')))
                total_score = (
                    title_score * 0.5 +
                    topic_score * 0.3 +
                    desc_score * 0.2
                )
                if total_score >= threshold:
                    results.append({
                        'id': doc.id,
                        'document': data.get('document', ''),
                        'metadata': self._format_article_metadata(data),
                        'match_type': 'fuzzy',
                        'score': total_score
                    })
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        except Exception as e:
            logging.error(f"Error in fuzzy search: {e}")
            return []

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for fuzzy matching"""
        # Convert to lowercase and split into words
        words = str(text).lower().split()
        # Remove common stop words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        return [word.strip('.,!?()[]{}') for word in words if word not in stop_words]

    def _fuzzy_match_score(self, query_terms: List[str], target_terms: List[str]) -> float:
        """Calculate fuzzy match score between query and target terms"""
        if not query_terms or not target_terms:
            return 0.0
            
        scores = []
        for q_term in query_terms:
            term_scores = []
            for t_term in target_terms:
                # Calculate Levenshtein distance
                distance = self._levenshtein_distance(q_term, t_term)
                max_len = max(len(q_term), len(t_term))
                similarity = 1 - (distance / max_len)
                term_scores.append(similarity)
            
            # Use best match for this query term
            scores.append(max(term_scores) if term_scores else 0)
            
        # Average the scores
        return sum(scores) / len(scores)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    def _format_article_metadata(self, data: Dict) -> Dict:
        """Format article metadata consistently"""
        return {
            'title': data.get('title', 'Unknown Title'),
            'description': data.get('description', 'No description available'),
            'link': data.get('link', '#'),
            'pub_date': self._format_date(self._parse_date(data.get('pub_date'))),
            'topic': data.get('topic', 'Uncategorized'),
            'source': data.get('source', 'Unknown source'),
            'summary': data.get('summary', None),
            'image_url': data.get('image_url', ''),
            'has_full_content': bool(data.get('document', '')),
            'reading_time': data.get('reading_time', 0)
        }

    async def _semantic_search(self, query: str, **kwargs) -> List[Dict]:
        """Placeholder for future semantic search implementation (async)."""
        logging.warning("Semantic search not yet implemented")
        return []

    def check_articles_exist(self, article_ids: List[str]) -> Dict[str, bool]:
        """
        Check if a list of article IDs exist in the Firestore collection.

        Args:
            article_ids: A list of article document IDs to check.

        Returns:
            A dictionary mapping each article ID to a boolean (True if exists, False otherwise).
        """
        if not article_ids:
            return {}

        results = {article_id: False for article_id in article_ids}
        try:
            # Firestore allows fetching multiple documents by ID efficiently
            doc_refs = [self.articles_collection.document(id) for id in article_ids]
            
            # Use get_all to fetch documents in batches (handles potential large lists)
            # Note: get_all might not be directly available on the client, depends on library version.
            # If get_all is not available, we might need to fetch in batches manually.
            # Let's assume get_all exists for now based on common patterns.
            # If it fails, we'll need to adjust.
            docs = []
            # Firestore client library usually requires get_all to be called within a transaction or batch
            # Let's try fetching directly first, might need adjustment
            
            # Let's use a simple loop for now, can optimize later if needed.
            for doc_ref in doc_refs:
                # Remove await, doc_ref.get() is synchronous
                doc = doc_ref.get()
                if doc.exists:
                    docs.append(doc)

            for doc in docs:
                if doc.exists:
                    results[doc.id] = True
                    
            return results
        except Exception as e:
            logging.error(f"Error checking article existence: {e}", exc_info=True)
            # On error, return the initial dict (all False) or raise an exception
            # Returning False might be safer for the frontend's current logic
            return results

    async def get_rag_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent RAG query history
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of history items (most recent first)
        """
        try:
            # Check if we have a history collection
            if not hasattr(self, 'db') or isinstance(self.db, MagicMock):
                logging.warning("get_rag_history called but no Firestore DB available")
                return []
                
            # Query the rag_history collection
            history_collection = self.db.collection('rag_history')
            query = history_collection.order_by('timestamp', direction='DESCENDING').limit(limit)
            
            # Get results
            docs = query.get()
            
            history = []
            for doc in docs:
                history_item = doc.to_dict()
                history_item['id'] = doc.id
                history.append(history_item)
                
            return history
            
        except Exception as e:
            logging.error(f"Error getting RAG history: {e}")
            return []

    # --- Methods for Saved Analyses ---

    async def save_analysis(self, analysis_data: Dict) -> str:
        """Save analysis results to the 'saved_analyses' collection."""
        try:
            if not self.db or isinstance(self.db, MagicMock):
                 raise StorageException("Firestore is not available.")
                 
            # Ensure required fields are present (adjust as needed)
            required = ['id', 'query', 'response', 'analysis_type', 'confidence', 'timestamp']
            if not all(field in analysis_data for field in required):
                raise StorageException("Missing required fields in analysis data.")

            doc_ref = self.db.collection('saved_analyses').document(analysis_data['id'])
            # Use set() which creates or overwrites
            await doc_ref.set({
                'id': analysis_data['id'],
                'query': analysis_data['query'],
                'response': analysis_data['response'],
                'analysis_type': analysis_data['analysis_type'],
                'confidence': analysis_data['confidence'],
                'timestamp': analysis_data['timestamp'],
                'sources': analysis_data.get('sources', []),
                'created_at': firestore.SERVER_TIMESTAMP # Add creation timestamp
            })
            logging.info(f"Successfully saved analysis: {analysis_data['id']}")
            return analysis_data['id']
        except Exception as e:
            logging.error(f"Database error saving analysis: {e}")
            raise StorageException(f"Failed to save analysis: {str(e)}")

    def get_saved_analyses(self, limit: int = 10, offset: int = 0, sort_by: str = 'timestamp', sort_order: str = 'desc') -> Dict:
        """Get a list of saved analyses with pagination and sorting."""
        try:
            if not self.db or isinstance(self.db, MagicMock):
                 raise StorageException("Firestore is not available.")

            # Validate sort parameters
            valid_sort_fields = ['timestamp', 'analysis_type', 'confidence', 'created_at']
            if sort_by not in valid_sort_fields:
                sort_by = 'created_at' # Default to creation time
            
            valid_sort_orders = ['asc', 'desc']
            if sort_order not in valid_sort_orders:
                sort_order = 'desc'

            # Query saved analyses
            query = self.db.collection('saved_analyses').order_by(
                sort_by,
                direction=firestore.Query.DESCENDING if sort_order == 'desc' else firestore.Query.ASCENDING
            ).limit(limit).offset(offset)

            # Execute query
            docs = query.get()

            # Format results
            analyses = []
            for doc in docs:
                analysis_data = doc.to_dict()
                # Ensure timestamps are handled correctly if they are Firestore Timestamps
                if 'created_at' in analysis_data and hasattr(analysis_data['created_at'], 'isoformat'):
                     analysis_data['created_at'] = analysis_data['created_at'].isoformat()
                if 'timestamp' in analysis_data and hasattr(analysis_data['timestamp'], 'isoformat'):
                     analysis_data['timestamp'] = analysis_data['timestamp'].isoformat()

                analyses.append({
                    'id': analysis_data.get('id'),
                    'query': analysis_data.get('query'),
                    'response': analysis_data.get('response'), # Consider adding truncation for list view
                    'analysis_type': analysis_data.get('analysis_type'),
                    'confidence': analysis_data.get('confidence'),
                    'timestamp': analysis_data.get('timestamp'),
                    'created_at': analysis_data.get('created_at'),
                    'sources_count': len(analysis_data.get('sources', [])) # Add source count instead of full sources
                })

            # Get total count efficiently
            # Note: Firestore count() aggregation might require specific setup/indexes
            # Using get() for total count as fallback
            total_query = self.db.collection('saved_analyses')
            # Remove await as get() is synchronous
            total_docs = total_query.get()
            total_count = len(total_docs)

            return {
                'analyses': analyses,
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'page': (offset // limit) + 1,
                'total_pages': (total_count + limit - 1) // limit
            }

        except Exception as e:
            logging.error(f"Error getting saved analyses: {e}")
            raise StorageException(f"Failed to retrieve saved analyses: {str(e)}")

    def get_saved_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get a specific saved analysis by ID."""
        try:
            if not self.db or isinstance(self.db, MagicMock):
                 raise StorageException("Firestore is not available.")

            doc_ref = self.db.collection('saved_analyses').document(analysis_id)
            # Remove await as get() is synchronous
            doc = doc_ref.get()

            if not doc.exists:
                logging.warning(f"Saved analysis not found: {analysis_id}")
                return None

            analysis_data = doc.to_dict()
            # Ensure timestamps are handled correctly if they are Firestore Timestamps
            if 'created_at' in analysis_data and hasattr(analysis_data['created_at'], 'isoformat'):
                 analysis_data['created_at'] = analysis_data['created_at'].isoformat()
            if 'timestamp' in analysis_data and hasattr(analysis_data['timestamp'], 'isoformat'):
                 analysis_data['timestamp'] = analysis_data['timestamp'].isoformat()
                 
            return analysis_data # Return the full data

        except Exception as e:
            logging.error(f"Error getting saved analysis {analysis_id}: {e}")
            raise StorageException(f"Failed to retrieve saved analysis {analysis_id}: {str(e)}")

    def delete_saved_analysis(self, analysis_id: str) -> bool:
        """Delete a saved analysis by ID."""
        try:
            if not self.db or isinstance(self.db, MagicMock):
                 raise StorageException("Firestore is not available.")

            doc_ref = self.db.collection('saved_analyses').document(analysis_id)
            # Remove await as get() is synchronous
            doc = doc_ref.get()

            if not doc.exists:
                logging.warning(f"Analysis {analysis_id} not found for deletion.")
                return False # Indicate not found

            # Delete analysis
            doc_ref.delete()
            logging.info(f"Successfully deleted analysis {analysis_id}")
            return True

        except Exception as e:
            logging.error(f"Error deleting saved analysis {analysis_id}: {e}")
            raise StorageException(f"Failed to delete saved analysis {analysis_id}: {str(e)}")

    async def get_articles(self, page=1, limit=10, topic=None, search_query=None, sort_by=None, sort_order=None):
        """Get articles with optional pagination, topic filtering, and search (async)."""
        from datetime import datetime, timedelta
        from utils.date_utils import normalize_datetime
        import logging
        if search_query:
            results = await self.enhanced_search(search_query)
            # Optionally filter by topic
            if topic and topic != 'All':
                results = [a for a in results if a.get('metadata', {}).get('topic') == topic]
            # Recency filter for search results
            # (Optional: can be added if desired)
            total = len(results)
            start = (page - 1) * limit
            end = start + limit
            paginated = results[start:end]
            return {
                'articles': paginated,
                'total': total
            }
        # Fallback to previous logic
        all_articles = await self.get_all_articles()
        if topic and topic != 'All':
            all_articles = [a for a in all_articles if a.get('metadata', {}).get('topic') == topic]
        # Recency filter unless all_time=true is in request args
        try:
            from quart import request
            all_time = request.args.get('all_time', '').lower() == 'true'
        except Exception:
            all_time = False
        if not all_time:
            cutoff = datetime.utcnow() - timedelta(days=365 * 2)
            def is_recent(article):
                pub_date = article.get('metadata', {}).get('pub_date')
                dt = normalize_datetime(pub_date)
                return dt and dt >= cutoff
            filtered_articles = [a for a in all_articles if is_recent(a)]
            logging.info(f"Recency filter applied: {len(filtered_articles)} of {len(all_articles)} articles are recent.")
            all_articles = filtered_articles
        total = len(all_articles)
        start = (page - 1) * limit
        end = start + limit
        paginated = all_articles[start:end]
        return {
            'articles': paginated,
            'total': total
        }

    def _parse_date(self, date_str):
        """Parse dates in various formats and return a datetime object. Falls back to current time if parsing fails."""
        parsed = normalize_datetime(date_str)
        return parsed if parsed else datetime.now()

    def _format_date(self, dt):
        """Format a datetime object as ISO string, or return as-is if already a string."""
        if not dt:
            return ""
        if isinstance(dt, str):
            return dt
        try:
            return dt.isoformat()
        except Exception:
            return str(dt)

    async def get_topic_distribution(self):
        """Return a dict of topic -> stats (count, percentage) for all articles."""
        all_articles = await self.get_all_articles()
        topic_counts = {}
        for article in all_articles:
            topic = article.get('metadata', {}).get('topic', 'Uncategorized')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        total = sum(topic_counts.values())
        result = {}
        for topic, count in topic_counts.items():
            result[topic] = {
                'count': count,
                'percentage': (count / total * 100) if total else 0.0
            }
        return result

    async def cleanup_duplicate_articles(self):
        """Remove duplicate articles by URL, keeping only one of each."""
        articles = await self.get_all_articles()
        seen_links = set()
        duplicates_removed = 0
        for article in articles:
            link = article.get('metadata', {}).get('link')
            if not link:
                continue
            if link in seen_links:
                await self.delete_article(article['id'])
                duplicates_removed += 1
            else:
                seen_links.add(link)
        return {
            'duplicates_removed': duplicates_removed,
            'total_articles': len(articles)
        }

    async def get_articles_without_summaries(self) -> list:
        """Return all articles that have full content but no summary."""
        articles = await self.get_all_articles()
        return [
            article for article in articles
            if article.get('metadata', {}).get('has_full_content')
            and not article.get('metadata', {}).get('summary')
        ]