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
            # Create a unique ID from title, date, and link (link is more reliable than source)
            title_part = article['title'][:50].strip().lower()  # First 50 chars of title
            date_part = article.get('pub_date', datetime.now().isoformat())
            link_part = article.get('link', '')[:30]  # First 30 chars of link
            
            # Clean the ID components
            title_part = "".join(c for c in title_part if c.isalnum() or c.isspace())
            title_part = title_part.replace(" ", "_")
            link_part = "".join(c for c in link_part if c.isalnum() or c in ['/', '.', '-', '_'])
            
            # Create a reliable date part
            try:
                if isinstance(date_part, str):
                    parsed_date = datetime.fromisoformat(date_part.replace('Z', '+00:00'))
                else:
                    parsed_date = date_part
                date_part = parsed_date.strftime('%Y-%m-%d_%H-%M-%S')
            except Exception:
                date_part = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # Generate a hash of the full content to ensure uniqueness
            content_hash = hashlib.md5(
                f"{title_part}{date_part}{link_part}".encode()
            ).hexdigest()[:8]
            
            return f"{content_hash}_{date_part}"
            
        except Exception as e:
            logging.error(f"Error generating article ID: {e}")
            # Fallback to a timestamp-based ID with MD5 hash
            fallback_id = f"{article['title']}_{time.time()}"
            hash_id = hashlib.md5(fallback_id.encode()).hexdigest()[:10]
            return f"article_{hash_id}"
            
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
        
    async def store_article(self, article: Dict) -> bool:
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
                    await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Error storing article: {e}")
            raise StorageException(f"Failed to store article: {str(e)}")
            
    async def batch_store_articles(self, articles: List[Dict]) -> bool:
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
                        await asyncio.sleep(1)
                        
            # Report results
            logging.info(f"Successfully stored {stored_count} articles, skipped {skipped_count} duplicates")
            
            # Run cleanup periodically rather than after every batch
            # Only run cleanup if we stored a significant number of articles
            if stored_count > 10:
                try:
                    await self.cleanup_duplicate_articles()
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
    
    async def query_articles(self, query: str, n_results: int = 5, exclude_ids: List[str] = None) -> List[Dict]:
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
                return await self.get_recent_articles(limit=n_results)
                
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
                return await self.get_recent_articles(limit=n_results)
            
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
                return await self.get_recent_articles(limit=n_results)
            except Exception as fallback_error:
                logging.error(f"Error getting recent articles as fallback: {fallback_error}")
                return []
            
    async def get_recent_articles(self, limit: int = 10, topic: str = None) -> List[Dict]:
        """Get recent articles, optionally filtered by topic"""
        try:
            # Start with a base query ordered by publication date
            query = self.articles_collection.order_by('pub_date', direction=firestore.Query.DESCENDING)
            
            # Apply topic filter if provided
            if topic:
                query = query.where('topic', '==', topic)
                
            # Apply limit
            query = query.limit(limit)
            
            # Execute query
            docs = query.stream()
            
            # Format results
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
            
    async def get_similar_articles(self, article_id: str, n_results: int = 3) -> List[Dict]:
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
            
    async def backup_to_gcs(self, data_file: str) -> str:
        """Backup data file to Google Cloud Storage"""
        if self.storage_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            # Read the file
            with open(data_file, 'r') as f:
                data = f.read()
                
            # Determine file type and store accordingly
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{os.path.basename(data_file)}_{timestamp}"
            
            blob = self.backup_bucket.blob(f"backups/{filename}")
            
            if data_file.endswith('.csv'):
                blob.upload_from_string(data, content_type='text/csv')
            elif data_file.endswith('.json'):
                blob.upload_from_string(data, content_type='application/json')
            else:
                blob.upload_from_string(data)
                
            return blob.public_url
                
        except Exception as e:
            logging.error(f"Error backing up to GCS: {e}")
            raise StorageException(f"Failed to backup to GCS: {str(e)}")
            
    async def list_gcs_backups(self) -> List[Dict]:
        """List all backups in GCS"""
        if self.storage_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            blobs = self.storage_client.list_blobs(self.backup_bucket.name, prefix="backups/")
            
            backups = []
            for blob in blobs:
                backups.append({
                    'name': blob.name,
                    'size': blob.size,
                    'updated': blob.updated,
                    'url': blob.public_url
                })
                
            return backups
            
        except Exception as e:
            logging.error(f"Error listing GCS backups: {e}")
            return []
            
    async def restore_from_gcs(self, filename: str, target_path: str) -> bool:
        """Restore a backup from GCS to local file"""
        if self.storage_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            blob = self.backup_bucket.blob(filename)
            
            with open(target_path, 'wb') as f:
                blob.download_to_file(f)
                
            return True
            
        except Exception as e:
            logging.error(f"Error restoring from GCS: {e}")
            return False

    async def sync_from_csv(self, csv_file_path: str) -> Dict:
        """Synchronize Firestore with CSV data to ensure all CSV articles are in Firestore
        
        Args:
            csv_file_path: Path to the CSV file containing articles
            
        Returns:
            Dict with statistics about the operation
        """
        stats = {
            "total_csv": 0,
            "already_in_db": 0,
            "added_to_db": 0,
            "failed": 0
        }
        
        try:
            # Read CSV into DataFrame
            if not os.path.exists(csv_file_path):
                logging.error(f"CSV file not found: {csv_file_path}")
                return stats
                
            df = pd.read_csv(csv_file_path)
            stats["total_csv"] = len(df)
            
            if stats["total_csv"] == 0:
                logging.info("CSV file is empty, nothing to sync")
                return stats
                
            # Convert to list of dictionaries for processing
            articles = df.to_dict('records')
            logging.info(f"Processing {len(articles)} articles from CSV")
            
            # Process each article
            for article in articles:
                try:
                    # Generate article ID the same way as store_article
                    article_id = self._generate_article_id(article)
                    
                    # Check if article already exists
                    doc_ref = self.articles_collection.document(article_id)
                    doc = doc_ref.get()
                    
                    if doc.exists:
                        stats["already_in_db"] += 1
                        continue
                    
                    # Create document with enhanced context
                    document_text = f"""
                    Title: {article['title']}
                    Topic: {article.get('topic', '')}
                    Description: {article.get('description', '')}
                    Key Findings: {article.get('summary', '')}
                    Source: {article.get('source', '')}
                    Industry Impact: This article relates to {article.get('topic', '')} 
                    in the pharmaceutical industry.
                    """
                    
                    # Store article data
                    doc_ref.set({
                        'title': article['title'],
                        'link': article.get('link', ''),
                        'pub_date': article.get('pub_date', ''),
                        'topic': article.get('topic', ''),
                        'source': article.get('source', ''),
                        'description': article.get('description', ''),
                        'summary': article.get('summary', ''),
                        'industry_segment': self._categorize_segment(article.get('topic', '')),
                        'document': document_text,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                    
                    stats["added_to_db"] += 1
                    
                except Exception as e:
                    logging.error(f"Error syncing article {article.get('title', 'Unknown')}: {e}")
                    stats["failed"] += 1
            
            logging.info(f"Sync completed. Stats: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Error during CSV to Firestore sync: {e}")
            stats["failed"] = stats["total_csv"]
            return stats 

    async def delete_article(self, article_id: str) -> bool:
        """Delete an article from Firestore by ID
        
        Args:
            article_id: The ID of the article to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logging.warning(f"Article {article_id} not found for deletion")
                return False
                
            # Delete the document
            doc_ref.delete()
            logging.info(f"Successfully deleted article {article_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting article {article_id}: {e}")
            return False

    async def export_to_csv(self, csv_file_path: str, limit: int = 0) -> Dict:
        """Export articles from Firestore to CSV for archival or backup purposes
        
        Args:
            csv_file_path: Path where CSV file should be saved
            limit: Maximum number of articles to export (0 for unlimited)
            
        Returns:
            Dict with statistics about the operation
        """
        stats = {
            "total_exported": 0,
            "success": False
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_file_path) if os.path.dirname(csv_file_path) else '.', exist_ok=True)
            
            # Fetch articles from Firestore
            if limit > 0:
                articles_ref = self.articles_collection.limit(limit).stream()
            else:
                articles_ref = self.articles_collection.stream()
            
            articles = []
            for doc in articles_ref:
                article_data = doc.to_dict()
                
                # Add doc ID
                article_data['id'] = doc.id
                
                # Remove document field generated for embedding
                if 'document' in article_data:
                    del article_data['document']
                    
                # Normalize timestamp fields to strings
                for key, value in article_data.items():
                    if hasattr(value, 'timestamp'):
                        article_data[key] = value.isoformat()
                        
                articles.append(article_data)
            
            stats["total_exported"] = len(articles)
            
            if len(articles) > 0:
                # Create DataFrame
                df = pd.DataFrame(articles)
                
                # Save to CSV
                df.to_csv(csv_file_path, index=False)
                logging.info(f"Exported {len(articles)} articles to {csv_file_path}")
                
                stats["success"] = True
            else:
                logging.info(f"No articles to export")
            
            return stats
            
        except Exception as e:
            logging.error(f"Error exporting to CSV: {e}")
            return stats

    async def cleanup_duplicate_articles(self) -> Dict:
        """Clean up duplicate articles with enhanced error handling"""
        try:
            # Get all articles
            articles = self.articles_collection.stream()
            article_map = {}  # Map of link -> (doc_id, article_data)
            content_map = {}  # Map of content_hash -> (doc_id, article_data)
            duplicates = []
            
            for doc in articles:
                data = doc.to_dict()
                link = data.get('link')
                content_hash = data.get('content_hash')
                
                if not link:
                    continue
                    
                # Check for duplicate links
                if link in article_map:
                    # Compare timestamps to keep the most recent
                    existing_doc_id, existing_data = article_map[link]
                    existing_time = datetime.fromisoformat(existing_data.get('last_updated', '1970-01-01'))
                    current_time = datetime.fromisoformat(data.get('last_updated', '1970-01-01'))
                    
                    if current_time > existing_time:
                        duplicates.append(existing_doc_id)
                        article_map[link] = (doc.id, data)
                    else:
                        duplicates.append(doc.id)
                else:
                    article_map[link] = (doc.id, data)
                
                # Check for similar content if we have a content hash
                if content_hash:
                    is_similar = False
                    for existing_hash, (existing_doc_id, existing_data) in content_map.items():
                        similarity = self._calculate_similarity(content_hash, existing_hash)
                        if similarity > 0.9:  # 90% similarity threshold
                            # Compare timestamps to keep the most recent
                            existing_time = datetime.fromisoformat(existing_data.get('last_updated', '1970-01-01'))
                            current_time = datetime.fromisoformat(data.get('last_updated', '1970-01-01'))
                            
                            if current_time > existing_time:
                                duplicates.append(existing_doc_id)
                                content_map[content_hash] = (doc.id, data)
                            else:
                                duplicates.append(doc.id)
                            is_similar = True
                            break
                    
                    if not is_similar:
                        content_map[content_hash] = (doc.id, data)
            
            # Remove duplicates from duplicates list
            duplicates = list(set(duplicates))
            
            # Delete duplicates in batches
            batch = self.db.batch()
            batch_count = 0
            deleted_count = 0
            
            for doc_id in duplicates:
                doc_ref = self.articles_collection.document(doc_id)
                batch.delete(doc_ref)
                batch_count += 1
                deleted_count += 1
                
                if batch_count >= 500:  # Firestore batch limit
                    batch.commit()
                    batch = self.db.batch()
                    batch_count = 0
            
            # Commit any remaining deletions
            if batch_count > 0:
                batch.commit()
            
            logging.info(f"Cleaned up {deleted_count} duplicate articles")
            return {
                "duplicates_removed": deleted_count,
                "total_processed": len(article_map),
                "similar_content_removed": len(duplicates) - len(set(doc_id for doc_id, _ in article_map.values()))
            }
            
        except Exception as e:
            logging.error(f"Error cleaning up duplicates: {e}")
            raise StorageException(f"Cleanup failed: {str(e)}")

    async def get_topic_counts(self) -> Dict[str, int]:
        """Get counts of articles per topic"""
        try:
            # Get all articles
            docs = self.articles_collection.stream()
            topic_counts = {}
            
            for doc in docs:
                data = doc.to_dict()
                topic = data.get('topic', 'Uncategorized')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
            return topic_counts
            
        except Exception as e:
            logging.error(f"Error getting topic counts: {e}")
            return {}

    def _parse_date(self, date_value) -> datetime:
        """Parse date from various formats to datetime object"""
        try:
            if not date_value or date_value == 'Unknown date' or pd.isna(date_value):
                return datetime.now()
            
            if isinstance(date_value, datetime):
                return self._ensure_naive_datetime(date_value)
            
            if isinstance(date_value, (int, float)):
                if pd.isna(date_value):  # Handle NaN
                    return datetime.now()
                return datetime.fromtimestamp(date_value)
        
            # Define timezone mappings
            tzinfos = {
                'EST': -18000,  # UTC-5
                'EDT': -14400,  # UTC-4
                'CST': -21600,  # UTC-6
                'CDT': -18000,  # UTC-5
                'MST': -25200,  # UTC-7
                'MDT': -21600,  # UTC-6
                'PST': -28800,  # UTC-8
                'PDT': -25200,  # UTC-7
            }
            
            # Try different date formats
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%b %d, %Y %H:%M:%S',
                '%B %d, %Y %H:%M:%S',
                '%a, %d %b %Y %H:%M:%S %Z',  # RFC 2822 format
                '%A, %d %B %Y %H:%M:%S %Z',
            ]
            
            # First try parsing with standard formats
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(str(date_value).strip(), fmt)
                    return self._ensure_naive_datetime(parsed_date)
                except ValueError:
                    continue
                
            try:
                # Try parsing with dateutil as a fallback, with timezone info
                from dateutil import parser
                parsed_date = parser.parse(str(date_value), tzinfos=tzinfos)
                return self._ensure_naive_datetime(parsed_date)
            except Exception as e:
                logging.debug(f"dateutil parser failed for '{date_value}': {e}")
                
            # If all parsing attempts fail, return current time
            logging.warning(f"Failed to parse date '{date_value}': Unknown string format")
            return datetime.now()
            
        except Exception as e:
            logging.warning(f"Error parsing date '{date_value}': {e}")
            return datetime.now()

    def _ensure_naive_datetime(self, dt):
        """Ensure datetime is naive (no timezone) by converting to UTC and removing tzinfo"""
        if dt is None:
            return datetime.now()
        try:
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception as e:
            logging.warning(f"Error converting timezone: {e}")
            return datetime.now()

    def _format_date(self, dt: datetime) -> str:
        """Format datetime to consistent string format"""
        try:
            if dt is None:
                dt = datetime.now()
            
            # Ensure datetime is naive (no timezone)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            
            # Format as human-readable date
            return dt.strftime('%b %d, %Y %H:%M')
        except Exception as e:
            logging.warning(f"Error formatting date: {e}")
            return datetime.now().strftime('%b %d, %Y %H:%M')

    async def get_topic_distribution(self) -> Dict:
        """Get enhanced topic distribution statistics"""
        try:
            # Check if we're using Firestore
            if self.config.storage_backend == 'firestore':
                # Get all articles from Firestore
                docs = self.articles_collection.stream()
                topic_stats = {}
                total_articles = 0
                
                # Get current time for recent article calculation - ensure naive datetime
                current_time = self._ensure_naive_datetime(datetime.now())
                thirty_days_ago = current_time - timedelta(days=30)
                
                for doc in docs:
                    data = doc.to_dict()
                    topic = data.get('topic', 'Uncategorized')
                    total_articles += 1
                    
                    # Initialize topic stats if not exists
                    if topic not in topic_stats:
                        topic_stats[topic] = {
                            'count': 0,
                            'recent_count': 0,
                            'trend': 'stable',
                            'growth_rate': 0
                        }
                    
                    topic_stats[topic]['count'] += 1
                    
                    # Check if article is recent - ensure naive datetime for comparison
                    try:
                        pub_date = self._ensure_naive_datetime(self._parse_date(data.get('pub_date')))
                        if pub_date > thirty_days_ago:
                            topic_stats[topic]['recent_count'] += 1
                    except Exception as e:
                        logging.warning(f"Error processing date for topic distribution: {e}")
                        continue
                
                # Calculate percentages and trends
                for topic, stats in topic_stats.items():
                    stats['percentage'] = (stats['count'] / total_articles) * 100 if total_articles > 0 else 0
                    
                    # Calculate growth rate
                    if stats['count'] > 0:
                        recent_ratio = stats['recent_count'] / stats['count']
                        if recent_ratio > 0.5:
                            stats['trend'] = 'up'
                            stats['growth_rate'] = recent_ratio * 100
                        elif recent_ratio > 0.2:
                            stats['trend'] = 'stable'
                            stats['growth_rate'] = recent_ratio * 100
                        else:
                            stats['trend'] = 'down'
                            stats['growth_rate'] = recent_ratio * 100
                
                return topic_stats
            else:
                # Handle ChromaDB or other non-Firestore backends
                filename = self.config.settings.ARTICLES_FILE_PATH
                
                # If file doesn't exist, return empty stats
                if not os.path.isfile(filename):
                    logging.warning(f"Articles file not found: {filename}")
                    return {}
                    
                try:
                    # Use pandas to read the CSV
                    import pandas as pd
                    df = pd.read_csv(filename, quotechar='"', escapechar='\\')
                    
                    # Check if 'topic' column exists
                    if 'topic' not in df.columns:
                        logging.error(f"No 'topic' column found in CSV file: {filename}")
                        return {}
                    
                    # Get current time for recent article calculation
                    current_time = datetime.now()
                    thirty_days_ago = current_time - timedelta(days=30)
                    
                    # Calculate topic distribution
                    topic_stats = {}
                    total_articles = len(df)
                    
                    # Group by topic and count
                    topic_counts = df['topic'].value_counts().reset_index()
                    topic_counts.columns = ['topic', 'count']
                    
                    # Get recent articles
                    # Handle different date formats
                    try:
                        df['pub_date_obj'] = pd.to_datetime(df['pub_date'], errors='coerce')
                        recent_df = df[df['pub_date_obj'] > pd.Timestamp(thirty_days_ago)]
                        recent_counts = recent_df['topic'].value_counts().to_dict()
                    except Exception as date_err:
                        logging.warning(f"Error parsing dates: {date_err}. Using default recent counts.")
                        recent_counts = {}
                    
                    # Create topic stats
                    for _, row in topic_counts.iterrows():
                        topic = row['topic']
                        if pd.isna(topic) or topic == '':
                            continue
                            
                        count = row['count']
                        recent_count = recent_counts.get(topic, 0)
                        
                        # Calculate percentage and trends
                        percentage = (count / total_articles) * 100 if total_articles > 0 else 0
                        
                        # Calculate growth rate and trend
                        if count > 0:
                            recent_ratio = recent_count / count
                            if recent_ratio > 0.5:
                                trend = 'up'
                            elif recent_ratio > 0.2:
                                trend = 'stable'
                            else:
                                trend = 'down'
                            growth_rate = recent_ratio * 100
                        else:
                            trend = 'stable'
                            growth_rate = 0
                            
                        topic_stats[topic] = {
                            'count': int(count),
                            'percentage': percentage,
                            'recent_count': int(recent_count),
                            'trend': trend,
                            'growth_rate': growth_rate
                        }
                    
                    return topic_stats
                except Exception as e:
                    logging.error(f"Error reading articles file: {e}")
                    return {}
            
        except Exception as e:
            logging.error(f"Error getting topic distribution: {e}")
            return {}

    async def get_articles(self, page: int = 1, limit: int = 10, topic: str = None, search_query: str = None, sort_by: str = 'pub_date', sort_order: str = 'desc') -> List[Dict]:
        """Get articles with pagination, topic filtering, search functionality, and sorting"""
        try:
            # Start with base query
            query = self.articles_collection

            # Apply topic filter if provided
            if topic and topic != 'All':
                query = query.where('topic', '==', topic)

            # Get all matching documents first (for accurate pagination)
            docs = query.stream()
            articles = []

            for doc in docs:
                data = doc.to_dict()
                article_id = doc.id
                
                # Parse the publication date
                pub_date = self._parse_date(data.get('pub_date'))
                    
                metadata = {
                    'title': data.get('title', 'Unknown Title'),
                    'description': data.get('description', 'No description available'),
                    'link': data.get('link', '#'),
                    'pub_date': self._format_date(pub_date),
                    'topic': data.get('topic', 'Uncategorized'),
                    'source': data.get('source', 'Unknown source'),
                    'summary': data.get('summary', None),
                    'image_url': data.get('image_url', ''),
                    'has_full_content': bool(data.get('document', '')),
                    'reading_time': data.get('reading_time', 0),
                    'relevance_score': 0  # Default relevance score
                }

                # Apply search filter if provided
                if search_query:
                    search_terms = search_query.lower().split()
                    searchable_text = ' '.join([
                        str(metadata['title']),
                        str(metadata['description']),
                        str(metadata['topic']),
                        str(metadata['source'])
                    ]).lower()
                    
                    # Calculate relevance score based on term matches
                    relevance_score = 0
                    for term in search_terms:
                        if term in metadata['title'].lower():
                            relevance_score += 3
                        if term in metadata['description'].lower():
                            relevance_score += 2
                        if term in metadata['topic'].lower():
                            relevance_score += 1
                    metadata['relevance_score'] = relevance_score
                    
                    # Check if all search terms are present
                    if not all(term in searchable_text for term in search_terms):
                        continue

                articles.append({
                    'id': article_id,
                    'metadata': metadata,
                    'document': data.get('document', '')
                })

            # Sort articles based on sort_by and sort_order
            reverse = sort_order.lower() == 'desc'
            
            if sort_by == 'pub_date':
                articles.sort(key=lambda x: self._parse_date(x['metadata']['pub_date']), reverse=reverse)
            elif sort_by == 'title':
                articles.sort(key=lambda x: x['metadata']['title'].lower(), reverse=reverse)
            elif sort_by == 'relevance' and search_query:
                articles.sort(key=lambda x: x['metadata']['relevance_score'], reverse=True)  # Always sort relevance in descending order
            elif sort_by == 'reading_time':
                articles.sort(key=lambda x: x['metadata']['reading_time'], reverse=reverse)

            # Apply pagination
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_articles = articles[start_idx:end_idx]

            return {
                'articles': paginated_articles,
                'total': len(articles),
                'page': page,
                'total_pages': (len(articles) + limit - 1) // limit,
                'query_time': 0  # Add query time field for consistency
            }

        except Exception as e:
            logging.error(f"Error getting articles: {e}")
            # Return empty result instead of raising exception
            return {
                'articles': [],
                'total': 0,
                'page': page,
                'total_pages': 0,
                'query_time': 0
            }

    async def get_article(self, article_id: str) -> Optional[Dict]:
        """Get a single article by ID
        
        Args:
            article_id: The ID of the article to retrieve
            
        Returns:
            Dict containing the article data if found, None otherwise
        """
        try:
            # Get the document reference
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            
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

    async def update_article_metadata(self, article_id: str, metadata: Dict) -> bool:
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
            # Query articles with matching URL
            docs = self.articles_collection.where('link', '==', url).stream()
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
        """Get all articles from the database without pagination
        
        Returns:
            List[Dict]: List of all articles with their complete data
        """
        try:
            # Get all documents from the articles collection
            docs = self.articles_collection.stream()
            articles = []

            for doc in docs:
                data = doc.to_dict()
                # Parse the publication date
                pub_date = self._parse_date(data.get('pub_date'))
                
                articles.append({
                    'id': doc.id,
                    'document': data.get('document', ''),
                    'metadata': {
                        'title': data.get('title', 'Unknown Title'),
                        'description': data.get('description', 'No description available'),
                        'link': data.get('link', '#'),
                        'pub_date': self._format_date(pub_date),
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
        """Enhanced search with multiple strategies based on query type
        
        Args:
            query: Search query string
            search_type: One of 'auto', 'exact', 'fuzzy', 'semantic'
            **kwargs: Additional search parameters
        
        Returns:
            List[Dict]: Search results with relevance scores
        """
        try:
            # Analyze query to determine best search strategy if auto
            if search_type == 'auto':
                search_type = self._determine_search_strategy(query)

            if search_type == 'exact':
                return await self._exact_field_search(query, **kwargs)
            elif search_type == 'fuzzy':
                return await self._enhanced_fuzzy_search(query, **kwargs)
            else:  # semantic
                return await self._semantic_search(query, **kwargs)
        except Exception as e:
            logging.error(f"Error in enhanced search: {e}")
            return []

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
        """Perform exact field matching with index optimization"""
        try:
            if not fields:
                fields = ['title', 'topic', 'source']  # Default searchable fields
                
            # Remove quotes if present
            clean_query = query.strip('"\'').lower()
            
            # Start with base query
            base_query = self.articles_collection
            
            # Apply field filters
            results = []
            for field in fields:
                # Use proper index-based query
                docs = base_query.where(field, '==', clean_query).stream()
                
                for doc in docs:
                    data = doc.to_dict()
                    results.append({
                        'id': doc.id,
                        'document': data.get('document', ''),
                        'metadata': self._format_article_metadata(data),
                        'match_type': 'exact',
                        'matched_field': field,
                        'score': 1.0  # Exact matches get perfect score
                    })
                    
            return results
            
        except Exception as e:
            logging.error(f"Error in exact field search: {e}")
            return []

    async def _enhanced_fuzzy_search(self, query: str, threshold: float = 0.6, **kwargs) -> List[Dict]:
        """Improved fuzzy search with better scoring and matching"""
        try:
            # Normalize query
            query_terms = self._normalize_text(query)
            
            # Get all articles for fuzzy matching
            docs = self.articles_collection.stream()
            results = []
            
            for doc in docs:
                data = doc.to_dict()
                # Calculate fuzzy match scores for different fields
                title_score = self._fuzzy_match_score(
                    query_terms,
                    self._normalize_text(data.get('title', ''))
                )
                topic_score = self._fuzzy_match_score(
                    query_terms,
                    self._normalize_text(data.get('topic', ''))
                )
                desc_score = self._fuzzy_match_score(
                    query_terms,
                    self._normalize_text(data.get('description', ''))
                )
                
                # Weighted scoring
                total_score = (
                    title_score * 0.5 +    # Title matches are most important
                    topic_score * 0.3 +    # Topic matches are second
                    desc_score * 0.2       # Description matches are third
                )
                
                if total_score >= threshold:
                    results.append({
                        'id': doc.id,
                        'document': data.get('document', ''),
                        'metadata': self._format_article_metadata(data),
                        'match_type': 'fuzzy',
                        'score': total_score
                    })
            
            # Sort by score
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
        """Placeholder for future semantic search implementation"""
        logging.warning("Semantic search not yet implemented")
        return []

    async def check_articles_exist(self, article_ids: List[str]) -> Dict[str, bool]:
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

    async def get_saved_analyses(self, limit: int = 10, offset: int = 0, sort_by: str = 'timestamp', sort_order: str = 'desc') -> Dict:
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

    async def get_saved_analysis(self, analysis_id: str) -> Optional[Dict]:
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

    async def delete_saved_analysis(self, analysis_id: str) -> bool:
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
            await doc_ref.delete()
            logging.info(f"Successfully deleted analysis {analysis_id}")
            return True

        except Exception as e:
            logging.error(f"Error deleting saved analysis {analysis_id}: {e}")
            raise StorageException(f"Failed to delete saved analysis {analysis_id}: {str(e)}")