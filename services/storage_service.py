from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import hashlib
import time
import json
import os
import asyncio
import re
import calendar

# Google Cloud imports
from google.cloud import storage
from google.cloud import firestore

from .config_service import ConfigService

class StorageException(Exception):
    """Custom exception for storage operations"""
    pass

class StorageService:
    """Simplified storage service using Firestore as the primary backend"""
    
    def __init__(self):
        """Initialize storage service with proper error handling"""
        try:
            config = ConfigService.get_instance()
            self.db = firestore.Client()
            self.articles_collection = self.db.collection('articles')
            self.backup_bucket = None
            
            if config.use_gcs:
                self.storage_client = storage.Client()
                self.backup_bucket = self.storage_client.bucket(config.gcs_bucket_name)
                
            logging.info("StorageService initialized successfully")
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
    
    async def query_articles(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query articles using Firestore with improved matching for vague queries"""
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
        import pandas as pd
        
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
        import pandas as pd
        
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
        """Parse date from various formats into a datetime object"""
        if date_value is None:
            return datetime.now()
            
        # If it's already a datetime
        if isinstance(date_value, datetime):
            return date_value.replace(tzinfo=None)
            
        # If it's a float/int timestamp
        if isinstance(date_value, (float, int)):
            try:
                return datetime.fromtimestamp(date_value)
            except (ValueError, OSError, TypeError):
                return datetime.now()
                
        # If it's a string
        if isinstance(date_value, str):
            if not date_value or date_value.lower() == 'unknown date':
                return datetime.now()
                
            try:
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
                    match = re.match(pattern, date_value)
                    if match:
                        groups = match.groups()
                        weekday, day, month, year, hour, minute, second = groups[:7]
                        # Convert month name to number
                        month_num = list(calendar.month_abbr).index(month[:3].title())
                        # Create datetime object
                        dt = datetime(int(year), month_num, int(day), 
                                    int(hour), int(minute), int(second))
                        
                        # Handle timezone adjustments
                        if len(groups) > 7:
                            tz = groups[7]
                            if tz == 'EST':
                                dt = dt + timedelta(hours=5)  # EST is UTC-5
                            elif tz == 'EDT':
                                dt = dt + timedelta(hours=4)  # EDT is UTC-4
                        else:
                            # Check if there's a timezone offset in the original string
                            tz_match = re.search(r'([+-])(\d{2})(\d{2})$', date_value)
                            if tz_match:
                                sign, tz_hours, tz_minutes = tz_match.groups()
                                offset = int(tz_hours) * 60 + int(tz_minutes)
                                if sign == '-':
                                    dt = dt + timedelta(minutes=offset)
                                else:
                                    dt = dt - timedelta(minutes=offset)
                        
                        return dt
                        
                # Try ISO format
                if 'T' in date_value:
                    # Handle timezone
                    if date_value.endswith('Z'):
                        date_value = date_value.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(date_value)
                    return dt.replace(tzinfo=None)
                    
                # Try email.utils parser for RFC format
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(date_value)
                    if dt:
                        return dt.replace(tzinfo=None)
                except Exception:
                    pass
                    
                # Try other common formats
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
                    '%Y-%m-%d %H:%M:%S.%f',
                    # Add explicit timezone formats
                    '%a, %d %b %Y %H:%M:%S GMT',
                    '%a, %d %b %Y %H:%M:%S +0000'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
                        
            except Exception as e:
                logging.warning(f"Error parsing date '{date_value}': {e}")
                
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

    async def get_topic_distribution(self) -> List[Dict]:
        """Get enhanced topic distribution statistics"""
        try:
            # Get all articles
            docs = self.articles_collection.stream()
            topic_stats = {}
            total_articles = 0
            
            # Get current time for recent article calculation
            current_time = datetime.now()
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
                
                # Check if article is recent
                try:
                    pub_date = self._parse_date(data.get('pub_date'))
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
                        stats['trend'] = 'increasing'
                        stats['growth_rate'] = recent_ratio * 100
                    elif recent_ratio > 0.2:
                        stats['trend'] = 'stable'
                        stats['growth_rate'] = recent_ratio * 100
                    else:
                        stats['trend'] = 'decreasing'
                        stats['growth_rate'] = recent_ratio * 100
            
            return topic_stats
            
        except Exception as e:
            logging.error(f"Error getting topic distribution: {e}")
            return {}

    def _ensure_naive_datetime(self, dt):
        """Convert a datetime to naive (no timezone) if it isn't already"""
        if dt is None:
            return datetime.now()
        if isinstance(dt, str):
            try:
                if dt.endswith('Z'):
                    dt = dt.replace('Z', '+00:00')
                dt = datetime.fromisoformat(dt)
            except ValueError:
                return datetime.now()
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

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