from typing import List, Dict, Optional
import logging
from datetime import datetime
import hashlib
import time
import json
import os

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
        self.config = ConfigService()
        
        # Initialize Firestore client
        self.db = firestore.Client(project=self.config.gcp_project_id)
        self.articles_collection = self.db.collection('news_articles')
        logging.info("Using Firestore backend for article storage")
        
        # Initialize GCS client if configured
        if self.config.use_gcs and self.config.gcp_project_id:
            self.gcs_client = storage.Client(project=self.config.gcp_project_id)
            self.bucket_name = self.config.gcs_bucket_name
            
            # Create bucket if it doesn't exist
            try:
                self.bucket = self.gcs_client.get_bucket(self.bucket_name)
            except Exception:
                # Create bucket with standard storage class in us-central1
                self.bucket = self.gcs_client.create_bucket(
                    self.bucket_name, 
                    location="us-central1",
                    storage_class="STANDARD"
                )
            logging.info("Using Google Cloud Storage for backups")
        else:
            self.gcs_client = None
            
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
        """Store article in Firestore"""
        try:
            article_id = self._generate_article_id(article)
            
            # Check if article already exists
            doc_ref = self.articles_collection.document(article_id)
            doc = doc_ref.get()
            if doc.exists:
                logging.info(f"Article already exists with ID: {article_id}")
                return True
                
            # Create document with enhanced context
            document_text = f"""
            Title: {article['title']}
            Topic: {article.get('topic', '')}
            Description: {article['description']}
            Key Findings: {article.get('summary', '')}
            Source: {article['source']}
            Industry Impact: This article relates to {article.get('topic', '')} 
            in the pharmaceutical industry.
            """
            
            # Store article data
            doc_ref.set({
                'title': article['title'],
                'link': article['link'],
                'pub_date': article['pub_date'],
                'topic': article.get('topic', ''),
                'source': article['source'],
                'description': article.get('description', ''),
                'summary': article.get('summary', ''),
                'industry_segment': self._categorize_segment(article.get('topic', '')),
                'document': document_text,
                'created_at': firestore.SERVER_TIMESTAMP
            })
            
            logging.info(f"Successfully stored article in Firestore: {article_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error storing article in Firestore: {e}")
            return False
            
    async def batch_store_articles(self, articles: List[Dict]) -> bool:
        """Store multiple articles at once with duplicate handling"""
        try:
            successful = 0
            skipped = 0
            failed = 0
            
            # Use a batch write for better performance
            batch = self.db.batch()
            batch_count = 0
            max_batch_size = 500  # Firestore limit
            
            # Track unique articles by link (most reliable identifier)
            unique_links = set()
            
            for article in articles:
                try:
                    article_id = self._generate_article_id(article)
                    article_link = article.get('link', '')
                    
                    # Skip if we've already seen this link
                    if article_link in unique_links:
                        skipped += 1
                        continue
                        
                    # Check if article already exists by link
                    existing_docs = self.articles_collection.where('link', '==', article_link).limit(1).stream()
                    if any(True for _ in existing_docs):
                        skipped += 1
                        continue
                    
                    # Add to unique links set
                    unique_links.add(article_link)
                    
                    # Create document with enhanced context
                    document_text = f"""
                    Title: {article['title']}
                    Topic: {article.get('topic', '')}
                    Description: {article['description']}
                    Key Findings: {article.get('summary', '')}
                    Source: {article['source']}
                    Industry Impact: This article relates to {article.get('topic', '')} 
                    in the pharmaceutical industry.
                    """
                    
                    # Add to batch
                    batch.set(self.articles_collection.document(article_id), {
                        'title': article['title'],
                        'link': article['link'],
                        'pub_date': article['pub_date'],
                        'topic': article.get('topic', ''),
                        'source': article['source'],
                        'description': article.get('description', ''),
                        'summary': article.get('summary', ''),
                        'industry_segment': self._categorize_segment(article.get('topic', '')),
                        'document': document_text,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                    
                    batch_count += 1
                    successful += 1
                    
                    # Commit batch if we've reached the limit
                    if batch_count >= max_batch_size:
                        batch.commit()
                        batch = self.db.batch()
                        batch_count = 0
                        
                except Exception as e:
                    failed += 1
                    logging.error(f"Failed to store article in batch: {e}")
            
            # Commit any remaining articles
            if batch_count > 0:
                batch.commit()
                
            logging.info(f"Batch storage complete: {successful} stored, {skipped} skipped, {failed} failed")
            
            # Run cleanup if we added any new articles
            if successful > 0:
                await self.cleanup_duplicate_articles()
            
            return failed == 0  # Return True only if no failures
            
        except Exception as e:
            logging.error(f"Error in batch store: {e}")
            return False
    
    async def query_articles(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query articles using Firestore"""
        try:
            # Simple keyword search implementation
            query_terms = query.lower().split()
            
            # Search by keywords in title and topic
            docs = self.articles_collection.limit(50).stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                score = 0
                
                # Simple keyword matching
                for term in query_terms:
                    if term in data.get('title', '').lower():
                        score += 3
                    if term in data.get('topic', '').lower():
                        score += 2
                    if term in data.get('document', '').lower():
                        score += 1
                        
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
            
            # Sort by score (higher is better) and limit results
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logging.error(f"Error querying articles from Firestore: {e}")
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
        if self.gcs_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            # Read the file
            with open(data_file, 'r') as f:
                data = f.read()
                
            # Determine file type and store accordingly
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{os.path.basename(data_file)}_{timestamp}"
            
            blob = self.bucket.blob(f"backups/{filename}")
            
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
        if self.gcs_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            blobs = self.gcs_client.list_blobs(self.bucket_name, prefix="backups/")
            
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
        if self.gcs_client is None:
            raise StorageException("GCS backup is not enabled")
            
        try:
            blob = self.bucket.blob(filename)
            
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
        """Find and remove duplicate articles from Firestore
        
        This method will:
        1. Fetch all articles from Firestore
        2. Group them by title and link
        3. For each group with multiple articles, keep the most recent one and delete the others
        
        Returns:
            Dict with statistics about removed duplicates
        """
        stats = {
            "total_articles": 0,
            "duplicate_groups": 0,
            "duplicates_removed": 0,
            "errors": 0
        }
        
        try:
            # Fetch all articles
            articles_ref = self.articles_collection.stream()
            
            # Group articles by title+link (which should be unique)
            article_groups = {}
            
            for doc in articles_ref:
                stats["total_articles"] += 1
                article_data = doc.to_dict()
                article_data['id'] = doc.id
                
                # Create key for grouping (title + link)
                if 'title' not in article_data or 'link' not in article_data:
                    logging.warning(f"Article {doc.id} missing title or link, skipping")
                    continue
                    
                group_key = f"{article_data['title']}|{article_data['link']}"
                
                if group_key not in article_groups:
                    article_groups[group_key] = []
                    
                article_groups[group_key].append(article_data)
            
            # Process groups with duplicates
            for group_key, articles in article_groups.items():
                if len(articles) > 1:
                    stats["duplicate_groups"] += 1
                    
                    # Sort by created_at timestamp if available, newest first
                    try:
                        articles.sort(key=lambda x: x.get('created_at', 0), reverse=True)
                    except Exception as e:
                        logging.warning(f"Error sorting articles by timestamp: {e}")
                    
                    # Keep the first article (newest), delete the rest
                    for article in articles[1:]:
                        try:
                            await self.delete_article(article['id'])
                            stats["duplicates_removed"] += 1
                        except Exception as e:
                            logging.error(f"Error removing duplicate {article['id']}: {e}")
                            stats["errors"] += 1
            
            logging.info(f"Duplicate cleanup complete. Stats: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Error cleaning up duplicates: {e}")
            stats["errors"] += 1
            return stats