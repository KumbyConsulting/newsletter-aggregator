from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import pandas as pd
from datetime import datetime
import logging
from .config_service import ConfigService
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import hashlib  # Using built-in hashlib instead

class StorageException(Exception):
    """Custom exception for storage operations"""
    pass

class StorageService:
    def __init__(self):
        self._initialize_client()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _initialize_client(self):
        """Initialize ChromaDB client with retry logic"""
        config = ConfigService()
        self.client = chromadb.Client(
            Settings(
                persist_directory=config.chroma_persist_dir,
                anonymized_telemetry=False
            )
        )
        self.collection = self.client.get_or_create_collection("news_articles")

    def _generate_article_id(self, article: Dict) -> str:
        """Generate a unique ID for an article"""
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
            logging.error(f"Error generating article ID: {e}")
            # Fallback to a timestamp-based ID with MD5 hash
            fallback_id = f"{article['title']}_{time.time()}"
            hash_id = hashlib.md5(fallback_id.encode()).hexdigest()[:10]
            return f"article_{hash_id}"

    async def _store_with_validation(self, article: Dict, article_id: str):
        """Store article with validation and duplicate handling"""
        try:
            # Check if article already exists
            existing = self.collection.get(
                ids=[article_id],
                include=['metadatas']
            )
            
            if existing and existing['ids']:
                logging.info(f"Article already exists with ID: {article_id}")
                return  # Skip if already exists
            
            # Create document text with enhanced context
            document_text = f"""
            Title: {article['title']}
            Topic: {article.get('topic', '')}
            Description: {article['description']}
            Key Findings: {article.get('summary', '')}
            Source: {article['source']}
            Industry Impact: This article relates to {article.get('topic', '')} 
            in the pharmaceutical industry.
            """
            
            # Store with enhanced metadata
            self.collection.add(
                documents=[document_text],
                metadatas=[{
                    'title': article['title'],
                    'link': article['link'],
                    'pub_date': article['pub_date'],
                    'topic': article.get('topic', ''),
                    'source': article['source'],
                    'industry_segment': self._categorize_segment(article.get('topic', ''))
                }],
                ids=[article_id]
            )
            logging.info(f"Successfully stored article: {article_id}")
            
        except Exception as e:
            logging.error(f"Error storing article {article_id}: {e}")
            raise StorageException(f"Failed to store article: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def store_article(self, article: Dict) -> bool:
        """Store article with retry mechanism"""
        try:
            article_id = self._generate_article_id(article)
            await self._store_with_validation(article, article_id)
            return True
        except Exception as e:
            logging.error(f"Failed to store article: {e}")
            raise StorageException(f"Storage operation failed: {str(e)}")
    
    def query_articles(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query articles using semantic search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            articles = []
            for i in range(len(results['ids'][0])):
                articles.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return articles
            
        except Exception as e:
            logging.error(f"Error querying articles: {e}")
            return []
    
    async def batch_store_articles(self, articles: List[Dict]) -> bool:
        """Store multiple articles at once with duplicate handling"""
        try:
            successful = 0
            skipped = 0
            failed = 0
            
            for article in articles:
                try:
                    await self.store_article(article)
                    successful += 1
                except StorageException as e:
                    if "already exists" in str(e).lower():
                        skipped += 1
                    else:
                        failed += 1
                        logging.error(f"Failed to store article: {e}")
                except Exception as e:
                    failed += 1
                    logging.error(f"Unexpected error storing article: {e}")
            
            logging.info(f"Batch storage complete: {successful} stored, {skipped} skipped, {failed} failed")
            return failed == 0  # Return True only if no failures
            
        except Exception as e:
            logging.error(f"Error in batch store: {e}")
            return False
            
    def get_similar_articles(self, article_id: str, n_results: int = 3) -> List[Dict]:
        """Find articles similar to a given article"""
        try:
            # Get the original article
            result = self.collection.get(ids=[article_id])
            if not result['documents']:
                return []
                
            # Query using the article's text
            similar = self.collection.query(
                query_texts=[result['documents'][0]],
                n_results=n_results + 1  # Add 1 to account for the article itself
            )
            
            # Remove the original article from results
            articles = []
            for i in range(len(similar['ids'][0])):
                if similar['ids'][0][i] != article_id:
                    articles.append({
                        'id': similar['ids'][0][i],
                        'document': similar['documents'][0][i],
                        'metadata': similar['metadatas'][0][i],
                        'distance': similar['distances'][0][i] if 'distances' in similar else None
                    })
            
            return articles[:n_results]
            
        except Exception as e:
            logging.error(f"Error finding similar articles: {e}")
            return []

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