from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from .storage_service import StorageService
import logging
from datetime import datetime
import json
import asyncio
from .config_service import ConfigService
from newsLetter import RateLimitException
from dataclasses import dataclass
from enum import Enum
import time

class AIServiceException(Exception):
    pass

class PromptTemplate(Enum):
    RAG_QUERY = """
    As a pharmaceutical industry expert, please answer this question:
    Question: {query}
    Context: {context}
    """
    SUMMARY = """
    Please summarize the following pharmaceutical text:
    Text: {text}
    Context: {context}
    """

@dataclass
class AIResponse:
    text: str
    sources: List[Dict]
    confidence: float
    timestamp: str = None  # Add timestamp instead of processing_time
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class AIService:
    def __init__(self, storage_service: StorageService):
        config = ConfigService()
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.storage_service = storage_service
        self.history = []
        self.max_history = 5
        self.relevant_articles = []
        
    async def generate_rag_response(self, query: str, use_history: bool = True) -> AIResponse:
        """Generate a response using RAG with streaming"""
        try:
            # Get relevant articles
            self.relevant_articles = self.storage_service.query_articles(
                query, 
                n_results=5
            )
            
            # Build context from relevant articles
            context = self._build_context(self.relevant_articles)
            
            # Include conversation history if requested
            if use_history and self.history:
                history_context = self._format_history()
                context = f"{context}\n\nPrevious conversation:\n{history_context}"

            # Generate prompt
            prompt = self._build_prompt(query, context)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Update conversation history
            self._update_history(query, response.text)
            
            return response

        except Exception as e:
            logging.error(f"Error in RAG response generation: {e}")
            raise

    def _build_context(self, articles: List[Dict]) -> str:
        """Build context from relevant articles"""
        context_parts = []
        for i, article in enumerate(articles, 1):
            context_parts.append(f"""
            Article {i}:
            Title: {article['metadata']['title']}
            Topic: {article['metadata']['topic']}
            Date: {article['metadata']['pub_date']}
            Content: {article['document']}
            Source: {article['metadata']['source']}
            """)
        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build a comprehensive prompt"""
        return f"""
        As a pharmaceutical industry expert, please provide a detailed response to this query:

        Query: {query}

        Use the following articles as context:
        {context}

        Please provide:
        1. A comprehensive answer
        2. Industry implications
        3. Regulatory considerations (if applicable)
        4. Future outlook
        5. Any potential risks or limitations
        6. References to specific sources

        Format the response in markdown with clear sections.
        """

    async def _generate_response(self, prompt: str) -> AIResponse:
        """Generate response with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Convert synchronous call to async
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                # Extract sources from the response
                sources = self._extract_sources(response.text)
                
                return AIResponse(
                    text=response.text,
                    sources=sources,
                    confidence=0.8  # TODO: Implement confidence scoring
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    def _extract_sources(self, response_text: str) -> List[Dict]:
        """Extract source references from response"""
        try:
            sources = []
            for article in self.relevant_articles:  # Store articles as class property
                sources.append({
                    'title': article['metadata']['title'],
                    'source': article['metadata']['source'],
                    'date': article['metadata']['pub_date'],
                    'link': article['metadata'].get('link', '#'),
                    'relevance_score': article.get('distance', 1.0)
                })
            return sources
        except Exception as e:
            logging.error(f"Error extracting sources: {e}")
            return []

    def _format_history(self) -> str:
        """Format conversation history"""
        return "\n".join([
            f"Q: {h['query']}\nA: {h['response']}"
            for h in self.history[-self.max_history:]
        ])

    def _update_history(self, query: str, response: str):
        """Update conversation history"""
        self.history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        # Keep only recent history
        self.history = self.history[-self.max_history:]

    async def clear_history(self):
        """Clear conversation history"""
        self.history = []

    async def summarize_with_context(self, text: str, topic: str) -> str:
        """Generate a summary with context from similar articles"""
        try:
            # Get similar articles synchronously
            similar_articles = self.storage_service.query_articles(topic, n_results=2)
            
            context = "\n\n".join([
                f"Related Article:\n{article['document']}"
                for article in similar_articles
            ])
            
            prompt = f"""
            Please summarize the following pharmaceutical industry text, taking into account these related articles for context:
            
            Text to Summarize:
            {text}
            
            Related Context:
            {context}
            
            Provide a concise, informative summary that captures the main points and relates them to the broader context.
            """
            
            response = await self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error in contextual summarization: {e}")
            return "Error generating summary" 