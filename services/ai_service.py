from typing import List, Dict, Optional
import google.generativeai as genai
from .storage_service import StorageService
import logging
from datetime import datetime
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
    processing_time: float

class AIService:
    def __init__(self, storage_service: StorageService):
        config = ConfigService()
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.storage_service = storage_service
        self.history = []  # Store conversation history
        
    async def generate_rag_response(self, query: str, n_context: int = 5, use_history: bool = True) -> AIResponse:
        """Generate a response using pharmaceutical-specific RAG"""
        start_time = time.time()
        try:
            # Get relevant articles synchronously
            relevant_articles = self.storage_service.query_articles(query, n_context)
            
            # Add conversation history context
            history_context = ""
            if use_history and self.history:
                history_context = "\nPrevious conversation:\n" + "\n".join(
                    [f"Q: {h['query']}\nA: {h['response']}" for h in self.history[-2:]]
                )

            if not relevant_articles:
                return AIResponse(
                    text="No relevant pharmaceutical articles found.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Enhanced prompt with industry context
            prompt = f"""
            As a pharmaceutical industry expert, please answer this question using the provided articles:

            Question: {query}

            {history_context}

            Consider these relevant pharmaceutical articles:
            {self._format_articles_for_prompt(relevant_articles)}

            Please provide:
            1. A comprehensive answer using the information from these articles
            2. Industry implications and impact
            3. Relevant regulatory considerations if applicable
            4. Future outlook based on the information
            5. Any contradictions or conflicts in the sources
            6. Confidence level in the response (High/Medium/Low)

            Include specific references to the sources when possible.
            """
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Convert response to proper format
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return AIResponse(
                text=response_text,
                sources=self._format_sources(relevant_articles),
                confidence=getattr(response, 'confidence', 0.8),  # Default confidence if not available
                processing_time=time.time() - start_time
            )
            
        except RateLimitException as e:
            logging.error(f"Rate limit exceeded: {e}")
            raise AIServiceException("Rate limit exceeded")
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            raise AIServiceException(f"Error generating response: {str(e)}")

    def _format_articles_for_prompt(self, articles: List[Dict]) -> str:
        """Format articles for prompt context"""
        formatted = []
        for i, article in enumerate(articles):
            formatted.append(f"""
            Article {i+1}:
            Title: {article['metadata']['title']}
            Topic: {article['metadata']['topic']}
            Content: {article['document']}
            Industry Segment: {article['metadata'].get('industry_segment', 'general')}
            """)
        return "\n\n".join(formatted)
    
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