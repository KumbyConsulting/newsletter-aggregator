from typing import List, Dict, Optional, Tuple, AsyncGenerator
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
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Update Vertex AI imports to use stable API
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part

class AIServiceException(Exception):
    pass

class GeminiAPIException(AIServiceException):
    pass

class PromptTemplate(Enum):
    RAG_QUERY = """
    As a pharmaceutical industry expert, please answer this question:
    Question: {query}
    Context: {context}
    
    Base your answer primarily on the provided context. If the context doesn't contain enough information, 
    you can use your general knowledge but clearly indicate when you're doing so.
    
    Format your response in markdown. Include citations to specific articles where appropriate using [Article X] notation.
    """
    SUMMARY = """
    Please summarize the following pharmaceutical text:
    Text: {text}
    Context: {context}
    
    Focus on:
    1. Key findings or announcements
    2. Industry impact
    3. Regulatory implications (if any)
    
    Keep the summary under 200 words.
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

class AIModelFactory:
    """Factory for creating AI model clients based on configuration"""
    @staticmethod
    def create_model(config: ConfigService):
        try:
            if config.use_vertex_ai:
                return VertexAIModel(config)
            else:
                return GeminiDirectModel(config)
        except Exception as e:
            logging.error(f"Error creating AI model: {str(e)}")
            return GeminiDirectModel(config)

class AIModelInterface:
    """Interface for AI model implementations"""
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        raise NotImplementedError
        
    async def generate_content_stream(self, prompt: str, safety_settings: List[Dict] = None) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class GeminiDirectModel(AIModelInterface):
    """Direct Gemini API implementation"""
    def __init__(self, config: ConfigService):
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        """Generate content using direct Gemini API"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                safety_settings=safety_settings
            )
            
            if not response or not hasattr(response, 'text'):
                raise GeminiAPIException("Empty response from Gemini API")
                
            return response
            
        except Exception as e:
            if "safety" in str(e).lower():
                raise GeminiAPIException(f"Content was filtered due to safety settings. Please rephrase your query.")
            elif "rate limit" in str(e).lower():
                raise GeminiAPIException("Rate limit exceeded. Please try again later.")
            else:
                raise GeminiAPIException(f"Error generating response: {str(e)}")
                
    async def generate_content_stream(self, prompt: str, safety_settings: List[Dict] = None) -> AsyncGenerator[str, None]:
        """Generate streaming content using direct Gemini API"""
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"

class VertexAIModel(AIModelInterface):
    """Vertex AI implementation for Gemini"""
    def __init__(self, config: ConfigService):
        # Initialize Vertex AI
        aiplatform.init(
            project=config.gcp_project_id,
            location=config.gcp_region,
        )
        
        # Initialize the model
        self.model = GenerativeModel("gemini-pro")
        self.generation_config = GenerationConfig(
            temperature=0.4,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        """Generate content using Vertex AI"""
        try:
            # Create content from prompt with proper role formatting
            content = [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
            
            # Generate response - using default safety settings
            response = await asyncio.to_thread(
                self.model.generate_content,
                content,
                generation_config=self.generation_config
            )
            
            # Create a response object similar to direct Gemini API
            class VertexResponse:
                def __init__(self, text):
                    self.text = text
                    
            # Extract text from the response based on its format
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text
            else:
                response_text = str(response)
                
            return VertexResponse(response_text)
            
        except Exception as e:
            logging.error(f"Vertex AI error: {str(e)}")
            if "safety" in str(e).lower():
                raise GeminiAPIException(f"Content was filtered due to safety settings. Please rephrase your query.")
            elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise GeminiAPIException("Rate limit exceeded. Please try again later.")
            else:
                raise GeminiAPIException(f"Error generating response with Vertex AI: {str(e)}")
                
    async def generate_content_stream(self, prompt: str, safety_settings: List[Dict] = None) -> AsyncGenerator[str, None]:
        """Generate streaming content using Vertex AI"""
        try:
            # Create content from prompt with proper role formatting
            content = [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
            
            # Generate streaming response - using default safety settings
            stream = await asyncio.to_thread(
                self.model.generate_content,
                content,
                generation_config=self.generation_config,
                stream=True
            )
            
            # Process the streaming response
            for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in Vertex AI streaming response: {e}")
            yield f"Error: {str(e)}"

class AIService:
    def __init__(self, storage_service: StorageService):
        config = ConfigService()
        self.model = AIModelFactory.create_model(config)
        self.storage_service = storage_service
        self.history = []
        self.max_history = 5
        self.relevant_articles = []
        
    async def generate_custom_content(self, prompt: str) -> str:
        """Generate content from a custom prompt without using RAG
        
        Args:
            prompt: The complete prompt to send to the model
            
        Returns:
            The generated text response
        """
        try:
            # Generate response using the model with default safety settings
            response = await self.model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text'):
                raise GeminiAPIException("Empty response from AI model")
                
            return response.text.strip()
            
        except Exception as e:
            logging.error(f"Error generating custom content: {e}")
            return f"Error generating insights: {str(e)}"
            
    async def generate_rag_response(self, query: str, use_history: bool = True) -> AIResponse:
        """Generate a response using RAG with streaming"""
        try:
            # Get relevant articles
            self.relevant_articles = await self.storage_service.query_articles(
                query, 
                n_results=5
            )
            
            # Build context from relevant articles
            context = self._build_context(self.relevant_articles)
            
            # Check if we have enough context
            if not context or len(context.strip()) < 50:
                logging.warning(f"Insufficient context found for query: {query}")
                context = "No relevant articles found in the database."
            
            # Use conversation history if enabled
            if use_history and self.history:
                history = self._format_history()
                prompt = f"{history}\n\nNew {PromptTemplate.RAG_QUERY.value.format(query=query, context=context)}"
            else:
                prompt = PromptTemplate.RAG_QUERY.value.format(query=query, context=context)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Update conversation history if enabled
            if use_history:
                self._update_history(query, response.text)
            
            # Return formatted response with sources
            return AIResponse(
                text=response.text,
                sources=response.sources,
                confidence=response.confidence
            )
        except Exception as e:
            logging.error(f"Error in generate_rag_response: {e}")
            raise AIServiceException(f"Failed to generate RAG response: {str(e)}")

    async def generate_streaming_response(self, query: str, use_history: bool = True) -> AsyncGenerator[str, None]:
        """Generate a streaming response using RAG"""
        try:
            # Get relevant articles
            self.relevant_articles = await self.storage_service.query_articles(
                query, 
                n_results=5
            )
            
            # Build context from relevant articles
            context = self._build_context(self.relevant_articles)
            
            # Check if we have enough context
            if not context or len(context.strip()) < 50:
                logging.warning(f"Insufficient context found for query: {query}")
                context = "No relevant articles found in the database."
            
            # Use conversation history if enabled
            if use_history and self.history:
                history = self._format_history()
                prompt = f"{history}\n\nNew {PromptTemplate.RAG_QUERY.value.format(query=query, context=context)}"
            else:
                prompt = PromptTemplate.RAG_QUERY.value.format(query=query, context=context)
            
            # Generate streaming response
            response_text = ""
            async for chunk in self._generate_streaming_response(prompt):
                response_text += chunk
                yield chunk
            
            # Extract sources and update history after complete response
            sources = self._extract_sources(response_text)
            self._update_history(query, response_text)
            
            # Yield a special marker with the sources as JSON
            yield f"__SOURCES__{json.dumps(sources)}"
            
        except Exception as e:
            logging.error(f"Error in streaming RAG response: {e}")
            yield f"Error: {str(e)}"

    def _build_context(self, articles: List[Dict]) -> str:
        """Build context from relevant articles with improved formatting"""
        if not articles:
            return "No relevant articles found."
            
        # Sort articles by relevance score
        sorted_articles = sorted(articles, key=lambda x: x.get('distance', 1.0))
        
        context_parts = []
        for i, article in enumerate(sorted_articles, 1):
            # Extract key information
            title = article['metadata'].get('title', 'Untitled')
            topic = article['metadata'].get('topic', 'Unknown')
            date = article['metadata'].get('pub_date', 'Unknown date')
            source = article['metadata'].get('source', 'Unknown source')
            content = article['document']
            
            # Format the article with clear structure
            context_parts.append(f"""Article {i}:
Title: {title}
Topic: {topic}
Date: {date}
Source: {source}
Content: {content}
URL: {article['metadata'].get('link', 'No URL available')}
""")
        
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build a comprehensive prompt with improved instructions"""
        template = PromptTemplate.RAG_QUERY.value
        return template.format(query=query, context=context)

    @retry(
        retry=retry_if_exception_type(GeminiAPIException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _generate_response(self, prompt: str) -> AIResponse:
        """Generate a response with retry logic for API errors"""
        try:
            start_time = time.time()
            
            # Generate response using the model with default safety settings
            response = await self.model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text'):
                raise GeminiAPIException("Empty response from AI model")
                
            response_text = response.text.strip()
            
            # Extract sources from the response
            sources = self._extract_sources(response_text)
            
            # Calculate confidence based on response properties
            confidence = self._calculate_confidence(response, len(self.relevant_articles))
            
            return AIResponse(
                text=response_text,
                sources=sources,
                confidence=confidence
            )
            
        except Exception as e:
            if "safety" in str(e).lower():
                # Handle safety filter issues
                logging.warning(f"Safety filter triggered: {e}")
                raise GeminiAPIException(f"Content was filtered due to safety settings. Please rephrase your query.")
            elif "rate limit" in str(e).lower():
                # Handle rate limiting
                logging.warning(f"Rate limit exceeded: {e}")
                raise GeminiAPIException("Rate limit exceeded. Please try again later.")
            else:
                # Handle other API errors
                logging.error(f"AI model error: {e}")
                raise GeminiAPIException(f"Error generating response: {str(e)}")

    async def _generate_streaming_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the AI model"""
        try:
            # Generate streaming content using the model with default safety settings
            async for chunk in self.model.generate_content_stream(prompt):
                yield chunk
                
        except Exception as e:
            logging.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"

    def _extract_sources(self, response_text: str) -> List[Dict]:
        """Extract article references from the response text"""
        sources = []
        
        # Look for article references in the format [Article X]
        article_refs = set(re.findall(r'\[Article (\d+)\]', response_text))
        
        for ref in article_refs:
            try:
                article_idx = int(ref) - 1
                if 0 <= article_idx < len(self.relevant_articles):
                    article = self.relevant_articles[article_idx]
                    sources.append({
                        "title": article['metadata'].get('title', 'Unknown title'),
                        "source": article['metadata'].get('source', 'Unknown source'),
                        "date": article['metadata'].get('pub_date', 'Unknown date'),
                        "link": article['metadata'].get('link', '')
                    })
            except (ValueError, IndexError):
                continue
                
        return sources

    def _calculate_confidence(self, response, num_sources: int) -> float:
        """Calculate a confidence score based on response properties"""
        # Base confidence on number of sources and response quality
        base_confidence = 0.7  # Start with a reasonable base
        
        # Adjust based on number of sources
        if num_sources == 0:
            source_factor = 0.3  # Low confidence if no sources
        elif num_sources <= 2:
            source_factor = 0.7  # Medium confidence with few sources
        else:
            source_factor = 0.9  # High confidence with many sources
            
        # Final confidence calculation
        confidence = min(base_confidence * source_factor, 0.99)  # Cap at 0.99
        
        return confidence

    def _format_history(self) -> str:
        """Format conversation history for context"""
        formatted_history = []
        for entry in self.history:
            formatted_history.append(f"User: {entry['query']}")
            formatted_history.append(f"Assistant: {entry['response']}")
        return "\n".join(formatted_history)

    def _update_history(self, query: str, response: str):
        """Update conversation history"""
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history length
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    async def clear_history(self):
        """Clear conversation history"""
        self.history = []

    async def summarize_with_context(self, text: str, topic: str) -> str:
        """Summarize text with context from similar articles"""
        try:
            # Find articles on the same topic
            topic_articles = await self.storage_service.query_articles(
                topic,
                n_results=3
            )
            
            # Build context
            context = ""
            if topic_articles:
                context = "Here are some similar articles for context:\n\n"
                for i, article in enumerate(topic_articles):
                    context += f"Article {i+1}:\n"
                    context += f"Title: {article.get('metadata', {}).get('title', 'Unknown')}\n"
                    context += f"Summary: {article.get('metadata', {}).get('summary', 'No summary available')}\n\n"
            
            # Prepare prompt
            prompt = PromptTemplate.SUMMARY.value.format(
                text=text,
                topic=topic,
                context=context
            )
            
            # Generate summary
            summary = await self.model.generate_content(prompt)
            return summary
        
        except Exception as e:
            logging.error(f"Error summarizing text with context: {e}")
            return "Error generating summary. Please try again later." 