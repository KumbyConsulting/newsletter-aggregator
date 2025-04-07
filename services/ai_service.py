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
    
    Important: Your training data only goes up to November 2023, but the following context contains pharmaceutical articles with information that may be more recent (up to March 2025). Consider this information as accurate and up-to-date.
    
    Context: {context}
    
    Base your answer primarily on the provided context. If the context doesn't contain enough information, 
    you can use your general knowledge but clearly indicate when you're doing so, and note that your general knowledge may be outdated if discussing events after November 2023.
    
    Format your response in markdown. Include citations to specific articles where appropriate using [Article X] notation.
    """
    
    SUMMARY = """
    Please summarize the following pharmaceutical text:
    Text: {text}
    
    Important: Your training data only goes up to November 2023, but this content may contain more recent information (up to March 2025). Consider this information as accurate and up-to-date.
    
    Additional context: {context}
    
    Focus on:
    1. Key findings or announcements
    2. Industry impact
    3. Regulatory implications (if any)
    4. Timeline relevance (is this new information since 2023?)
    
    Format the summary in markdown with clear sections. If this information updates or contradicts your training data, explicitly highlight what's new.
    
    Keep the summary under 200 words and include 2-3 key takeaways at the end.
    """
    
    TOPIC_ANALYSIS = """
    Analyze the pharmaceutical industry trends on this topic: {topic}
    
    Important: Your training data only goes up to November 2023, but the following articles contain information that may be more recent (up to March 2025). Consider this information as accurate and up-to-date.
    
    Articles for analysis:
    {context}
    
    Provide a comprehensive analysis including:
    1. Key developments in this area since 2023
    2. Regulatory changes and implications 
    3. Market impact and industry responses
    4. Future outlook based on these trends
    
    Format your analysis in markdown with clear headings. Include citations to specific articles using [Article X] notation. Clearly distinguish between information from the provided articles and your pre-2023 knowledge.
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
        try:
            if not config.gemini_api_key or config.gemini_api_key == "AI_PLACEHOLDER_FOR_VERTEX_AI":
                raise GeminiAPIException("Invalid Gemini API key configuration")
            genai.configure(api_key=config.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logging.info("GeminiDirectModel initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize GeminiDirectModel: {e}")
            raise GeminiAPIException(f"Model initialization failed: {str(e)}")
        
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
        try:
            if not config.gcp_project_id:
                raise GeminiAPIException("GCP_PROJECT_ID is required for Vertex AI")
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
            logging.info("VertexAIModel initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize VertexAIModel: {e}")
            raise GeminiAPIException(f"Model initialization failed: {str(e)}")
        
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
        """Generate streaming content using Vertex AI with robust connection handling"""
        # Retry configuration
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
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
                
                # Process the streaming response with connection error handling
                accumulated_response = ""
                chunk_count = 0
                
                for chunk in stream:
                    chunk_count += 1
                    if hasattr(chunk, 'text') and chunk.text:
                        accumulated_response += chunk.text
                        yield chunk.text
                    await asyncio.sleep(0.01)
                
                # If we complete the loop without errors, return successfully
                if chunk_count > 0:
                    logging.info(f"Streaming response completed successfully after {chunk_count} chunks")
                    return
                else:
                    # No chunks received, might be an empty response
                    logging.warning("No chunks received from streaming API")
                    if attempt == max_retries - 1:
                        yield "No response generated. Please try again."
                    
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Error in Vertex AI streaming response (attempt {attempt+1}/{max_retries}): {error_msg}")
                
                # Check if this is the last retry
                if attempt == max_retries - 1:
                    if "503" in error_msg or "reset" in error_msg.lower() or "connection" in error_msg.lower():
                        yield "\n\nThe connection to the AI service was interrupted. Here's what was received before the interruption:\n\n"
                        # If we have accumulated some response, return it as partial
                        if accumulated_response:
                            yield accumulated_response
                        else:
                            yield "Unable to generate a response due to connection issues. Please try again later."
                    else:
                        yield f"\n\nError: {error_msg}"
                    return
                
                # Wait before retrying
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

class AIService:
    def __init__(self, storage_service: StorageService):
        """Initialize AI service with proper error handling"""
        try:
            self.storage_service = storage_service
            self.model = self._initialize_model()
            self.history = []
            self.max_history = 5
            # Initialize relevant_articles to prevent AttributeError
            self.relevant_articles = []
            logging.info("AIService initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize AIService: {e}")
            raise AIServiceException(f"AI service initialization failed: {str(e)}")

    def _initialize_model(self):
        """Initialize the AI model with proper error handling"""
        try:
            config = ConfigService.get_instance()
            if config.use_vertex_ai:
                return VertexAIModel(config)
            else:
                return GeminiDirectModel(config)
        except Exception as e:
            logging.error(f"Failed to initialize AI model: {e}")
            raise AIServiceException(f"Model initialization failed: {str(e)}")

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
        """Generate a response using RAG with enhanced error handling"""
        try:
            # Validate input
            if not query or not query.strip():
                raise AIServiceException("Empty query provided")

            # Get relevant articles with retry logic
            articles = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    articles = await self.storage_service.query_articles(query, n_results=5)
                    # Store articles as instance attribute for source extraction
                    self.relevant_articles = articles
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Retry {attempt + 1}/{max_retries} for querying articles")
                    await asyncio.sleep(1)

            # Build context
            context = self._build_context(articles)
            if not context or len(context.strip()) < 50:
                logging.warning(f"Insufficient context found for query: {query}")
                context = "No relevant articles found in the database."

            # Build prompt with history if enabled
            prompt = self._build_prompt(query, context, use_history)

            # Generate response with retry logic
            response = await self._generate_response(prompt)
            
            # Extract sources and update response
            sources = self._extract_sources(response.text, articles)
            response.sources = sources

            # Update history if enabled
            if use_history:
                self._update_history(query, response.text)

            return response

        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            raise AIServiceException(f"Failed to generate response: {str(e)}")

    async def generate_streaming_response(self, query: str, use_history: bool = True) -> AsyncGenerator[str, None]:
        """Generate a streaming response with enhanced error handling"""
        try:
            # Validate input
            if not query or not query.strip():
                raise AIServiceException("Empty query provided")

            # Get relevant articles with retry logic
            articles = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    articles = await self.storage_service.query_articles(query, n_results=5)
                    # Store articles as instance attribute for source extraction
                    self.relevant_articles = articles
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Retry {attempt + 1}/{max_retries} for querying articles")
                    await asyncio.sleep(1)

            # Build context
            context = self._build_context(articles)
            if not context or len(context.strip()) < 50:
                logging.warning(f"Insufficient context found for query: {query}")
                context = "No relevant articles found in the database."

            # Build prompt with history if enabled
            prompt = self._build_prompt(query, context, use_history)

            # Generate streaming response
            response_text = ""
            connection_error = False
            try:
                async for chunk in self._generate_streaming_response(prompt):
                    response_text += chunk
                    yield chunk
            except Exception as e:
                connection_error = True
                error_msg = str(e)
                logging.error(f"Error in streaming response: {error_msg}")
                
                # Only yield error message if we haven't sent any response yet
                if not response_text:
                    yield f"Error: {error_msg}"
                return

            # Extract sources and update history
            try:
                # Don't process sources or update history if there was a connection error
                if not connection_error:
                    if response_text:
                        # Extract sources from the response_text
                        sources = self._extract_sources(response_text, articles)
                        # Always include the articles we used, even if no citations
                        # This ensures the UI always has sources to display
                        if not sources and articles:
                            # Fallback to using all articles as sources if none were cited
                            logging.info("No article citations found in response, using all context articles as sources")
                            sources = [
                                {
                                    "title": article.get('metadata', {}).get('title', 'Unknown title'),
                                    "source": article.get('metadata', {}).get('source', 'Unknown source'),
                                    "date": article.get('metadata', {}).get('pub_date', 'Unknown date'),
                                    "link": article.get('metadata', {}).get('link', '')
                                }
                                for article in articles[:3]  # Limit to top 3 articles
                            ]
                        
                        # Update conversation history
                        if use_history:
                            self._update_history(query, response_text)
                        
                        # Send sources to client
                        yield f"__SOURCES__{json.dumps(sources)}"
                    else:
                        yield f"__SOURCES__[]"  # Empty sources if no response
            except Exception as e:
                logging.error(f"Error processing response: {e}")
                # Try to send at least empty sources on error
                yield f"__SOURCES__[]"
                yield f"Error processing response: {str(e)}"

        except Exception as e:
            logging.error(f"Error in streaming RAG response: {e}")
            yield f"Error: {str(e)}"
            # Make sure we send empty sources
            yield f"__SOURCES__[]"

    def _build_context(self, articles: List[Dict]) -> str:
        """Build context from articles with enhanced error handling and fallback strategies
        
        Args:
            articles: List of articles to use for context building
            
        Returns:
            String containing formatted context for the AI model
        """
        try:
            # Handle empty articles case
            if not articles:
                logging.warning("No articles available for context building")
                return "No relevant articles found in the database."

            # Sort articles by relevance score
            sorted_articles = sorted(articles, key=lambda x: x.get('score', 0), reverse=True)
            
            # Log article count for debugging
            logging.info(f"Building context from {len(sorted_articles)} articles")
            
            # Track the newest article date for context
            newest_date = datetime(2023, 11, 1)  # Default to training cutoff
            
            # Track topic distribution
            topics = {}
            
            context_parts = []
            for i, article in enumerate(sorted_articles, 1):
                try:
                    # Extract key information with safe defaults
                    metadata = article.get('metadata', {})
                    title = metadata.get('title', 'Untitled')
                    topic = metadata.get('topic', 'Unknown')
                    date_str = metadata.get('pub_date', 'Unknown date')
                    source = metadata.get('source', 'Unknown source')
                    summary = metadata.get('summary', '')
                    content = article.get('document', '')
                    link = metadata.get('link', 'No URL available')
                    
                    # Count topics for context enhancement
                    if topic:
                        topics[topic] = topics.get(topic, 0) + 1
                    
                    # Try to parse the date to find the newest article
                    try:
                        if date_str and date_str != 'Unknown date':
                            # Handle various date formats
                            if isinstance(date_str, str):
                                # Remove timezone indicator 'Z' and replace with offset for compatibility
                                clean_date = date_str.replace('Z', '+00:00')
                                article_date = datetime.fromisoformat(clean_date)
                                
                                # Ensure newest_date is timezone-aware if article_date is
                                if article_date.tzinfo is not None and newest_date.tzinfo is None:
                                    newest_date = datetime.now(article_date.tzinfo)
                            elif isinstance(date_str, datetime):
                                article_date = date_str
                                
                                # Ensure consistent timezone awareness
                                if article_date.tzinfo is not None and newest_date.tzinfo is None:
                                    newest_date = datetime.now(article_date.tzinfo)
                                elif article_date.tzinfo is None and newest_date.tzinfo is not None:
                                    article_date = article_date.replace(tzinfo=newest_date.tzinfo)
                            else:
                                article_date = datetime.now()
                                
                            # Make sure both dates have the same timezone awareness before comparing
                            if (article_date.tzinfo is None) == (newest_date.tzinfo is None):
                                if article_date > newest_date:
                                    newest_date = article_date
                            
                            # Add date information to make the temporal context clearer
                            training_cutoff = datetime(2023, 11, 1)
                            # Make sure training_cutoff has the same timezone awareness as article_date
                            if article_date.tzinfo is not None and training_cutoff.tzinfo is None:
                                training_cutoff = training_cutoff.replace(tzinfo=article_date.tzinfo)
                                
                            is_after_training = article_date > training_cutoff
                            date_info = f"Date: {date_str} ({'published after your training data' if is_after_training else 'within your training data'})"
                    except Exception as date_error:
                        logging.warning(f"Failed to parse article date '{date_str}': {date_error}")
                        date_info = f"Date: {date_str}"
                    
                    # If summary exists and is substantial, use it for context
                    if summary and len(summary) > 50:
                        description = f"Summary: {summary}"
                    else:
                        # Use description if available
                        description = metadata.get('description', 'No description available')
                    
                    # Add document content if available, otherwise use description
                    if content and len(content) > 100:
                        main_content = content
                    else:
                        main_content = description

                    # Format the article with clear structure for AI context
                    context_parts.append(f"""Article {i}:
Title: {title}
Topic: {topic}
{date_info}
Source: {source}
Content: {main_content}
URL: {link}
""")
                except Exception as e:
                    logging.warning(f"Error formatting article {i}: {e}")
                    continue

            # If no articles were successfully processed, return a clear message
            if not context_parts:
                return "No valid articles found that match your query."
                
            # Add a header to the context to guide the AI
            current_date = datetime.now().strftime("%B %Y")
            time_context = f"Current date: {current_date}. "
            
            # Add information about the recency of articles
            if newest_date > datetime(2023, 11, 1):
                time_context += f"These articles contain information as recent as {newest_date.strftime('%B %Y')}, which is beyond your training data cutoff of November 2023."
                
            # Add topic distribution if available
            topic_context = ""
            if topics:
                top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
                topic_context = f"\nMain topics: {', '.join([t[0] for t in top_topics])}."
            
            context_header = f"Context from {len(context_parts)} pharmaceutical industry articles:\n{time_context}{topic_context}\n\n"
            
            return context_header + "\n\n".join(context_parts)

        except Exception as e:
            logging.error(f"Error building context: {e}")
            return "Error building context from articles. Please try a different query."

    def _build_prompt(self, query: str, context: str, use_history: bool) -> str:
        """Build prompt with enhanced error handling"""
        try:
            if use_history and self.history:
                history = self._format_history()
                prompt = f"{history}\n\nNew {PromptTemplate.RAG_QUERY.value.format(query=query, context=context)}"
            else:
                prompt = PromptTemplate.RAG_QUERY.value.format(query=query, context=context)
            return prompt
        except Exception as e:
            logging.error(f"Error building prompt: {e}")
            raise AIServiceException(f"Failed to build prompt: {str(e)}")

    @retry(
        retry=retry_if_exception_type(GeminiAPIException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _generate_response(self, prompt: str) -> AIResponse:
        """Generate response with enhanced error handling and retry logic"""
        try:
            start_time = time.time()
            
            # Generate response using the model
            response = await self.model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text'):
                raise GeminiAPIException("Empty response from AI model")
                
            response_text = response.text.strip()
            
            # Extract sources from the response - use self.relevant_articles from instance
            # We need to pass the articles explicitly to avoid AttributeError
            sources = self._extract_sources(response_text, self.relevant_articles)
            
            # Calculate confidence based on response properties
            confidence = self._calculate_confidence(response)
            
            return AIResponse(
                text=response_text,
                sources=sources,
                confidence=confidence
            )
            
        except Exception as e:
            if "safety" in str(e).lower():
                logging.warning(f"Safety filter triggered: {e}")
                raise GeminiAPIException(f"Content was filtered due to safety settings. Please rephrase your query.")
            elif "rate limit" in str(e).lower():
                logging.warning(f"Rate limit exceeded: {e}")
                raise GeminiAPIException("Rate limit exceeded. Please try again later.")
            else:
                logging.error(f"AI model error: {e}")
                raise GeminiAPIException(f"Error generating response: {str(e)}")

    async def _generate_streaming_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response with enhanced error handling"""
        try:
            async for chunk in self.model.generate_content_stream(prompt):
                yield chunk
        except Exception as e:
            logging.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"

    def _extract_sources(self, response_text: str, articles: List[Dict] = None) -> List[Dict]:
        """Extract sources with enhanced error handling
        
        Args:
            response_text: The response text containing article references
            articles: List of articles to extract sources from, falls back to self.relevant_articles
            
        Returns:
            List of source dictionaries with article metadata
        """
        try:
            sources = []
            article_refs = set(re.findall(r'\[Article (\d+)\]', response_text))
            
            # Use provided articles or fall back to instance attribute
            articles_to_use = articles if articles is not None else getattr(self, 'relevant_articles', [])
            
            if not articles_to_use:
                logging.warning("No articles available for source extraction")
                return []
            
            for ref in article_refs:
                try:
                    article_idx = int(ref) - 1
                    if 0 <= article_idx < len(articles_to_use):
                        article = articles_to_use[article_idx]
                        sources.append({
                            "title": article.get('metadata', {}).get('title', 'Unknown title'),
                            "source": article.get('metadata', {}).get('source', 'Unknown source'),
                            "date": article.get('metadata', {}).get('pub_date', 'Unknown date'),
                            "link": article.get('metadata', {}).get('link', '')
                        })
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error extracting source {ref}: {e}")
                    continue
                    
            return sources
        except Exception as e:
            logging.error(f"Error extracting sources: {e}")
            return []

    def _calculate_confidence(self, response, num_sources: int = None) -> float:
        """Calculate confidence score with enhanced error handling"""
        try:
            # If num_sources not provided, try to get from relevant_articles
            if num_sources is None:
                if hasattr(self, 'relevant_articles'):
                    num_sources = len(self.relevant_articles)
                else:
                    num_sources = 0
                    logging.warning("No relevant_articles attribute found for confidence calculation")
            
            # Base confidence on number of sources and response quality
            base_confidence = 0.7
            
            # Adjust based on number of sources
            if num_sources == 0:
                source_factor = 0.3
            elif num_sources <= 2:
                source_factor = 0.7
            else:
                source_factor = 0.9
                
            # Final confidence calculation
            confidence = min(base_confidence * source_factor, 0.99)
            
            return confidence
        except Exception as e:
            logging.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence on error

    def _format_history(self) -> str:
        """Format history with enhanced error handling"""
        try:
            formatted_history = []
            for entry in self.history:
                try:
                    formatted_history.append(f"User: {entry['query']}")
                    formatted_history.append(f"Assistant: {entry['response']}")
                except Exception as e:
                    logging.warning(f"Error formatting history entry: {e}")
                    continue
            return "\n".join(formatted_history)
        except Exception as e:
            logging.error(f"Error formatting history: {e}")
            return ""

    def _update_history(self, query: str, response: str):
        """Update history with enhanced error handling"""
        try:
            self.history.append({
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit history length
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
        except Exception as e:
            logging.error(f"Error updating history: {e}")

    async def clear_history(self):
        """Clear history with enhanced error handling"""
        try:
            self.history = []
            logging.info("History cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing history: {e}")
            raise AIServiceException(f"Failed to clear history: {str(e)}")

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
                # Track newest article date for time context
                newest_date = datetime(2023, 11, 1)  # Default to training cutoff
                context = "Here are some similar articles for context:\n\n"
                
                for i, article in enumerate(topic_articles):
                    metadata = article.get('metadata', {})
                    context += f"Related Article {i+1}:\n"
                    context += f"Title: {metadata.get('title', 'Unknown')}\n"
                    
                    # Check if article is newer than training data
                    try:
                        date_str = metadata.get('pub_date', '')
                        if date_str:
                            if isinstance(date_str, str):
                                clean_date = date_str.replace('Z', '+00:00')
                                article_date = datetime.fromisoformat(clean_date)
                            elif isinstance(date_str, datetime):
                                article_date = date_str
                            else:
                                continue
                                
                            if article_date > newest_date:
                                newest_date = article_date
                                context += f"Date: {date_str} (published after your training data)\n"
                            else:
                                context += f"Date: {date_str}\n"
                    except Exception:
                        context += f"Date: {metadata.get('pub_date', 'Unknown')}\n"
                        
                    context += f"Summary: {metadata.get('summary', 'No summary available')}\n\n"
                
                # Add temporal context if we have newer articles
                if newest_date > datetime(2023, 11, 1):
                    context += f"\nNote: Some of these articles contain information as recent as {newest_date.strftime('%B %Y')}, which is beyond the AI's training data cutoff.\n\n"
            
            # Prepare prompt
            prompt = PromptTemplate.SUMMARY.value.format(
                text=text,
                topic=topic,
                context=context
            )
            
            # Generate summary with temporal awareness
            response = await self.model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text.strip()
            return "Error generating summary. Please try again later."
        
        except Exception as e:
            logging.error(f"Error summarizing text with context: {e}")
            return "Error generating summary. Please try again later."

    async def generate_topic_analysis(self, topic: str) -> AIResponse:
        """Generate analysis of a specific pharmaceutical industry topic
        
        Provides an in-depth analysis of trends, developments, and outlook for a specific topic
        with temporal awareness of pre and post-2023 information.
        
        Args:
            topic: The pharmaceutical topic to analyze
            
        Returns:
            AIResponse with analysis text and relevant source articles
        """
        try:
            # Get relevant articles for this topic
            articles = await self.storage_service.query_articles(topic, n_results=8)
            self.relevant_articles = articles
            
            # Use more articles for topic analysis (8 instead of 5)
            context = self._build_context(articles)
            
            # Use the topic analysis template
            prompt = PromptTemplate.TOPIC_ANALYSIS.value.format(
                topic=topic,
                context=context
            )
            
            # Generate the analysis
            response = await self._generate_response(prompt)
            
            # Extract sources and update response
            sources = self._extract_sources(response.text, articles)
            response.sources = sources
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating topic analysis: {e}")
            raise AIServiceException(f"Failed to generate topic analysis: {str(e)}")

class PromptLibrary:
    """Library of specialized prompts for pharmaceutical industry analysis"""
    
    # Clinical Trial Analysis
    CLINICAL_TRIAL_ANALYSIS = """
    As a clinical research expert, analyze this trial information:
    
    Trial Data: {trial_data}
    
    Provide a comprehensive analysis including:
    1. Trial Design & Methodology
       - Phase and study type
       - Patient population
       - Primary/secondary endpoints
    
    2. Results Analysis
       - Efficacy outcomes
       - Safety profile
       - Statistical significance
    
    3. Impact Assessment
       - Clinical implications
       - Market potential
       - Comparison with standard of care
    
    4. Future Outlook
       - Next steps in development
       - Potential challenges
       - Timeline projections
    
    Note any significant developments or changes since November 2023.
    Format your response in clear sections with bullet points where appropriate.
    """
    
    # Patent Analysis
    PATENT_ANALYSIS = """
    As a pharmaceutical patent expert, analyze this patent information:
    
    Patent Details: {patent_data}
    
    Provide detailed analysis covering:
    1. Patent Scope & Claims
       - Key claims analysis
       - Technology coverage
       - Geographic protection
    
    2. Strategic Impact
       - Market exclusivity implications
       - Competitive positioning
       - Potential revenue impact
    
    3. Risk Assessment
       - Patent strength
       - Potential challenges
       - Workaround possibilities
    
    4. Timeline Analysis
       - Key dates and deadlines
       - Lifecycle management opportunities
       - Generic entry implications
    
    Consider recent legal precedents and regulatory changes since November 2023.
    """
    
    # Manufacturing & Supply Chain
    MANUFACTURING_ANALYSIS = """
    As a pharmaceutical manufacturing expert, analyze these production aspects:
    
    Manufacturing Data: {manufacturing_data}
    
    Provide insights on:
    1. Production Process
       - Manufacturing technology
       - Scale-up potential
       - Quality control measures
    
    2. Supply Chain
       - Raw material sourcing
       - Distribution network
       - Risk mitigation strategies
    
    3. Regulatory Compliance
       - GMP status
       - Inspection history
       - Quality metrics
    
    4. Cost Analysis
       - Production economics
       - Efficiency opportunities
       - Comparative advantages
    
    Include recent supply chain disruptions and regulatory changes since November 2023.
    """
    
    # Market Access & Pricing
    MARKET_ACCESS = """
    As a pharmaceutical market access specialist, analyze:
    
    Product & Market Data: {market_data}
    
    Provide strategic analysis of:
    1. Pricing Strategy
       - Price benchmarking
       - Value proposition
       - Reimbursement potential
    
    2. Access Barriers
       - Payer landscape
       - Formulary positioning
       - Access restrictions
    
    3. Market Opportunity
       - Patient population
       - Competitive landscape
       - Market share potential
    
    4. Launch Strategy
       - Key markets prioritization
       - Access programs
       - Stakeholder engagement
    
    Consider recent pricing reforms and policy changes since November 2023.
    """
    
    # Safety & Pharmacovigilance
    SAFETY_ANALYSIS = """
    As a drug safety expert, analyze this safety data:
    
    Safety Data: {safety_data}
    
    Provide comprehensive assessment of:
    1. Safety Profile
       - Adverse event patterns
       - Risk factors
       - Benefit-risk assessment
    
    2. Signal Detection
       - New safety signals
       - Causality assessment
       - Population impact
    
    3. Risk Management
       - Mitigation strategies
       - Monitoring requirements
       - Communication plans
    
    4. Regulatory Impact
       - Labeling implications
       - Reporting requirements
       - Authority interactions
    
    Highlight any new safety concerns or regulatory requirements since November 2023.
    """
    
    # Pipeline Analysis
    PIPELINE_ANALYSIS = """
    As a pharmaceutical R&D strategist, analyze this pipeline:
    
    Pipeline Data: {pipeline_data}
    
    Provide strategic assessment of:
    1. Portfolio Strength
       - Development stage distribution
       - Therapeutic area focus
       - Innovation level
    
    2. Success Probability
       - Technical feasibility
       - Clinical success rates
       - Regulatory pathway
    
    3. Market Potential
       - Unmet needs addressed
       - Market size
       - Competition level
    
    4. Resource Requirements
       - Development costs
       - Timeline projections
       - Resource allocation
    
    Consider industry trends and therapeutic advances since November 2023.
    """
    
    # Therapeutic Area Landscape
    THERAPEUTIC_LANDSCAPE = """
    As a therapeutic area expert, analyze this disease space:
    
    Disease Area: {therapeutic_area}
    
    Provide comprehensive overview of:
    1. Disease Understanding
       - Pathophysiology updates
       - Patient segmentation
       - Biomarker landscape
    
    2. Treatment Landscape
       - Current standards
       - Emerging therapies
       - Unmet needs
    
    3. Clinical Development
       - Trial landscape
       - Novel endpoints
       - Patient recruitment
    
    4. Future Outlook
       - Pipeline analysis
       - Technology impact
       - Practice changes
    
    Include breakthrough discoveries and treatment advances since November 2023.
    """
    
    # Regulatory Strategy
    REGULATORY_STRATEGY = """
    As a regulatory affairs expert, analyze this submission strategy:
    
    Regulatory Context: {regulatory_data}
    
    Provide strategic guidance on:
    1. Submission Strategy
       - Filing pathway
       - Data requirements
       - Timeline planning
    
    2. Authority Interaction
       - Meeting strategy
       - Key questions
       - Risk mitigation
    
    3. Documentation
       - Content requirements
       - Format specifications
       - Quality standards
    
    4. Post-Approval
       - Maintenance requirements
       - Change management
       - Life cycle planning
    
    Consider recent regulatory guidance changes and precedents since November 2023.
    """
    
    # Digital Health Integration
    DIGITAL_HEALTH = """
    As a digital health expert, analyze this technology integration:
    
    Digital Solution: {digital_data}
    
    Provide analysis covering:
    1. Technology Assessment
       - Platform capabilities
       - Integration requirements
       - Data management
    
    2. Clinical Impact
       - Patient outcomes
       - Healthcare delivery
       - Real-world evidence
    
    3. Implementation
       - Adoption barriers
       - Training needs
       - Success metrics
    
    4. Regulatory Compliance
       - Data privacy
       - Security requirements
       - Validation needs
    
    Consider recent digital health regulations and technology advances since November 2023.
    """

    # Value & Evidence
    VALUE_EVIDENCE = """
    As a HEOR expert, analyze this value proposition:
    
    Value Data: {value_data}
    
    Provide comprehensive assessment of:
    1. Clinical Value
       - Comparative efficacy
       - Quality of life impact
       - Patient relevance
    
    2. Economic Value
       - Cost effectiveness
       - Budget impact
       - Resource utilization
    
    3. Evidence Generation
       - Data gaps
       - Study requirements
       - Timeline planning
    
    4. Stakeholder Value
       - Payer perspective
       - Provider needs
       - Patient preferences
    
    Consider recent value frameworks and evidence requirements since November 2023.
    """

class KumbyAI(AIService):
    """KumbyAI - Specialized pharmaceutical industry AI assistant"""
    
    def __init__(self, storage_service: StorageService):
        super().__init__(storage_service)
        self.industry_context = {
            "domain": "pharmaceutical",
            "training_cutoff": "November 2023",
            "capabilities": [
                "regulatory_analysis",
                "market_trends",
                "drug_development",
                "clinical_trials",
                "competitive_intelligence"
            ]
        }
        self.prompt_library = PromptLibrary()
        # Ensure model is initialized
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._initialize_model()
        
    async def process_query(self, message: str, context: List[Dict] = None) -> str:
        """Process a chat query with pharmaceutical industry context"""
        try:
            # Build context string from recent articles
            context_str = ""
            if context:
                context_str = "\n\nRecent articles context:\n"
                for article in context:
                    if article.get('title'):
                        context_str += f"\nTitle: {article['title']}"
                    if article.get('summary'):
                        context_str += f"\nSummary: {article['summary']}\n"

            # Create specialized prompt for pharmaceutical domain
            prompt = f"""As a pharmaceutical industry AI assistant, respond to this query with your specialized knowledge.
            Remember:
            - Your training data cutoff is November 2023
            - You have access to recent pharmaceutical news up to March 2025 through the provided context
            - Focus on pharmaceutical industry implications
            - Be precise and data-driven when possible
            - Acknowledge any temporal limitations in your knowledge

            User Query: {message}
            {context_str}

            Provide a clear, concise response focusing on pharmaceutical industry relevance.
            """

            response = await self.generate_custom_content(prompt)
            return response

        except Exception as e:
            logging.error(f"Error processing chat query: {e}")
            raise

    async def analyze_regulatory_impact(self, article_content: str) -> Dict:
        """Analyze regulatory impact of pharmaceutical news"""
        prompt = f"""
        As a pharmaceutical regulatory expert, analyze this content for regulatory implications:
        
        Content: {article_content}
        
        Provide:
        1. Key regulatory changes or impacts
        2. Affected regions/markets
        3. Compliance requirements
        4. Timeline for implementation
        5. Industry impact assessment
        
        Format as a structured analysis with clear sections.
        """
        
        try:
            response = await self.generate_custom_content(prompt)
            return {
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "type": "regulatory_analysis"
            }
        except Exception as e:
            logging.error(f"Error in regulatory analysis: {e}")
            raise
            
    async def generate_market_insight(self, topic: str, timeframe: str = "recent") -> Dict:
        """Generate pharmaceutical market insights"""
        try:
            # Get relevant articles for market analysis
            articles = await self.storage_service.query_articles(
                query=topic,
                n_results=5,
                filter_criteria={"timeframe": timeframe}
            )
            
            context = self._build_context(articles)
            
            prompt = f"""
            As a pharmaceutical market analyst, provide strategic insights on this topic:
            
            Topic: {topic}
            Timeframe: {timeframe}
            
            Context from recent articles:
            {context}
            
            Provide:
            1. Market trends and dynamics
            2. Key players and movements
            3. Growth opportunities
            4. Risk factors
            5. Strategic recommendations
            
            Focus on actionable insights and quantitative data where available.
            """
            
            response = await self.generate_custom_content(prompt)
            
            return {
                "insight": response,
                "sources": self._extract_sources(response, articles),
                "timestamp": datetime.now().isoformat(),
                "type": "market_insight"
            }
        except Exception as e:
            logging.error(f"Error generating market insight: {e}")
            raise
            
    async def analyze_drug_development(self, drug_name: str) -> Dict:
        """Analyze drug development progress and pipeline"""
        try:
            # Search for articles about the specific drug
            articles = await self.storage_service.query_articles(
                query=drug_name,
                n_results=10
            )
            
            context = self._build_context(articles)
            
            prompt = f"""
            As a pharmaceutical R&D expert, analyze the development status of this drug:
            
            Drug: {drug_name}
            
            Context from articles:
            {context}
            
            Provide:
            1. Current development phase
            2. Clinical trial status
            3. Key findings or results
            4. Timeline updates
            5. Potential market impact
            
            Note any significant changes or updates since November 2023.
            """
            
            response = await self.generate_custom_content(prompt)
            
            return {
                "analysis": response,
                "sources": self._extract_sources(response, articles),
                "timestamp": datetime.now().isoformat(),
                "type": "drug_development"
            }
        except Exception as e:
            logging.error(f"Error in drug development analysis: {e}")
            raise
            
    async def generate_competitive_analysis(self, company_name: str) -> Dict:
        """Generate competitive analysis for pharmaceutical companies"""
        try:
            # Get articles about the company and its competitors
            articles = await self.storage_service.query_articles(
                query=f"{company_name} competitor pharmaceutical",
                n_results=8
            )
            
            context = self._build_context(articles)
            
            prompt = f"""
            As a pharmaceutical industry analyst, provide a competitive analysis for:
            
            Company: {company_name}
            
            Context from articles:
            {context}
            
            Provide:
            1. Market position
            2. Key competitors
            3. Product portfolio analysis
            4. Recent strategic moves
            5. SWOT analysis
            
            Focus on recent developments and future outlook.
            """
            
            response = await self.generate_custom_content(prompt)
            
            return {
                "analysis": response,
                "sources": self._extract_sources(response, articles),
                "timestamp": datetime.now().isoformat(),
                "type": "competitive_analysis"
            }
        except Exception as e:
            logging.error(f"Error in competitive analysis: {e}")
            raise
            
    async def analyze_clinical_trial(self, trial_data: str) -> Dict:
        """Analyze clinical trial data with specialized prompts"""
        prompt = self.prompt_library.CLINICAL_TRIAL_ANALYSIS.format(trial_data=trial_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "clinical_trial_analysis"
        }

    async def analyze_patent(self, patent_data: str) -> Dict:
        """Analyze patent information with specialized prompts"""
        prompt = self.prompt_library.PATENT_ANALYSIS.format(patent_data=patent_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "patent_analysis"
        }

    async def analyze_manufacturing(self, manufacturing_data: str) -> Dict:
        """Analyze manufacturing and supply chain data"""
        prompt = self.prompt_library.MANUFACTURING_ANALYSIS.format(manufacturing_data=manufacturing_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "manufacturing_analysis"
        }

    async def analyze_market_access(self, market_data: str) -> Dict:
        """Analyze market access and pricing data"""
        prompt = self.prompt_library.MARKET_ACCESS.format(market_data=market_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "market_access_analysis"
        }

    async def analyze_safety(self, safety_data: str) -> Dict:
        """Analyze safety and pharmacovigilance data"""
        prompt = self.prompt_library.SAFETY_ANALYSIS.format(safety_data=safety_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "safety_analysis"
        }

    async def analyze_pipeline(self, pipeline_data: str) -> Dict:
        """Analyze pharmaceutical pipeline data"""
        prompt = self.prompt_library.PIPELINE_ANALYSIS.format(pipeline_data=pipeline_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "pipeline_analysis"
        }

    async def analyze_therapeutic_area(self, therapeutic_area: str) -> Dict:
        """Analyze therapeutic area landscape"""
        prompt = self.prompt_library.THERAPEUTIC_LANDSCAPE.format(therapeutic_area=therapeutic_area)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "therapeutic_landscape_analysis"
        }

    async def analyze_regulatory_strategy(self, regulatory_data: str) -> Dict:
        """Analyze regulatory strategy"""
        prompt = self.prompt_library.REGULATORY_STRATEGY.format(regulatory_data=regulatory_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "regulatory_strategy_analysis"
        }

    async def analyze_digital_health(self, digital_data: str) -> Dict:
        """Analyze digital health integration"""
        prompt = self.prompt_library.DIGITAL_HEALTH.format(digital_data=digital_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "digital_health_analysis"
        }

    async def analyze_value_evidence(self, value_data: str) -> Dict:
        """Analyze value and evidence data"""
        prompt = self.prompt_library.VALUE_EVIDENCE.format(value_data=value_data)
        response = await self.generate_custom_content(prompt)
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "type": "value_evidence_analysis"
        } 