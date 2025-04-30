from typing import List, Dict, Optional, Tuple, AsyncGenerator
import google.generativeai as genai
from .storage_service import StorageService
import logging
from datetime import datetime, timedelta
import json
import asyncio
from .config_service import ConfigService
from newsLetter import RateLimitException
from dataclasses import dataclass
from enum import Enum
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp
import math
import httpx

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
    
    {output_format}
    Include citations to specific articles where appropriate using [Article X] notation.
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
    
    {output_format}
    If this information updates or contradicts your training data, explicitly highlight what's new.
    
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
    
    {output_format}
    Include citations to specific articles using [Article X] notation. Clearly distinguish between information from the provided articles and your pre-2023 knowledge.
    """
    
    COMPREHENSIVE_ANALYSIS = """
    Analyze this article comprehensively, considering your pharmaceutical industry expertise:
    
    Article Content:
    {content}
    
    Topic: {topic}
    Date: {date}
    
    Important: Your training data goes up to November 2023, but you have access to more recent information through the provided context. Consider all information as accurate and up-to-date.
    
    Related Articles Context:
    {context}
    
    Provide a comprehensive analysis including:
    1. Executive Summary (2-3 sentences)
    2. Key Findings and Implications
    3. Industry Impact Analysis
       - Market dynamics
       - Competitive landscape
       - Regulatory considerations
    4. Future Outlook
       - Short-term implications (0-6 months)
       - Long-term considerations (6+ months)
    5. Related Trends and Patterns
    6. Action Points for Stakeholders
    
    Format your analysis in markdown. Include citations to specific articles using [Article X] notation.
    Clearly distinguish between information from the provided article, related context, and your pre-2023 knowledge.
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
        model_type = config.ai_model_type.lower()
        
        if model_type == "gemini_direct":
            return GeminiDirectModel(config)
        elif model_type == "vertex_ai":
            return VertexAIModel(config)
        elif model_type == "gemini_fine_tuned":
            return GeminiFineTunedModel(config)
        else:
            logging.warning(f"Unknown model type {model_type}, defaulting to GeminiDirectModel")
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
        """Initialize the model with appropriate configuration."""
        try:
            self.config = config
            self.api_key = config.gemini_api_key
            
            # Log API key configuration (safely)
            if not self.api_key:
                logging.error("No API key provided. Check GEMINI_API_KEY in environment/config")
            elif self.api_key == "AI_PLACEHOLDER_FOR_VERTEX_AI":
                logging.error("Using placeholder API key. GEMINI_API_KEY is not properly configured")
            else:
                masked_key = self.api_key[:4] + '*' * 6 + self.api_key[-4:] if len(self.api_key) > 10 else '***'
                logging.info(f"Gemini API key configured: {masked_key}")
            
            # Initialize the model settings
            self.model_name = "gemini-2.0-flash"  # Using the latest model version
            self.api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            logging.info(f"Using Gemini model: {self.model_name}")
            
            # Initialize Gemini API for streaming requests
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.model_name)
            
            # Safety settings - use defaults
            self.default_safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            # Additional configuration parameters
            self.temperature = 0.4  # Slightly lower for more deterministic responses
            self.max_tokens = 2048
            
            logging.info(f"GeminiDirectModel initialized with temperature={self.temperature}, max_tokens={self.max_tokens}")
            
        except Exception as e:
            logging.error(f"Error initializing GeminiDirectModel: {e}", exc_info=True)
            raise GeminiAPIException(f"Failed to initialize Gemini model: {e}")
        
    @retry(
        retry=retry_if_exception_type((GeminiAPIException, asyncio.TimeoutError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        try:
            # Log the request parameters
            logging.info(f"Gemini API request starting - prompt length: {len(prompt)}")
            logging.debug(f"Gemini API request prompt (first 100 chars): {prompt[:100]}...")
            
            # Apply timeout
            start_time = time.time()
            timeout = self.config.ai_timeout

            # Use safety settings if provided, otherwise use defaults
            effective_safety_settings = safety_settings or self.default_safety_settings
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Prepare request payload
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "safetySettings": effective_safety_settings,
                    "generationConfig": {
                        "temperature": self.temperature,
                        "maxOutputTokens": self.max_tokens,
                        "topP": 0.95,
                        "topK": 40
                    }
                }
                
                # Log the full request payload for debugging
                logging.debug(f"Gemini API full request payload: {json.dumps(payload)}")
                
                # Making API request
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                
                response = await client.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers
                )
                
                # Log raw response for debugging
                logging.debug(f"Gemini API raw response status: {response.status_code}")
                logging.debug(f"Gemini API raw response headers: {response.headers}")
                try:
                    response_data = response.json()
                    logging.debug(f"Gemini API response data: {json.dumps(response_data)}")
                except Exception as json_err:
                    logging.error(f"Failed to parse response as JSON: {json_err}")
                    logging.debug(f"Raw response text: {response.text}")
                
                # Error handling
                if response.status_code != 200:
                    error_message = f"Gemini API error ({response.status_code}): {response.text}"
                    logging.error(error_message)
                    if response.status_code == 400:
                        try:
                            error_detail = response.json().get("error", {}).get("message", "Unknown 400 error")
                            logging.error(f"Gemini API 400 error detail: {error_detail}")
                        except:
                            pass
                    raise GeminiAPIException(error_message)
                
                # Extract text from response
                response_data = response.json()
                try:
                    # Log extraction attempts
                    if 'candidates' not in response_data:
                        logging.error("No 'candidates' field in Gemini API response")
                        logging.debug(f"Response data keys: {response_data.keys()}")
                        raise GeminiAPIException("Missing candidates in response")
                    
                    candidates = response_data.get("candidates", [])
                    if not candidates:
                        logging.error("Empty candidates list in Gemini API response")
                        raise GeminiAPIException("Empty candidates list")
                        
                    logging.debug(f"Number of candidates: {len(candidates)}")
                    
                    candidate = candidates[0]
                    logging.debug(f"First candidate data: {json.dumps(candidate)}")
                    
                    if 'content' not in candidate:
                        logging.error("No 'content' field in candidate")
                        raise GeminiAPIException("Missing content in candidate")
                        
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    if not parts:
                        logging.error("Empty parts list in content")
                        logging.debug(f"Content data: {json.dumps(content)}")
                        raise GeminiAPIException("Empty parts list")
                    
                    text = parts[0].get("text", "")
                    
                    # Check if text is empty
                    if not text.strip():
                        logging.error("Empty text response from Gemini API")
                        logging.debug(f"Full response data: {json.dumps(response_data)}")
                        raise GeminiAPIException("Empty text response")
                        
                    logging.info(f"Gemini API request completed successfully - response length: {len(text)}")
                    logging.debug(f"Response text (first 100 chars): {text[:100]}...")
                    
                    # Calculate request duration
                    duration = time.time() - start_time
                    logging.info(f"Gemini API request duration: {duration:.2f}s")
                    
                    return text
                except (KeyError, IndexError) as e:
                    logging.error(f"Error extracting text from Gemini API response: {e}")
                    logging.error(f"Response structure: {json.dumps(response_data)}")
                    raise GeminiAPIException(f"Failed to extract text from response: {e}")
                
        except httpx.HTTPError as e:
            logging.error(f"HTTP error during Gemini API request: {e}")
            raise GeminiAPIException(f"HTTP error: {e}")
        except asyncio.TimeoutError:
            logging.error(f"Timeout during Gemini API request after {time.time() - start_time:.2f}s")
            raise asyncio.TimeoutError(f"Gemini API request timed out after {timeout}s")
        except Exception as e:
            logging.error(f"Unexpected error in generate_content: {e}", exc_info=True)
            raise GeminiAPIException(f"Unexpected error: {e}")
    
    def _safe_generate_content(self, prompt, safety_settings):
        """Safely make the API call with extra error handling for None responses"""
        try:
            # Make the actual API call - GeminiDirectModel doesn't use generation_config
            if safety_settings:
                response = self.model.generate_content(prompt, safety_settings=safety_settings)
            else:
                response = self.model.generate_content(prompt)
                
            # Validate response isn't None
            if response is None:
                logging.error("Model returned None response")
                # Create a basic response object with error message
                return self._create_fallback_response("Model returned empty response")
                
            return response
        except Exception as e:
            logging.error(f"Error in direct API call: {e}")
            # Return a structured error response instead of raising
            return self._create_fallback_response(f"API error: {str(e)}")
        
    def _create_fallback_response(self, message):
        """Create a fallback response object when the API fails"""
        # Create a minimal response-like object
        class FallbackResponse:
            def __init__(self, message):
                self.text = f"Error: {message}"
                self.candidates = [self]
                self.content = self
                self.parts = [self]
                
        return FallbackResponse(message)

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
            
            # Set generation config first
            self.generation_config = GenerationConfig(
                temperature=0.4,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
            
            # Define prioritized models to try
            model_candidates = [
                "models/gemini-2.0-flash",
                "models/gemini-1.5-pro",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-flash-latest",
                "models/gemini-1.0-pro",
                "models/gemini-pro"
            ]
            
            # Try each model in order
            model_initialized = False
            for model_name in model_candidates:
                try:
                    self.model = GenerativeModel(model_name)
                    logging.info(f"Successfully initialized Vertex AI with {model_name}")
                    model_initialized = True
                    break
                except Exception as model_error:
                    logging.warning(f"Failed to initialize {model_name}: {model_error}")
            
            # If no models from candidates worked, try listing available models
            if not model_initialized:
                try:
                    available_models = GenerativeModel.list_models()
                    model_names = [model.name for model in available_models]
                    logging.info(f"Available Vertex AI models: {model_names}")
                    
                    # Try to find a Gemini model
                    gemini_models = [m for m in model_names if 'gemini' in m.lower()]
                    if gemini_models:
                        self.model = GenerativeModel(gemini_models[0])
                        logging.info(f"Using alternative Vertex AI model: {gemini_models[0]}")
                        model_initialized = True
                    else:
                        raise GeminiAPIException(f"No Gemini models found. Available models: {model_names}")
                except Exception as list_error:
                    raise GeminiAPIException(f"Failed to initialize models and list available models. List error: {str(list_error)}")
            
            # Test the model with a simple prompt
            if model_initialized:
                try:
                    test_response = self.model.generate_content(
                        "Test prompt to verify model initialization.",
                        generation_config=self.generation_config
                    )
                    if not test_response or not test_response.text:
                        raise GeminiAPIException("Model initialization test failed: empty response")
                    logging.info("Vertex AI model test successful")
                except Exception as test_error:
                    logging.error(f"Model test failed: {test_error}")
                    raise GeminiAPIException(f"Model test failed: {str(test_error)}")
            
        except Exception as e:
            logging.error(f"Failed to initialize VertexAIModel: {e}")
            raise GeminiAPIException(f"Model initialization failed: {str(e)}")
    
    @retry(
        retry=retry_if_exception_type((GeminiAPIException, asyncio.TimeoutError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        """Generate content with improved error handling and retry logic"""
        try:
            # Set a timeout for the API call
            timeout = aiohttp.ClientTimeout(total=20)  # Reduced from 30 to 20 seconds
            
            # Validate and truncate prompt if needed
            if isinstance(prompt, str) and len(prompt) > 12000:
                logging.warning(f"Truncating prompt from {len(prompt)} chars to 12000")
                prompt = prompt[:12000] + "...[truncated]"
            
            # Create a task for the API call
            api_task = asyncio.create_task(
                asyncio.to_thread(
                    lambda: self._safe_generate_content(prompt, safety_settings)
                )
            )
            
            # Wait for the task with timeout
            try:
                response = await asyncio.wait_for(api_task, timeout=timeout.total)
            except asyncio.TimeoutError:
                # Try to cancel the task if it's still running
                if not api_task.done():
                    logging.warning("Cancelling timed out API call")
                    api_task.cancel()
                raise
            
            # If we get None response, raise appropriate exception    
            if response is None:
                raise GeminiAPIException("Model returned None response")
                
            # Handle empty candidates
            if not hasattr(response, 'candidates') or not response.candidates or len(response.candidates) == 0:
                raise GeminiAPIException("No response candidates generated")
            
            # Extract text with better error handling
            text = None
            try:
                text = response.text
            except (AttributeError, TypeError) as attr_error:
                # If .text fails, try multiple ways to get text
                try:
                    if hasattr(response.candidates[0], 'content'):
                        text = response.candidates[0].content.parts[0].text
                    elif hasattr(response.candidates[0], 'text'):
                        text = response.candidates[0].text
                    else:
                        # Last attempt - try to extract raw text from any property
                        for prop in ['content', 'message', 'output', 'result']:
                            if hasattr(response, prop):
                                candidate_content = getattr(response, prop)
                                if isinstance(candidate_content, str):
                                    text = candidate_content
                                    break
                except Exception as extract_error:
                    logging.error(f"Failed to extract text from response: {extract_error}")
                    raise GeminiAPIException(f"Failed to extract text from response: {str(extract_error)}")
            
            if not text or text.strip() == "":
                raise GeminiAPIException("Empty response text")
            
            return text
            
        except asyncio.TimeoutError:
            logging.error("Response generation timed out")
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                logging.warning(f"Rate limit exceeded: {e}")
                raise RateLimitException("Rate limit exceeded. Please try again later.")
            elif "safety" in error_msg:
                logging.warning(f"Safety filter triggered: {e}")
                raise GeminiAPIException("Content was filtered due to safety settings. Please rephrase your query.")
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                logging.error(f"Authentication error: {e}")
                raise GeminiAPIException(f"API authentication error: {e}")
            else:
                logging.error(f"Error generating content: {e}")
                raise GeminiAPIException(f"Error generating response: {str(e)}")

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

    def _safe_generate_content(self, prompt, safety_settings):
        """Safely make the API call with extra error handling for None responses"""
        try:
            # Make the actual API call - VertexAIModel uses generation_config
            if safety_settings:
                response = self.model.generate_content(
                    prompt,
                    safety_settings=safety_settings,
                    generation_config=self.generation_config
                )
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
            # Validate response isn't None
            if response is None:
                logging.error("Model returned None response")
                # Create a basic response object with error message
                return self._create_fallback_response("Model returned empty response")
                
            return response
        except Exception as e:
            logging.error(f"Error in direct API call: {e}")
            # Return a structured error response instead of raising
            return self._create_fallback_response(f"API error: {str(e)}")
        
    def _create_fallback_response(self, message):
        """Create a fallback response object when the API fails"""
        # Create a minimal response-like object
        class FallbackResponse:
            def __init__(self, message):
                self.text = f"Error: {message}"
                self.candidates = [self]
                self.content = self
                self.parts = [self]
                
        return FallbackResponse(message)

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
            # Don't raise an exception, set a flag that AI features are disabled
            self.ai_enabled = False
            self.error_message = str(e)
            raise AIServiceException(f"AI service initialization failed: {str(e)}")

    def _initialize_model(self):
        """Initialize the AI model with proper error handling"""
        try:
            config = ConfigService()
            logging.info(f"Initializing AI model with config: use_vertex_ai={config.use_vertex_ai}")
            
            if config.use_vertex_ai:
                logging.info("Using VertexAIModel for Gemini access")
                try:
                    return VertexAIModel(config)
                except Exception as vertex_error:
                    logging.error(f"Failed to initialize VertexAIModel: {vertex_error}, falling back to GeminiDirectModel")
                    # Fall back to direct API if Vertex AI fails
                    return GeminiDirectModel(config)
            else:
                logging.info("Using GeminiDirectModel for direct API access")
                return GeminiDirectModel(config)
        except Exception as e:
            logging.error(f"Failed to initialize AI model: {e}")
            raise AIServiceException(f"Model initialization failed: {str(e)}")

    async def generate_custom_content(self, prompt: str) -> str:
        """Generate custom content with robust error handling
        
        Args:
            prompt: Input prompt for the AI model
            
        Returns:
            Generated text content
        """
        try:
            logging.info(f"Generating custom content with prompt (first 500 chars): {prompt[:500]}")
            
            # Add timing information
            start_time = time.time()
            
            # Check if model is properly initialized
            if not hasattr(self, 'model') or self.model is None:
                logging.error("AI model not initialized properly")
                return ""
                
            # Call the model's generate_content method directly
            try:
                response_text = await self.model.generate_content(prompt)
                
                # Check response validity
                if response_text is None:
                    logging.error("Model returned None for generate_custom_content")
                    return ""
                    
                if not response_text.strip():
                    logging.error("Model returned empty string for generate_custom_content")
                    return ""
                    
                # Log success and timing
                duration = time.time() - start_time
                logging.info(f"Custom content generated successfully in {duration:.2f}s, length: {len(response_text)}")
                return response_text
                
            except Exception as model_error:
                # More specific logging based on error type
                logging.error(f"Error generating custom content: {str(model_error)}")
                
                # Check for common error patterns
                error_msg = str(model_error).lower()
                if "api key" in error_msg:
                    logging.error("API key issue detected. Check GEMINI_API_KEY in configuration.")
                elif "timeout" in error_msg:
                    logging.error("Timeout issue detected. Consider increasing AI_TIMEOUT in configuration.")
                elif "safety" in error_msg:
                    logging.error("Safety filter triggered. The test prompt was filtered by content safety settings.")
                
                return ""
                
        except Exception as e:
            logging.error(f"Unexpected error in generate_custom_content: {e}", exc_info=True)
            return ""

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
            context = await self._build_context(articles)
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
        start_time = time.time()
        partial_response = None
        stream_timeout = False
        fallback_task = None
        generator = None
        
        try:
            # Validate input
            if not query or not query.strip():
                raise AIServiceException("Empty query provided")

            # Get relevant articles with retry logic and timeout
            articles = []
            try:
                # Set shorter timeout for article retrieval
                articles = await asyncio.wait_for(
                    self.storage_service.query_articles(query, n_results=4),  # Reduced from 5 to 4
                    timeout=3  # Very short timeout for article query
                )
            except asyncio.TimeoutError:
                logging.warning("Timeout getting articles for streaming response")
                # Continue with empty articles
            except Exception as e:
                logging.warning(f"Error getting articles for streaming response: {e}")
                # Continue with empty articles

            # Store articles as instance attribute for source extraction
            self.relevant_articles = articles or []

            # Build context with better error handling and timeout
            try:
                context = await asyncio.wait_for(
                    self._build_context(self.relevant_articles),
                    timeout=3  # Short timeout for context building
                )
            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"Error or timeout building context: {e}")
                # Use minimal context for faster response
                context = "\n".join([
                    f"Article {i+1}: {article.get('metadata', {}).get('title', 'Unknown')}"
                    for i, article in enumerate(self.relevant_articles[:2])
                ])
                
            if not context or len(context.strip()) < 50:
                context = "Limited context available for this query."

            # Build prompt with history if enabled (with size limits)
            try:
                prompt = self._build_prompt(query, context, use_history)
                # Truncate if too long
                if len(prompt) > 12000:
                    logging.warning(f"Truncating streaming prompt from {len(prompt)} chars to 12000")
                    # Keep the query and a portion of the context
                    query_part = f"Question: {query}\n\n"
                    context_limit = 12000 - len(query_part) - 100  # Allow some buffer
                    truncated_context = context[:context_limit] + "...[truncated]"
                    prompt = f"{query_part}Context: {truncated_context}"
            except Exception as e:
                logging.warning(f"Error building prompt: {e}")
                # Fallback to simplified prompt
                prompt = f"Question: {query}\n\nContext: {context[:1000] if context else 'Limited context available.'}"

            # Start background task to prepare a partial response
            async def prepare_fallback():
                nonlocal partial_response
                try:
                    # Wait 10 seconds before preparing fallback
                    await asyncio.sleep(10)
                    partial_response = await self.get_partial_response()
                except Exception as e:
                    logging.warning(f"Error preparing fallback response: {e}")
                    
            fallback_task = asyncio.create_task(prepare_fallback())
            
            # Set a maximum streaming time to prevent excessive resource usage
            max_streaming_time = 30  # seconds
            stream_start = time.time()
            
            # Use the model's streaming capability with monitoring
            try:
                generator = self.model.generate_content_stream(prompt)
                async for chunk in generator:
                    # Check if we've exceeded the maximum streaming time
                    if time.time() - stream_start > max_streaming_time:
                        stream_timeout = True
                        logging.warning(f"Streaming response exceeded {max_streaming_time}s time limit")
                        # Send a special marker for the frontend
                        yield "\n\n[Streaming timed out. Truncating response for performance reasons.]"
                        break
                        
                    yield chunk
                    
                # If streaming completed successfully, cancel the fallback task
                if not stream_timeout and not fallback_task.done():
                    fallback_task.cancel()
                    
                # Try to extract and append sources after streaming
                try:
                    sources = self._extract_sources("", self.relevant_articles)
                    if sources:
                        # Send sources as a special marker that the frontend can parse
                        yield f"__SOURCES__{json.dumps(sources)}"
                except Exception as e:
                    logging.warning(f"Error extracting sources for streaming response: {e}")
                    
            except Exception as streaming_error:
                logging.error(f"Error in streaming response: {streaming_error}")
                
                # Wait for fallback if it's still running
                if not fallback_task.done():
                    try:
                        await asyncio.wait_for(fallback_task, timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                        
                # Use fallback response if available
                if partial_response:
                    yield f"\n\nStream interrupted. {partial_response}"
                else:
                    yield f"\n\nError: {str(streaming_error)}"
                
                # Try to add sources even in error case
                try:
                    sources = self._extract_sources("", self.relevant_articles)
                    if sources:
                        yield f"__SOURCES__{json.dumps(sources)}"
                except Exception:
                    pass

        except Exception as e:
            logging.error(f"Critical error in streaming response: {e}")
            
            # Wait for fallback if it's still running
            if fallback_task and not fallback_task.done():
                try:
                    await asyncio.wait_for(fallback_task, timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
            # Use fallback response if available
            if partial_response:
                yield f"\n\nStream interrupted. {partial_response}"
            else:
                yield f"Error: {str(e)}"
        finally:
            # Clean up resources
            if fallback_task and not fallback_task.done():
                fallback_task.cancel()
                
            # Close the generator if it exists
            if generator and hasattr(generator, 'aclose'):
                try:
                    await generator.aclose()
                except Exception as close_error:
                    logging.error(f"Error closing content generator: {close_error}")

    async def _generate_streaming_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Internal method to generate streaming response from the model
        
        Args:
            prompt: The complete prompt to send to the model
            
        Yields:
            Chunks of the generated response
        """
        generator = None
        try:
            generator = self.model.generate_content_stream(prompt)
            async for chunk in generator:
                yield chunk
                
            # After content generation, try to extract and append sources
            try:
                sources = self._extract_sources("", self.relevant_articles)
                if sources:
                    # Send sources as a special marker that the frontend can parse
                    yield f"__SOURCES__{json.dumps(sources)}"
            except Exception as e:
                logging.warning(f"Error extracting sources for streaming response: {e}")
                
        except Exception as e:
            logging.error(f"Error in streaming response generation: {e}")
            yield f"Error: {str(e)}"
        finally:
            # Close the generator if it exists
            if generator and hasattr(generator, 'aclose'):
                try:
                    await generator.aclose()
                except Exception as close_error:
                    logging.error(f"Error closing generator in _generate_streaming_response: {close_error}")

    def _validate_date(self, date_obj: datetime, article_index: int) -> tuple[bool, str]:
        """Validate date is within reasonable range
        
        Args:
            date_obj: The datetime object to validate
            article_index: Index of article for logging
            
        Returns:
            tuple(is_valid, message): Validation result and message
        """
        try:
            # Ensure both dates are timezone-naive for comparison
            if date_obj.tzinfo is not None:
                date_obj = date_obj.replace(tzinfo=None)
            
            now = datetime.now()
            
            # Define reasonable date ranges
            earliest_date = datetime(1990, 1, 1)  # Earliest reasonable date for articles
            max_future = now + timedelta(days=30)  # Allow up to 1 month in future for pre-published articles
            
            if date_obj < earliest_date:
                logging.warning(f"Article {article_index}: Date too old: {date_obj.strftime('%Y-%m-%d')}")
                return False, f"Invalid date (before {earliest_date.year})"
                
            if date_obj > max_future:
                logging.warning(f"Article {article_index}: Future date detected: {date_obj.strftime('%Y-%m-%d')}")
                return True, f"Future date ({date_obj.strftime('%Y-%m-%d')})"
                
            return True, date_obj.strftime('%Y-%m-%d')
            
        except Exception as e:
            logging.error(f"Error validating date for article {article_index}: {e}")
            return False, "Date unknown"

    def _parse_date(self, date_str: any, article_index: int) -> str:
        """Helper method to parse dates in various formats"""
        if not date_str:
            return "Date unknown"
            
        try:
            # Handle numeric timestamp (including NaN check)
            if isinstance(date_str, (int, float)):
                if not isinstance(date_str, bool) and not math.isnan(float(date_str)):
                    date_obj = datetime.fromtimestamp(float(date_str))
                    is_valid, result = self._validate_date(date_obj, article_index)
                    return result
                return "Date unknown"
                
            if not isinstance(date_str, str):
                return "Date unknown"
                
            date_str = date_str.strip()
            
            # Try parsing RFC format first (handles timezone)
            try:
                from email.utils import parsedate_to_datetime
                date_obj = parsedate_to_datetime(date_str)
                is_valid, result = self._validate_date(date_obj, article_index)
                return result
            except (TypeError, ValueError):
                pass
            
            # Try ISO format
            try:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                is_valid, result = self._validate_date(date_obj, article_index)
                return result
            except ValueError:
                pass
                
            # List of date formats to try
            date_formats = [
                # RFC 2822 and variations
                ('%a, %d %b %Y %H:%M:%S %z', True),  # RFC 2822
                ('%a, %d %b %Y %H:%M:%S %Z', True),  # RFC 2822 with timezone name
                ('%a, %d %b %Y %H:%M:%S', False),    # RFC 2822 without timezone
                ('%a, %m/%d/%Y - %H:%M', False),     # Wed, 03/26/2025 - 08:23
                ('%a, %d/%m/%Y - %H:%M', False),     # Wed, 26/03/2025 - 08:23
                ('%a, %m/%d/%Y %H:%M', False),       # Wed, 03/26/2025 08:23
                ('%a %m/%d/%Y - %H:%M', False),      # Wed 03/26/2025 - 08:23
                ('%a, %m/%d/%Y', False),             # Wed, 03/26/2025
                
                # Common formats
                ('%m/%d/%Y', False),                 # 03/26/2025
                ('%d/%m/%Y', False),                 # 26/03/2025
                ('%Y-%m-%d', False),                 # 2025-03-26
                ('%B %d, %Y', False),                # March 26, 2025
                ('%d %B %Y', False),                 # 26 March 2025
                ('%d-%b-%Y', False),                 # 26-Mar-2025
                ('%Y%m%d', False),                   # 20250326
                
                # With time components
                ('%m/%d/%Y %H:%M:%S', False),       # 03/26/2025 08:23:45
                ('%Y-%m-%d %H:%M:%S', False),       # 2025-03-26 08:23:45
                ('%d.%m.%Y %H:%M', False),          # 26.03.2025 08:23
            ]
            
            # Try each format
            for date_format, has_timezone in date_formats:
                try:
                    parsed_str = date_str
                    if has_timezone:
                        # Handle both comma and no-comma cases
                        if ',' in parsed_str:
                            parsed_str = parsed_str.split(',')[1].strip()
                        else:
                            # Try to remove day name without comma
                            parts = parsed_str.split()
                            if len(parts) > 1:
                                parsed_str = ' '.join(parts[1:])
                    
                    # Remove everything after hyphen if present
                    if ' - ' in parsed_str:
                        parsed_str = parsed_str.split(' - ')[0].strip()
                    
                    date_obj = datetime.strptime(parsed_str, date_format)
                    is_valid, result = self._validate_date(date_obj, article_index)
                    return result
                except ValueError:
                    continue
            
            logging.warning(f"Error parsing date for article {article_index}: Unrecognized format: {date_str}")
            return "Date unknown"
            
        except Exception as e:
            logging.warning(f"Error parsing date for article {article_index}: {str(e)}")
            return "Date unknown"

    async def _build_context(self, articles: List[Dict], query: str = None) -> str:
        """
        Build context from articles using semantic chunking and relevance scoring.
        
        Args:
            articles: List of article dictionaries
            query: Optional query string for relevance scoring
            
        Returns:
            A formatted context string with the most relevant chunks
        """
        if not articles:
            return ""
            
        try:
            # Configure chunk size and overlap for better context
            CHUNK_SIZE = 1000
            CHUNK_OVERLAP = 200
            MAX_CHUNKS = 10
            MAX_CONTEXT_LENGTH = 12000
            
            # Track chunks and their relevance scores
            chunks = []
            
            for idx, article in enumerate(articles, 1):
                try:
                    # Get article content
                    content = article.get('metadata', {}).get('description', '')
                    title = article.get('metadata', {}).get('title', 'Untitled Article')
                    source = article.get('metadata', {}).get('source', 'Unknown Source')
                    pub_date = article.get('metadata', {}).get('pub_date', None)
                    url = article.get('metadata', {}).get('link', '')
                    
                    # Format date for readability if available
                    date_str = ""
                    if pub_date:
                        # Try to standardize date format
                        try:
                            date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            # Format as YYYY-MM-DD
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except Exception:
                            date_str = pub_date
                    
                    # Skip article if no content
                    if not content or len(content) < 50:
                        continue
            
                    # Create semantic chunks rather than arbitrary splits
                    # Use natural boundaries like paragraphs and sentences
                    paragraphs = re.split(r'\n\n+', content)
                    
                    current_chunk = ""
                    for paragraph in paragraphs:
                        # Skip empty paragraphs
                        if not paragraph.strip():
                            continue
                            
                        # If adding this paragraph would exceed chunk size, store current chunk and start new one
                        if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
                            if current_chunk:
                                chunk_data = {
                                    'content': current_chunk,
                                    'article_id': article.get('id', ''),
                                    'title': title,
                                    'source': source,
                                    'date': date_str,
                                    'url': url,
                                    'article_index': idx
                                }
                                chunks.append(chunk_data)
                            current_chunk = paragraph
                        else:
                            # Add paragraph to current chunk
                            if current_chunk:
                                current_chunk += "\n\n" + paragraph
                            else:
                                current_chunk = paragraph
                    
                    # Add the last chunk if it exists
                    if current_chunk:
                        chunk_data = {
                            'content': current_chunk,
                            'article_id': article.get('id', ''),
                            'title': title,
                            'source': source,
                            'date': date_str,
                            'url': url,
                            'article_index': idx
                        }
                        chunks.append(chunk_data)
                        
                except Exception as e:
                    logging.error(f"Error processing article {idx} for context: {e}")
                    continue
                    
            # If query is provided, calculate relevance score for each chunk
            if query and len(chunks) > MAX_CHUNKS:
                # Implement semantic similarity using Gemini or other embedding model
                try:
                    import google.generativeai as genai
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np
                    
                    # Configure embedding model
                    genai.configure(api_key=self.config.google_api_key)
                    
                    # Get embedding for query
                    query_embedding = None
                    try:
                        embedding_model = "models/embedding-001"
                        query_embedding = genai.embed_content(
                            model=embedding_model,
                            content=query,
                            task_type="retrieval_query")["embedding"]
                    except Exception as e:
                        logging.warning(f"Error getting query embedding: {e}")
                    
                    # If we have query embedding, score chunks by similarity
                    if query_embedding:
                        for chunk in chunks:
                            try:
                                chunk_content = f"{chunk['title']} {chunk['content']}"
                                chunk_embedding = genai.embed_content(
                                    model=embedding_model,
                                    content=chunk_content,
                                    task_type="retrieval_document")["embedding"]
                                    
                                # Calculate cosine similarity
                                similarity = cosine_similarity(
                                    [query_embedding], 
                                    [chunk_embedding]
                                )[0][0]
                                
                                chunk['score'] = float(similarity)
                            except Exception as e:
                                logging.warning(f"Error calculating chunk similarity: {e}")
                                chunk['score'] = 0.0
                        
                        # Sort chunks by relevance score
                        chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
                except ImportError:
                    # Fallback to simpler relevance calculation if sklearn not available
                    logging.warning("Sklearn not available for similarity calculation. Using simpler relevance scoring.")
                    for chunk in chunks:
                        # Calculate simple relevance based on keyword matches
                        query_terms = query.lower().split()
                        content = (chunk['title'] + " " + chunk['content']).lower()
                        score = sum(1 for term in query_terms if term in content)
                        chunk['score'] = score / len(query_terms) if query_terms else 0
                        
                    # Sort chunks by score
                    chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
                except Exception as e:
                    logging.error(f"Error calculating relevance scores: {e}")
                    # If relevance scoring fails, use chronological order (newest first)
                    chunks = sorted(chunks, key=lambda x: x.get('date', ''), reverse=True)
            else:
                # If no query or few chunks, sort by date (newest first)
                chunks = sorted(chunks, key=lambda x: x.get('date', ''), reverse=True)
            
            # Limit number of chunks to prevent context being too large
            chunks = chunks[:MAX_CHUNKS]
            
            # Build context string from chunks
            context_parts = []
            total_length = 0
            
            for i, chunk in enumerate(chunks, 1):
                # Format chunk with metadata
                chunk_text = f"""
[Article {chunk['article_index']}] {chunk['title']}
Source: {chunk['source']} | Date: {chunk['date']}
URL: {chunk['url']}

{chunk['content']}
"""
                # Check if adding this chunk would exceed max context length
                if total_length + len(chunk_text) <= MAX_CONTEXT_LENGTH:
                    context_parts.append(chunk_text)
                    total_length += len(chunk_text)
                else:
                    # Stop adding chunks if we're reached max context length
                    break
                    
            # Join all context parts with separators
            context = "\n\n---\n\n".join(context_parts)
            
            return context
            
        except Exception as e:
            logging.error(f"Error building context: {e}")
            # Fallback to a simpler context building approach
            return self._build_minimal_context(articles)

    def _build_prompt(self, query: str, context: str, use_history: bool) -> str:
        """
        Build a prompt for the AI model using the query, context, and history.
        
        Args:
            query: The user's query
            context: The retrieved context articles
            use_history: Whether to include conversation history
            
        Returns:
            A formatted prompt string
        """
        try:
            # Get the appropriate output format
            output_format = PromptLibrary.get_output_format("default")
            
            # Format the prompt template with query, context, and output_format
            prompt = PromptTemplate.RAG_QUERY.value.format(
                query=query, 
                context=context,
                output_format=output_format
            )
            
            # Add history if needed
            if use_history and self.history:
                history_str = self._format_history()
                prompt = f"{prompt}\n\nPrevious conversation for context (consider this when relevant):\n{history_str}"
            
            return prompt
        except Exception as e:
            logging.error(f"Error building prompt: {e}")
            raise AIServiceException(f"Failed to build prompt: {str(e)}")

    @retry(
        retry=retry_if_exception_type(GeminiAPIException),
        stop=stop_after_attempt(2),  # Reduce retries to prevent cascading timeouts
        wait=wait_exponential(multiplier=1, min=1, max=5)  # Shorter retry wait times
    )
    async def _generate_response(self, prompt: str) -> AIResponse:
        """Generate response with enhanced error handling and retry logic"""
        try:
            start_time = time.time()
            
            # Validate prompt - truncate if too long
            if not prompt or not prompt.strip():
                raise AIServiceException("Empty prompt provided")
            
            # Calculate prompt length - truncate if excessively long
            if len(prompt) > 12000:  # Arbitrary limit to prevent timeouts
                logging.warning(f"Truncating excessively long prompt from {len(prompt)} chars to 12000")
                prompt = prompt[:12000] + "...[truncated for performance]"
            
            # Set a reasonable timeout that is progressively shortened with retries
            timeout_duration = 45  # Reduced from 60 to 45 seconds
            
            # Create a future for partial response that we can cancel if full response succeeds
            partial_future = None
            
            # Use a proxy response result we can fill if we need to cancel the main request
            response_proxy = {"text": None, "is_partial": False}
            
            # Function to generate a partial response in the background
            async def generate_partial():
                try:
                    # Wait 15 seconds before trying partial response
                    await asyncio.sleep(15)
                    
                    # Only proceed if main response hasn't completed yet
                    response_proxy["text"] = await self.get_partial_response()
                    response_proxy["is_partial"] = True
                    logging.info("Partial response prepared in background")
                except Exception as e:
                    logging.warning(f"Background partial response task failed: {e}")
            
            try:
                # Start partial response generation in the background
                partial_future = asyncio.create_task(generate_partial())
                
                # Set a flag to track if we've handled the result
                result_handled = False
                
                # Use asyncio.wait with timeout instead of wait_for
                # This allows us to get partial results without cancelling the task
                done, pending = await asyncio.wait(
                    [asyncio.create_task(self.model.generate_content(prompt))],
                    timeout=timeout_duration
                )
                
                # Check if we got a valid result
                if done:
                    task = list(done)[0]
                    try:
                        # Get the result if completed successfully
                        response_text = task.result()
                        result_handled = True
                        
                        # Make sure to cancel the partial response task
                        if partial_future and not partial_future.done():
                            partial_future.cancel()
                        
                        # Validate response text
                        if not response_text or not isinstance(response_text, str):
                            logging.error(f"Invalid response format: {response_text}")
                            raise GeminiAPIException("Invalid response format from AI model")
                        
                        response_text = response_text.strip()
                        
                        if not response_text:
                            logging.error("Empty text in response")
                            raise GeminiAPIException("Empty text in AI model response")
                        
                        # Validate response length
                        if len(response_text) < 10:  # Arbitrary minimum length
                            raise GeminiAPIException("Response too short, likely invalid")
                        
                        # Extract sources from the response
                        sources = self._extract_sources(response_text, self.relevant_articles)
                        
                        # Calculate confidence based on response properties
                        confidence = self._calculate_confidence(response_text, len(sources))
                        
                        # Log success with timing information
                        duration = time.time() - start_time
                        logging.info(f"Successfully generated response of length {len(response_text)} in {duration:.2f}s")
                        
                        return AIResponse(
                            text=response_text,
                            sources=sources,
                            confidence=confidence
                        )
                        
                    except Exception as result_error:
                        # Handle any errors from the completed task
                        logging.error(f"Error getting result from completed task: {result_error}")
                        if not result_handled:
                            raise GeminiAPIException(f"Error processing model response: {result_error}")
                
                # If we get here, the main request timed out
                # Cancel any remaining tasks
                for p in pending:
                    p.cancel()
                
                # Wait for partial response to complete if it's still running
                if partial_future and not partial_future.done():
                    try:
                        await asyncio.wait_for(partial_future, timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                
                # Check if we got a partial response
                if response_proxy["text"]:
                    logging.warning("Main request timed out, using partial response")
                    return AIResponse(
                        text=f"{response_proxy['text']}\n\n[Note: This is a partial response due to processing timeout.]",
                        sources=self._extract_sources(response_proxy["text"], self.relevant_articles),
                        confidence=0.5,  # Lower confidence for partial response
                        timestamp=datetime.now().isoformat()
                    )
                
                # If we got here with no partial response, raise timeout exception
                raise asyncio.TimeoutError("Request timed out and no partial response available")
                
            except asyncio.TimeoutError:
                logging.error(f"Response generation timed out after {timeout_duration}s")
                
                # Cancel the main request and any other tasks
                if partial_future and not partial_future.done():
                    try:
                        # Give partial response a bit more time to complete
                        await asyncio.wait_for(partial_future, timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                
                # Check if partial response is available
                if response_proxy["text"]:
                    logging.warning("Using partial response after timeout")
                    return AIResponse(
                        text=f"{response_proxy['text']}\n\n[Note: This is a partial response due to processing timeout.]",
                        sources=self._extract_sources(response_proxy["text"], self.relevant_articles),
                        confidence=0.4,  # Lower confidence for partial response
                        timestamp=datetime.now().isoformat()
                    )
                
                # If no partial response is available, raise timeout error for retry
                raise GeminiAPIException("Content generation timed out, consider simplifying your query")
                
        except asyncio.TimeoutError:
            logging.error("Response generation timed out")
            # Try to get a placeholder response before giving up
            placeholder = "The request timed out. Please try a more specific query or break it into smaller parts."
            return AIResponse(
                text=placeholder,
                sources=[],
                confidence=0.1,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if "safety" in error_msg:
                logging.warning(f"Safety filter triggered: {e}")
                raise GeminiAPIException("Content was filtered due to safety settings. Please rephrase your query.")
            elif any(term in error_msg for term in ["rate limit", "quota", "resource exhausted"]):
                logging.warning(f"Rate limit exceeded: {e}")
                raise RateLimitException("Rate limit exceeded. Please try again later.")
            else:
                logging.error(f"AI model error: {e}")
                raise GeminiAPIException(f"Error generating response: {str(e)}")

    def _calculate_confidence(self, response: str, num_sources: int) -> float:
        """Calculate confidence score based on response quality and sources"""
        try:
            base_confidence = 0.7
            
            # Adjust based on response length
            length_factor = min(1.0, len(response) / 1000)  # Normalize by expected length
            
            # Adjust based on number of sources
            source_factor = min(1.0, num_sources / 3)  # Normalize by expected sources
            
            # Combine factors
            confidence = base_confidence * (length_factor * 0.5 + source_factor * 0.5)
            
            # Cap at 0.95
            return min(0.95, confidence)
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
            context = await self._build_context(articles)
            
            # Use the topic analysis template, now including output_format
            prompt = PromptTemplate.TOPIC_ANALYSIS.value.format(
                topic=topic,
                context=context,
                output_format=PromptLibrary.get_output_format("default")
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

    async def generate_comprehensive_analysis(self, article_id: str, analysis_type: str = 'general', output_format: str = 'default') -> Dict:
        """
        Generate comprehensive analysis of an article with improved context handling and structured output.
        
        Args:
            article_id: ID of the article to analyze
            analysis_type: Type of analysis to perform (clinical_trial, patent, market, etc)
            output_format: Format to return results in (default, json, table, bullets)
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        start_time = time.time()
        logging.info(f"Generating comprehensive analysis for article {article_id}, type={analysis_type}, format={output_format}")
        
        try:
            # Get article from storage
            article = await self.storage_service.get_article(article_id)
            if not article:
                raise AIServiceException(f"Article {article_id} not found")
                
            # Get related articles for context
            related_articles = await self._fetch_related_articles(article_id, article)
            
            # Select the appropriate prompt template based on analysis type
            prompt_template = None
            template_args = {}
            
            # Get article content
            article_content = article.get('document', '') or article.get('metadata', {}).get('description', '')
            
            # Map analysis type to PromptLibrary templates
            if analysis_type == 'clinical_trial':
                prompt_template = PromptLibrary.CLINICAL_TRIAL_ANALYSIS
                template_args['trial_data'] = article_content
            elif analysis_type == 'patent':
                prompt_template = PromptLibrary.PATENT_ANALYSIS
                template_args['patent_data'] = article_content
            elif analysis_type == 'manufacturing':
                prompt_template = PromptLibrary.MANUFACTURING_ANALYSIS
                template_args['manufacturing_data'] = article_content
            elif analysis_type == 'market_access':
                prompt_template = PromptLibrary.MARKET_ACCESS
                template_args['market_data'] = article_content
            elif analysis_type == 'safety':
                prompt_template = PromptLibrary.SAFETY_ANALYSIS
                template_args['safety_data'] = article_content
            elif analysis_type == 'pipeline':
                prompt_template = PromptLibrary.PIPELINE_ANALYSIS
                template_args['pipeline_data'] = article_content
            elif analysis_type == 'therapeutic_area':
                prompt_template = PromptLibrary.THERAPEUTIC_LANDSCAPE
                template_args['therapeutic_area'] = article_content
            elif analysis_type == 'regulatory_strategy':
                prompt_template = PromptLibrary.REGULATORY_STRATEGY
                template_args['regulatory_data'] = article_content
            elif analysis_type == 'digital_health':
                prompt_template = PromptLibrary.DIGITAL_HEALTH
                template_args['digital_data'] = article_content
            else:
                # Default to TOPIC_ANALYSIS for general analysis
                prompt_template = PromptTemplate.TOPIC_ANALYSIS.value
                # Extract topic from article metadata or use a generic topic
                topic = article.get('metadata', {}).get('topic', 'pharmaceutical research')
                template_args = {
                    'topic': topic,
                    'context': self._build_minimal_context([article] + related_articles)
                }
            
            # Format the prompt with structured output formatting
            if prompt_template != PromptTemplate.TOPIC_ANALYSIS.value:
                # Use PromptLibrary's formatting capabilities
                prompt = PromptLibrary.format_prompt(
                    prompt_template, 
                    format_type=output_format,
                    **template_args
                )
            else:
                # Handle Enum templates with output_format
                template_args['output_format'] = PromptLibrary.get_output_format(output_format)
                prompt = prompt_template.format(**template_args)
            
            # Generate analysis with the AI model
            response_text = await self.model.generate_content(prompt)
            
            # Calculate confidence score
            confidence = self._calculate_analysis_confidence(article, related_articles)
            
            # Extract article sources from the response
            sources = self._extract_sources(response_text, [article] + related_articles)
            
            # Format the result
            result = {
                'text': response_text,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'article_id': article_id,
                    'analysis_type': analysis_type,
                    'context_articles_count': len(related_articles),
                    'processing_time': round(time.time() - start_time, 2),
                    'output_format': output_format
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error generating comprehensive analysis: {e}")
            # Create a fallback response
            return self._create_fallback_response(article_id, article, analysis_type, output_format, start_time, e)
    
    async def _fetch_related_articles(self, article_id: str, article: Dict) -> List[Dict]:
        """Fetch related articles for context building."""
        try:
            # Get topic
            topic = article.get('metadata', {}).get('topic')
            
            # Query parameters for finding related articles
            query_params = {
                'topic': topic,
                'limit': 5,  # Get 5 most relevant related articles
                # 'exclude_ids': [article_id]  # Exclude the current article - get_articles doesn't support this
            }
            
            # Fetch related articles without exclude_ids
            # Need to add +1 to limit temporarily if we want 5 *excluding* the original
            query_params['limit'] += 1 
            related_results = await self.storage_service.get_articles(**query_params)
            
            # Filter out the original article ID manually
            related = [
                art for art in related_results.get('articles', []) 
                if art.get('id') != article_id
            ]
            
            # Ensure we return the correct number of articles
            related = related[:5] # Limit back to 5 after filtering

            # Log the results
            logging.info(f"Found {len(related)} related articles for context")
            
            return related
        except Exception as e:
            logging.error(f"Error fetching related articles: {e}")
            return []
    
    def _create_fallback_response(self, article_id: str, article: Dict, analysis_type: str, 
                                 output_format: str, start_time: float, error: Exception) -> Dict:
        """Create a fallback response when analysis generation fails."""
        try:
            # Create a minimal fallback response
            return {
                'text': "We couldn't generate a complete analysis at this time. Please try again later.",
                'sources': [{'id': article_id, 'title': article.get('metadata', {}).get('title', 'Unknown')}] if article else [],
                'confidence': 0.1,
                'metadata': {
                    'article_id': article_id,
                    'analysis_type': analysis_type,
                    'context_articles_count': 0,
                    'processing_time': round(time.time() - start_time, 2),
                    'is_fallback': True,
                    'output_format': output_format,
                    'error': str(error)
                }
            }
        except Exception as fallback_error:
            logging.error(f"Error creating fallback response: {fallback_error}")
            # Absolute minimal response
            return {
                'text': "Analysis generation failed. Please try again later.",
                'sources': [],
                'confidence': 0.0,
                'metadata': {
                    'is_fallback': True,
                    'error': str(fallback_error)
                }
            }
        
    def _build_minimal_context(self, articles: List[Dict]) -> str:
        """Build minimal context from articles to avoid timeouts"""
        try:
            context_parts = []
            for i, article in enumerate(articles, 1):
                metadata = article.get('metadata', {})
                title = metadata.get('title', 'Untitled')
                source = metadata.get('source', 'Unknown')
                
                # Just include basic metadata without full content
                context_parts.append(f"Related Article {i}:\nTitle: {title}\nSource: {source}")
                
            return "\n\n".join(context_parts)
        except Exception as e:
            logging.warning(f"Error building minimal context: {e}")
            return ""

    def _calculate_analysis_confidence(self, main_article: Dict, related_articles: List[Dict]) -> float:
        """Calculate confidence score for the analysis based on available data"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on article content
        if main_article.get('document'):  # Full content available
            confidence += 0.1
        if main_article.get('metadata', {}).get('summary'):  # Has AI summary
            confidence += 0.05
            
        # Adjust based on related articles
        if related_articles:
            confidence += min(0.1, len(related_articles) * 0.02)  # Up to 0.1 for 5+ related articles
            
        # Adjust based on recency
        try:
            pub_date = datetime.fromisoformat(main_article.get('metadata', {}).get('pub_date', ''))
            if pub_date > datetime(2024, 11, 1):  # After training cutoff
                confidence -= 0.05  # Slightly lower confidence for very recent information
        except (ValueError, TypeError):
            pass
            
        return round(min(0.95, confidence), 2)  # Cap at 0.95 and round to 2 decimal places

    async def get_partial_response(self) -> Optional[str]:
        """Get partial response if available when a timeout occurs"""
        try:
            # Start with empty response
            partial_content = None
            
            # First check if we have any cached partial response
            if hasattr(self, '_partial_response_cache'):
                cached_response = getattr(self, '_partial_response_cache', None)
                if cached_response and isinstance(cached_response, str) and len(cached_response) > 20:
                    logging.info("Using cached partial response")
                    return cached_response
                
            # If we have no articles, return simple error message
            if not hasattr(self, 'relevant_articles') or not self.relevant_articles:
                return "Unable to generate a response due to insufficient context. Please try a more specific query."
            
            # Create hardcoded fallback that always works even if model fails completely
            hardcoded_fallback = self._create_minimal_fallback()
                    
            # Use only metadata to create minimal context (avoid sending full documents)
            try:
                context_items = []
                
                for i, article in enumerate(self.relevant_articles[:2]):  # Only use top 2 articles for speed
                    metadata = article.get('metadata', {})
                    title = metadata.get('title', 'Untitled')
                    
                    # Add only the title to reduce complexity
                    context_items.append(f"Article {i+1}: {title}")
                
                context = "\n".join(context_items)
                
                # Create a minimal-sized prompt to avoid timeout
                prompt = f"""
                Based on these article titles, provide a 1-2 sentence response:

                {context}

                Keep it very brief.
                """
                
                # Use a very short timeout
                try:
                    # First try direct model access with minimal processing
                    response = None
                    try:
                        # Use bare model with timeout guard
                        response_text = await asyncio.wait_for(
                            self.model.generate_content(prompt),
                            timeout=7
                        )
                        
                        if response_text and isinstance(response_text, str) and len(response_text.strip()) > 5:
                            result = f"[Partial Response] {response_text.strip()}\n\nNote: This is a partial response due to processing constraints. Please try a more specific query."
                            
                            # Cache this response for future use
                            setattr(self, '_partial_response_cache', result)
                            
                            return result
                    except Exception as e:
                        logging.warning(f"Error generating partial response: {e}")
                        return hardcoded_fallback
                except Exception as e:
                    logging.warning(f"Failed to generate partial response: {e}")
                    return hardcoded_fallback
            except Exception as e:
                logging.error(f"Critical error in get_partial_response: {e}")
                return hardcoded_fallback
        except Exception as e:
            logging.error(f"Critical error in get_partial_response: {e}")
            return "Processing timed out. Please try again with a more specific request."
        
    def _create_minimal_fallback(self) -> str:
        """Create a minimal fallback response based on relevant articles metadata only"""
        try:
            if not hasattr(self, 'relevant_articles') or not self.relevant_articles:
                return "Unable to process your request. Please try again with a different query."
                
            # Try to extract useful information from article metadata
            article_titles = []
            for article in self.relevant_articles[:3]:  # Use up to 3 articles
                metadata = article.get('metadata', {})
                title = metadata.get('title', '')
                if title:
                    article_titles.append(title)
                    
            if article_titles:
                return f"[System Message] The request timed out, but we found articles that might be relevant: {', '.join(article_titles)}. Please try a more specific query."
            else:
                return "The request timed out. Please try a more specific query or check back later."
        except Exception:
            return "The request timed out. Please try again with a more specific query."

    def _extract_sources(self, response_text: str, articles: List[Dict] = None) -> List[Dict]:
        """Extract sources with enhanced error handling for pharmaceutical content
        
        Args:
            response_text: The response text containing article references
            articles: List of articles to extract sources from, falls back to self.relevant_articles
            
        Returns:
            List of source dictionaries with article metadata
        """
        try:
            sources = []
            # Match both [Article X] and [Source X] references
            article_refs = set(re.findall(r'\[(Article|Source) (\d+)\]', response_text))
            
            # Use provided articles or fall back to instance attribute
            articles_to_use = articles if articles is not None else getattr(self, 'relevant_articles', [])
            
            if not articles_to_use:
                logging.warning("No articles available for source extraction")
                return []
            
            for _, ref_num in article_refs:
                try:
                    article_idx = int(ref_num) - 1
                    if 0 <= article_idx < len(articles_to_use):
                        article = articles_to_use[article_idx]
                        metadata = article.get('metadata', {})
                        sources.append({
                            "title": metadata.get('title', 'Unknown title'),
                            "source": metadata.get('source', 'Unknown source'),
                            "date": self._parse_date(metadata.get('pub_date', ''), article_idx + 1),
                            "link": metadata.get('link', ''),
                            "topic": metadata.get('topic', 'Unknown topic'),
                            "type": metadata.get('type', 'article')
                        })
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error extracting source {ref_num}: {e}")
                    continue
                    
            return sources
        except Exception as e:
            logging.error(f"Error extracting sources: {e}")
            return []

class PromptLibrary:
    """Library of specialized prompts for pharmaceutical industry analysis with structured output support"""
    
    # Define output formats for structured responses
    OUTPUT_FORMATS = {
        # Default markdown format
        "default": "Format your response in markdown with clear sections.",
        
        # JSON structured format
        "json": """
        Format your output as a valid JSON object with the following structure:
        {
            "main_findings": "Primary conclusions from the analysis",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "industry_impact": "Assessment of industry implications",
            "regulatory_implications": "Any regulatory considerations",
            "timeline": "Relevant timeline information",
            "sources": ["Article references used"]
        }
        
        IMPORTANT: Ensure the output is valid JSON with proper escaping of special characters.
        """,
        
        # Table format for comparative data
        "table": """
        Format your response with a markdown table for any comparative data,
        followed by analysis in clear sections.
        
        Example table format:
        | Criteria | Value | Comparison | Impact |
        | -------- | ----- | ---------- | ------ |
        | Data 1   | X     | Higher     | Strong |
        """,
        
        # Bullet point format for concise information
        "bullets": """
        Format your response using bullet points for key information:
        
        ## Main Finding
        * Primary conclusion
        
        ## Key Points
        * Point 1
        * Point 2
        * Point 3
        
        ## Industry Impact
        * Impact 1
        * Impact 2
        
        ## Regulatory Implications
        * Implication 1
        * Implication 2
        """
    }
    
    @classmethod
    def get_output_format(cls, format_type: str = "default") -> str:
        """Get the specified output format instruction"""
        return cls.OUTPUT_FORMATS.get(format_type.lower(), cls.OUTPUT_FORMATS["default"])
    
    @classmethod
    def format_prompt(cls, template: str, format_type: str = "default", **kwargs) -> str:
        """
        Format a prompt template with variables and add structured output instructions
        
        Args:
            template: The prompt template string
            format_type: Type of structured output format to use (default, json, table, bullets)
            **kwargs: Variables to insert into the template
            
        Returns:
            Formatted prompt with output instructions
        """
        # Get the output format instructions
        output_format = cls.get_output_format(format_type)
        
        # Add output format to kwargs
        kwargs['output_format'] = output_format
        
        # Format the template with all variables
        return template.format(**kwargs)
    
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
    
    {output_format}
    """
    
    # Add structured_clinical_trial_analysis method
    @classmethod
    def structured_clinical_trial_analysis(cls, trial_data: str, format_type: str = "default") -> str:
        """
        Generate a structured clinical trial analysis prompt
        
        Args:
            trial_data: The clinical trial data to analyze
            format_type: Output format type (default, json, table, bullets)
            
        Returns:
            Formatted prompt with structured output instructions
        """
        return cls.format_prompt(
            cls.CLINICAL_TRIAL_ANALYSIS,
            format_type=format_type,
            trial_data=trial_data
        )
    
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
    
    {output_format}
    """
    
    # Add structured_patent_analysis method
    @classmethod
    def structured_patent_analysis(cls, patent_data: str, format_type: str = "default") -> str:
        """
        Generate a structured patent analysis prompt
        
        Args:
            patent_data: The patent data to analyze
            format_type: Output format type (default, json, table, bullets)
            
        Returns:
            Formatted prompt with structured output instructions
        """
        return cls.format_prompt(
            cls.PATENT_ANALYSIS,
            format_type=format_type,
            patent_data=patent_data
        )
    
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
    
    {output_format}
    """
    
    # Add structured_manufacturing_analysis method
    @classmethod
    def structured_manufacturing_analysis(cls, manufacturing_data: str, format_type: str = "default") -> str:
        """
        Generate a structured manufacturing analysis prompt
        
        Args:
            manufacturing_data: The manufacturing data to analyze
            format_type: Output format type (default, json, table, bullets)
            
        Returns:
            Formatted prompt with structured output instructions
        """
        return cls.format_prompt(
            cls.MANUFACTURING_ANALYSIS,
            format_type=format_type,
            manufacturing_data=manufacturing_data
        )
    
    # Continue this pattern for other prompt templates...

    DIGITAL_HEALTH = """
    As a digital health expert, analyze the following data:

    Digital Health Data: {digital_data}

    Provide a comprehensive analysis including:
    1. Key digital health interventions and technologies used
    2. Impact on patient outcomes and engagement
    3. Regulatory and privacy considerations
    4. Market trends and adoption barriers
    5. Future outlook for digital health in this context

    {output_format}
    """

    MARKET_ACCESS = """
    As a market access expert, analyze the following data:

    Market Access Data: {market_data}

    Provide a comprehensive analysis including:
    1. Market access challenges and opportunities
    2. Pricing and reimbursement landscape
    3. Payer and stakeholder perspectives
    4. Barriers to adoption and strategies to overcome them
    5. Regulatory and policy considerations
    6. Future outlook for market access in this context

    {output_format}
    """

    SAFETY_ANALYSIS = """
    As a safety and pharmacovigilance expert, analyze the following data:

    Safety Data: {safety_data}

    Provide a comprehensive analysis including:
    1. Key safety findings and adverse events
    2. Risk mitigation strategies
    3. Regulatory safety requirements and reporting
    4. Impact on patient outcomes and clinical practice
    5. Recommendations for ongoing safety monitoring

    {output_format}
    """

    PIPELINE_ANALYSIS = """
    As a pharmaceutical pipeline expert, analyze the following data:

    Pipeline Data: {pipeline_data}

    Provide a comprehensive analysis including:
    1. Overview of pipeline assets and development stages
    2. Competitive landscape and differentiation
    3. Key milestones and upcoming catalysts
    4. Risks and opportunities in the pipeline
    5. Strategic recommendations for portfolio management

    {output_format}
    """

    THERAPEUTIC_LANDSCAPE = """
    As a therapeutic area expert, analyze the following data:

    Therapeutic Area: {therapeutic_area}

    Provide a comprehensive analysis including:
    1. Disease burden and unmet needs
    2. Current standard of care and emerging therapies
    3. Key players and competitive dynamics
    4. Regulatory and market access considerations
    5. Future trends and innovation opportunities

    {output_format}
    """

    REGULATORY_STRATEGY = """
    As a regulatory affairs expert, analyze the following data:

    Regulatory Data: {regulatory_data}

    Provide a comprehensive analysis including:
    1. Key regulatory requirements and pathways
    2. Recent changes in regulatory policy
    3. Impact on product development and approval timelines
    4. Risk mitigation and compliance strategies
    5. Recommendations for successful regulatory submissions

    {output_format}
    """

class KumbyAI(AIService):
    """Enhanced AI service with specialized pharmaceutical industry capabilities"""
    
    def __init__(self, storage_service: StorageService):
        try:
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
            self.ai_enabled = True
        except AIServiceException as e:
            # Handle AI service initialization failure
            logging.error(f"KumbyAI initialization failed: {e}")
            self.storage_service = storage_service
            self.ai_enabled = False
            self.error_message = str(e)
            self.industry_context = {}
            self.prompt_library = PromptLibrary()
        
    async def is_healthy(self):
        """Check if the AI service is healthy and responding."""
        if not hasattr(self, 'ai_enabled') or not self.ai_enabled:
            logging.error("AI service is disabled due to initialization failure")
            return False
            
        try:
            logging.info("Checking AI service health with test prompt")
            test_prompt = "Write a one-word response to confirm you're working: 'OPERATIONAL'"
            
            # Try to generate a simple response
            start_time = time.time()
            response = await self.generate_custom_content(test_prompt)
            duration = time.time() - start_time
            
            # Check if response is valid
            if not response or len(response.strip()) == 0:
                logging.error(f"AI health check failed: Empty response received after {duration:.2f}s")
                return False
                
            # Success - healthy service
            logging.info(f"AI health check passed in {duration:.2f}s, received valid response")
            return True
        except GeminiAPIException as api_err:
            # Specific API errors - likely configuration issues
            logging.error(f"AI health check failed - API error: {api_err}")
            return False
        except asyncio.TimeoutError:
            # Timeout errors - likely service overload or unresponsive
            logging.error(f"AI health check failed - timeout error")
            return False
        except Exception as e:
            # Unexpected errors
            logging.error(f"AI health check failed - unexpected error: {e}", exc_info=True)
            return False
            
    async def generate_custom_content(self, prompt):
        """Generate content with AI, with fallback if AI is not available"""
        if not hasattr(self, 'ai_enabled') or not self.ai_enabled:
            return f"AI services are currently unavailable: {getattr(self, 'error_message', 'Unknown error')}"
        
        try:
            return await self.model.generate_content(prompt)
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"

    # Override other methods that use the AI model to check ai_enabled first

class GeminiFineTunedModel(AIModelInterface):
    """Implementation of Gemini model with pharmaceutical domain fine-tuning capabilities"""
    
    def __init__(self, config: ConfigService):
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            self.config = config
            self.genai = genai
            self.HarmCategory = HarmCategory
            self.HarmBlockThreshold = HarmBlockThreshold
            self.genai.configure(api_key=config.google_api_key)
            
            # Load the fine-tuned model
            self.model_name = config.fine_tuned_model_name or "gemini-pro"
            logging.info(f"Initializing fine-tuned Gemini model: {self.model_name}")
            
            # Initialize with appropriate safety settings
            self.default_safety_settings = [
                {
                    "category": self.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": self.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": self.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": self.HarmBlockThreshold.BLOCK_ONLY_HIGH
                },
                {
                    "category": self.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": self.HarmBlockThreshold.BLOCK_ONLY_HIGH
                },
                {
                    "category": self.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": self.HarmBlockThreshold.BLOCK_ONLY_HIGH
                }
            ]
            
            # Set model parameters for pharma domain
            self.generation_config = {
                "temperature": 0.1,  # Lower temperature for more factual outputs
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            logging.info(f"Gemini fine-tuned model initialized successfully")
        except ImportError as e:
            logging.error(f"Failed to import Google Generative AI: {e}")
            raise ImportError(f"Google Generative AI module not available: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            raise GeminiAPIException(f"Gemini model initialization failed: {e}")
    
    @retry(
        retry=retry_if_exception_type((GeminiAPIException, asyncio.TimeoutError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def generate_content(self, prompt: str, safety_settings: List[Dict] = None) -> str:
        """Generate content using the fine-tuned Gemini model with pharmaceutical domain knowledge."""
        try:
            # Add pharmaceutical domain context to the prompt if not already present
            if not any(marker in prompt for marker in ["pharmaceutical", "pharma industry", "drug development"]):
                prompt = f"As a pharmaceutical industry expert, please respond to the following: {prompt}"
            
            # Use a timeout to prevent hanging requests
            async with asyncio.timeout(self.config.ai_timeout):
                response = self._safe_generate_content(prompt, safety_settings)
                if not response or not hasattr(response, 'text'):
                    raise GeminiAPIException("Empty or invalid response from Gemini API")
                return response.text
        except asyncio.TimeoutError:
            logging.error(f"Timeout exceeded for Gemini API call ({self.config.ai_timeout}s)")
            raise asyncio.TimeoutError(f"Gemini API call timed out after {self.config.ai_timeout}s")
        except Exception as e:
            logging.error(f"Error generating content with fine-tuned Gemini model: {e}")
            raise GeminiAPIException(f"Gemini API error: {e}")
    
    def _safe_generate_content(self, prompt, safety_settings):
        """Safely generate content with error handling."""
        try:
            # Initialize model with specific parameters for pharmaceutical domain
            model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=safety_settings or self.default_safety_settings
            )
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Check for response issues
            if not response:
                logging.warning("Empty response from Gemini API")
                return self._create_fallback_response("No response generated.")
                
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if response.prompt_feedback.block_reason:
                    logging.warning(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                    return self._create_fallback_response(
                        f"Unable to process request: {response.prompt_feedback.block_reason}"
                    )
            
            return response
        except Exception as e:
            logging.error(f"Error in _safe_generate_content: {e}")
            return self._create_fallback_response(f"Error generating response: {e}")
    
    def _create_fallback_response(self, message):
        """Create a fallback response object with the same interface as a successful response."""
        class FallbackResponse:
            def __init__(self, message):
                self.text = f"I apologize, but I couldn't generate a response. {message}"
                self.prompt_feedback = None
        
        return FallbackResponse(message)
    
    async def generate_content_stream(self, prompt: str, safety_settings: List[Dict] = None) -> AsyncGenerator[str, None]:
        """Stream content generation from the fine-tuned model."""
        try:
            # Add pharmaceutical domain context if needed
            if not any(marker in prompt for marker in ["pharmaceutical", "pharma industry", "drug development"]):
                prompt = f"As a pharmaceutical industry expert, please respond to the following: {prompt}"
            
            # Initialize model with specific parameters for pharmaceutical domain
            model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=safety_settings or self.default_safety_settings
            )
            
            # Stream the response
            async with asyncio.timeout(self.config.ai_timeout):
                response = model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
        except Exception as e:
            logging.error(f"Error streaming content: {e}")
            yield f"I apologize, but I couldn't generate a streaming response: {e}"