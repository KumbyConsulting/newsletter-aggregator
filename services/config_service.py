import os
import json
import logging
import time
from threading import Lock
from typing import Dict, Any, Optional, List
from pydantic import validator, Field
from pydantic_settings import BaseSettings
from google.auth.exceptions import DefaultCredentialsError
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class ConfigException(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigSettings(BaseSettings):
    USE_VERTEX_AI: bool = False  # Moved above GEMINI_API_KEY for validator order
    GEMINI_API_KEY: str = "AI_PLACEHOLDER_FOR_VERTEX_AI"
    FLASK_SECRET_KEY: str = "dev-secret-key-placeholder"
    
    # Optional fields with defaults
    CHROMA_PERSIST_DIR: str = "chroma_db"
    RATE_LIMIT_DELAY: float = 1.0  # Changed to float based on later usage
    FLASK_ENV: str = "production"
    CACHE_FILE_PATH: str = "summary_cache.json"
    ARTICLES_FILE_PATH: str = "news_alerts.csv"
    
    # Maintenance mode settings
    MAINTENANCE_MODE: bool = False
    MAINTENANCE_END_TIME: Optional[str] = None
    MAINTENANCE_MESSAGE: Optional[str] = None
    
    # Google Cloud settings
    GCP_PROJECT_ID: Optional[str] = None
    GCP_REGION: str = "us-central1"
    GCS_BUCKET_NAME: Optional[str] = None
    USE_GCS_BACKUP: bool = False
    USE_CLOUD_LOGGING: bool = False
    STORAGE_BACKEND: str = "firestore"  # Default to Firestore
    
    # New fields based on os.getenv usage in ConfigService
    DEBUG: bool = False
    # ENVIRONMENT: str = "development" # Redundant with FLASK_ENV? Using FLASK_ENV
    PORT: int = 5000
    HOST: str = "0.0.0.0"
    # GOOGLE_API_KEY: Optional[str] = None # Handled by GEMINI_API_KEY
    # CACHE_FILE: Optional[str] = None # Handled by CACHE_FILE_PATH
    AI_MODEL_TYPE: str = "gemini_direct"
    AI_TIMEOUT: int = 30
    FINE_TUNED_MODEL_NAME: Optional[str] = None
    EMBEDDING_MODEL: str = "models/embedding-001"
    DEFAULT_OUTPUT_FORMAT: str = "default"
    ALLOWED_OUTPUT_FORMATS: List[str] = ['default', 'json', 'table', 'bullets']
    DATABASE_URI: str = "sqlite:///news_aggregator.db"
    ARTICLES_PER_PAGE: int = 20
    # RATE_LIMIT_DELAY handled above
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # USE_CLOUD_LOGGING handled above
    # GCP_PROJECT_ID handled above
    RSS_FEEDS_JSON: str = "feeds.json"
    FEED_UPDATE_INTERVAL: int = 3600
    # MAINTENANCE_MODE handled above
    # USE_GCS_BACKUP handled above
    # GCS_BUCKET_NAME handled above

    # Cache specific settings (if not defined elsewhere)
    CACHE_TTL: int = 3600
    CACHE_MAX_SIZE: int = 1000

    # Rate limit specific settings (if not defined elsewhere)
    RATE_LIMIT_DEFAULT_CALLS: int = 10
    RATE_LIMIT_DEFAULT_PERIOD: float = 1.0

    @validator('GEMINI_API_KEY')
    def validate_api_key(cls, v: str, values: Dict[str, Any]) -> str:
        # Skip validation if using Vertex AI
        use_vertex_ai = values.get('USE_VERTEX_AI', False)
        flask_env = values.get('FLASK_ENV', 'production')
        if use_vertex_ai and v == "AI_PLACEHOLDER_FOR_VERTEX_AI":
            return v
        
        # Validate API key if not using Vertex AI
        if not v or v == "AI_PLACEHOLDER_FOR_VERTEX_AI":
            if flask_env == 'production':
                raise ValueError("GEMINI_API_KEY is required when not using Vertex AI")
            else:
                print("\u26A0\uFE0F  Warning: GEMINI_API_KEY is missing. Some features may not work.")
                return v
        return v
        
    @validator('ALLOWED_OUTPUT_FORMATS', pre=True, always=True)
    def parse_allowed_formats(cls, v):
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(',')]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"  # Ignore extra fields from .env


class ConfigService:
    """Service for managing application configuration using Pydantic settings."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize config service using ConfigSettings."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            self.settings = ConfigSettings()
        except ValueError as e:
            logging.error(f"Configuration validation error: {e}")
            raise ConfigException(f"Configuration error: {e}") from e
            
        # Attributes derived from settings for convenience or compatibility
        self.debug = self.settings.DEBUG
        self.environment = self.settings.FLASK_ENV # Use FLASK_ENV
        self.app_port = self.settings.PORT
        self.app_host = self.settings.HOST
        
        self.gemini_api_key = self.settings.GEMINI_API_KEY
        self.cache_file_path = self.settings.CACHE_FILE_PATH
        self.flask_secret_key = self.settings.FLASK_SECRET_KEY
        
        self.ai_model_type = self.settings.AI_MODEL_TYPE
        self.ai_timeout = self.settings.AI_TIMEOUT
        self.use_vertex_ai = self.settings.USE_VERTEX_AI
        
        self.fine_tuned_model_name = self.settings.FINE_TUNED_MODEL_NAME
        self.embedding_model = self.settings.EMBEDDING_MODEL
        
        self.default_output_format = self.settings.DEFAULT_OUTPUT_FORMAT
        self.allowed_output_formats = self.settings.ALLOWED_OUTPUT_FORMATS
        
        self.db_uri = self.settings.DATABASE_URI
        self.articles_per_page = self.settings.ARTICLES_PER_PAGE
        
        self.storage_backend = self.settings.STORAGE_BACKEND
        
        self.rate_limit_delay = self.settings.RATE_LIMIT_DELAY
        
        self.log_level = self.settings.LOG_LEVEL
        self.log_format = self.settings.LOG_FORMAT
        self.use_cloud_logging = self.settings.USE_CLOUD_LOGGING
        self.gcp_project_id = self.settings.GCP_PROJECT_ID
        self.gcp_region = self.settings.GCP_REGION
        
        self.rss_feeds_json = self.settings.RSS_FEEDS_JSON
        self.feed_update_interval = self.settings.FEED_UPDATE_INTERVAL
        
        self.is_maintenance_mode = self.settings.MAINTENANCE_MODE
        
        self.use_gcs_backup = self.settings.USE_GCS_BACKUP
        self.gcs_bucket_name = self.settings.GCS_BUCKET_NAME
        
        # Setup logging using loaded settings
        self._setup_logging()
        self._initialized = True
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.settings.LOG_LEVEL.upper(), logging.INFO)
        # TODO: Integrate with Google Cloud Logging if self.settings.USE_CLOUD_LOGGING is True
        logging.basicConfig(level=log_level, format=self.settings.LOG_FORMAT)
        
    @property
    def is_production(self):
        """Check if environment is production."""
        return self.settings.FLASK_ENV.lower() == 'production'
    
    @property
    def is_development(self):
        """Check if environment is development."""
        return self.settings.FLASK_ENV.lower() == 'development'
    
    @property
    def is_test(self):
        """Check if environment is test."""
        return self.settings.FLASK_ENV.lower() == 'test'
    
    @property
    def is_gcp_enabled(self):
        """Check if GCP integration is enabled (based on project ID)"""
        return bool(self.settings.GCP_PROJECT_ID)
    
    def validate_output_format(self, format_type: str) -> str:
        """
        Validate the requested output format, returning a valid format type.
        
        Args:
            format_type: The requested output format type
            
        Returns:
            A valid output format type (defaulting to the default format if invalid)
        """
        if not format_type or format_type not in self.settings.ALLOWED_OUTPUT_FORMATS:
            logging.warning(f"Invalid output format '{format_type}' requested. Defaulting to '{self.settings.DEFAULT_OUTPUT_FORMAT}'.")
            return self.settings.DEFAULT_OUTPUT_FORMAT
        return format_type
        
    def get_model_config(self) -> dict:
        """
        Get configuration for the AI model based on current settings.
        
        Returns:
            Dictionary with model configuration parameters
        """
        config = {
            'model_type': self.settings.AI_MODEL_TYPE,
            'timeout': self.settings.AI_TIMEOUT
        }
        
        # Add specific configurations based on model type
        if self.settings.AI_MODEL_TYPE == 'gemini_direct':
            config['api_key'] = self.settings.GEMINI_API_KEY
        elif self.settings.AI_MODEL_TYPE == 'gemini_fine_tuned':
            config['api_key'] = self.settings.GEMINI_API_KEY
            config['model_name'] = self.settings.FINE_TUNED_MODEL_NAME or 'gemini-pro' # Consider adding default model name to ConfigSettings
            config['embedding_model'] = self.settings.EMBEDDING_MODEL
        elif self.settings.AI_MODEL_TYPE == 'vertex_ai':
            # Vertex AI specific configuration needs to be added
            # This might involve checking for Application Default Credentials (ADC)
            config['project_id'] = self.settings.GCP_PROJECT_ID
            config['location'] = self.settings.GCP_REGION
            # Potentially remove api_key or handle differently for Vertex
            pass 
            
        return config
        
    def get_cache_config(self) -> dict:
        """
        Get configuration for the cache service using ConfigSettings.
        
        Returns:
            Dictionary with cache configuration parameters
        """
        return {
            'ttl': self.settings.CACHE_TTL,
            'max_size': self.settings.CACHE_MAX_SIZE,
            'cache_file_path': self.settings.CACHE_FILE_PATH # Pass the path too
        }
        
    def get_rate_limit_config(self) -> dict:
        """
        Get configuration for rate limiting using ConfigSettings.
        
        Returns:
            Dictionary with rate limit configuration parameters
        """
        return {
            'default_calls': self.settings.RATE_LIMIT_DEFAULT_CALLS,
            'default_period': self.settings.RATE_LIMIT_DEFAULT_PERIOD,
            'delay': self.settings.RATE_LIMIT_DELAY
        }

    def get_source_urls(self) -> dict:
        """
        Get a dictionary of source names to their URLs.
        
        Returns:
            dict: Dictionary mapping source names to their URLs
        """
        try:
            # Check if we have source URLs in config
            if hasattr(self, 'config') and 'source_urls' in self.config:
                return self.config['source_urls']
            
            # Default URLs for common sources
            return {
                'Pharmaceutical Technology': 'https://www.pharmaceutical-technology.com/',
                'FiercePharma': 'https://www.fiercepharma.com/',
                'FDA News': 'https://www.fda.gov/news-events',
                'BioPharmaDive': 'https://www.biopharmadive.com/',
                'DrugDiscoveryToday': 'https://www.drugdiscoverytoday.com/',
                'PharmaTimes': 'https://www.pharmatimes.com/',
                'European Pharmaceutical Review': 'https://www.europeanpharmaceuticalreview.com/',
                'PharmaManufacturing': 'https://www.pharmamanufacturing.com/',
                'The Lancet': 'https://www.thelancet.com/',
                'New England Journal of Medicine': 'https://www.nejm.org/',
                'JAMA': 'https://jamanetwork.com/',
                'Nature Biotechnology': 'https://www.nature.com/nbt/',
                'British Medical Journal': 'https://www.bmj.com/',
                'EMA News': 'https://www.ema.europa.eu/en/news-events',
                'WHO News': 'https://www.who.int/news-room',
                'CDC': 'https://www.cdc.gov/media/index.html',
            }
        except Exception as e:
            logging.error(f"Error getting source URLs: {e}")
            return {}

# Singleton instance (optional, consider dependency injection)
# config_service = ConfigService()
