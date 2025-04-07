import os
import json
import logging
import time
from threading import Lock
from typing import Dict, Any, Optional, List
from pydantic import  validator
from pydantic_settings import BaseSettings
from google.auth.exceptions import DefaultCredentialsError

class ConfigException(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigSettings(BaseSettings):
    # Required fields with defaults for Vertex AI mode
    GEMINI_API_KEY: str = "AI_PLACEHOLDER_FOR_VERTEX_AI"
    FLASK_SECRET_KEY: str = "dev-secret-key-placeholder"
    
    # Optional fields with defaults
    CHROMA_PERSIST_DIR: str = "chroma_db"
    RATE_LIMIT_DELAY: int = 1
    FLASK_ENV: str = "production"
    CACHE_FILE_PATH: str = "summary_cache.json"
    ARTICLES_FILE_PATH: str = "news_alerts.csv"
    
    # Google Cloud settings
    GCP_PROJECT_ID: Optional[str] = None
    GCP_REGION: str = "us-central1"
    GCS_BUCKET_NAME: Optional[str] = None
    USE_GCS_BACKUP: bool = False
    USE_VERTEX_AI: bool = False
    USE_CLOUD_LOGGING: bool = False
    STORAGE_BACKEND: str = "chromadb"  # Options: chromadb, firestore
    
    @validator('GEMINI_API_KEY')
    def validate_api_key(cls, v: str, values: Dict[str, Any]) -> str:
        # Skip validation if using Vertex AI
        use_vertex_ai = values.get('USE_VERTEX_AI', False)
        if use_vertex_ai and v == "AI_PLACEHOLDER_FOR_VERTEX_AI":
            return v
        
        # Validate API key if not using Vertex AI
        if not v or v == "AI_PLACEHOLDER_FOR_VERTEX_AI":
            raise ValueError("GEMINI_API_KEY is required when not using Vertex AI")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields in case we need to add more later


class ConfigService:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance with thread safety"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ConfigService()
            return cls._instance
    
    def __new__(cls):
        """Create singleton instance with thread safety"""
        with cls._lock:
            if not hasattr(cls, '_instance') or cls._instance is None:
                cls._instance = super(ConfigService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize configuration with enhanced error handling"""
        with self._lock:
            if not hasattr(self, '_initialized') or not self._initialized:
                try:
                    self._initialize()
                    self._initialized = True
                    logging.info("ConfigService initialized successfully")
                except Exception as e:
                    logging.error(f"Failed to initialize ConfigService: {e}")
                    raise ConfigException(f"Configuration initialization failed: {str(e)}")
    
    def _initialize(self):
        """Initialize configuration with proper error handling"""
        try:
            # Load configuration from environment variables
            self.settings = ConfigSettings()
            
            # Initialize basic attributes
            self.use_vertex_ai = self.settings.USE_VERTEX_AI
            self.gemini_api_key = self.settings.GEMINI_API_KEY
            self.flask_secret_key = self.settings.FLASK_SECRET_KEY
            self.storage_backend = self.settings.STORAGE_BACKEND
            self.use_cloud_logging = self.settings.USE_CLOUD_LOGGING
            
            # Load secrets from GCP if configured
            if self.settings.GCP_PROJECT_ID:
                self._load_secrets_from_gcp()
            
            # Set up file paths with validation
            self._setup_file_paths()
            
            # Set up Google Cloud configuration
            self._setup_gcp_config()
            
            # Validate the configuration
            if not self.validate_configuration():
                raise ConfigException("Configuration validation failed")
                
        except Exception as e:
            logging.error(f"Error initializing configuration: {e}")
            raise ConfigException(f"Configuration initialization failed: {str(e)}")
    
    def _setup_file_paths(self):
        """Set up file paths with validation"""
        try:
            self.chroma_persist_dir = self.settings.CHROMA_PERSIST_DIR
            self.cache_file_path = self.settings.CACHE_FILE_PATH
            self.articles_file_path = self.settings.ARTICLES_FILE_PATH
            
            # Validate paths
            for path in [self.chroma_persist_dir, self.cache_file_path, self.articles_file_path]:
                if not path:
                    raise ConfigException(f"Invalid file path: {path}")
                    
        except Exception as e:
            logging.error(f"Error setting up file paths: {e}")
            raise ConfigException(f"File path setup failed: {str(e)}")
    
    def _setup_gcp_config(self):
        """Set up Google Cloud configuration with validation"""
        try:
            self.gcp_project_id = self.settings.GCP_PROJECT_ID
            self.gcp_region = self.settings.GCP_REGION
            self.gcs_bucket_name = self.settings.GCS_BUCKET_NAME
            self.use_gcs = self.settings.USE_GCS_BACKUP
            
            # Check if Vertex AI should be used
            if self.settings.USE_VERTEX_AI:
                self._setup_vertex_ai()
                
        except Exception as e:
            logging.error(f"Error setting up GCP configuration: {e}")
            raise ConfigException(f"GCP configuration setup failed: {str(e)}")
    
    def _setup_vertex_ai(self):
        """Set up Vertex AI with proper validation"""
        try:
            if not self.gcp_project_id:
                raise ConfigException("GCP_PROJECT_ID is required for Vertex AI")
                
            # Try to get default credentials
            from google.auth import default
            try:
                credentials, project = default()
                
                # Check if we're running on GCP
                import requests
                try:
                    response = requests.get(
                        'http://metadata.google.internal/computeMetadata/v1/instance/id',
                        headers={'Metadata-Flavor': 'Google'},
                        timeout=1
                    )
                    is_on_gcp = response.status_code == 200
                except requests.exceptions.RequestException:
                    is_on_gcp = False
                    
                if is_on_gcp:
                    self.use_vertex_ai = True
                    logging.info("Running on GCP with service account, enabling Vertex AI")
                else:
                    # Local development - check credentials
                    from google.cloud import storage
                    client = storage.Client()
                    list(client.list_buckets(max_results=1))
                    self.use_vertex_ai = True
                    logging.info("Found valid Google Cloud credentials, enabling Vertex AI")
                    
            except Exception as e:
                logging.warning(f"Google Cloud authentication failed, disabling Vertex AI: {e}")
                self.use_vertex_ai = False
                
        except Exception as e:
            logging.error(f"Error setting up Vertex AI: {e}")
            self.use_vertex_ai = False
    
    def _load_secrets_from_gcp(self):
        """Load secrets from Google Secret Manager with proper error handling"""
        try:
            if not self.is_production:
                return
                
            from google.cloud import secretmanager_v1
            client = secretmanager_v1.SecretManagerServiceClient()
            
            # Load GEMINI_API_KEY if needed
            if self.settings.GEMINI_API_KEY == "AI_PLACEHOLDER_FOR_VERTEX_AI" and not self.settings.USE_VERTEX_AI:
                try:
                    name = f"projects/{self.settings.GCP_PROJECT_ID}/secrets/GEMINI_API_KEY/versions/latest"
                    response = client.access_secret_version(request={"name": name})
                    self.settings.GEMINI_API_KEY = response.payload.data.decode("UTF-8")
                    logging.info("Loaded GEMINI_API_KEY from Secret Manager")
                except Exception as e:
                    logging.warning(f"Failed to load GEMINI_API_KEY from Secret Manager: {e}")
                    
        except ImportError:
            logging.warning("google-cloud-secret-manager not installed, skipping secret loading")
        except Exception as e:
            logging.warning(f"Error loading secrets from GCP: {e}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.settings.FLASK_ENV == "production"
    
    @property
    def is_gcp_enabled(self) -> bool:
        """Check if GCP integration is enabled"""
        return bool(self.settings.GCP_PROJECT_ID)
    
    def validate_configuration(self) -> bool:
        """Validate configuration with enhanced error handling"""
        try:
            # Validate secrets
            if not self.flask_secret_key or self.flask_secret_key == "dev-secret-key-placeholder":
                if self.is_production:
                    logging.error("FLASK_SECRET_KEY is required in production")
                    return False
                else:
                    logging.warning("Using default FLASK_SECRET_KEY in development")
            
            # Validate Vertex AI configuration
            if self.use_vertex_ai:
                try:
                    from google.auth import default
                    credentials, project = default()
                except Exception as e:
                    logging.error(f"Vertex AI is enabled but credentials are missing: {e}")
                    return False
            else:
                # Validate Gemini API key
                if not self.gemini_api_key or self.gemini_api_key == "AI_PLACEHOLDER_FOR_VERTEX_AI":
                    logging.error("GEMINI_API_KEY is required when not using Vertex AI")
                    return False
            
            # Validate storage backend
            if self.storage_backend not in ["chromadb", "firestore"]:
                logging.error(f"Invalid STORAGE_BACKEND: {self.storage_backend}")
                return False
                
            # Validate Firestore configuration
            if self.storage_backend == "firestore" and not self.gcp_project_id:
                logging.error("GCP_PROJECT_ID is required when using Firestore backend")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating configuration: {e}")
            return False
    
    def get_vertex_ai_config(self) -> Dict[str, Any]:
        """Get Vertex AI configuration with validation"""
        try:
            if not self.use_vertex_ai:
                raise ConfigException("Vertex AI is not enabled")
                
            return {
                "project_id": self.gcp_project_id,
                "region": self.gcp_region,
                "use_vertex_ai": self.use_vertex_ai
            }
        except Exception as e:
            logging.error(f"Error getting Vertex AI config: {e}")
            raise ConfigException(f"Failed to get Vertex AI config: {str(e)}")
