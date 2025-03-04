import os
import json
import logging
import time
from threading import Lock
from typing import Dict, Any, Optional, List
from pydantic import  validator
from pydantic_settings import BaseSettings
from google.auth.exceptions import DefaultCredentialsError

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
        with cls._lock:
            if cls._instance is None:
                cls._instance = ConfigService()
        return cls._instance
    
    def __new__(cls):
        with cls._lock:
            if not hasattr(cls, '_instance') or cls._instance is None:
                cls._instance = super(ConfigService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        with self._lock:
            if not hasattr(self, '_initialized') or not self._initialized:
                self._initialize()
                self._initialized = True
    
    def _initialize(self):
        # Load configuration from environment variables
        try:
            self.settings = ConfigSettings()
            self._load_secrets_from_gcp()
            
            # Set up file paths
            self.chroma_persist_dir = self.settings.CHROMA_PERSIST_DIR
            self.cache_file_path = self.settings.CACHE_FILE_PATH
            self.articles_file_path = self.settings.ARTICLES_FILE_PATH
            
            # API keys and secrets
            self.gemini_api_key = self.settings.GEMINI_API_KEY
            self.flask_secret_key = self.settings.FLASK_SECRET_KEY
            
            # Google Cloud Configuration
            self.gcp_project_id = self.settings.GCP_PROJECT_ID
            self.gcp_region = self.settings.GCP_REGION
            self.gcs_bucket_name = self.settings.GCS_BUCKET_NAME
            self.use_gcs = self.settings.USE_GCS_BACKUP
            
            # Check if Vertex AI should be used
            vertex_ai_setting = self.settings.USE_VERTEX_AI
            if vertex_ai_setting and self.settings.GCP_PROJECT_ID:
                try:
                    # In Cloud Run, let's trust the env var and perform a simple auth check
                    # This avoids the false warning about missing key.json
                    from google.auth import default
                    try:
                        # Try to get default credentials to validate
                        credentials, project = default()
                        # Check if we're running on GCP by checking metadata
                        import requests
                        try:
                            # Try to access GCP metadata server - will succeed on GCP, fail locally
                            # This URL is only accessible from within GCP services
                            response = requests.get(
                                'http://metadata.google.internal/computeMetadata/v1/instance/id',
                                headers={'Metadata-Flavor': 'Google'},
                                timeout=1
                            )
                            is_on_gcp = response.status_code == 200
                        except requests.exceptions.RequestException:
                            is_on_gcp = False
                            
                        if is_on_gcp:
                            # If we're on GCP, trust the attached service account
                            self.use_vertex_ai = True
                            logging.info("Running on GCP with service account, enabling Vertex AI")
                        else:
                            # Local development - check if we can actually use the credentials
                            from google.cloud import storage
                            client = storage.Client()
                            # Just try a simple operation 
                            list(client.list_buckets(max_results=1))
                            self.use_vertex_ai = True
                            logging.info("Found valid Google Cloud credentials, enabling Vertex AI")
                    except Exception as e:
                        logging.warning(f"Google Cloud authentication failed, disabling Vertex AI: {str(e)}")
                        self.use_vertex_ai = False
                except (ImportError, Exception) as e:
                    logging.warning(f"Could not initialize Google Cloud SDK, disabling Vertex AI: {str(e)}")
                    self.use_vertex_ai = False
            else:
                self.use_vertex_ai = False
                if not vertex_ai_setting:
                    logging.info("Vertex AI disabled by configuration")
                else:
                    logging.warning("Vertex AI disabled: missing GCP_PROJECT_ID")
            
            # Additional settings
            self.use_cloud_logging = self.settings.USE_CLOUD_LOGGING
            self.storage_backend = self.settings.STORAGE_BACKEND
            
            # Validate the configuration
            self.validate_configuration()
            
        except Exception as e:
            logging.error(f"Error initializing configuration: {e}")
            raise

    def _load_secrets_from_gcp(self):
        """
        Attempt to load secrets from Google Secret Manager if running in GCP
        """
        try:
            # Only try to load secrets if GCP project is configured
            if not self.settings.GCP_PROJECT_ID:
                return
                
            # Only try to load secrets if in production
            if not self.is_production:
                return
                
            from google.cloud import secretmanager_v1
            
            client = secretmanager_v1.SecretManagerServiceClient()
            
            # Try to load GEMINI_API_KEY from Secret Manager if needed
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

    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable or raise an exception"""
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    def _get_env(self, key: str, default: str) -> str:
        """Get an environment variable with a default value"""
        return os.environ.get(key, default)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.settings.FLASK_ENV == "production"

    @property
    def is_gcp_enabled(self) -> bool:
        """Check if GCP integration is enabled"""
        return bool(self.settings.GCP_PROJECT_ID)

    def validate_configuration(self) -> bool:
        """Validate that the configuration is correct"""
        try:
            # Validate that secrets are set
            if not self.flask_secret_key or self.flask_secret_key == "dev-secret-key-placeholder":
                if self.is_production:
                    logging.error("FLASK_SECRET_KEY is required in production")
                    return False
                else:
                    logging.warning("Using default FLASK_SECRET_KEY in development")
            
            # Validate that if Vertex AI is enabled, we have credentials
            if self.use_vertex_ai:
                try:
                    from google.auth import default
                    credentials, project = default()
                except Exception as e:
                    logging.error(f"Vertex AI is enabled but credentials are missing: {e}")
                    return False
            else:
                # Validate that we have a Gemini API key if not using Vertex AI
                if not self.gemini_api_key or self.gemini_api_key == "AI_PLACEHOLDER_FOR_VERTEX_AI":
                    logging.error("GEMINI_API_KEY is required when not using Vertex AI")
                    return False
            
            # Validate storage backend
            if self.storage_backend not in ["chromadb", "firestore"]:
                logging.error(f"Invalid STORAGE_BACKEND: {self.storage_backend}. Valid options: chromadb, firestore")
                return False
                
            # Validate that if using Firestore, we have GCP project ID
            if self.storage_backend == "firestore" and not self.gcp_project_id:
                logging.error("GCP_PROJECT_ID is required when using Firestore backend")
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            logging.error(f"Error validating configuration: {e}")
            return False

    def get_vertex_ai_config(self) -> Dict[str, Any]:
        """Get configuration for Vertex AI"""
        return {
            "project_id": self.gcp_project_id,
            "region": self.gcp_region,
            "use_vertex_ai": self.use_vertex_ai
        }
