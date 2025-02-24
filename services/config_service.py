from typing import Optional, Any, Dict
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from threading import Lock
from pydantic_settings import BaseSettings
from pydantic import validator
from dynaconf import Dynaconf

class ConfigSettings(BaseSettings):
    # Required fields
    GEMINI_API_KEY: str
    FLASK_SECRET_KEY: str
    
    # Optional fields with defaults
    CHROMA_PERSIST_DIR: str = "chroma_db"
    RATE_LIMIT_DELAY: int = 1
    FLASK_ENV: str = "production"
    CACHE_FILE_PATH: str = "summary_cache.json"
    ARTICLES_FILE_PATH: str = "news_alerts.csv"
    
    @validator('GEMINI_API_KEY')
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith('AI'):
            raise ValueError("Invalid API key format")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"  # Allow extra fields in case we need to add more later

class ConfigService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def __init__(self):
        self.settings = Dynaconf(
            environments=True,
            settings_files=['settings.yaml', '.secrets.yaml'],
            env_switcher='NEWSLETTER_ENV'
        )

    def _initialize(self):
        """Initialize configuration with environment variables"""
        try:
            self.settings = ConfigSettings()
            
            # API Keys
            self.gemini_api_key = self.settings.GEMINI_API_KEY
            self.flask_secret_key = self.settings.FLASK_SECRET_KEY
            
            # Storage Configuration
            self.chroma_persist_dir = self.settings.CHROMA_PERSIST_DIR
            self.cache_file_path = self.settings.CACHE_FILE_PATH
            self.articles_file_path = self.settings.ARTICLES_FILE_PATH
            
            # Rate Limiting
            self.rate_limit_delay = self.settings.RATE_LIMIT_DELAY
            
            # Environment
            self.flask_env = self.settings.FLASK_ENV
            self.debug_mode = self.flask_env == 'development'
            
        except Exception as e:
            logging.error(f"Failed to initialize configuration: {e}")
            raise

    def _get_required_env(self, key: str) -> str:
        """Get a required environment variable or raise an error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    def _get_env(self, key: str, default: str) -> str:
        """Get an environment variable with a default value"""
        return os.getenv(key, default)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.flask_env == 'production'

    def validate_configuration(self) -> bool:
        """Validate all required configuration is present"""
        try:
            required_vars = [
                ('GEMINI_API_KEY', self.gemini_api_key),
                ('FLASK_SECRET_KEY', self.flask_secret_key),
                ('CHROMA_PERSIST_DIR', self.chroma_persist_dir),
            ]

            for var_name, var_value in required_vars:
                if not var_value:
                    logging.error(f"Missing required configuration: {var_name}")
                    return False

            return True

        except Exception as e:
            logging.error(f"Error validating configuration: {e}")
            return False 