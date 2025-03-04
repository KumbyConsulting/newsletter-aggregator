import logging
import sys
from .config_service import ConfigService

# Add Google Cloud Logging imports
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
import google.cloud.logging_v2.handlers.transports.sync as sync_transport

class LoggingService:
    """Service for configuring application logging"""
    
    @staticmethod
    def configure_logging():
        """Configure logging based on environment"""
        config = ConfigService()
        
        # Set up basic logging format
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
        
        # Add Google Cloud Logging if enabled
        if config.use_cloud_logging and config.gcp_project_id:
            try:
                # Initialize Cloud Logging client
                client = cloud_logging.Client(project=config.gcp_project_id)
                
                # Create a Cloud Logging handler with synchronous transport
                cloud_handler = CloudLoggingHandler(
                    client,
                    name="newsletter_aggregator",
                    transport=sync_transport.SyncTransport
                )
                
                # Add handler to the root logger
                root_logger.addHandler(cloud_handler)
                
                # Log successful setup
                logging.info("Google Cloud Logging configured successfully")
                
            except Exception as e:
                logging.error(f"Failed to configure Google Cloud Logging: {e}")
                logging.warning("Falling back to console logging only")
        
        # Set higher log level for noisy libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
        
        return root_logger 