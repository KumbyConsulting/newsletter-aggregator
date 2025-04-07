"""
Startup script for running the Flask app with Waitress WSGI server.
This provides better error handling and logging for Cloud Run deployments.
"""

import os
import sys
import logging
import time
import threading
from waitress import serve

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def check_required_env_vars():
    """Check if all required environment variables are set"""
    required_vars = [
        # 'PORT',  # Removed as Cloud Run sets this automatically
        'GCP_PROJECT_ID',
        'GCS_BUCKET_NAME',
    ]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    return True

def verify_service_initialization(app):
    """Verify that all critical services are initialized"""
    try:
        with app.app_context():
            # Initialize empty list to collect all missing services
            missing_services = []
            
            # Check storage service
            if not hasattr(app, 'storage_service'):
                logging.error("Storage service not initialized")
                missing_services.append("storage_service")
            else:
                logging.info("Storage service initialized successfully")
            
            # Check AI service
            if not hasattr(app, 'ai_service'):
                logging.error("AI service not initialized")
                missing_services.append("ai_service")
            else:
                logging.info("AI service initialized successfully")
            
            # Check monitoring service
            if not hasattr(app, 'monitoring_service'):
                logging.error("Monitoring service not initialized")
                missing_services.append("monitoring_service")
            else:
                logging.info("Monitoring service initialized successfully")
            
            # Return True if all services are initialized, or if we're in production,
            # allow the app to start even with some non-critical services missing
            if len(missing_services) == 0:
                logging.info("All services initialized successfully")
                return True
            
            # For now, let's continue even if some services aren't initialized
            # This ensures the app can start in Cloud Run
            if os.environ.get('FLASK_ENV') == 'production':
                logging.warning(f"Running in production with missing services: {', '.join(missing_services)}")
                logging.warning("Continuing startup despite missing services")
                return True
            
            logging.critical(f"Service initialization failed: {', '.join(missing_services)}")
            return False
    except Exception as e:
        logging.error(f"Service initialization verification failed: {e}")
        # In production, continue anyway
        if os.environ.get('FLASK_ENV') == 'production':
            logging.warning("Continuing startup despite service verification error")
            return True
        return False

def start_health_check_server():
    """Start a simple health check server on a separate thread"""
    from flask import Flask
    health_app = Flask('health_check')
    
    @health_app.route('/_ah/health')
    def health():
        return '{"status": "healthy"}', 200
    
    health_port = int(os.environ.get('HEALTH_PORT', 8081))
    threading.Thread(
        target=lambda: serve(health_app, host='0.0.0.0', port=health_port),
        daemon=True
    ).start()
    logging.info(f"Health check server started on port {health_port}")

def main():
    """Main entry point for the application"""
    try:
        # Get the port from the environment variable
        port = int(os.environ.get('PORT', 8080))
        logging.info(f"Starting server on port {port}")
        
        # Add timeout for initialization
        start_time = time.time()
        max_init_time = 50  # seconds, leaving 10s buffer for Cloud Run's 60s timeout
        
        # Check environment variables
        if not check_required_env_vars():
            logging.critical("Required environment variables missing")
            sys.exit(1)
        
        logging.info("Importing Flask app...")
        
        # Import the Flask app with better error handling
        try:
            from app import app
            logging.info("Flask app imported successfully")
        except Exception as e:
            logging.critical(f"Failed to import Flask app: {e}", exc_info=True)
            sys.exit(1)
        
        # Log import time
        import_time = time.time() - start_time
        logging.info(f"Flask app imported in {import_time:.2f} seconds")
        
        # Check initialization timeout
        if time.time() - start_time > max_init_time:
            logging.critical(f"Initialization taking too long (>{max_init_time}s)")
            sys.exit(1)
        
        # Verify service initialization
        if not verify_service_initialization(app):
            logging.critical("Service initialization failed")
            sys.exit(1)
            
        # Start health check server
        try:
            start_health_check_server()
            logging.info("Health check server started")
        except Exception as e:
            logging.error(f"Health check server failed: {e}")
            # Continue anyway
        
        # Final timeout check before starting server
        if time.time() - start_time > max_init_time:
            logging.critical(f"Pre-server initialization exceeded {max_init_time}s")
            sys.exit(1)
        
        # Start the waitress server
        logging.info(f"Starting waitress server on port {port}...")
        serve(app, host='0.0.0.0', port=port, threads=8, url_scheme='https')
        
    except Exception as e:
        logging.critical(f"Fatal error starting server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 