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

# Configure logging first with more verbose output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Log startup immediately
logging.info("======= CONTAINER STARTING =======")
logging.info(f"Environment: PORT={os.environ.get('PORT', 'not set')}")

def create_minimal_app():
    """Create a minimal Flask app just to bind to the port quickly"""
    from flask import Flask, jsonify
    minimal_app = Flask('minimal_app')
    
    @minimal_app.route('/_ah/health')
    def health():
        return jsonify({"status": "healthy"}), 200
    
    @minimal_app.route('/')
    def home():
        return jsonify({"status": "starting"}), 200
        
    return minimal_app

def main():
    """Main entry point for the application"""
    try:
        # Get the port immediately
        port = int(os.environ.get('PORT', 8080))
        logging.info(f"Starting on port {port}...")
        
        # First create and start a minimal app to bind to the port immediately
        minimal_app = create_minimal_app()
        
        # Start minimal app in a background thread
        threading_server = threading.Thread(
            target=lambda: serve(minimal_app, host='0.0.0.0', port=port, threads=2),
            daemon=True
        )
        threading_server.start()
        logging.info(f"Minimal app started on port {port}")
        
        # Now start loading the real app in background
        def load_real_app():
            try:
                logging.info("Loading actual Flask app...")
                from app import app
                logging.info("Flask app loaded successfully")
                
                # Wait a moment to ensure minimal app is fully initialized
                time.sleep(2)
                
                # Start the real server
                logging.info(f"Starting main application on port {port}...")
                serve(app, host='0.0.0.0', port=port, threads=8, url_scheme='https')
            except Exception as e:
                logging.critical(f"Fatal error loading real app: {e}", exc_info=True)
                sys.exit(1)

        # Start initialization in separate thread
        threading.Thread(target=load_real_app).start()
        
        # Keep main thread alive to handle signals
        while True:
            time.sleep(10)
            
    except Exception as e:
        logging.critical(f"Fatal error starting server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 