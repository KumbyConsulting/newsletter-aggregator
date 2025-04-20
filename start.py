"""
Simplified startup script for running the Flask app with Waitress WSGI server.
"""

import os
import sys
import logging
import traceback
from waitress import serve

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Log startup immediately
logging.info("======= CONTAINER STARTING =======")
PORT = int(os.environ.get('PORT', 8080))
logging.info(f"PORT set to {PORT}")
logging.info(f"Working directory: {os.getcwd()}")
logging.info("Attempting to load main Flask app...")

try:
    # Import the main Flask app instance
    from app import app
    logging.info("Flask app imported successfully.")

    # Run the Waitress server with the main app
    logging.info(f"Starting Waitress server on host 0.0.0.0 port {PORT}...")
    # Output more diagnostic info about the environment
    logging.info(f"Environment variables: PORT={os.environ.get('PORT')}, FLASK_ENV={os.environ.get('FLASK_ENV')}")
    
    # Make sure waitress handles connections properly
    serve(
        app, 
        host='0.0.0.0', 
        port=PORT, 
        threads=8, 
        url_scheme='https',
        clear_untrusted_proxy_headers=True,
        channel_timeout=30
    )

except ImportError as e:
    logging.critical(f"Failed to import Flask app: {e}", exc_info=True)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    logging.critical(f"Fatal error starting server: {e}", exc_info=True)
    traceback.print_exc()
    sys.exit(1) 