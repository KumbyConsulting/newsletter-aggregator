"""
Startup script for running the Flask app with Waitress WSGI server.
This provides better error handling and logging for Cloud Run deployments.
"""

import os
import sys
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Get the port from the environment variable
port = int(os.environ.get('PORT', 8080))
logging.info(f"Starting server on port {port}")

try:
    # Log start time
    start_time = time.time()
    logging.info("Importing Flask app...")
    
    # Import the Flask app
    from app import app
    
    # Log import time
    import_time = time.time() - start_time
    logging.info(f"Flask app imported successfully in {import_time:.2f} seconds")
    
    # Log routes
    logging.info(f"Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
    
    # Start the waitress server
    logging.info(f"Starting waitress server on port {port}...")
    from waitress import serve
    serve(app, host='0.0.0.0', port=port, threads=8)
    
except Exception as e:
    logging.error(f"Error starting server: {e}", exc_info=True)
    sys.exit(1) 