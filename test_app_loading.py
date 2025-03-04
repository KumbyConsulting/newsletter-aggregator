"""
Simple test script to verify the Flask application can be loaded without errors.
This helps diagnose initialization issues before deployment.
"""

import logging
import sys

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logging.info("Testing application loading...")

try:
    # Just try to import the app and verify it initializes
    from app import app
    
    logging.info("✅ Success! The Flask app loaded successfully")
    logging.info(f"App routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
    
except Exception as e:
    logging.error(f"❌ Error loading the Flask app: {e}", exc_info=True)
    sys.exit(1) 