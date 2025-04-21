from typing import List, Dict, Optional, Any, Union, Tuple
from flask import Flask, render_template, request, jsonify, make_response, abort, redirect, url_for, session, flash, Response
from flask_cors import CORS
import pandas as pd
import json
import logging
import time
import asyncio
import os
import re
import csv
from functools import wraps
from datetime import datetime, timedelta
from urllib.parse import urlparse
import threading
from threading import Lock
import argparse
from pathlib import Path
import io
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import uuid
from google.cloud import firestore

# Import services
from services.config_service import ConfigService
from services.cache_service import CacheService
from services.rate_limiting_service import RateLimitingService
from services.storage_service import StorageService, StorageException
from services.monitoring_service import MonitoringService
from services.logging_service import LoggingService
from newsLetter import scrape_news, RateLimitException, generate_missing_summaries, verify_database_consistency
from services.constants import TOPICS
from services.ai_service import KumbyAI, AIServiceException, GeminiAPIException
from io import StringIO

# Only import Google Cloud Storage if it's enabled
if os.environ.get('USE_GCS_BACKUP') == 'true':
    from google.cloud import storage
from cachetools import TTLCache, LRUCache
from google.auth.exceptions import DefaultCredentialsError

# Load environment variables
load_dotenv()

# Initialize configuration
config = ConfigService()
# Remove the validation call as the method does not exist
# if not config.validate_configuration():
#     raise ValueError("Invalid configuration. Please check your .env file.")

# Configure logging
LoggingService.configure_logging()
logging.info("Starting Newsletter Aggregator application")

# Initialize monitoring
monitoring_service = MonitoringService()
logging.info("Monitoring service initialized")

# Initialize services in the correct order
cache_service = CacheService()
rate_limiting_service = RateLimitingService()

# Configure rate limits for different endpoints
rate_limiting_service.configure_limit("api", calls=100, period=60)  # 100 calls per minute
rate_limiting_service.configure_limit("scrape", calls=1, period=300)  # 1 scrape per 5 minutes
rate_limiting_service.configure_limit("summarize", calls=10, period=60)  # 10 summaries per minute

# Define async route decorator
def async_route(f):
    """Decorator for async routes that provides a unique endpoint name for each wrapped function
    and ensures event loops are properly handled in a multi-threaded environment"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if none exists in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the coroutine in this loop
        return loop.run_until_complete(f(*args, **kwargs))
    
    # Use a unique name for the wrapped function based on the original function's name
    wrapped.__name__ = f"{f.__name__}_async_wrapper"
    return wrapped

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = config.flask_secret_key

# Initialize services
storage_service = StorageService()

try:
    ai_service = KumbyAI(storage_service)  # Use KumbyAI instead of AIService
    logging.info("AI service initialized successfully")
except Exception as e:
    # Critical error but don't crash the app
    ai_service = None
    logging.critical(f"Fatal error starting AI service: {str(e)}")
    # Continue without AI service - the application will have limited functionality

# Custom API Error Class
class APIError(Exception):
    """Custom exception class for API errors"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
        self.payload['status'] = 'error'

    def to_dict(self):
        return {
            'error': self.message,
            'status': 'error',
            'payload': self.payload,
            'status_code': self.status_code
        }

# Error Handler Registration
def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """Handle custom API errors"""
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(400)
    def bad_request_error(error):
        """Handle 400 Bad Request errors with consistent format"""
        return jsonify({
            'error': str(error.description),
            'status': 'error',
            'status_code': 400
        }), 400

    @app.errorhandler(401)
    def unauthorized_error(error):
        """Handle 401 Unauthorized errors with consistent format"""
        return jsonify({
            'error': 'Unauthorized access',
            'status': 'error',
            'status_code': 401
        }), 401

    @app.errorhandler(403)
    def forbidden_error(error):
        """Handle 403 Forbidden errors with consistent format"""
        return jsonify({
            'error': 'Access forbidden',
            'status': 'error',
            'status_code': 403
        }), 403

    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 Not Found errors with consistent format"""
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Resource not found',
                'status': 'error',
                'status_code': 404
            }), 404
        return render_template('errors/404.html', error=error), 404

    @app.errorhandler(429)
    def too_many_requests_error(error):
        """Handle 429 Too Many Requests errors with consistent format"""
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Rate limit exceeded',
                'status': 'error',
                'status_code': 429,
                'retry_after': error.description.get('retry_after', 60)
            }), 429
        return render_template('errors/429.html', error=error), 429

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error with consistent format"""
        logging.error(f"Internal Server Error: {error}", exc_info=True)
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal server error',
                'status': 'error',
                'status_code': 500
            }), 500
        return render_template('errors/500.html', error=error), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """Handle any unhandled exceptions with consistent format"""
        logging.error(f"Unhandled Exception: {error}", exc_info=True)
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'An unexpected error occurred',
                'status': 'error',
                'status_code': 500
            }), 500
        return render_template('errors/500.html', error=error), 500

# Register error handlers now that the function is defined
register_error_handlers(app)

# Request Validation Middleware
def validate_json_payload(schema=None):
    """Decorator to validate JSON payload against a schema"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                raise APIError("Content-Type must be application/json", status_code=400)
            
            data = request.get_json()
            if not data:
                raise APIError("No JSON data provided", status_code=400)
            
            if schema:
                try:
                    validate_schema(data, schema)
                except Exception as e:
                    raise APIError(f"Invalid request data: {str(e)}", status_code=400)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_query_params(*required_params):
    """Decorator to validate query parameters"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            missing_params = [param for param in required_params if param not in request.args]
            if missing_params:
                raise APIError(
                    f"Missing required query parameters: {', '.join(missing_params)}", 
                    status_code=400
                )
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_pagination(f):
    """Decorator to validate and normalize pagination parameters"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            page = max(1, request.args.get('page', 1, type=int))
            per_page = min(50, max(10, request.args.get('per_page', 10, type=int)))
        except ValueError:
            raise APIError("Invalid pagination parameters", status_code=400)
        
        # Add validated parameters to request object
        request.pagination = {
            'page': page,
            'per_page': per_page,
            'offset': (page - 1) * per_page
        }
        return f(*args, **kwargs)
    return decorated_function

def validate_sort_params(valid_fields, valid_orders=None):
    """Decorator to validate sorting parameters"""
    if valid_orders is None:
        valid_orders = {'asc', 'desc'}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            sort_by = request.args.get('sort_by')
            sort_order = request.args.get('sort_order', 'desc').lower()
            
            if sort_by and sort_by not in valid_fields:
                raise APIError(
                    f"Invalid sort field. Must be one of: {', '.join(valid_fields)}", 
                    status_code=400
                )
            
            if sort_order not in valid_orders:
                raise APIError(
                    f"Invalid sort order. Must be one of: {', '.join(valid_orders)}", 
                    status_code=400
                )
            
            # Add validated parameters to request object
            request.sort = {
                'field': sort_by,
                'order': sort_order
            }
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Schema validation helper
def validate_schema(data, schema):
    """Validate data against a schema"""
    for field, rules in schema.items():
        if 'required' in rules and rules['required'] and field not in data:
            raise ValueError(f"Missing required field: {field}")
        
        if field in data:
            value = data[field]
            
            # Type validation
            if 'type' in rules:
                expected_type = rules['type']
                if not isinstance(value, expected_type):
                    raise ValueError(f"Field {field} must be of type {expected_type.__name__}")
            
            # Length validation
            if 'min_length' in rules and len(value) < rules['min_length']:
                raise ValueError(f"Field {field} must be at least {rules['min_length']} characters long")
            if 'max_length' in rules and len(value) > rules['max_length']:
                raise ValueError(f"Field {field} must be at most {rules['max_length']} characters long")
            
            # Range validation
            if 'min_value' in rules and value < rules['min_value']:
                raise ValueError(f"Field {field} must be greater than or equal to {rules['min_value']}")
            if 'max_value' in rules and value > rules['max_value']:
                raise ValueError(f"Field {field} must be less than or equal to {rules['max_value']}")
            
            # Pattern validation
            if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                raise ValueError(f"Field {field} does not match required pattern")

def cache_key(*args, **kwargs):
    """Generate a cache key from request parameters"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

# Request middleware
@app.before_request
def before_request():
    """Actions to perform before each request."""
    # Add request timestamp for performance monitoring
    request.start_time = time.time()
    
    # Log request details in development
    if app.debug:
        logging.debug(f"Request: {request.method} {request.url}")
        logging.debug(f"Headers: {dict(request.headers)}")
    
    # Check maintenance mode
    if config.is_maintenance_mode and request.endpoint != 'maintenance':
        return render_template('errors/maintenance.html'), 503

@app.after_request
def after_request(response):
    """Actions to perform after each request."""
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Calculate and log request duration
    try:
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            try:
                monitoring_service.record_request_duration(
                    request.endpoint or 'unknown',
                    duration,
                    response.status_code
                )
            except Exception as e:
                logging.error(f"Error recording request duration: {e}")
            
            # Log response in development
            if app.debug:
                logging.debug(f"Response: {response.status}")
                logging.debug(f"Duration: {duration:.2f}s")
    except Exception as e:
        logging.error(f"Error in after_request: {e}")
    
    return response

@app.teardown_request
def teardown_request(exception):
    """Clean up after each request."""
    if exception:
        logging.error(f"Request teardown error: {exception}")

@app.teardown_appcontext
def teardown_appcontext(exception):
    """Clean up application context."""
    if exception:
        logging.error(f"App context teardown error: {exception}")

# Log GCP configuration status
if config.is_gcp_enabled:
    logging.info(f"Google Cloud Platform enabled with project: {config.gcp_project_id}")
    logging.info(f"Using storage backend: {config.storage_backend}")
    if config.use_gcs_backup:
        logging.info(f"GCS backup enabled with bucket: {config.gcs_bucket_name}")
    if config.use_vertex_ai:
        logging.info("Using Vertex AI for Gemini model access")
    if config.use_cloud_logging:
        logging.info("Cloud Logging enabled")
else:
    logging.info("Google Cloud Platform integration not enabled")

# Add custom template filters
@app.template_filter('max_value')
def max_value(a, b):
    return max(a, b)

@app.template_filter('min_value')
def min_value(a, b):
    return min(a, b)

@app.route('/_ah/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200

@app.route('/_ah/warmup')
def warmup():
    """Warmup endpoint for Cloud Run"""
    return jsonify({"status": "warmed_up"}), 200

def clean_html(raw_html):
    """Sanitize HTML content for safe display"""
    if pd.isna(raw_html) or not raw_html or raw_html == 'nan':
        return "No description available"
    try:
        # For descriptions that should be plain text
        if isinstance(raw_html, str) and ('<' in raw_html and '>' in raw_html):
            # Create a whitelist of allowed tags
            allowed_tags = ['p', 'br', 'b', 'i', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            
            # Parse HTML
            soup = BeautifulSoup(raw_html, "html.parser")
            
            # Remove unwanted tags
            for tag in soup.find_all():
                if tag.name.lower() not in allowed_tags:
                    # Replace with its text content
                    tag.replace_with(tag.get_text())
            
            # Convert back to HTML
            sanitized_html = str(soup)
            return sanitized_html if sanitized_html.strip() else "No description available"
        else:
            # If it's not HTML, just return as is
            text = str(raw_html)
            text = ' '.join(text.split())  # Clean up whitespace
            return text if text.strip() else "No description available"
    except Exception as e:
        logging.error(f"Error cleaning HTML: {e}")
        # Fallback to simple stripping of all tags
        text = BeautifulSoup(raw_html, "html.parser").get_text()
        return ' '.join(text.split()) if text.strip() else "No description available"

def strip_html_tags(raw_html):
    """Completely strip all HTML tags, returning only text"""
    if pd.isna(raw_html) or not raw_html or raw_html == 'nan':
        return "No description available"
    try:
        # Remove all HTML tags
        text = BeautifulSoup(raw_html, "html.parser").get_text()
        # Clean up whitespace
        text = ' '.join(text.split())
        return text if text.strip() else "No description available"
    except Exception as e:
        logging.error(f"Error stripping HTML: {e}")
        return str(raw_html)

@app.route('/', methods=['GET', 'POST'], endpoint='index')
@async_route
@cache_service.cache_decorator(namespace="index", ttl=300)  # Cache for 5 minutes
@rate_limiting_service.rate_limit_decorator(key="api")
@validate_pagination
@validate_sort_params(valid_fields={'pub_date', 'title', 'source', 'topic'})
async def index():
    """Main index route with improved search functionality and caching"""
    try:
        # Get validated pagination and sort parameters
        pagination = request.pagination
        sort = request.sort
        
        # Get and sanitize query parameters
        search_query = request.args.get('search', '').strip()
        selected_topic = request.args.get('topic', 'All')
        
        # Validate topic if provided
        if selected_topic != 'All' and selected_topic not in TOPICS:
            raise APIError(f"Invalid topic. Must be one of: All, {', '.join(TOPICS.keys())}")
        
        # Get articles with enhanced parameters
        try:
            result = await storage_service.get_articles(
                page=pagination['page'],
                limit=pagination['per_page'],
                topic=selected_topic if selected_topic != 'All' else None,
                search_query=search_query,
                sort_by=sort['field'],
                sort_order=sort['order']
            )
        except Exception as e:
            logging.error(f"Error fetching articles: {e}")
            raise APIError("Failed to fetch articles", status_code=500)
        
        # Process articles
        try:
            formatted_articles = []
            for article in result['articles']:
                metadata = article.get('metadata', {})
                formatted_articles.append({
                    'id': article.get('id'),
                    'title': metadata.get('title', 'Unknown Title'),
                    'description': clean_html(metadata.get('description', 'No description available')),
                    'link': metadata.get('link', '#'),
                    'pub_date': metadata.get('pub_date', 'Unknown date'),
                    'topic': metadata.get('topic', 'Uncategorized'),
                    'source': metadata.get('source', 'Unknown source'),
                    'summary': metadata.get('summary'),
                    'image_url': metadata.get('image_url', ''),
                    'has_full_content': metadata.get('has_full_content', False),
                    'reading_time': metadata.get('reading_time', calculate_reading_time(metadata.get('description', ''))),
                    'relevance_score': metadata.get('relevance_score', None),
                    'is_recent': is_recent_article(metadata.get('pub_date', '')),
                })
        except Exception as e:
            logging.error(f"Error formatting articles: {e}")
            raise APIError("Failed to process articles", status_code=500)
        
        # Calculate pagination
        total = result['total']
        showing_from = (pagination['page'] - 1) * pagination['per_page'] + 1 if total > 0 else 0
        showing_to = min(pagination['page'] * pagination['per_page'], total)
        last_page = (total + pagination['per_page'] - 1) // pagination['per_page']
        
        # Calculate page range
        if last_page <= 5:
            start_page = 1
            end_page = last_page
        else:
            if pagination['page'] <= 3:
                start_page = 1
                end_page = 5
            elif pagination['page'] >= last_page - 2:
                start_page = last_page - 4
                end_page = last_page
            else:
                start_page = pagination['page'] - 2
                end_page = pagination['page'] + 2
        
        # Get topic distribution
        try:
            topic_distribution = await get_topic_distribution()
        except Exception as e:
            logging.error(f"Error getting topic distribution: {e}")
            topic_distribution = []  # Fallback to empty list
        
        template_data = {
            'articles': formatted_articles,
            'total': total,
            'showing_from': showing_from,
            'showing_to': showing_to,
            'page': pagination['page'],
            'last_page': last_page,
            'start_page': start_page,
            'end_page': end_page,
            'selected_topic': selected_topic,
            'search_query': search_query,
            'sort_by': sort['field'],
            'sort_order': sort['order'],
            'per_page': pagination['per_page'],
            'topic_distribution': topic_distribution,
            'topics': TOPICS.keys()
        }
        
        # Record metrics
        monitoring_service.record_count("page_views", labels={"page": "index"})
        
        return render_template('index.html', **template_data)
        
    except APIError as e:
        # Let the error handler handle API errors
        raise
    except RateLimitException as e:
        logging.warning(f"Rate limit exceeded: {e}")
        raise APIError(str(e), status_code=429)
    except Exception as e:
        logging.error(f"Unhandled error in index route: {e}", exc_info=True)
        raise APIError("An unexpected error occurred", status_code=500)

async def get_topics_with_counts():
    """Get topics list with article counts"""
    try:
        # Get all topics from the database with their counts
        topics_result = await storage_service.get_topic_counts()
        
        # Format topics with counts
        topics = []
        for topic, count in topics_result.items():
            topics.append({
                'name': topic,
                'count': count,
                'percentage': round((count / max(1, sum(topics_result.values()))) * 100, 1)
            })
        
        # Sort topics by count in descending order
        return sorted(topics, key=lambda x: x['count'], reverse=True)
    except Exception as e:
        logging.error(f"Error getting topics with counts: {e}")
        return sorted(TOPICS.keys())  # Fallback to basic topics list

async def get_topic_distribution():
    """Get enhanced topic distribution statistics"""
    try:
        # Get topic distribution with time-based analysis
        distribution = await storage_service.get_topic_distribution()
        
        # Enhanced statistics for each topic
        topic_stats = []
        for topic, stats in distribution.items():
            topic_stats.append({
                'name': topic,
                'count': stats['count'],
                'percentage': round(stats['percentage'], 1),
                'trend': stats.get('trend', 'stable'),  # Topic trend (increasing/decreasing/stable)
                'recent_count': stats.get('recent_count', 0),  # Articles in last 30 days
                'growth_rate': stats.get('growth_rate', 0),  # Growth rate compared to previous period
            })
        
        return sorted(topic_stats, key=lambda x: x['count'], reverse=True)
    except Exception as e:
        logging.error(f"Error getting topic distribution: {e}")
        return []

def calculate_reading_time(text):
    """Calculate estimated reading time in minutes"""
    if not text:
        return 1
    words = len(text.split())
    reading_time = max(1, round(words / 200))  # Assume 200 words per minute reading speed
    return reading_time

def is_recent_article(pub_date):
    """Check if an article is recent (published in 2024 or 2025)"""
    if not pub_date:
        return False
    return any(year in pub_date for year in ['2024', '2025'])

# Global variable to track update status
update_status = {
    "in_progress": False,
    "last_update": None,
    "status": "idle",
    "progress": 0,
    "message": "",
    "error": None,
    "sources_processed": 0,
    "total_sources": 0,
    "articles_found": 0,
    "errors": [],
    "started_at": None,
    "estimated_completion_time": None,
    "can_be_cancelled": True
}

# Lock for thread-safe update status operations
update_status_lock = Lock()

def update_status_safely(updates):
    """Thread-safe update of the status dictionary with improved concurrency handling"""
    global update_status
    with update_status_lock:
        # Make a copy of the current status to avoid partial updates
        current_status = dict(update_status)
        
        # Calculate estimated completion time if progress changed
        if 'progress' in updates and updates['progress'] > 0:
            if not current_status.get('started_at'):
                updates['started_at'] = time.time()
            else:
                elapsed_time = time.time() - current_status['started_at']
                if updates['progress'] > 0 and updates['progress'] > current_status.get('progress', 0):
                    # Only update estimated time if progress is increasing
                    total_estimated_time = (elapsed_time / updates['progress']) * 100
                    updates['estimated_completion_time'] = current_status['started_at'] + total_estimated_time
                elif 'estimated_completion_time' in current_status:
                    # Keep the existing estimate if progress hasn't changed
                    updates['estimated_completion_time'] = current_status['estimated_completion_time']
        
        # Track history of status changes
        if 'status' in updates and updates['status'] != current_status.get('status'):
            history = current_status.get('status_history', [])
            history.append({
                'timestamp': time.time(),
                'old_status': current_status.get('status'),
                'new_status': updates['status']
            })
            updates['status_history'] = history[:10]  # Keep last 10 status changes
        
        # Set in_progress flag consistently based on status
        if 'status' in updates:
            if updates['status'] in ['completed', 'failed', 'cancelled']:
                updates['in_progress'] = False
            elif updates['status'] in ['starting', 'running']:
                updates['in_progress'] = True
        
        # Update the status atomically
        update_status.update(updates)
        
        # Return a new copy to avoid external modifications
        return dict(update_status)

def get_update_status_safely():
    """Thread-safe access to the status dictionary with minimal locking time"""
    with update_status_lock:
        # Return a deep copy to prevent external modification
        return dict(update_status)  # Return a copy

def async_update(app_context):
    """Enhanced background task for updating articles"""
    global update_status, update_thread
    
    try:
        with app_context:
            update_status_safely({
                "status": "running",
                "progress": 0,
                "message": "Starting update process...",
                "started_at": time.time(),
                "can_be_cancelled": True
            })
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the update process with a fresh coroutine
                success = loop.run_until_complete(scrape_news(status_callback=update_callback))
                
                if success:
                    # Verify database consistency with a fresh coroutine
                    update_status_safely({
                        "progress": 85,
                        "message": "Verifying database consistency..."
                    })
                    
                    # Create a new coroutine for verification
                    verify_success, stats = loop.run_until_complete(verify_database_consistency())
                    
                    if verify_success:
                        if stats["repaired"] > 0:
                            final_status = {
                                "status": "completed_with_repair",
                                "message": f"Articles updated. Repaired {stats['repaired']} inconsistencies."
                            }
                        else:
                            final_status = {
                                "status": "completed",
                                "message": "Articles updated and database is consistent."
                            }
                    else:
                        final_status = {
                            "status": "completed_with_warnings",
                            "message": "Articles updated but verification failed."
                        }
                    
                    # Invalidate caches
                    invalidate_caches()
                    
                    # Record success metric
                    monitoring_service.record_count("article_updates_completed", labels={"success": "true"})
                else:
                    final_status = {
                        "status": "completed_with_errors",
                        "message": "Some errors occurred while updating articles."
                    }
                    monitoring_service.record_count("article_updates_completed", labels={"success": "false"})
                
                final_status.update({
                    "progress": 100,
                    "last_update": time.time(),
                    "in_progress": False,
                    "can_be_cancelled": False
                })
                update_status_safely(final_status)
                
            except asyncio.CancelledError:
                update_status_safely({
                    "status": "cancelled",
                    "message": "Update was cancelled",
                    "in_progress": False,
                    "progress": 0,
                    "can_be_cancelled": False
                })
                monitoring_service.record_count("article_updates_completed", labels={"success": "false", "reason": "cancelled"})
            finally:
                loop.close()
                
    except Exception as e:
        logging.error(f"Error in background update: {e}", exc_info=True)
        update_status_safely({
            "status": "failed",
            "message": "Update failed",
            "error": str(e),
            "progress": 0,
            "in_progress": False,
            "can_be_cancelled": False
        })
        monitoring_service.record_count("article_updates_completed", labels={"success": "false", "error": "exception"})
    finally:
        # Clear thread reference
        with update_thread_lock:
            global update_thread
            update_thread = None

def invalidate_caches():
    """Invalidate all caches after articles are updated"""
    try:
        # Clear article cache
        article_cache.clear()
        logging.info("Article cache cleared")
        
        # Clear search cache
        search_cache.clear()
        logging.info("Search cache cleared")
        
        # Clear suggestions cache
        suggestion_cache.clear()
        logging.info("Suggestion cache cleared")
        
        logging.info("All caches successfully invalidated")
    except Exception as e:
        logging.error(f"Error invalidating caches: {e}")
        # Don't raise the exception - we don't want cache invalidation
        # failures to affect the main update flow

def update_callback(progress, message, sources_processed=None, total_sources=None, articles_found=None):
    """Callback function to update status during scraping"""
    global update_status
    
    update_status["progress"] = progress
    update_status["message"] = message
    
    if sources_processed is not None:
        update_status["sources_processed"] = sources_processed
    
    if total_sources is not None:
        update_status["total_sources"] = total_sources
    
    if articles_found is not None:
        update_status["articles_found"] = articles_found

@app.route('/api/update/start', methods=['POST'])
def start_update():
    """Enhanced API endpoint to start a background update"""
    global update_thread, update_status
    
    try:
        # Thread-safe check if update is already running
        if is_update_running():
            return jsonify({
                "success": False,
                "message": "Update already in progress",
                "status": get_update_status_safely()
            })
        
        # Reset status with thread safety
        update_status_safely({
            "in_progress": True,
            "last_update": None,
            "status": "starting",
            "progress": 0,
            "message": "Starting update process...",
            "error": None,
            "sources_processed": 0,
            "total_sources": 0,
            "articles_found": 0,
            "errors": [],
            "started_at": time.time(),
            "estimated_completion_time": None,
            "can_be_cancelled": True
        })
        
        # Start background update with thread safety
        with update_thread_lock:
            update_thread = threading.Thread(
                target=async_update,
                args=(app.app_context(),),
                daemon=True
            )
            update_thread.start()
        
        # Record metric
        monitoring_service.record_count("article_updates_started")
        
        return jsonify({
            "success": True,
            "message": "Update started in background",
            "status": get_update_status_safely()
        })
        
    except Exception as e:
        logging.error(f"Error starting update: {e}", exc_info=True)
        update_status_safely({
            "in_progress": False,
            "status": "failed",
            "error": str(e)
        })
        return jsonify({
            "success": False,
            "message": f"Error starting update: {str(e)}",
            "error": str(e)
        }), 500

@app.route('/api/update/cancel', methods=['POST'])
def cancel_update():
    """New endpoint to cancel an ongoing update"""
    try:
        if not is_update_running():
            return jsonify({
                "success": False,
                "message": "No update in progress",
                "status": get_update_status_safely()
            })
        
        if not update_status.get("can_be_cancelled", True):
            return jsonify({
                "success": False,
                "message": "Update cannot be cancelled at this stage",
                "status": get_update_status_safely()
            })
        
        if cancel_update():
            return jsonify({
                "success": True,
                "message": "Update cancellation initiated",
                "status": get_update_status_safely()
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to cancel update",
                "status": get_update_status_safely()
            })
            
    except Exception as e:
        logging.error(f"Error cancelling update: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error cancelling update: {str(e)}",
            "error": str(e)
        }), 500

@app.route('/api/update/status', methods=['GET'])
def get_update_status():
    """Enhanced API endpoint to get the current update status"""
    try:
        status = get_update_status_safely()
        status['is_running'] = is_update_running()
        return jsonify(status)
    except Exception as e:
        logging.error(f"Error getting update status: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Legacy route that redirects to the unified endpoint
@app.route('/update', methods=['POST'])
def update_articles():
    """Legacy update route - redirects to the unified API endpoint"""
    try:
        # Just redirect to the unified API endpoint
        result = start_update()
        
        # Since this is called from the UI, redirect back to index with a flash message
        if isinstance(result, tuple):
            flash(f"Error: {result[0]['message']}", "error")
        else:
            flash("Update started in background. You can continue using the app.", "info")
        
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error in legacy update route: {e}")
        flash("Error starting update. Please try again later.", "error")
        return redirect(url_for('index'))

# Legacy API endpoint that redirects to the unified endpoint
@app.route('/api/update', methods=['POST'])
def api_update_articles():
    """Legacy API endpoint - redirects to the unified API endpoint"""
    # Simply call start_update() directly, without the async decorator
    return start_update()


@app.route('/summarize', methods=['POST'])
@async_route
@rate_limiting_service.rate_limit_decorator(key="summarize")
async def summarize_article():
    """Generate a summary for an article."""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            abort(400, description="No data provided")
            
        description = data.get('description', '').strip()
        topic = data.get('topic', '').strip()
        article_link = data.get('link', '').strip()
        
        if not description:
            abort(400, description="No description provided")
            
        if not article_link:
            abort(400, description="No article link provided")
            
        # Check cache first
        cache_key = f"summary:{article_link}"
        cached_summary = cache_service.get(cache_key)
        if cached_summary:
            monitoring_service.record_count("summary_cache_hits")
            return jsonify({"summary": cached_summary})
            
        try:
            # Use AIService for summarization
            summary = await ai_service.summarize_with_context(description, topic)
            
            if not summary or summary.strip() == '':
                abort(500, description="Could not generate summary")
                
            # Cache the result
            cache_service.set(cache_key, summary, ttl=3600)  # Cache for 1 hour
            
            # Save summary to storage
            try:
                await storage_service.update_article_summary(article_link, summary)
                monitoring_service.record_count("summaries_generated", labels={"success": "true"})
            except Exception as e:
                logging.error(f"Error saving summary to storage: {e}")
                # Don't fail the request if storage update fails
            
            return jsonify({"summary": summary})
            
        except AIServiceException as e:
            logging.error(f"AI service error: {e}")
            abort(503, description="AI service temporarily unavailable")
        except RateLimitException as e:
            logging.warning(f"Rate limit exceeded: {e}")
            abort(429, description=str(e))
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            abort(500, description="Error generating summary")
            
    except Exception as e:
        logging.error(f"Unhandled error in summarize_article: {e}", exc_info=True)
        abort(500, description="An unexpected error occurred")

@app.route('/generate-summaries', methods=['POST'], endpoint='generate_summaries_endpoint')
@monitoring_service.time_async_function("generate_summaries")
@rate_limiting_service.rate_limit_decorator(key="summarize")
@async_route
async def generate_summaries():
    """Generate summaries for articles that don't have them."""
    try:
        # Check if there's already a generation in progress
        if getattr(generate_summaries, 'in_progress', False):
            abort(429, description="Summary generation already in progress")
            
        generate_summaries.in_progress = True
        
        try:
            # Start the generation process
            success = await generate_missing_summaries()
            
            if success:
                monitoring_service.record_count("batch_summaries_generated", labels={"success": "true"})
                flash("Summaries generated successfully!", "success")
            else:
                monitoring_service.record_count("batch_summaries_generated", labels={"success": "false"})
                flash("Some errors occurred while generating summaries.", "warning")
                
            return redirect(url_for('index'))
            
        except RateLimitException as e:
            logging.warning(f"Rate limit exceeded during batch summary generation: {e}")
            abort(429, description=str(e))
        except AIServiceException as e:
            logging.error(f"AI service error during batch summary generation: {e}")
            abort(503, description="AI service temporarily unavailable")
        except Exception as e:
            logging.error(f"Error in batch summary generation: {e}", exc_info=True)
            abort(500, description="Failed to generate summaries")
            
    except Exception as e:
        logging.error(f"Unhandled error in generate_summaries: {e}", exc_info=True)
        abort(500, description="An unexpected error occurred")
    finally:
        generate_summaries.in_progress = False

# Add debug route to test API
@app.route('/test_summary', methods=['GET'])
def test_summary():
    return render_template('test_summary.html')

@app.route('/api/rag', methods=['POST'], endpoint='rag_query_endpoint')
@async_route
@monitoring_service.time_async_function("rag_query")
@rate_limiting_service.rate_limit_decorator(key="rag")
async def rag_query():
    """Handle RAG-based queries."""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            abort(400, description="No data provided")
            
        query = data.get('query', '').strip()
        use_history = data.get('use_history', True)
        show_sources = data.get('show_sources', True)
        
        if not query:
            abort(400, description="No query provided")
            
        # Check cache first
        cache_key = f"rag:{query}:{use_history}"
        cached_response = cache_service.get(cache_key)
        if cached_response:
            monitoring_service.record_count("rag_cache_hits")
            return jsonify(cached_response)
            
        try:
            # Set a longer timeout for complex queries
            timeout = 60  # 60 seconds timeout
            
            # Get response from AI service with timeout
            response = await asyncio.wait_for(
                ai_service.generate_rag_response(query, use_history),
                timeout=timeout
            )
            
            result = {
                "query": query,
                "response": response.text,
                "sources": response.sources if show_sources else [],
                "timestamp": response.timestamp,
                "confidence": response.confidence
            }
            
            # Cache the result
            cache_service.set(cache_key, result, ttl=1800)  # Cache for 30 minutes
            monitoring_service.record_count("rag_queries", labels={"success": "true"})
            
            return jsonify(result)
            
        except asyncio.TimeoutError:
            logging.error(f"Request timed out after {timeout} seconds")
            # Try to get a partial response if available
            try:
                partial_response = await ai_service.get_partial_response()
                if partial_response:
                    return jsonify({
                        "query": query,
                        "response": partial_response,
                        "sources": [],
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.5,
                        "is_partial": True
                    })
            except Exception as pe:
                logging.warning(f"Failed to get partial response: {pe}")
            
            abort(504, description="Request timed out. Please try a more specific query or break it into smaller parts.")
            
        except GeminiAPIException as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg:
                logging.warning(f"Rate limit exceeded: {e}")
                abort(429, description=str(e))
            elif "safety" in error_msg:
                logging.warning(f"Safety filter triggered: {e}")
                abort(400, description=str(e))
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                logging.error(f"Authentication error: {e}")
                abort(503, description="AI service configuration error")
            else:
                logging.error(f"AI service error: {e}")
                abort(503, description="AI service temporarily unavailable")
        except AIServiceException as e:
            logging.error(f"AI service error: {e}")
            abort(503, description="AI service temporarily unavailable")
        except RateLimitException as e:
            logging.warning(f"Rate limit exceeded: {e}")
            abort(429, description=str(e))
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            abort(500, description="Error generating response")
            
    except Exception as e:
        logging.error(f"Unhandled error in RAG query: {e}", exc_info=True)
        abort(500, description="An unexpected error occurred")

@app.route('/api/rag/stream', methods=['POST'], endpoint='stream_rag_query_endpoint')
@async_route
@monitoring_service.time_async_function("rag_stream_query")
@rate_limiting_service.rate_limit_decorator(key="rag")
async def stream_rag_query():
    """Handle RAG-based queries with streaming response."""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400
            
        query = data.get('query', '').strip()
        use_history = data.get('use_history', True)
        insight_mode = data.get('insight_mode', False)
        time_aware = data.get('time_aware', True)
        analysis_type = data.get('analysis_type')
        
        if not query:
            return jsonify({"error": "No query provided", "status": "error"}), 400
            
        # Check AI service health before proceeding
        if not await ai_service.is_healthy():
            return jsonify({"error": "AI service is currently unavailable", "status": "error"}), 503
            
        # Create async generator for streaming
        async def async_generate():
            generator = None
            try:
                # Create partial JSON content with metadata first
                partial_metadata = {
                    "type": "metadata",
                    "status": "processing",
                    "confidence": 0.0,
                    "startTime": datetime.now().isoformat()
                }
                yield json.dumps(partial_metadata) + "\n"
                
                # Stream the response content
                generator = ai_service.generate_streaming_response(query, use_history)
                async for chunk in generator:
                    # Send the content chunk
                    yield json.dumps({
                        "type": "content",
                        "content": chunk,
                        "done": False
                    }) + "\n"
                
                # After streaming, fetch sources data for the response
                try:
                    history = await storage_service.get_rag_history(limit=1)
                    sources = []
                    confidence = 0.0
                    
                    if history and len(history) > 0:
                        latest_item = history[0]
                        sources = latest_item.get('sources', [])
                        confidence = latest_item.get('confidence', 0.0)
                except (AttributeError, Exception) as e:
                    # Fallback if method doesn't exist or fails
                    logging.warning(f"Error getting rag history: {e}, using fallback")
                    sources = getattr(ai_service, 'relevant_articles', [])
                    confidence = 0.7  # Default confidence value
                
                # Send sources data
                sources_data = {
                    "type": "sources",
                    "sources": sources
                }
                yield json.dumps(sources_data) + "\n"
                        
                # Send final metadata
                final_metadata = {
                    "type": "metadata",
                    "status": "complete",
                    "confidence": confidence,
                    "endTime": datetime.now().isoformat()
                }
                yield json.dumps(final_metadata) + "\n"
                
                # Send done signal
                yield json.dumps({
                    "type": "content",
                    "content": "",
                    "done": True
                }) + "\n"
                
            except Exception as e:
                # Log the error
                logging.error(f"Error in streaming response: {str(e)}", exc_info=True)
                
                # Send error message to client
                error_message = {
                    "type": "error",
                    "error": str(e),
                    "status": "error"
                }
                yield json.dumps(error_message) + "\n"
                
                # Try to get partial response on error
                try:
                    partial_response = await ai_service.get_partial_response()
                    if partial_response:
                        # Send partial content
                        yield json.dumps({
                            "type": "content",
                            "content": partial_response,
                            "is_partial": True,
                            "done": True
                        }) + "\n"
                except Exception as partial_error:
                    logging.error(f"Error getting partial response: {str(partial_error)}")
            finally:
                # Ensure the generator is properly closed
                if generator and hasattr(generator, 'aclose'):
                    try:
                        await generator.aclose()
                    except Exception as close_error:
                        logging.error(f"Error closing generator: {close_error}")
        
        # Convert the async generator to a sync generator for Flask
        def sync_generator():
            loop = asyncio.new_event_loop()
            async_gen = async_generate()
            
            async def consume():
                try:
                    async for chunk in async_gen:
                        yield chunk
                finally:
                    # Make sure to close the generator even on early exit
                    if hasattr(async_gen, 'aclose'):
                        try:
                            await async_gen.aclose()
                        except Exception as e:
                            logging.error(f"Error closing generator in consume: {e}")
            
            consumer = consume()
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(consumer.__anext__().__await__())
                        yield chunk
                    except StopAsyncIteration:
                        # Finish the consumer to trigger the finally block
                        loop.run_until_complete(asyncio.sleep(0))
                        break
                    except Exception as e:
                        logging.error(f"Error in sync generator: {e}")
                        yield f"data: {json.dumps({'error': 'Stream interrupted', 'done': True})}\n\n"
                        break
            finally:
                # Run any pending tasks to ensure cleanup
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
        
        # Return the streaming response
        response = Response(
            sync_generator(), 
            mimetype='application/json',
            status=200,
            headers={
                'X-Accel-Buffering': 'no',  # Disable proxy buffering
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        return response
        
    except RateLimitException as e:
        # Handle rate limiting
        return jsonify({"error": str(e), "status": "error"}), 429
    except AIServiceException as e:
        # Handle AI service errors
        return jsonify({"error": str(e), "status": "error"}), 503
    except Exception as e:
        # Log and handle any other errors
        logging.error(f"Error in stream_rag_query: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/api/rag/stream', methods=['GET'], endpoint='stream_rag_query_get_endpoint')
@async_route
@monitoring_service.time_async_function("rag_stream_query_get")
@rate_limiting_service.rate_limit_decorator(key="rag")
async def stream_rag_query_get():
    """Handle streaming RAG-based queries via GET for EventSource compatibility."""
    try:
        # Validate input
        query = request.args.get('query', '').strip()
        use_history = request.args.get('use_history', 'true').lower() == 'true'
        
        if not query:
            abort(400, description="No query provided")
        
        # Record metrics
        monitoring_service.record_count("rag_stream_queries", labels={"method": "get"})
        
        # Create a streaming response
        async def generate():
            sources = []
            response_text = ""
            generator = None
            
            try:
                generator = ai_service.generate_streaming_response(
                    query=query,
                    use_history=use_history
                )
                
                async for chunk in generator:
                    # Check if this is the sources marker
                    if chunk.startswith("__SOURCES__"):
                        try:
                            sources = json.loads(chunk[11:])  # Extract JSON after marker
                        except json.JSONDecodeError:
                            logging.warning("Failed to parse sources JSON")
                        continue
                        
                    response_text += chunk
                    # Corrected f-string: Use single quotes outside, double quotes inside json.dumps
                    yield f'data: {json.dumps({"chunk": chunk, "done": False})}\n\n'
                    
                # Send final message with sources
                response_data = {
                    'chunk': '',
                    'done': True,
                    'sources': sources,
                    'full_response': response_text
                }
                # Corrected f-string
                yield f'data: {json.dumps(response_data)}\n\n'
                
                # Record success metric
                monitoring_service.record_count(
                    "rag_stream_queries",
                    labels={"success": "true", "method": "get"}
                )
                
            except AIServiceException as e:
                logging.error(f"AI service error during streaming: {e}")
                # Try to get partial response
                try:
                    partial_response = await ai_service.get_partial_response()
                    if partial_response:
                        # Corrected nested f-string issue
                        chunk_content = f"\n\nAI service temporarily unavailable. Partial response: {partial_response}"
                        yield f'data: {json.dumps({"chunk": chunk_content, "done": True})}\n\n'
                    else:
                        # Corrected f-string
                        yield f'data: {json.dumps({"error": "AI service temporarily unavailable", "done": True})}\n\n'
                except Exception as partial_error:
                    logging.error(f"Error getting partial response: {str(partial_error)}")
                    # Corrected f-string
                    yield f'data: {json.dumps({"error": "AI service temporarily unavailable", "done": True})}\n\n'
            except RateLimitException as e:
                logging.warning(f"Rate limit exceeded during streaming: {e}")
                # Corrected f-string
                yield f'data: {json.dumps({"error": str(e), "done": True})}\n\n'
            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                # Try to get partial response
                try:
                    partial_response = await ai_service.get_partial_response()
                    if partial_response:
                        # Corrected nested f-string issue
                        chunk_content = f"\n\nError occurred. Partial response: {partial_response}"
                        yield f'data: {json.dumps({"chunk": chunk_content, "done": True})}\n\n'
                    else:
                        # Corrected f-string
                        yield f'data: {json.dumps({"error": "Error generating response", "done": True})}\n\n'
                except Exception as partial_error:
                    logging.error(f"Error getting partial response: {str(partial_error)}")
                    # Corrected f-string
                    yield f'data: {json.dumps({"error": "Error generating response", "done": True})}\n\n'
            finally:
                # Ensure the generator is properly closed
                if generator and hasattr(generator, 'aclose'):
                    try:
                        await generator.aclose()
                    except Exception as close_error:
                        logging.error(f"Error closing generator in stream_rag_query_get: {close_error}")
        
        # Use a synchronous generator that runs the async one
        def sync_generator():
            loop = asyncio.new_event_loop()
            async_gen = generate()
            
            async def consume():
                try:
                    async for chunk in async_gen:
                        yield chunk
                finally:
                    # Make sure to close the generator even on early exit
                    if hasattr(async_gen, 'aclose'):
                        try:
                            await async_gen.aclose()
                        except Exception as e:
                            logging.error(f"Error closing generator in consume: {e}")
            
            consumer = consume()
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(consumer.__anext__().__await__())
                        yield chunk
                    except StopAsyncIteration:
                        # Finish the consumer to trigger the finally block
                        loop.run_until_complete(asyncio.sleep(0))
                        break
                    except Exception as e:
                        logging.error(f"Error in sync generator: {e}")
                        yield f"data: {json.dumps({'error': 'Stream interrupted', 'done': True})}\n\n"
                        break
            finally:
                # Run any pending tasks to ensure cleanup
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
        
        return Response(sync_generator(), mimetype='text/event-stream')
        
    except Exception as e:
        logging.error(f"Unhandled error in streaming RAG query: {e}", exc_info=True)
        monitoring_service.record_count("rag_stream_queries", labels={"success": "false"})
        abort(500, description="An unexpected error occurred")

@app.route('/api/rag/history', methods=['GET'])
def get_rag_history():
    """Get conversation history"""
    try:
        return jsonify({"history": ai_service.history})
    except Exception as e:
        logging.error(f"Error getting history: {e}")
        return jsonify({"error": str(e), "history": []}), 500

@app.route('/api/rag/history', methods=['DELETE'])
@async_route
async def clear_rag_history():
    """Clear conversation history"""
    try:
        await ai_service.clear_history()
        return jsonify({"message": "History cleared"})
    except Exception as e:
        logging.error(f"Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/backup', methods=['POST'])
@async_route
async def backup_data():
    """Backup current data to Google Cloud Storage"""
    try:
        # Backup articles file and cache file
        articles_url = await storage_service.backup_to_gcs(config.articles_file_path)
        logging.info(f"Articles backup created: {articles_url}")
        
        # Backup cache file if it exists
        cache_url = None
        if os.path.exists(config.cache_file_path):
            cache_url = await storage_service.backup_to_gcs(config.cache_file_path)
            logging.info(f"Cache backup created: {cache_url}")
        
        return jsonify({"success": True, "articles_url": articles_url, "cache_url": cache_url})
    except Exception as e:
        logging.error(f"Error in backup: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/backups', methods=['GET'])
@async_route
async def list_backups():
    """List all backups in Google Cloud Storage"""
    try:
        backups = await storage_service.list_gcs_backups()
        return jsonify({"backups": backups})
    except Exception as e:
        logging.error(f"Error listing backups: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/restore', methods=['POST'])
@async_route
async def restore_backup():
    """Restore a backup from Google Cloud Storage"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({"error": "No filename provided"}), 400
            
        # Determine target path based on backup type
        if "news_alerts" in filename or "articles" in filename:
            target_path = config.articles_file_path
        elif "cache" in filename:
            target_path = config.cache_file_path
        else:
            return jsonify({"error": "Unrecognized backup file type"}), 400
            
        # Restore the backup
        success = await storage_service.restore_from_gcs(filename, target_path)
        
        if success:
            return jsonify({"success": True, "message": f"Restored {filename} to {target_path}"})
        else:
            return jsonify({"error": "Failed to restore backup"}), 500
            
    except Exception as e:
        logging.error(f"Error in restore: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similar-articles/<article_id>', methods=['GET'])
@async_route
async def get_similar_articles(article_id):
    """Get similar articles to the given article ID"""
    try:
        similar_articles = await storage_service.get_similar_articles(article_id)
        return jsonify({"articles": similar_articles})
    except Exception as e:
        logging.error(f"Error getting similar articles: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similar-articles/', methods=['GET'])
@app.route('/similar-articles', methods=['GET'])
@async_route
async def get_recent_articles():
    """Get recent articles when no specific article ID is provided"""
    try:
        # Get most recent articles instead
        recent_articles = await storage_service.get_recent_articles(limit=10)
        return jsonify({
            "message": "No article ID provided. Showing recent articles instead.",
            "articles": recent_articles
        })
    except Exception as e:
        logging.error(f"Error getting recent articles: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag')
def rag_interface():
    return render_template('rag_interface.html')

# Cache configurations
article_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL
search_cache = TTLCache(maxsize=50, ttl=60)     # 1 minute TTL
suggestion_cache = LRUCache(maxsize=100)         # LRU cache for suggestions

@app.route('/api/articles', methods=['GET'], endpoint='get_articles_endpoint')
async def get_articles():
    """Get articles with pagination, filtering, and enhanced search"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        topic = request.args.get('topic', 'All')
        search_query = request.args.get('search', '')
        sort_by = request.args.get('sort_by', 'pub_date')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Enhanced search parameters
        search_type = request.args.get('searchType', 'auto')
        threshold = float(request.args.get('threshold', 0.6))
        fields = request.args.get('fields', '').split(',') if request.args.get('fields') else None
        
        # Initialize storage service
        storage = StorageService()
        
        if search_query:
            # Use enhanced search if query exists
            results = await storage.enhanced_search(
                search_query,
                search_type=search_type,
                threshold=threshold,
                fields=fields
            )
            
            # Apply pagination to search results
            total_results = len(results)
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_results = results[start_idx:end_idx]
            
            return jsonify({
                'articles': paginated_results,
                'total': total_results,
                'page': page,
                'total_pages': (total_results + limit - 1) // limit,
                'query_time': 0  # Placeholder for future performance tracking
            })
        else:
            # Use regular article retrieval for non-search requests
            results = await storage.get_articles(
                page=page,
                limit=limit,
                topic=topic if topic != 'All' else None,
                sort_by=sort_by,
                sort_order=sort_order
            )
            return jsonify(results)
            
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error in get_articles: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """API endpoint to get topic distribution"""
    try:
        # Use the storage service to get comprehensive topic distribution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        topic_stats = loop.run_until_complete(storage_service.get_topic_distribution())
        loop.close()
        
        if not topic_stats:
            # Fallback to CSV approach if storage service failed
            from services.constants import TOPICS
            
            filename = config.articles_file_path
            
            # If file doesn't exist, return all topics with zero counts
            if not os.path.isfile(filename):
                all_topics = []
                for topic in TOPICS.keys():
                    all_topics.append({
                        "topic": topic,
                        "count": 0,
                        "percentage": 0.0
                    })
                return jsonify({"topics": all_topics})
                
            df = pd.read_csv(filename)
            
            # Calculate topic distribution
            topic_counts = df['topic'].value_counts().reset_index()
            topic_counts.columns = ['topic', 'count']
            topic_counts['percentage'] = (topic_counts['count'] / len(df) * 100).round(1)
            
            # Sort by count (descending)
            topic_counts = topic_counts.sort_values('count', ascending=False)
            
            return jsonify({
                "topics": topic_counts.to_dict('records')
            })
        
        # Format the result from storage service
        formatted_topics = []
        for topic, stats in topic_stats.items():
            formatted_topics.append({
                "topic": topic,
                "count": stats["count"],
                "percentage": stats["percentage"]
            })
            
        # Sort by count (descending)
        formatted_topics = sorted(formatted_topics, key=lambda x: x["count"], reverse=True)
        
        return jsonify({
            "topics": formatted_topics
        })
        
    except Exception as e:
        logging.error(f"Error in get_topics API: {e}")
        return jsonify({
            "error": str(e),
            "topics": []
        }), 500

@app.route('/api/storage/sync', methods=['POST'], endpoint='sync_csv_to_database_endpoint')
@async_route
async def sync_csv_to_database():
    """API endpoint for legacy CSV to Firestore sync (deprecated)
    
    This endpoint is maintained for backward compatibility but is no longer needed
    since articles are now saved directly to Firestore.
    """
    try:
        # Get parameters
        data = request.get_json() or {}
        force = data.get('force', False)
        
        # Return information about the new direct storage approach
        return jsonify({
            "success": True,
            "message": "CSV sync is no longer needed. Articles are now saved directly to Firestore.",
            "info": "This endpoint is maintained for backward compatibility only."
        })
        
    except Exception as e:
        logging.error(f"Error in sync endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/api/storage/export', methods=['POST'], endpoint='export_database_to_csv_endpoint')
@async_route
async def export_database_to_csv():
    """API endpoint to export articles from Firestore to CSV
    
    This allows for creating backups or archives of the articles.
    """
    try:
        # Get parameters
        data = request.get_json() or {}
        limit = data.get('limit', 0)  # 0 means all articles
        file_path = data.get('file_path', config.articles_file_path)
        
        # Export articles to CSV
        stats = await storage_service.export_to_csv(file_path, limit)
        
        if stats["success"]:
            message = f"Successfully exported {stats['total_exported']} articles to {file_path}"
        else:
            message = f"Failed to export articles to CSV"
            
        return jsonify({
            "success": stats["success"],
            "message": message,
            "stats": stats
        })
        
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        return jsonify({
            "success": False,
            "message": f"Error exporting to CSV: {str(e)}"
        }), 500

@app.route('/api/storage/cleanup', methods=['POST'])
@async_route
async def cleanup_duplicate_articles():
    """API endpoint to clean up duplicate articles in Firestore
    
    This is a one-time operation to fix the duplicate entries issue.
    """
    try:
        stats = await storage_service.cleanup_duplicate_articles()
        
        return jsonify({
            'success': True,
            'message': f"Cleanup successful. Removed {stats['duplicates_removed']} duplicate articles out of {stats['total_articles']} total articles.",
            'stats': stats
        })
    except Exception as e:
        logging.error(f"Error cleaning up duplicates: {e}")
        return jsonify({
            'success': False,
            'message': f"Error cleaning up duplicates: {str(e)}"
        }), 500

@app.route('/api/search/suggestions', methods=['GET'])
@async_route
async def get_search_suggestions():
    try:
        query = request.args.get('q', '').strip()
        
        if not query or len(query) < 2:
            return jsonify({"suggestions": []})
            
        # Try to get from cache
        cache_key_str = f"suggestions:{query.lower()}"
        cached_suggestions = suggestion_cache.get(cache_key_str)
        if cached_suggestions:
            return jsonify({"suggestions": cached_suggestions})
            
        # Get suggestions from storage
        suggestions = set()
        result = await storage_service.get_articles(limit=50)
        articles = result.get('articles', []) if isinstance(result, dict) else []
        
        for article in articles:
            if not isinstance(article, dict):
                continue
                
            metadata = article.get('metadata', {})
            if not isinstance(metadata, dict):
                continue
            
            # Check title
            title = metadata.get('title', '')
            if title and isinstance(title, str):
                title_lower = title.lower()
                if query.lower() in title_lower:
                    suggestions.add(title)
                
            # Check topic
            topic = metadata.get('topic', '')
            if topic and isinstance(topic, str):
                topic_lower = topic.lower()
                if query.lower() in topic_lower:
                    suggestions.add(topic)
                
            # Check for company names and drug names in title
            if title and isinstance(title, str):
                capitalized_terms = re.findall(r'\b[A-Z][a-zA-Z]*\b', title)
                for term in capitalized_terms:
                    if query.lower() in term.lower():
                        suggestions.add(term)
        
        # Sort and limit suggestions
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: (
                not x.lower().startswith(query.lower()),
                not query.lower() in x.lower(),
                x.lower()
            )
        )[:10]
        
        # Cache the result
        suggestion_cache[cache_key_str] = sorted_suggestions
        
        return jsonify({"suggestions": sorted_suggestions})
        
    except Exception as e:
        logging.error(f"Error getting search suggestions: {e}")
        return jsonify({
            "error": str(e),
            "suggestions": []
        }), 500

@app.route('/api/topic-analysis', methods=['POST'])
@async_route
async def topic_analysis():
    """API endpoint for in-depth pharmaceutical topic analysis with time awareness"""
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({"error": "No topic provided"}), 400
            
        # Get topic analysis from AI service
        response = await ai_service.generate_topic_analysis(topic)
        
        return jsonify({
            "topic": topic,
            "analysis": response.text,
            "sources": response.sources,
            "confidence": response.confidence,
            "timestamp": response.timestamp
        })
        
    except Exception as e:
        logging.error(f"Error in topic analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/regulatory', methods=['POST'])
@async_route
async def analyze_regulatory_impact():
    """Analyze regulatory impact of pharmaceutical content"""
    try:
        data = request.get_json()
        content = data.get('content')
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
            
        result = await ai_service.analyze_regulatory_impact(content)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in regulatory analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/market', methods=['POST'])
@async_route
async def generate_market_insight():
    """Generate pharmaceutical market insights"""
    try:
        data = request.get_json()
        topic = data.get('topic')
        timeframe = data.get('timeframe', 'recent')
        
        if not topic:
            return jsonify({"error": "No topic provided"}), 400
            
        result = await ai_service.generate_market_insight(topic, timeframe)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error generating market insight: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/drug', methods=['POST'])
@async_route
async def analyze_drug_development():
    """Analyze drug development progress"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name')
        
        if not drug_name:
            return jsonify({"error": "No drug name provided"}), 400
            
        result = await ai_service.analyze_drug_development(drug_name)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in drug development analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/competitive', methods=['POST'])
@async_route
async def generate_competitive_analysis():
    """Generate competitive analysis for pharmaceutical companies"""
    try:
        data = request.get_json()
        company_name = data.get('company_name')
        
        if not company_name:
            return jsonify({"error": "No company name provided"}), 400
            
        result = await ai_service.generate_competitive_analysis(company_name)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in competitive analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/clinical-trial', methods=['POST'])
@async_route
async def analyze_clinical_trial():
    """Analyze clinical trial data"""
    try:
        data = request.get_json()
        trial_data = data.get('trial_data')
        
        if not trial_data:
            return jsonify({"error": "No trial data provided"}), 400
            
        result = await ai_service.analyze_clinical_trial(trial_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in clinical trial analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/patent', methods=['POST'])
@async_route
async def analyze_patent():
    """Analyze patent information"""
    try:
        data = request.get_json()
        patent_data = data.get('patent_data')
        
        if not patent_data:
            return jsonify({"error": "No patent data provided"}), 400
            
        result = await ai_service.analyze_patent(patent_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in patent analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/manufacturing', methods=['POST'])
@async_route
async def analyze_manufacturing():
    """Analyze manufacturing and supply chain data"""
    try:
        data = request.get_json()
        manufacturing_data = data.get('manufacturing_data')
        
        if not manufacturing_data:
            return jsonify({"error": "No manufacturing data provided"}), 400
            
        result = await ai_service.analyze_manufacturing(manufacturing_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in manufacturing analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/market-access', methods=['POST'])
@async_route
async def analyze_market_access():
    """Analyze market access and pricing data"""
    try:
        data = request.get_json()
        market_data = data.get('market_data')
        
        if not market_data:
            return jsonify({"error": "No market data provided"}), 400
            
        result = await ai_service.analyze_market_access(market_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in market access analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/safety', methods=['POST'], endpoint='analyze_safety_endpoint')
@async_route
async def analyze_safety():
    """Analyze safety and pharmacovigilance data"""
    try:
        data = request.get_json()
        safety_data = data.get('safety_data')
        
        if not safety_data:
            return jsonify({"error": "No safety data provided"}), 400
            
        result = await ai_service.analyze_safety(safety_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in safety analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/pipeline', methods=['POST'], endpoint='analyze_pipeline_endpoint')
@async_route
async def analyze_pipeline():
    """Analyze pharmaceutical pipeline data"""
    try:
        data = request.get_json()
        pipeline_data = data.get('pipeline_data')
        
        if not pipeline_data:
            return jsonify({"error": "No pipeline data provided"}), 400
            
        result = await ai_service.analyze_pipeline(pipeline_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in pipeline analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/therapeutic-area', methods=['POST'], endpoint='analyze_therapeutic_area_endpoint')
@async_route
async def analyze_therapeutic_area():
    """Analyze therapeutic area landscape"""
    try:
        data = request.get_json()
        therapeutic_area = data.get('therapeutic_area')
        
        if not therapeutic_area:
            return jsonify({"error": "No therapeutic area provided"}), 400
            
        result = await ai_service.analyze_therapeutic_area(therapeutic_area)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in therapeutic area analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/regulatory-strategy', methods=['POST'], endpoint='analyze_regulatory_strategy_endpoint')
@async_route
async def analyze_regulatory_strategy():
    """Analyze regulatory strategy"""
    try:
        data = request.get_json()
        regulatory_data = data.get('regulatory_data')
        
        if not regulatory_data:
            return jsonify({"error": "No regulatory data provided"}), 400
            
        result = await ai_service.analyze_regulatory_strategy(regulatory_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in regulatory strategy analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/digital-health', methods=['POST'], endpoint='analyze_digital_health_endpoint')
@async_route
async def analyze_digital_health():
    """Analyze digital health integration"""
    try:
        data = request.get_json()
        digital_data = data.get('digital_data')
        
        if not digital_data:
            return jsonify({"error": "No digital health data provided"}), 400
            
        result = await ai_service.analyze_digital_health(digital_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in digital health analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kumby/value-evidence', methods=['POST'], endpoint='analyze_value_evidence_endpoint')
@async_route
async def analyze_value_evidence():
    """Analyze value and evidence data"""
    try:
        data = request.get_json()
        value_data = data.get('value_data')
        
        if not value_data:
            return jsonify({"error": "No value/evidence data provided"}), 400
            
        result = await ai_service.analyze_value_evidence(value_data)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in value/evidence analysis: {e}")
        return jsonify({"error": str(e)}), 500

# KumbyAI Chat Routes
@app.route('/api/kumbyai/chat', methods=['POST'], endpoint='kumbyai_chat_endpoint')
@async_route
@monitoring_service.time_async_function("kumbyai_chat")
async def kumbyai_chat():
    """KumbyAI chat endpoint"""
    try:
        # Check if AI service is available
        if ai_service is None:
            return jsonify({
                'error': 'AI service is not available',
                'success': False
            }), 503
            
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        
        # Get relevant articles for context
        articles = await storage_service.get_articles(limit=5, sort_by='date', sort_order='desc')
        context = []
        if articles and 'articles' in articles:
            for article in articles['articles']:
                if article.get('metadata'):
                    context.append({
                        'title': article['metadata'].get('title', ''),
                        'summary': article['metadata'].get('summary', ''),
                        'description': article['metadata'].get('description', '')
                    })

        # Process the message with RAG context
        response = await ai_service.process_query(message, context)
        
        return jsonify({
            'response': response,
            'success': True
        })
    except Exception as e:
        logging.error(f"Error in KumbyAI chat: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'success': False
        }), 500

# Admin Routes
@app.route('/api/admin/export-csv', endpoint='admin_export_csv_endpoint')
@async_route
@monitoring_service.time_async_function("admin_export_csv")
async def admin_export_csv():
    """Export articles as CSV"""
    try:
        # Get all articles
        articles = await storage_service.get_articles(limit=0)
        if not articles or 'articles' not in articles:
            return jsonify({'error': 'No articles found'}), 404

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        headers = ['Title', 'Source', 'Date', 'Topic', 'URL', 'Summary']
        writer.writerow(headers)
        
        # Write article data
        for article in articles['articles']:
            metadata = article.get('metadata', {})
            writer.writerow([
                metadata.get('title', ''),
                metadata.get('source', ''),
                metadata.get('pub_date', ''),
                metadata.get('topic', ''),
                metadata.get('link', ''),
                metadata.get('summary', '')
            ])
        
        # Prepare response
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=pharmaceutical-news-{datetime.now().strftime("%Y-%m-%d")}.csv'
            }
        )
    except Exception as e:
        logging.error(f"Error exporting CSV: {str(e)}")
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/admin/force-sync', methods=['POST'], endpoint='admin_force_sync_endpoint')
@async_route
@monitoring_service.time_async_function("admin_force_sync")
async def admin_force_sync():
    """Force sync with all sources"""
    try:
        # Force sync with all sources
        result = await news_service.update_all(force=True)
        
        return jsonify({
            'success': True,
            'message': 'Sync completed successfully',
            'articles_synced': result.get('articles_found', 0)
        })
    except Exception as e:
        logging.error(f"Error in force sync: {str(e)}")
        return jsonify({
            'error': 'Failed to sync data',
            'success': False
        }), 500

@app.route('/api/admin/cleanup-duplicates', methods=['POST'], endpoint='admin_cleanup_duplicates_endpoint')
@async_route
@monitoring_service.time_async_function("admin_cleanup_duplicates")
async def admin_cleanup_duplicates():
    """Clean up duplicate articles"""
    try:
        # Get all articles
        articles = await storage_service.get_articles(limit=0)
        if not articles or 'articles' not in articles:
            return jsonify({'error': 'No articles found'}), 404

        # Track duplicates by URL
        seen_urls = set()
        duplicates = []
        
        for article in articles['articles']:
            metadata = article.get('metadata', {})
            url = metadata.get('link')
            
            if url:
                if url in seen_urls:
                    duplicates.append(article['id'])
                else:
                    seen_urls.add(url)

        # Remove duplicates
        for article_id in duplicates:
            await storage_service.delete_article(article_id)

        return jsonify({
            'success': True,
            'message': f'Successfully removed {len(duplicates)} duplicate articles',
            'removed_count': len(duplicates)
        })
    except Exception as e:
        logging.error(f"Error cleaning up duplicates: {str(e)}")
        return jsonify({
            'error': 'Failed to clean up duplicates',
            'success': False
        }), 500

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    # Get the origin from the request
    origin = request.headers.get('Origin')
    
    # List of allowed origins
    allowed_origins = [
        'https://newsletter-aggregator-kvpb.vercel.app',
        'https://newsletter-aggregator-knap-b125yaqk7-kumbyconsultings-projects.vercel.app',
        'https://newsletter-aggregator-knap.vercel.app'  # Added new Vercel domain
    ]
    
    # If the request origin is in our list of allowed origins, set it in the response
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '3600'  # Cache preflight requests for 1 hour
    
    # Handle OPTIONS request
    if request.method == 'OPTIONS':
        response.status_code = 204
    return response

@app.route('/api/topics/stats', methods=['GET'])
async def get_topic_stats():
    """API endpoint to get enhanced topic statistics with trends and growth rates"""
    try:
        # First try to get from storage service for enhanced statistics
        if hasattr(storage_service, 'get_topic_distribution'):
            try:
                topic_stats = await storage_service.get_topic_distribution()
                
                # Format for the frontend
                formatted_stats = []
                for topic, stats in topic_stats.items():
                    formatted_stats.append({
                        'topic': topic,
                        'count': stats.get('count', 0), # Use .get() for safety
                        'percentage': stats.get('percentage', 0.0), # Use .get() for safety
                        # Add new fields using .get() for safe access
                        'trend': stats.get('trend', 'stable'), 
                        'growth_rate': stats.get('growth_rate', 0),
                        'recent_count': stats.get('recent_count', 0) 
                    })
                
                # Sort by count (descending)
                formatted_stats.sort(key=lambda x: x['count'], reverse=True)
                
                return jsonify({"topics": formatted_stats})
            except Exception as storage_error:
                logging.error(f"Error getting topic distribution from storage service: {storage_error}")
                # Continue to fallback method
        
        # Fallback to the regular topics endpoint implementation
        from services.constants import TOPICS
        
        filename = config.articles_file_path
        
        # If file doesn't exist, return all topics with zero counts and default trends
        if not os.path.isfile(filename):
            all_topics = []
            for topic in TOPICS.keys():
                all_topics.append({
                    'topic': topic,
                    'count': 0,
                    'percentage': 0.0,
                    'trend': 'stable', # Default trend
                    'growth_rate': 0, # Default growth rate
                    'recent_count': 0 # Default recent count
                })
            return jsonify({"topics": all_topics})
            
        try:
            df = pd.read_csv(filename)
            
            # Calculate topic distribution
            topic_counts = df['topic'].value_counts().reset_index()
            topic_counts.columns = ['topic', 'count']
            total_articles = len(df)
            # Ensure division by zero doesn't occur
            topic_counts['percentage'] = (topic_counts['count'] / max(1, total_articles) * 100).round(1)
            
            # Add trend, growth rate, and recent count columns (with default values)
            topic_counts['trend'] = 'stable' 
            topic_counts['growth_rate'] = 0
            topic_counts['recent_count'] = 0
            
            # Make sure all possible topics are included
            all_defined_topics = set(TOPICS.keys())
            existing_topics_in_csv = set(topic_counts['topic'].tolist())
            
            # Add missing topics with zero counts and default trends
            for missing_topic in all_defined_topics - existing_topics_in_csv:
                # Create a DataFrame for the missing topic
                missing_df = pd.DataFrame([{
                    'topic': missing_topic, 
                    'count': 0, 
                    'percentage': 0.0,
                    'trend': 'stable', 
                    'growth_rate': 0,
                    'recent_count': 0 
                }])
                # Concatenate using pandas.concat
                topic_counts = pd.concat([topic_counts, missing_df], ignore_index=True)
            
            # Sort by count (descending)
            topic_counts = topic_counts.sort_values('count', ascending=False)
            
            return jsonify({
                "topics": topic_counts.to_dict('records')
            })
        except Exception as df_error:
            logging.error(f"Error processing CSV file: {df_error}")
            # Last resort fallback - return empty list with status 200 rather than 500 error
            return jsonify({
                "topics": [],
                "error": "Unable to process topic statistics",
                "status": "error"
            })
        
    except Exception as e:
        logging.error(f"Error in get_topic_stats API: {e}")
        # Don't return 500 status - return empty topics with an error message
        return jsonify({
            "error": str(e),
            "topics": [],
            "status": "error"
        })

@app.route('/api/articles/<article_id>/analyze', methods=['POST'], endpoint='analyze_article_endpoint')
@async_route
@rate_limiting_service.rate_limit_decorator(key="article_analysis")
async def analyze_article(article_id):
    """
    Generate a structured analysis of an article using specialized AI models.
    
    Accepts:
    - analysis_type: Type of analysis to perform (clinical_trial, patent, market, etc)
    - output_format: Format of response (json, markdown, table, bullets)
    
    Returns structured analysis in the requested format.
    """
    try:
        # Check if AI service is available
        if ai_service is None:
            raise APIError("AI service is not available", status_code=503)
            
        # Validate request payload
        data = request.get_json() or {}
        analysis_type = data.get('analysis_type', 'general')
        output_format = data.get('output_format', 'default')
        
        # Log analysis request
        logging.info(f"Analysis request for article_id={article_id}, type={analysis_type}, format={output_format}")
        
        # Get article from database
        article = await storage_service.get_article(article_id)
        if not article:
            raise APIError(f"Article with id {article_id} not found", status_code=404)
            
        # Verify AI health
        is_healthy = await ai_service.is_healthy()
        
        if not is_healthy:
            raise APIError("AI service is currently unavailable. Please try again later.", status_code=503)
            
        # Generate analysis
        analysis_result = await ai_service.generate_comprehensive_analysis(
            article_id=article_id,
            analysis_type=analysis_type,
            output_format=output_format  # Pass the output format to the AI service
        )
        
        # Log the raw analysis result from the AI service
        logging.debug(f"Raw analysis result for article {article_id}: {analysis_result}")

        # Add metadata about the analysis
        response_data = {
            "article_id": article_id,
            "title": article.get('metadata', {}).get('title', 'Unknown Title'),
            "analysis_type": analysis_type,
            "output_format": output_format,
            "timestamp": datetime.now().isoformat(),
            "result": analysis_result
        }
        
        # Parse JSON if the output format is JSON
        if output_format == 'json' and isinstance(analysis_result.get('text'), str):
            try:
                # Extract JSON from the text if wrapped in markdown code blocks
                json_text = analysis_result.get('text', '{}')
                if '```json' in json_text and '```' in json_text:
                    json_text = re.search(r'```json\s+(.*?)\s+```', json_text, re.DOTALL)
                    if json_text:
                        json_text = json_text.group(1)
                
                # Parse the JSON
                parsed_json = json.loads(json_text)
                response_data['result']['parsed_json'] = parsed_json
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON from response: {e}")
                response_data['result']['parsed_json'] = None
        
        return jsonify(response_data)
        
    except APIError as e:
        raise e
    except Exception as e:
        logging.exception(f"Error analyzing article: {e}")
        raise APIError(f"Error analyzing article: {str(e)}", status_code=500)

# Global variables for thread management
update_thread = None
update_thread_lock = Lock()

def is_update_running():
    """Thread-safe check if update is running"""
    global update_thread
    with update_thread_lock:
        return update_thread is not None and update_thread.is_alive()

def cancel_update():
    """Thread-safe update cancellation"""
    global update_thread, update_status
    with update_thread_lock:
        if update_thread and update_thread.is_alive():
            update_status_safely({
                "status": "cancelling",
                "message": "Cancelling update...",
                "can_be_cancelled": False
            })
            return True
    return False

@app.route('/api/analysis/save', methods=['POST'], endpoint='save_analysis_endpoint')
@async_route
@rate_limiting_service.rate_limit_decorator(key="analysis_save")
async def save_analysis():
    """Save analysis results for future reference"""
    try:
        # Validate input
        data = request.get_json()
        if not data:
            abort(400, description="No data provided")
            
        # Required fields
        required_fields = ['query', 'response', 'timestamp', 'analysis_type']
        for field in required_fields:
            if field not in data:
                abort(400, description=f"Missing required field: {field}")
        
        # Generate ID if not provided
        if 'id' not in data:
            data['id'] = f"analysis_{uuid.uuid4().hex}"
        
        # Default confidence if not provided
        if 'confidence' not in data:
            data['confidence'] = 0.5
            
        # Store analysis in database
        try:
            # Store in saved_analyses collection
            doc_ref = db.collection('saved_analyses').document(data['id'])
            await doc_ref.set({
                'id': data['id'],
                'query': data['query'],
                'response': data['response'],
                'analysis_type': data['analysis_type'],
                'confidence': data['confidence'],
                'timestamp': data['timestamp'],
                'sources': data.get('sources', []),
                'created_at': firestore.SERVER_TIMESTAMP
            })
            
            return jsonify({
                'success': True,
                'id': data['id'],
                'message': 'Analysis saved successfully'
            })
            
        except Exception as db_error:
            logging.error(f"Database error saving analysis: {db_error}")
            abort(500, description="Failed to save analysis")
            
    except Exception as e:
        logging.error(f"Error saving analysis: {e}")
        abort(500, description="An unexpected error occurred")


@app.route('/api/analysis/saved/<analysis_id>', methods=['GET'], endpoint='get_saved_analysis_endpoint')
@async_route
async def get_saved_analysis(analysis_id):
    """Get a specific saved analysis by ID"""
    try:
        # Get analysis
        doc_ref = db.collection('saved_analyses').document(analysis_id)
        doc = await doc_ref.get()
        
        if not doc.exists:
            abort(404, description="Analysis not found")
            
        analysis_data = doc.to_dict()
        
        return jsonify({
            'id': analysis_data.get('id'),
            'query': analysis_data.get('query'),
            'response': analysis_data.get('response'),
            'analysis_type': analysis_data.get('analysis_type'),
            'confidence': analysis_data.get('confidence'),
            'timestamp': analysis_data.get('timestamp'),
            'sources': analysis_data.get('sources', [])
        })
            
    except Exception as e:
        logging.error(f"Error getting saved analysis: {e}")
        abort(500, description="Failed to retrieve saved analysis")


@app.route('/api/articles/validate', methods=['POST'], endpoint='validate_articles_endpoint')
@async_route
@rate_limiting_service.rate_limit_decorator(key="api") # Apply general API rate limiting
async def validate_articles_endpoint():
    """Validate if a list of article IDs exist in the database."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    article_ids = data.get('article_ids')

    if not isinstance(article_ids, list):
        return jsonify({"error": "article_ids must be a list"}), 400

    if not article_ids:
        return jsonify({"results": {}, "count": {"total": 0, "valid": 0, "invalid": 0}})

    try:
        # Use the newly added method in StorageService
        validity_results = await storage_service.check_articles_exist(article_ids)

        valid_count = sum(1 for valid in validity_results.values() if valid)
        invalid_count = len(article_ids) - valid_count

        return jsonify({
            "results": validity_results,
            "count": {
                "total": len(article_ids),
                "valid": valid_count,
                "invalid": invalid_count
            }
        })
    except Exception as e:
        logging.error(f"Error validating articles: {e}", exc_info=True)
        return jsonify({"error": "Failed to validate articles"}), 500

@app.route('/api/sources', methods=['GET'], endpoint='get_sources_endpoint')
@async_route
@cache_service.cache_decorator(namespace="sources", ttl=3600)  # Cache for 1 hour
@rate_limiting_service.rate_limit_decorator(key="api")
async def get_sources():
    """
    Get all news sources with their metadata and article count.
    Returns:
        JSON response with sources data
    """
    try:
        # Get articles to count by source
        articles = await storage_service.get_all_articles()
        
        # Count articles by source
        source_counts = {}
        for article in articles:
            source = article.get('metadata', {}).get('source')
            if source:
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
        
        # Get source URLs from config if available
        source_urls = config.get_source_urls() or {}
        
        # Prepare sources data
        sources = []
        for source_name, count in source_counts.items():
            source_data = {
                'name': source_name,
                'count': count,
                'url': source_urls.get(source_name, '#')
            }
            
            # Add logo URL if available
            logo_path = f'static/logos/{source_name.lower().replace(" ", "_")}.png'
            if os.path.exists(logo_path):
                source_data['logo_url'] = f'/{logo_path}'
                
            sources.append(source_data)
        
        # Sort sources by count (descending)
        sources.sort(key=lambda x: x['count'], reverse=True)
        
        return jsonify({'sources': sources})
        
    except Exception as e:
        logging.error(f"Error getting sources: {e}")
        raise APIError(f"Failed to get sources: {str(e)}")

@app.route('/api/analysis/saved', methods=['GET'], endpoint='get_saved_analyses_endpoint')
@async_route
async def get_saved_analyses():
    """Get list of saved analyses using StorageService"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 10, type=int)
        offset = request.args.get('offset', 0, type=int)
        sort_by = request.args.get('sort_by', 'created_at') # Default sort by creation
        sort_order = request.args.get('sort_order', 'desc')

        # Ensure limit and offset are non-negative
        limit = max(1, limit)
        offset = max(0, offset)
            
        try:
            # Use StorageService to get saved analyses
            results = await storage_service.get_saved_analyses(
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            return jsonify(results)
            
        except StorageException as db_error:
            logging.error(f"Storage service error getting saved analyses: {db_error}")
            abort(500, description=f"Failed to retrieve saved analyses: {db_error}")
            
    except Exception as e:
        logging.error(f"Error getting saved analyses: {e}", exc_info=True)
        abort(500, description="An unexpected error occurred while retrieving saved analyses")



@app.route('/api/analysis/saved/<analysis_id>', methods=['DELETE'], endpoint='delete_saved_analysis_endpoint')
@async_route
async def delete_saved_analysis(analysis_id):
    """Delete a saved analysis by ID using StorageService"""
    try:
        if not analysis_id:
             abort(400, description="Analysis ID is required")
             
        try:
            # Use StorageService to delete the analysis
            deleted = await storage_service.delete_saved_analysis(analysis_id)
            
            if not deleted:
                # If the service returned False, it means the analysis wasn't found
                abort(404, description="Analysis not found")
            
            return jsonify({
                'success': True,
                'message': 'Analysis deleted successfully'
            })
            
        except StorageException as db_error:
            logging.error(f"Storage service error deleting analysis {analysis_id}: {db_error}")
            abort(500, description=f"Failed to delete analysis {analysis_id}: {db_error}")
            
    except Exception as e:
        logging.error(f"Error deleting saved analysis {analysis_id}: {e}", exc_info=True)
        abort(500, description=f"An unexpected error occurred while deleting analysis {analysis_id}")

@app.route('/api/status', methods=['GET'])
async def get_service_status():
    """Get the current status of app services"""
    try:
        status = {
            "status": "ok",
            "services": {
                "api": True,
                "storage": True,
                "ai": ai_service is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check if AI is available but not healthy
        if ai_service is not None:
            try:
                ai_health = await ai_service.is_healthy()
                status["services"]["ai_healthy"] = ai_health
                if not ai_health:
                    status["services"]["ai_error"] = getattr(ai_service, 'error_message', 'Unknown error')
            except Exception as e:
                status["services"]["ai_healthy"] = False
                status["services"]["ai_error"] = str(e)
        
        return jsonify(status)
    except Exception as e:
        logging.error(f"Error checking service status: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Newsletter Aggregator Application')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to run the application on')
    parser.add_argument('--import_articles', action='store_true', help='Import articles from CSV to database on startup')
    parser.add_argument('--verify_db', action='store_true', help='Verify and repair database consistency')
    args = parser.parse_args()
    
    # Run database verification if requested
    if args.verify_db:
        logging.info("Running database verification...")
        try:
            loop = asyncio.get_event_loop()
            success, stats = loop.run_until_complete(verify_database_consistency())
            
            if success:
                logging.info(f"Database verification completed: {stats}")
                if stats['repaired'] > 0:
                    logging.info(f"Repaired {stats['repaired']} inconsistencies")
            else:
                logging.error(f"Database verification failed: {stats}")
        except Exception as e:
            logging.error(f"Error during database verification: {e}")
    
    # Import articles if requested
    if args.import_articles:
        logging.info("Importing articles from CSV to database...")
        try:
            # Directly use import_articles.py logic
            # First check if the module exists
            try:
                from import_articles import import_articles
                
                loop = asyncio.get_event_loop()
                loop.run_until_complete(import_articles())
                logging.info("Articles imported successfully")
            except ImportError:
                logging.error("Could not import the import_articles module")
        except Exception as e:
            logging.error(f"Error importing articles: {e}")
    
    # Use debug mode only in development environment
    debug_mode = os.environ.get('FLASK_ENV', 'development').lower() != 'production'
    logging.info(f"Starting Flask application on port {args.port} with debug={debug_mode}")
    
    # Log important configuration information
    logging.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    logging.info(f"Storage backend: {os.environ.get('STORAGE_BACKEND', 'default')}")
    logging.info(f"Using Cloud Logging: {os.environ.get('USE_CLOUD_LOGGING', 'False')}")
    
    # Run the app - when run directly (not through waitress/gunicorn)
    try:
        app.run(debug=debug_mode, host='0.0.0.0', port=args.port)
    except Exception as e:
        logging.critical(f"Fatal error starting the application: {e}", exc_info=True)
        sys.exit(1)
