from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, current_app, Response
from flask_cors import CORS
import pandas as pd
import os
from bs4 import BeautifulSoup
from newsLetter import scrape_news, RateLimitException, generate_missing_summaries, TOPICS, verify_database_consistency
import logging
from dotenv import load_dotenv
from services.storage_service import StorageService
from services.ai_service import KumbyAI, AIServiceException
from services.config_service import ConfigService
from services.logging_service import LoggingService
from services.monitoring_service import MonitoringService
import asyncio
from functools import wraps
import json
import threading
import time
import argparse
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from cachetools import TTLCache, LRUCache
from datetime import datetime
import io
import csv

# Load environment variables
load_dotenv()

# Initialize configuration
config = ConfigService()
if not config.validate_configuration():
    raise ValueError("Invalid configuration. Please check your .env file.")

# Configure logging
LoggingService.configure_logging()
logging.info("Starting Newsletter Aggregator application")

# Initialize monitoring
monitoring_service = MonitoringService()
logging.info("Monitoring service initialized")

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = config.flask_secret_key

# Initialize services
storage_service = StorageService()
ai_service = KumbyAI(storage_service)  # Use KumbyAI instead of AIService

# Log GCP configuration status
if config.is_gcp_enabled:
    logging.info(f"Google Cloud Platform enabled with project: {config.gcp_project_id}")
    logging.info(f"Using storage backend: {config.storage_backend}")
    if config.use_gcs:
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

@app.route('/', methods=['GET', 'POST'])
async def index():
    """Main index route with improved search functionality and caching"""
    # Get query parameters with improved defaults and sanitization
    search_query = request.args.get('search', '').strip()
    selected_topic = request.args.get('topic', 'All')
    sort_by = request.args.get('sort_by', 'pub_date')  # New sorting parameter
    sort_order = request.args.get('sort_order', 'desc')  # New sort order parameter
    page = max(1, request.args.get('page', 1, type=int))  # Ensure page is at least 1
    per_page = min(50, max(10, request.args.get('per_page', 10, type=int)))  # Limit per_page between 10 and 50
    
    # Set default values
    formatted_articles = []
    total = 0
    showing_from = 0
    showing_to = 0
    last_page = 1
    start_page = 1
    end_page = 1
    topic_distribution = []
    
    # Generate cache key for this request
    cache_key_str = f"index:{selected_topic}:{page}:{search_query}:{sort_by}:{sort_order}:{per_page}"
    cached_result = article_cache.get(cache_key_str)
    
    if cached_result:
        # Use cached result if available
        return render_template('index.html', **cached_result)
    
    try:
        # Get articles with enhanced parameters
        result = await storage_service.get_articles(
            page=page,
            limit=per_page,
            topic=selected_topic if selected_topic != 'All' else None,
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        formatted_articles = []
        for article in result['articles']:
            metadata = article.get('metadata', {})
            # Enhanced article formatting with more metadata
            formatted_articles.append({
                'id': article.get('id'),  # Add article ID for similar articles feature
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
                'relevance_score': metadata.get('relevance_score', None),  # Add relevance score for search results
                'is_recent': is_recent_article(metadata.get('pub_date', '')),  # Add flag for recent articles
            })
        
        # Use pagination info from result
        total = result['total']
        showing_from = (page - 1) * per_page + 1 if total > 0 else 0
        showing_to = min(page * per_page, total)
        last_page = (total + per_page - 1) // per_page
        
        # Improved pagination range calculation
        if last_page <= 5:
            start_page = 1
            end_page = last_page
        else:
            if page <= 3:
                start_page = 1
                end_page = 5
            elif page >= last_page - 2:
                start_page = last_page - 4
                end_page = last_page
            else:
                start_page = page - 2
                end_page = page + 2
        
        # Get topics list with counts
        topics = await get_topics_with_counts()
        
        # Get topic distribution with enhanced statistics
        topic_distribution = await get_topic_distribution()
        
        # Prepare template data with enhanced parameters
        template_data = {
            'articles': formatted_articles,
            'search_query': search_query,
            'selected_topic': selected_topic,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'topics': topics,
            'page': page,
            'per_page': per_page,
            'total': total,
            'showing_from': showing_from,
            'showing_to': showing_to,
            'last_page': last_page,
            'start_page': start_page,
            'end_page': end_page,
            'topic_distribution': topic_distribution,
            'has_search_results': bool(search_query and formatted_articles),
            'search_time': result.get('query_time', 0),  # Add search performance metric
        }
        
        # Cache the result
        article_cache[cache_key_str] = template_data
        
        return render_template('index.html', **template_data)
        
    except Exception as e:
        logging.error(f"Error in index route: {e}")
        flash("An error occurred while loading articles. Please try again.", "error")
        return render_template('index.html', 
                             articles=[], 
                             topics=sorted(TOPICS.keys()),
                             search_query=search_query,
                             selected_topic=selected_topic,
                             error=str(e))

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
    "articles_found": 0
}

def async_update(app_context):
    """Background task for updating articles with improved error handling and cache invalidation"""
    global update_status
    
    with app_context:
        try:
            update_status["status"] = "running"
            update_status["progress"] = 10
            update_status["message"] = "Starting update process..."
            
            # Run the update process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(scrape_news(status_callback=update_callback))
            
            if success:
                # Verify database consistency after update
                update_status["progress"] = 85
                update_status["message"] = "Verifying database consistency..."
                
                # Verify and repair any inconsistencies
                verify_success, stats = loop.run_until_complete(verify_database_consistency())
                
                if verify_success:
                    if stats["repaired"] > 0:
                        update_status["status"] = "completed_with_repair"
                        update_status["message"] = f"Articles updated. Repaired {stats['repaired']} inconsistencies."
                    else:
                        update_status["status"] = "completed"
                        update_status["message"] = "Articles updated and database is consistent."
                else:
                    update_status["status"] = "completed_with_warnings"
                    update_status["message"] = "Articles updated but verification failed."
                    
                # Invalidate caches after successful update
                invalidate_caches()
                
                # Record success metric
                monitoring_service.record_count("article_updates_completed", labels={"success": "true"})
            else:
                update_status["status"] = "completed_with_errors"
                update_status["message"] = "Some errors occurred while updating articles."
                monitoring_service.record_count("article_updates_completed", labels={"success": "false"})
                
            update_status["progress"] = 100
            update_status["last_update"] = time.time()
            
        except Exception as e:
            logging.error(f"Error in background update: {e}")
            update_status["status"] = "failed"
            update_status["message"] = "Update failed"
            update_status["error"] = str(e)
            update_status["progress"] = 0
            monitoring_service.record_count("article_updates_completed", labels={"success": "false", "error": "exception"})
        finally:
            update_status["in_progress"] = False

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
    """Single unified API endpoint to start a background update"""
    global update_status
    
    try:
        # Check if update is already in progress
        if update_status["in_progress"]:
            return jsonify({
                "success": False,
                "message": "Update already in progress",
                "status": update_status
            })
        
        # Reset status
        update_status = {
            "in_progress": True,
            "last_update": None,
            "status": "starting",
            "progress": 0,
            "message": "Starting update process...",
            "error": None,
            "sources_processed": 0,
            "total_sources": 0,
            "articles_found": 0
        }
        
        # Start background update
        thread = threading.Thread(
            target=async_update,
            args=(app.app_context(),)
        )
        thread.daemon = True
        thread.start()
        
        # Record metric
        monitoring_service.record_count("article_updates_started")
        
        return jsonify({
            "success": True,
            "message": "Update started in background",
            "status": update_status
        })
    except Exception as e:
        logging.error(f"Error starting update: {e}")
        update_status["in_progress"] = False
        update_status["status"] = "failed"
        update_status["error"] = str(e)
        return jsonify({
            "success": False,
            "message": f"Error starting update: {str(e)}",
            "error": str(e)
        }), 500

# Legacy route that redirects to the unified endpoint
@app.route('/update', methods=['POST'])
async def update_articles():
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
async def api_update_articles():
    """Legacy API endpoint - redirects to the unified API endpoint"""
    return start_update()

@app.route('/api/update/status', methods=['GET'])
def get_update_status():
    """API endpoint to get the current update status"""
    return jsonify(update_status)

@app.route('/summarize', methods=['POST'])
async def summarize_article():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        description = data.get('description', '')
        topic = data.get('topic', '')
        article_link = data.get('link', '')
        
        if not description:
            return jsonify({"error": "No description provided", "summary": None}), 400
            
        try:
            # Use AIService for summarization
            summary = await ai_service.summarize_with_context(description, topic)
            
            if not summary or summary.strip() == '':
                return jsonify({"error": "Could not generate summary", "summary": None}), 400
            
            # Save summary to CSV
            filename = config.articles_file_path  # Use config for file path
            if os.path.isfile(filename):
                try:
                    df = pd.read_csv(filename)
                    df['summary'] = df['summary'].fillna('').astype(str)
                    mask = df['link'] == article_link
                    if mask.any():
                        df.loc[mask, 'summary'] = str(summary)
                        df.to_csv(filename, index=False)
                except Exception as e:
                    logging.error(f"Error updating CSV with summary: {e}")
            
            return jsonify({"summary": summary})
            
        except Exception as e:
            logging.error(f"Error in summarization: {e}")
            return jsonify({
                "error": "Error generating summary",
                "summary": None
            }), 500
            
    except Exception as e:
        logging.error(f"Error in summarize_article: {e}")
        return jsonify({
            "error": "Internal server error",
            "summary": None
        }), 500

@app.route('/generate-summaries', methods=['POST'])
@monitoring_service.time_function("generate_summaries")
def generate_summaries():
    try:
        success = generate_missing_summaries()
        if success:
            flash("Summaries generated successfully!", "success")
        else:
            flash("Some errors occurred while generating summaries.", "warning")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error generating summaries: {e}")
        flash("Error generating summaries. Please try again later.", "error")
        return redirect(url_for('index'))

# Add debug route to test API
@app.route('/test_summary', methods=['GET'])
def test_summary():
    return render_template('test_summary.html')

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@app.route('/api/rag', methods=['POST'])
@async_route
@monitoring_service.time_async_function("rag_query")
async def rag_query():
    """Handle RAG-based queries"""
    try:
        data = request.get_json()
        query = data.get('query')
        use_history = data.get('use_history', True)
        show_sources = data.get('show_sources', True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        # Get response from AI service
        response = await ai_service.generate_rag_response(query, use_history)
        
        return jsonify({
            "query": query,
            "response": response.text,
            "sources": response.sources if show_sources else [],
            "timestamp": response.timestamp
        })
        
    except Exception as e:
        logging.error(f"Error in RAG query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/rag/stream', methods=['POST'])
@async_route
@monitoring_service.time_async_function("rag_stream_query")
async def stream_rag_query():
    """Handle RAG-based queries with streaming response"""
    try:
        data = request.get_json()  # No need for await with Flask
        query = data.get('query')
        use_history = data.get('use_history', True)
        insight_mode = data.get('insight_mode', False)
        time_aware = data.get('time_aware', True)  # New parameter for time awareness
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Create response
        if insight_mode:
            # For insights, we want to use the most recent articles regardless of query relevance
            # and craft a more analytical response focused on trends
            recent_articles = await storage_service.get_recent_articles(limit=10)
            
            if not recent_articles:
                logging.warning("No articles found for insights mode")
                return jsonify({"error": "No articles available for analysis"}), 404
                
            # Format the articles as a proper context for the model
            context_parts = []
            for i, article in enumerate(recent_articles, 1):
                title = article.get('metadata', {}).get('title', 'Untitled')
                topic = article.get('metadata', {}).get('topic', 'Unknown')
                date = article.get('metadata', {}).get('pub_date', 'Unknown date')
                source = article.get('metadata', {}).get('source', 'Unknown source')
                content = article.get('document', '')
                
                # Format the article with clear structure
                context_parts.append(f"""Article {i}:
Title: {title}
Topic: {topic}
Date: {date}
Source: {source}
Content: {content}
URL: {article.get('metadata', {}).get('link', 'No URL available')}
""")
            
            articles_context = "\n\n".join(context_parts)
            
            # Custom prompt for insights with time awareness
            current_date = datetime.now().strftime("%B %Y")
            insight_prompt = f"""
                As a business intelligence expert, analyze these recent articles and identify key trends or patterns:
                
                Current date: {current_date}
                Important: Your training data only goes up to November 2023, but the following context contains pharmaceutical articles with information that may be more recent (up to March 2025). Consider this information as accurate and up-to-date.
                
                Query: {query}
                
                Articles to analyze:
                {articles_context}
                
                Provide one concise, data-driven insight. Include percentages or statistics if relevant.
                Keep your response under 150 words and focus on business implications.
                """
            
            # Use a different generator for insights
            response_text = await ai_service.generate_custom_content(insight_prompt)
            
            # Create a simple response object
            response = KumbyAI(
                text=response_text,
                sources=[{
                    'title': a.get('metadata', {}).get('title', 'Untitled'),
                    'source': a.get('metadata', {}).get('source', 'Unknown'),
                    'date': a.get('metadata', {}).get('pub_date', 'No date'),
                    'link': a.get('metadata', {}).get('link', '')
                } for a in recent_articles],
                confidence=0.85  # Higher confidence for insights as we're using most recent articles
            )
        else:
            # Regular RAG query
            response = await ai_service.generate_streaming_response(
                query=query,
                use_history=use_history
            )
        
        # Record metrics
        monitoring_service.record_count("rag_queries", labels={"success": "true", "insight_mode": str(insight_mode).lower()})
        
        # Format response for frontend
        with app.app_context():
            return jsonify({
                "response": response.text,
                "sources": response.sources or [],  # Ensure sources is never null
                "confidence": response.confidence,
                "timestamp": response.timestamp,
                "status": "success"
            })
        
    except Exception as e:
        logging.error(f"Error in RAG query: {e}")
        monitoring_service.record_count("rag_queries", labels={"success": "false"})
        with app.app_context():
            return jsonify({
                "error": str(e),
                "status": "error",
                "response": None,
                "sources": []
            }), 500

@app.route('/api/rag/stream', methods=['GET'])
@async_route
@monitoring_service.time_async_function("rag_stream_query_get")
async def stream_rag_query_get():
    """Handle streaming RAG-based queries via GET for EventSource compatibility"""
    try:
        query = request.args.get('query')
        use_history = request.args.get('use_history', 'true').lower() == 'true'
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Record metrics
        monitoring_service.record_count("rag_stream_queries", labels={"method": "get"})
        
        # Create a streaming response
        async def generate():
            sources = []
            response_text = ""
            
            async for chunk in ai_service.generate_streaming_response(
                query=query,
                use_history=use_history
            ):
                # Check if this is the sources marker
                if chunk.startswith("__SOURCES__"):
                    try:
                        sources = json.loads(chunk[11:])  # Extract JSON after marker
                    except:
                        pass
                    continue
                    
                response_text += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                
            # Send final message with sources
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'sources': sources, 'full_response': response_text})}\n\n"
            
        # Use a synchronous generator that runs the async one
        def sync_generator():
            loop = asyncio.new_event_loop()
            async_gen = generate()
            
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()
        
        return Response(sync_generator(), mimetype='text/event-stream')
        
    except Exception as e:
        logging.error(f"Error in streaming RAG query: {e}")
        monitoring_service.record_count("rag_stream_queries", labels={"success": "false"})
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

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

def validate_request(*required_fields):
    """Decorator to validate request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.is_json:
                data = request.get_json()
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": f"Missing required fields: {', '.join(missing_fields)}"
                    }), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def cache_key(*args, **kwargs):
    """Generate a cache key from request parameters"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

@app.route('/api/articles', methods=['GET'])
@async_route
async def get_articles():
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        topic = request.args.get('topic')
        search = request.args.get('search')
        
        # Validate parameters
        if page < 1 or limit < 1 or limit > 50:
            return jsonify({"error": "Invalid page or limit parameters"}), 400
            
        # Generate cache key
        cache_key_str = cache_key('articles', page, limit, topic, search)
        
        # Try to get from cache
        cached_result = article_cache.get(cache_key_str)
        if cached_result:
            return jsonify(cached_result)
            
        # Get articles from storage
        start_time = time.time()
        result = await storage_service.get_articles(
            page=page,
            limit=limit,
            topic=topic,
            search_query=search
        )
        query_time = int((time.time() - start_time) * 1000)
        
        # Add query time to result
        result['query_time'] = query_time
        
        # Cache the result
        article_cache[cache_key_str] = result
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": "Invalid parameters"}), 400
    except Exception as e:
        logging.error(f"Error getting articles: {e}")
        return jsonify({
            "error": str(e),
            "articles": [],
            "total": 0,
            "page": 1,
            "total_pages": 0
        }), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """API endpoint to get topic distribution"""
    try:
        filename = config.articles_file_path
        if not os.path.isfile(filename):
            return jsonify({"topics": []})
            
        df = pd.read_csv(filename)
        
        # Calculate topic distribution
        topic_counts = df['topic'].value_counts().reset_index()
        topic_counts.columns = ['topic', 'count']
        topic_counts['percentage'] = (topic_counts['count'] / len(df) * 100).round(1)
        
        return jsonify({
            "topics": topic_counts.to_dict('records')
        })
        
    except Exception as e:
        logging.error(f"Error in get_topics API: {e}")
        return jsonify({
            "error": str(e),
            "topics": []
        }), 500

@app.route('/api/storage/sync', methods=['POST'])
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

@app.route('/api/storage/export', methods=['POST'])
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
                import re
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

@app.route('/api/kumby/safety', methods=['POST'])
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

@app.route('/api/kumby/pipeline', methods=['POST'])
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

@app.route('/api/kumby/therapeutic-area', methods=['POST'])
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

@app.route('/api/kumby/regulatory-strategy', methods=['POST'])
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

@app.route('/api/kumby/digital-health', methods=['POST'])
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

@app.route('/api/kumby/value-evidence', methods=['POST'])
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
@app.route('/api/kumbyai/chat', methods=['POST'])
@async_route
@monitoring_service.time_async_function("kumbyai_chat")
async def kumbyai_chat():
    try:
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
@app.route('/api/admin/export-csv')
@async_route
@monitoring_service.time_async_function("admin_export_csv")
async def admin_export_csv():
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

@app.route('/api/admin/force-sync', methods=['POST'])
@async_route
@monitoring_service.time_async_function("admin_force_sync")
async def admin_force_sync():
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

@app.route('/api/admin/cleanup-duplicates', methods=['POST'])
@async_route
@monitoring_service.time_async_function("admin_cleanup_duplicates")
async def admin_cleanup_duplicates():
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

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

# Add after the app initialization
@app.after_request
def add_header(response):
    """Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Newsletter Aggregator Application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
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
    
    app.run(debug=True, port=args.port)
