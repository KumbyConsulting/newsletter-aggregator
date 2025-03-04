from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, current_app, Response
from flask_cors import CORS
import pandas as pd
import os
from bs4 import BeautifulSoup
from newsLetter import scrape_news, RateLimitException, generate_missing_summaries, TOPICS, verify_database_consistency
import logging
from dotenv import load_dotenv
from services.storage_service import StorageService
from services.ai_service import AIService, AIResponse
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
ai_service = AIService(storage_service)

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
    # Try to load articles from Firestore first
    try:
        # Get query parameters
        search_query = request.args.get('search', '')
        selected_topic = request.args.get('topic', 'All')
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        # Get recent articles from Firestore
        if selected_topic != 'All':
            articles = await storage_service.get_recent_articles(limit=100, topic=selected_topic)
        else:
            articles = await storage_service.get_recent_articles(limit=100)
            
        # Format articles from Firestore response
        formatted_articles = []
        for article in articles:
            metadata = article.get('metadata', {})
            formatted_articles.append({
                'title': metadata.get('title', 'Unknown Title'),
                'description': metadata.get('description', 'No description available'),
                'link': metadata.get('link', '#'),
                'pub_date': metadata.get('pub_date', 'Unknown date'),
                'topic': metadata.get('topic', 'Uncategorized'),
                'source': metadata.get('source', 'Unknown source'),
                'summary': metadata.get('summary', None)
            })
            
        # Apply search filter if needed
        if search_query:
            formatted_articles = [a for a in formatted_articles 
                                 if search_query.lower() in a['title'].lower()]
        
        # Calculate pagination
        total = len(formatted_articles)
        showing_from = (page - 1) * per_page + 1 if total > 0 else 0
        showing_to = min(page * per_page, total)
        last_page = (total + per_page - 1) // per_page  # Ceiling division
        
        # Calculate page range for pagination
        start_page = max(page - 2, 1)
        end_page = min(start_page + 4, last_page)
        start_page = max(end_page - 4, 1)
        
        # Apply pagination
        formatted_articles = formatted_articles[(page - 1) * per_page: page * per_page]
        
        # Get topics list
        topics = sorted(TOPICS.keys())
        
        # Calculate topic distribution from all articles
        topic_counts = {}
        for article in articles:
            topic = article.get('metadata', {}).get('topic', 'Uncategorized')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
        topic_distribution = [
            {'topic': topic, 'count': count, 
             'percentage': round(count / max(1, len(articles)) * 100, 1)}
            for topic, count in topic_counts.items()
        ]
        
    except Exception as e:
        # Fallback to CSV if Firestore query fails
        logging.error(f"Error loading articles from Firestore: {e}. Falling back to CSV.")
        
        # Load the CSV file as fallback
        filename = config.articles_file_path
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            
            # Ensure all columns are string type and handle NaN values
            string_columns = ['title', 'description', 'link', 'pub_date', 'summary', 'topic', 'source']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
                    # Clean HTML and handle empty descriptions
                    if col == 'description':
                        df[col] = df[col].apply(lambda x: clean_html(x) if x else "No description available")
                    # Handle summaries - these should be plain text
                    if col == 'summary':
                        df[col] = df[col].replace({'nan': None, '': None})
                        # Ensure summaries don't contain HTML
                        df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].apply(strip_html_tags)
            
            # Add image_url column if it doesn't exist
            if 'image_url' not in df.columns:
                df['image_url'] = ''
            
            # Remove duplicates
            df.drop_duplicates(subset=['title', 'link'], inplace=True)
            
            # Handle missing values with meaningful defaults
            df.fillna({
                'description': 'No description available', 
                'pub_date': 'Unknown date',
                'topic': 'Uncategorized',
                'source': 'Unknown source',
                'image_url': ''
            }, inplace=True)
            
            # Search functionality
            search_query = request.args.get('search', '')
            if search_query:
                df = df[df['title'].str.contains(search_query, case=False, na=False)]
            
            # Topic filter
            selected_topic = request.args.get('topic', 'All')
            if selected_topic != 'All':
                df = df[df['topic'] == selected_topic]
            
            # Calculate pagination values
            page = request.args.get('page', 1, type=int)
            per_page = 10
            total = len(df)
            showing_from = (page - 1) * per_page + 1 if total > 0 else 0
            showing_to = min(page * per_page, total)
            last_page = (total + per_page - 1) // per_page  # Ceiling division
            
            # Calculate page range for pagination
            start_page = max(page - 2, 1)
            end_page = min(start_page + 4, last_page)
            start_page = max(end_page - 4, 1)
            
            # Apply pagination
            df = df.iloc[(page - 1) * per_page: page * per_page]
            
            formatted_articles = df.to_dict(orient='records')
            topics = sorted(TOPICS.keys())
            
            # Calculate topic distribution
            topic_distribution = df['topic'].value_counts().reset_index()
            topic_distribution.columns = ['topic', 'count']
            topic_distribution['percentage'] = (topic_distribution['count'] / len(df) * 100).round(1)
            topic_distribution = topic_distribution.to_dict('records')
        else:
            formatted_articles = []
            total = 0
            page = 1
            per_page = 10
            topics = []
            selected_topic = 'All'
            search_query = ''
            showing_from = 0
            showing_to = 0
            last_page = 1
            start_page = 1
            end_page = 1
            topic_distribution = []

    return render_template('index.html', 
                         articles=formatted_articles, 
                         search_query=search_query,
                         selected_topic=selected_topic, 
                         topics=topics, 
                         page=page,
                         per_page=per_page, 
                         total=total,
                         showing_from=showing_from,
                         showing_to=showing_to,
                         last_page=last_page,
                         start_page=start_page,
                         end_page=end_page,
                         topic_distribution=topic_distribution)

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
    """Background task for updating articles"""
    global update_status
    
    with app_context:
        try:
            update_status["in_progress"] = True
            update_status["status"] = "running"
            update_status["progress"] = 10
            update_status["message"] = "Starting update process..."
            
            # Run the update process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(scrape_news(status_callback=update_callback))
            
            if success:
                # Note: scrape_news now handles both CSV and database updates
                # But for robustness, let's verify consistency after the update
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
            else:
                update_status["status"] = "completed_with_errors"
                update_status["message"] = "Some errors occurred while updating articles."
                
            update_status["progress"] = 100
            update_status["last_update"] = time.time()
            
        except Exception as e:
            logging.error(f"Error in background update: {e}")
            update_status["status"] = "failed"
            update_status["message"] = "Update failed"
            update_status["error"] = str(e)
            update_status["progress"] = 0
        finally:
            update_status["in_progress"] = False

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

@app.route('/update', methods=['POST'])
async def update_articles():
    """Legacy synchronous update route - redirects to async version"""
    try:
        # Check if update is already in progress
        if update_status["in_progress"]:
            flash("Update already in progress. Please wait.", "info")
            return redirect(url_for('index'))
        
        # Start background update
        thread = threading.Thread(
            target=async_update,
            args=(app.app_context(),)
        )
        thread.daemon = True
        thread.start()
        
        flash("Update started in background. You can continue using the app.", "info")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error starting update: {e}")
        flash("Error starting update. Please try again later.", "error")
        return redirect(url_for('index'))

@app.route('/api/update/status', methods=['GET'])
def get_update_status():
    """API endpoint to get the current update status"""
    return jsonify(update_status)

@app.route('/api/update/start', methods=['POST'])
def start_update():
    """API endpoint to start a background update"""
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
        update_status["error"] = None
        update_status["progress"] = 0
        update_status["sources_processed"] = 0
        update_status["articles_found"] = 0
        
        # Start background update
        thread = threading.Thread(
            target=async_update,
            args=(app.app_context(),)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Update started in background",
            "status": update_status
        })
    except Exception as e:
        logging.error(f"Error starting update: {e}")
        return jsonify({
            "success": False,
            "message": f"Error starting update: {str(e)}",
            "error": str(e)
        }), 500

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
            
            # Custom prompt for insights
            insight_prompt = f"""
                As a business intelligence expert, analyze these recent articles and identify key trends or patterns:
                
                Query: {query}
                
                Articles to analyze:
                {articles_context}
                
                Provide one concise, data-driven insight. Include percentages or statistics if relevant.
                Keep your response under 150 words and focus on business implications.
                """
            
            # Use a different generator for insights
            response_text = await ai_service.generate_custom_content(insight_prompt)
            
            # Create a simple response object
            response = AIResponse(
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

@app.route('/api/articles', methods=['GET'])
@async_route
async def get_articles():
    """Get articles with optional filtering"""
    try:
        # Get parameters
        topic = request.args.get('topic')
        limit = int(request.args.get('limit', 10))
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        sort_by = request.args.get('sort_by', 'pub_date')
        sort_order = request.args.get('sort_order', 'desc')
        search_query = request.args.get('search', '')
        
        try:
            # Try to get from Firestore first
            if topic and topic != 'All':
                articles = await storage_service.get_recent_articles(limit=limit * page, topic=topic)
            else:
                articles = await storage_service.get_recent_articles(limit=limit * page)
                
            # Format articles from Firestore response
            formatted_articles = []
            for article in articles:
                metadata = article.get('metadata', {})
                # Extract ID from article document
                article_id = article.get('id', None)
                
                formatted_articles.append({
                    'id': article_id,
                    'title': metadata.get('title', 'Unknown Title'),
                    'description': metadata.get('description', 'No description available'),
                    'link': metadata.get('link', '#'),
                    'pub_date': metadata.get('pub_date', 'Unknown date'),
                    'topic': metadata.get('topic', 'Uncategorized'),
                    'source': metadata.get('source', 'Unknown source'),
                    'summary': metadata.get('summary', None),
                    'image_url': metadata.get('image_url', ''),
                    'has_full_content': 'full_content' in metadata and metadata['full_content'] is not None
                })
                
            # Apply search filter if needed
            if search_query:
                formatted_articles = [a for a in formatted_articles 
                                     if search_query.lower() in a['title'].lower()]
            
            # Apply pagination - do this after filtering to ensure correct results
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_articles = formatted_articles[start_idx:end_idx]
            
            return jsonify({
                "articles": page_articles,
                "count": len(page_articles),
                "total": len(formatted_articles)
            })
            
        except Exception as e:
            # Fallback to CSV if Firestore query fails
            logging.error(f"Error loading articles from Firestore for API: {e}. Falling back to CSV.")
            
            # Load the CSV file as fallback
            filename = config.articles_file_path
            if os.path.isfile(filename):
                df = pd.read_csv(filename)
                
                # Ensure all columns are string type and handle NaN values
                string_columns = ['title', 'description', 'link', 'pub_date', 'summary', 'topic', 'source']
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].fillna('').astype(str)
                        # Clean HTML and handle empty descriptions
                        if col == 'description':
                            df[col] = df[col].apply(lambda x: clean_html(x) if x else "No description available")
                        # Handle summaries - these should be plain text
                        if col == 'summary':
                            df[col] = df[col].replace({'nan': None, '': None})
                            # Ensure summaries don't contain HTML
                            df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].apply(strip_html_tags)
                
                # Add image_url column if it doesn't exist
                if 'image_url' not in df.columns:
                    df['image_url'] = ''
                    
                # Add id column if it doesn't exist
                if 'id' not in df.columns:
                    # Generate simple IDs based on index
                    df['id'] = df.index.astype(str)
                
                # Add has_full_content flag
                df['has_full_content'] = df['description'].apply(
                    lambda x: len(x) > 300 if isinstance(x, str) else False
                )
                
                # Remove duplicates
                df.drop_duplicates(subset=['title', 'link'], inplace=True)
                
                # Handle missing values with meaningful defaults
                df.fillna({
                    'description': 'No description available', 
                    'pub_date': 'Unknown date',
                    'topic': 'Uncategorized',
                    'source': 'Unknown source',
                    'image_url': ''
                }, inplace=True)
                
                # Apply filters
                if topic and topic != 'All':
                    df = df[df['topic'] == topic]
                    
                if search_query:
                    df = df[df['title'].str.contains(search_query, case=False, na=False)]
                
                # Sort the data
                if sort_by in df.columns:
                    ascending = sort_order.lower() != 'desc'
                    df = df.sort_values(by=sort_by, ascending=ascending)
                
                # Apply pagination
                total_records = len(df)
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                df = df.iloc[start_idx:end_idx]
                
                formatted_articles = df.to_dict(orient='records')
                
                return jsonify({
                    "articles": formatted_articles,
                    "count": len(formatted_articles),
                    "total": total_records,
                    "source": "csv_fallback"
                })
            else:
                return jsonify({
                    "articles": [],
                    "count": 0,
                    "total": 0,
                    "error": "No data source available"
                })
            
    except Exception as e:
        logging.error(f"Error in API get_articles: {e}")
        return jsonify({"error": str(e), "articles": []}), 500

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

@app.route('/api/update', methods=['POST'])
async def api_update_articles():
    """API endpoint to trigger news scraping"""
    try:
        success = await scrape_news()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Articles updated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Some errors occurred while updating articles"
            }), 500
            
    except Exception as e:
        logging.error(f"Error in API update: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
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

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
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
