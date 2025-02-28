from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, current_app
from flask_cors import CORS
import pandas as pd
import os
from bs4 import BeautifulSoup
from newsLetter import scrape_news, RateLimitException, generate_missing_summaries, TOPICS
import logging
from dotenv import load_dotenv
from services.storage_service import StorageService
from services.ai_service import AIService
from services.config_service import ConfigService
import asyncio
from functools import wraps

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize configuration
config = ConfigService()
if not config.validate_configuration():
    raise ValueError("Invalid configuration. Please check your .env file.")

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = config.flask_secret_key

# Initialize services
storage_service = StorageService()
ai_service = AIService(storage_service)

# Add custom template filters
@app.template_filter('max_value')
def max_value(a, b):
    return max(a, b)

@app.template_filter('min_value')
def min_value(a, b):
    return min(a, b)

def clean_html(raw_html):
    """Remove HTML tags and clean text"""
    if pd.isna(raw_html) or not raw_html or raw_html == 'nan':
        return "No description available"
    try:
        # Remove HTML tags
        text = BeautifulSoup(raw_html, "html.parser").get_text()
        # Clean up whitespace
        text = ' '.join(text.split())
        return text if text.strip() else "No description available"
    except Exception as e:
        logging.error(f"Error cleaning HTML: {e}")
        return "No description available"

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the CSV file
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
                # Replace 'nan' strings and empty strings with None for summary column
                if col == 'summary':
                    df[col] = df[col].replace({'nan': None, '': None})
        
        # Remove duplicates
        df.drop_duplicates(subset=['title', 'link'], inplace=True)
        
        # Handle missing values with meaningful defaults
        df.fillna({
            'description': 'No description available', 
            'pub_date': 'Unknown date',
            'topic': 'Uncategorized',
            'source': 'Unknown source'
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
        showing_from = (page - 1) * per_page + 1
        showing_to = min(page * per_page, total)
        last_page = (total + per_page - 1) // per_page  # Ceiling division
        
        # Calculate page range for pagination
        start_page = max(page - 2, 1)
        end_page = min(start_page + 4, last_page)
        start_page = max(end_page - 4, 1)
        
        # Apply pagination
        df = df.iloc[(page - 1) * per_page: page * per_page]
        
        articles = df.to_dict(orient='records')
        topics = sorted(TOPICS.keys())
        
        # Calculate topic distribution
        topic_distribution = df['topic'].value_counts().reset_index()
        topic_distribution.columns = ['topic', 'count']
        topic_distribution['percentage'] = (topic_distribution['count'] / len(df) * 100).round(1)
        topic_distribution = topic_distribution.to_dict('records')
    else:
        articles = []
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
                         articles=articles, 
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

@app.route('/update', methods=['POST'])
async def update_articles():
    try:
        success = await scrape_news()
        if success:
            if os.path.exists(config.articles_file_path):
                df = pd.read_csv(config.articles_file_path)
                articles = df.to_dict('records')
                storage_success = await storage_service.batch_store_articles(articles)
                
                if storage_success:
                    flash("Articles updated and indexed successfully!", "success")
                else:
                    flash("Articles updated but some failed to index. Check logs for details.", "warning")
            else:
                flash("Articles updated but not indexed.", "warning")
        else:
            flash("Some errors occurred while updating articles.", "warning")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error updating articles: {e}")
        flash("Error updating articles. Please try again later.", "error")
        return redirect(url_for('index'))

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
async def rag_query():
    """Handle RAG-based queries with streaming response"""
    try:
        data = request.get_json()  # No need for await with Flask
        query = data.get('query')
        use_history = data.get('use_history', True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        # Create response
        response = await ai_service.generate_rag_response(
            query=query,
            use_history=use_history
        )
        
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
        with app.app_context():
            return jsonify({
                "error": str(e),
                "status": "error",
                "response": None,
                "sources": []
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
async def clear_rag_history():
    """Clear conversation history"""
    try:
        await ai_service.clear_history()
        return jsonify({"message": "History cleared"})
    except Exception as e:
        logging.error(f"Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/similar-articles/<article_id>', methods=['GET'])
def get_similar_articles(article_id):
    """Get articles similar to the given article"""
    try:
        similar_articles = storage_service.get_similar_articles(article_id)
        return jsonify({"articles": similar_articles})
    except Exception as e:
        logging.error(f"Error getting similar articles: {e}")
        return jsonify({"error": "Error finding similar articles"}), 500

@app.route('/rag')
def rag_interface():
    return render_template('rag_interface.html')

if __name__ == "__main__":
    app.run(debug=True)