from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_cors import CORS
import pandas as pd
import os
from bs4 import BeautifulSoup
from newsLetter import scrape_news, summarize_text, RateLimitException, generate_missing_summaries
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

def clean_html(raw_html):
    """Remove HTML tags from a string."""
    if pd.isna(raw_html):
        return "No description available"
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the CSV file
    filename = "news_alerts.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        
        # Ensure all columns are string type and handle NaN values
        string_columns = ['title', 'description', 'link', 'pub_date', 'summary', 'topic', 'source']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                # Replace 'nan' strings and empty strings with None for summary column
                if col == 'summary':
                    df[col] = df[col].replace({'nan': None, '': None})
        
        # Ensure 'summary' column exists
        if 'summary' not in df.columns:
            df['summary'] = None
        
        # Remove duplicates
        df.drop_duplicates(subset=['title', 'link'], inplace=True)
        
        # Clean HTML tags from the description
        df['description'] = df['description'].apply(clean_html)
        
        # Handle missing values
        df.fillna({
            'description': 'No description available', 
            'pub_date': 'Unknown date',
            # Don't fill summary - let it remain None if empty
        }, inplace=True)
        
        # Search functionality
        search_query = request.form.get('search', '')
        if search_query:
            df = df[df['title'].str.contains(search_query, case=False, na=False)]
        
        # Topic filter
        selected_topic = request.form.get('topic', 'All')
        if selected_topic != 'All':
            df = df[df['topic'] == selected_topic]
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10
        total = len(df)
        df = df.iloc[(page - 1) * per_page: page * per_page]
        
        articles = df.to_dict(orient='records')
        topics = df['topic'].unique().tolist()
    else:
        articles = []
        total = 0
        page = 1
        per_page = 10
        topics = []

    return render_template('index.html', articles=articles, search_query=search_query, 
                         selected_topic=selected_topic, topics=topics, page=page, 
                         per_page=per_page, total=total)

@app.route('/update', methods=['POST'])
def update_articles():
    try:
        success = scrape_news()
        if success:
            flash("Articles updated successfully!", "success")
        else:
            flash("Some errors occurred while updating articles.", "warning")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error updating articles: {e}")
        flash("Error updating articles. Please try again later.", "error")
        return redirect(url_for('index'))

@app.route('/summarize', methods=['POST'])
def summarize_article():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        description = data.get('description', '')
        article_link = data.get('link', '')
        
        if not description:
            return jsonify({"error": "No description provided", "summary": None}), 400
            
        try:
            summary = summarize_text(description)
            
            if not summary or summary.strip() == '':
                return jsonify({"error": "Could not generate summary", "summary": None}), 400
            
            # Save summary to CSV
            filename = "news_alerts.csv"
            if os.path.isfile(filename):
                try:
                    df = pd.read_csv(filename)
                    # Ensure summary column is string type
                    df['summary'] = df['summary'].fillna('').astype(str)
                    # Update summary for the specific article
                    mask = df['link'] == article_link
                    if mask.any():
                        df.loc[mask, 'summary'] = str(summary)
                        df.to_csv(filename, index=False)
                except Exception as e:
                    logging.error(f"Error updating CSV with summary: {e}")
            
            return jsonify({"summary": summary})
            
        except RateLimitException as e:
            logging.warning(f"Rate limit exceeded: {e}")
            return jsonify({
                "error": "Rate limit exceeded", 
                "summary": None
            }), 429
            
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

if __name__ == "__main__":
    app.run(debug=True)