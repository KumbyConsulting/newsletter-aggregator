# Pharmaceutical Industry News Aggregator

A Flask-based web application that aggregates and monitors news from various pharmaceutical industry sources, providing real-time updates, AI-powered summaries, and intelligent question answering through RAG (Retrieval Augmented Generation).

## Features

### Core Features
- **Automated News Scraping**: Collects news from multiple pharmaceutical industry sources via RSS feeds
- **Periodic Updates**: Configurable automatic news updates at specified intervals
- **Topic Classification**: Organizes news into 25 different pharmaceutical industry categories
- **Search Functionality**: Both keyword and semantic search capabilities
- **Article Summarization**: AI-powered text summarization with contextual awareness
- **Pagination**: Displays 10 articles per page for better readability
- **Duplicate Prevention**: Automatically removes duplicate articles
- **HTML Cleaning**: Removes HTML tags from article descriptions

### RAG Features
- **Vector Storage**: Uses ChromaDB for efficient semantic search and retrieval
- **Contextual Summarization**: Generates summaries using related articles as context
- **Intelligent Q&A**: Answer questions using relevant articles as context
- **Source Attribution**: Provides sources for generated answers
- **Similar Article Discovery**: Finds semantically similar articles

## Configuration

### Environment Variables
```env
# Required
GEMINI_API_KEY=your-api-key
FLASK_SECRET_KEY=your-secret-key

# Optional
RATE_LIMIT_DELAY=1
SCRAPE_INTERVAL_SECONDS=3600  # Default 1 hour
CACHE_EXPIRY_DAYS=30
DEBUG_MODE=False
```

### Periodic Scraping Configuration
```python
# Default settings in newsLetter.py
DEFAULT_SCRAPE_INTERVAL = 3600  # Default to running every hour (in seconds)
MINIMUM_SCRAPE_INTERVAL = 300   # Minimum 5 minutes between scrapes
```

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd newsletter
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Running the Application

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
# Start the Flask application
gunicorn app:app

# Start the news scraper service (in separate terminal)
python newsLetter.py
```

## Architecture

### Component Overview
```
newsletter/
├── app.py                 # Main Flask application
├── newsLetter.py         # News scraping and scheduler service
├── services/
│   ├── ai_service.py     # AI and RAG functionality
│   ├── storage_service.py # Vector storage and retrieval
│   ├── config_service.py # Configuration management
│   └── cache_service.py  # Caching functionality
├── templates/            # HTML templates
└── tests/               # Test suite
```

### Key Components

#### newsLetter.py
- Periodic news scraping scheduler
- RSS feed processing
- Article deduplication
- Error handling and retry logic
- SSL certificate handling

#### ai_service.py
- RAG implementation using Gemini AI
- Contextual summarization
- Question answering with source attribution
- Conversation history management

#### storage_service.py
- ChromaDB integration
- Vector storage and retrieval
- Similar article discovery
- Batch article processing

## Monitoring

### Key Metrics
- Feed success rate
- Article match rate
- Summary generation rate
- Cache hit rate
- Processing duration
- API rate limits
- Scraping interval adherence

### Health Checks
- Feed availability
- API status
- Storage capacity
- Cache performance
- Scheduler status

## Error Handling

### SSL Verification
```python
ssl_context = ssl.create_default_context(cafile=certifi.where())
if 'fda.gov' in url:
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
```

### Rate Limiting
```python
RATE_LIMIT_DELAY = 1  # seconds between API calls
await asyncio.sleep(RATE_LIMIT_DELAY)
```

### Retry Logic
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
```

## Production Deployment

### Requirements
- SSL certificates
- Process manager (e.g., supervisord)
- Monitoring setup
- Backup strategy

### Example Supervisor Configuration
```ini
[program:newsletter-web]
command=/path/to/venv/bin/gunicorn app:app
directory=/path/to/newsletter
user=newsletter
autostart=true
autorestart=true

[program:newsletter-scraper]
command=/path/to/venv/bin/python newsLetter.py
directory=/path/to/newsletter
user=newsletter
autostart=true
autorestart=true
```

## Security Considerations

### API Protection
- Rate limiting
- Input validation
- Request sanitization
- API key rotation

### Data Security
- Secure storage
- Regular backups
- Access control
- SSL/TLS encryption

## Contributing

### Development Guidelines
1. Add tests for new features
2. Document configuration changes
3. Update performance metrics
4. Follow code style guide

### Testing
```bash
pytest tests/
```


