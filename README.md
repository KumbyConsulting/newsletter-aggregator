# Pharmaceutical Industry News Aggregator

A Flask-based web application that aggregates and monitors news from various pharmaceutical industry sources, providing real-time updates and summaries of important industry developments.

## Features

- **Automated News Scraping**: Collects news from multiple pharmaceutical industry sources via RSS feeds
- **Topic Classification**: Organizes news into 25 different pharmaceutical industry categories
- **Search Functionality**: Search through collected articles by title
- **Article Summarization**: AI-powered text summarization for article descriptions
- **Pagination**: Displays 10 articles per page for better readability
- **Duplicate Prevention**: Automatically removes duplicate articles
- **HTML Cleaning**: Removes HTML tags from article descriptions for clean display

## Directory Structure

```
newsletter/
├── app.py                 # Main Flask application
├── newsLetter.py         # News scraping and processing logic
├── news_alerts.csv       # Current news articles database
├── templates/            # HTML templates
│   └── index.html       # Main page template
└── README.md            # This documentation
```

## Key Components

### app.py
- Flask web application setup
- Route handlers for main page, article updates, and summarization
- Article pagination and search functionality
- HTML cleaning utilities

### newsLetter.py
- RSS feed definitions for various news sources
- Topic definitions and keywords for article classification
- News scraping and processing functions
- Text summarization functionality

## Topics Monitored

The system monitors news across 25 key pharmaceutical industry areas including:
- 483 Notifications
- Adulterated Drugs
- AI in Medicine
- Contamination Control
- Regulatory Affairs
- Clinical Trials
- Mergers and Acquisitions
- Drug Development
- Market Trends
- And many more...

## News Sources

The application aggregates news from multiple reliable sources including:
- FDA RSS Feeds
- European Pharmaceutical Review
- Nature Biotechnology
- Reuters Health
- PharmaTimes
- And many other industry-specific sources

## Requirements

- Python 3.x
- Flask
- pandas
- BeautifulSoup4
- requests
- logging

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Access the web interface at `http://localhost:5000`
3. Use the search bar to find specific articles
4. Click "Update Articles" to fetch fresh news
5. Click "Summarize" on any article to get an AI-generated summary

## Data Storage

News articles are stored in CSV format with the following information:
- Title
- Description
- Link
- Publication Date

Historical data is preserved in dated CSV files for reference.

## Planned Improvements and Future Enhancements

### Error Handling and Resilience
- Implement comprehensive error handling for RSS feed failures
- Add retry mechanisms for failed requests
- Enhance logging system for failed article fetches
- Create health check endpoints for system monitoring

### Data Management
- Migrate from CSV to a proper database system (SQLite/PostgreSQL)
- Implement data retention policies
- Add article archiving functionality
- Enable data export in multiple formats (PDF, Excel)

### User Experience
- Add topic-based filtering
- Implement multiple sorting options (date, relevance, source)
- Add user preferences for favorite topics
- Enable email notifications for specific topics
- Implement dark mode
- Add article bookmarking feature

### Performance Optimizations
- Implement caching for frequently accessed articles
- Add background task processing for news scraping
- Implement rate limiting for external API calls
- Optimize database queries with proper indexing

### Security Enhancements
- Add input validation for search queries
- Implement rate limiting for API endpoints
- Add CORS policies
- Implement request sanitization

### API Features
- Create RESTful API for programmatic access
- Add Swagger/OpenAPI documentation
- Implement API authentication
- Add API usage rate limiting

### Testing
- Add unit tests for core functionality
- Implement integration tests for RSS feed processing
- Add end-to-end tests for web interface
- Set up continuous integration

### Monitoring and Analytics
- Implement metrics collection (article counts, search patterns)
- Add RSS feed health monitoring
- Track user engagement analytics
- Create system statistics dashboard

### Content Enhancement
- Add sentiment analysis for articles
- Implement keyword extraction
- Add related articles suggestions
- Enable ML-based article categorization
- Add multi-language support

### Documentation
- Add detailed API documentation
- Include development setup instructions
- Create contribution guidelines
- Add troubleshooting section
- Include configuration documentation

## Contributing

We welcome contributions to improve the newsletter application. Please feel free to submit issues and pull requests for any of the planned improvements listed above or your own ideas.
