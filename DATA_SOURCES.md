# Newsletter Aggregator Data Sources

This document provides an overview of the data sources used in the Newsletter Aggregator system, with a focus on biopharmaceutical news and information.

## Current Data Sources

### 1. RSS Feeds

The system currently pulls data from approximately 40 RSS feeds from various sources including:

- Medical journals (JAMA, The Lancet, New England Journal of Medicine)
- Pharmaceutical industry news (Pharmaceutical Technology, Contract Pharma)
- Regulatory agencies (FDA)
- General news sources with biopharmaceutical coverage (BBC, CNN)

### 2. NewsAPI Integration [NEW]

The system now integrates with [NewsAPI](https://newsapi.org/) to provide additional coverage of biopharmaceutical news:

- Queries tailored to biopharmaceutical topics
- Configurable time range and result limits
- Automatic categorization by topic using the same keyword matching as RSS feeds
- Content fetching for complete article text when available

## Data Processing Pipeline

1. **Collection**: Data is gathered from RSS feeds and NewsAPI in parallel
2. **Content Enhancement**: Where possible, full article content is fetched
3. **Categorization**: Articles are categorized using keyword matching against predefined topics
4. **Storage**: Articles are stored in both CSV format and ChromaDB vector database
5. **Summarization**: AI-powered summaries are generated for articles

## Configuration

NewsAPI and other data source configurations are stored in `context/api_config.py`:

- API keys and endpoints
- Query parameters
- Rate limiting settings
- Future data source configurations

## Testing

A test script (`test_newsapi.py`) is available to verify NewsAPI integration:

```bash
python test_newsapi.py
```

This will:
1. Perform a test query to NewsAPI
2. Display information about retrieved articles
3. Save results to `newsapi_test_results.json` for inspection

## Future Data Source Expansion

The system is designed to be extensible for additional data sources:

1. **PubMed API**: For medical research publications
2. **Web Scraping**: For sites without RSS feeds
3. **Social Media**: For monitoring industry announcements
4. **Email Integration**: For newsletter parsing 