"""
API Configuration for External Data Sources

This file contains configuration settings and API keys for external data sources
used by the newsletter aggregator system.
"""

# NewsAPI Configuration
NEWSAPI_KEY = "08c23ab8f583472d803e5403fbad0433"
NEWSAPI_ENABLED = True
NEWSAPI_BASE_URL = "https://newsapi.org/v2"

# NewsAPI query parameters for biopharmaceutical news
NEWSAPI_QUERIES = [
    "biopharmaceutical",
    "pharmaceutical industry",
    "drug development",
    "FDA approval",
    "clinical trials",
    "biotech innovation",
    "gene therapy",
    "pharmaceutical manufacturing",
    "drug delivery",
    "cell therapy",
    "regulatory compliance pharmaceutical",
    "biopharma"
]

# Maximum number of days to look back for news
NEWSAPI_DAYS_BACK = 3

# Max results per query
NEWSAPI_MAX_RESULTS = 20

# PubMed Configuration (for future implementation)
PUBMED_ENABLED = False
PUBMED_EMAIL = ""  # Required for PubMed API
PUBMED_API_KEY = ""  # Optional, increases rate limit

# Social Media Configuration (for future implementation)
TWITTER_ENABLED = False
TWITTER_API_KEY = ""
TWITTER_API_SECRET = ""
TWITTER_ACCESS_TOKEN = ""
TWITTER_ACCESS_SECRET = ""

# Biopharmaceutical Sources Configuration
# Key websites that might not have RSS or need special handling
BIOPHARMA_SPECIAL_SOURCES = {
    "FDA Press Announcements": "https://www.fda.gov/news-events/fda-newsroom/press-announcements",
    "EMA News": "https://www.ema.europa.eu/en/news-events/news",
    "BioPharmaDive": "https://www.biopharmadive.com/",
    "FiercePharma": "https://www.fiercepharma.com/",
    "PharmaTimes": "https://www.pharmatimes.com/"
} 