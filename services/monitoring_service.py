from prometheus_client import Counter, Histogram
import structlog

class MonitoringService:
    def __init__(self):
        self.logger = structlog.get_logger()
        self.scrape_counter = Counter('news_scrapes_total', 'Total number of scraping operations')
        self.summary_duration = Histogram('summary_generation_seconds', 'Time spent generating summaries') 