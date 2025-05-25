import asyncio
import pandas as pd
import logging
from services.storage_service import StorageService
from services.config_service import ConfigService

storage_service = StorageService()  # Singleton instance

async def import_articles():
    logging.basicConfig(level=logging.INFO)
    config = ConfigService()
    
    print('Reading articles from CSV...')
    df = pd.read_csv(config.articles_file_path)
    articles = df.to_dict('records')
    print(f'Found {len(articles)} articles to import')
    
    # Print sample of the first article to verify structure
    if articles:
        print(f'Sample article structure:')
        for key, value in articles[0].items():
            print(f'  {key}: {value[:100] if isinstance(value, str) else value}')
    
    print('Storing articles in the database...')
    success = await storage_service.batch_store_articles(articles)
    print(f'Import completed. Success: {success}')
    
    # Verify import with both methods
    query_articles = await storage_service.query_articles('pharmaceutical', n_results=5)
    print(f'Query articles found: {len(query_articles)}')
    
    recent_articles = await storage_service.get_recent_articles(limit=5)
    print(f'Recent articles found: {len(recent_articles)}')
    
    if query_articles:
        print(f'First query article title: {query_articles[0]["metadata"].get("title")}')
    if recent_articles:
        print(f'First recent article title: {recent_articles[0]["metadata"].get("title")}')

# Run the import
if __name__ == "__main__":
    asyncio.run(import_articles()) 