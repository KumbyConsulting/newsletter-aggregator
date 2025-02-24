from redis import Redis
from functools import lru_cache

class CacheService:
    def __init__(self):
        self.redis_client = Redis(host='localhost', port=6379, db=0)
        
    async def get_cached_summary(self, text_hash):
        """Get summary from Redis cache"""
        return await self.redis_client.get(text_hash)
        
    @lru_cache(maxsize=1000)
    def get_topic_keywords(self):
        """Cache topic keywords in memory"""
        return TOPICS 