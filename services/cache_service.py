"""Unified caching service for the application."""
import threading
import time
from typing import Any, Dict, Optional, Set
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import asyncio
from functools import wraps, lru_cache
from services.config_service import ConfigService
import os
from cachetools import TTLCache
from services.constants import TOPICS
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

class CacheService:
    """Thread-safe caching service with optional Redis backend and TTL support."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CacheService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, backend="memory", redis_url=None):
        if self._initialized:
            return
            
        # Get the singleton ConfigService instance
        config_service = ConfigService()
        self.settings = config_service.settings
        
        self.backend = backend
        self._lock = asyncio.Lock()
        self._locks = {}  # Initialize namespace locks dictionary
        if backend == "redis":
            if aioredis is None:
                raise ImportError("redis[async] is not installed. Please install with 'pip install redis[async]'.")
            self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        else:
            self._cache = {}
        
        self._max_size = self.settings.CACHE_MAX_SIZE
        self._global_lock = threading.Lock()  # Remove this if not used elsewhere
        # Ensure CACHE_FILE_PATH has a default in ConfigSettings if not in .env
        self._persistence_file = Path(self.settings.CACHE_FILE_PATH)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        # Load cached data from disk if available
        self._load_cache()
        self._initialized = True
        
        # Register cleanup on module unload
        import atexit
        atexit.register(self._save_cache)
    
    def _get_lock(self, namespace: str) -> threading.Lock:
        """Get or create a lock for a namespace."""
        # Use asyncio.Lock for async code
        if namespace not in self._locks:
            self._locks[namespace] = asyncio.Lock()
        return self._locks[namespace]
    
    def _maybe_cleanup(self) -> None:
        """Perform cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup()
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get a value from the cache."""
        self._maybe_cleanup()
        
        namespaced_key = f"{namespace}:{key}"
        if self.backend == "redis":
            value = await self.redis.get(namespaced_key)
            return json.loads(value) if value else None
        else:
            async with self._lock:
                entry = self._cache.get(namespaced_key)
                if entry is None:
                    return None
                value, expires_at = entry
                if expires_at and expires_at < asyncio.get_event_loop().time():
                    del self._cache[namespaced_key]
                    return None
                return value
    
    async def set(self, key: str, value: Any, namespace: str = "default", ttl: Optional[int] = None) -> None:
        """Set a value in the cache."""
        namespaced_key = f"{namespace}:{key}"
        if self.backend == "redis":
            value_str = json.dumps(value)
            if ttl:
                await self.redis.set(namespaced_key, value_str, ex=ttl)
            else:
                await self.redis.set(namespaced_key, value_str)
        else:
            async with self._lock:
                expires_at = None
                if ttl:
                    expires_at = asyncio.get_event_loop().time() + ttl
                self._cache[namespaced_key] = (value, expires_at)
    
    async def delete(self, key: str, namespace: str = "default") -> None:
        """Delete a value from the cache."""
        namespaced_key = f"{namespace}:{key}"
        if self.backend == "redis":
            await self.redis.delete(namespaced_key)
        else:
            async with self._lock:
                if namespaced_key in self._cache:
                    del self._cache[namespaced_key]
    
    def clear(self, namespace: str = "default") -> None:
        """Clear all values in a namespace."""
        # Remove threading lock usage for in-memory cache
        if self.backend == "redis":
            # Not implemented for Redis
            return
        self._cache.clear()
    
    def cleanup(self) -> int:
        """Remove expired entries and return count of removed items."""
        removed = 0
        current_time = time.time()
        
        # Remove threading lock usage for in-memory cache
        for namespace in list(self._cache.keys()):
            lock = self._get_lock(namespace)
            # Use async lock if needed elsewhere
            cache = self._cache[namespace]
            expired_keys = [
                k for k, (_, expiry) in cache.items()
                if expiry < current_time
            ]
            for k in expired_keys:
                del cache[k]
                removed += 1
        
        self._last_cleanup = current_time
        
        return removed
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if self._persistence_file.exists():
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)
                    
                current_time = time.time()
                for namespace, entries in data.items():
                    self._cache[namespace] = {
                        k: (v, e) for k, (v, e) in entries.items()
                        if e > current_time  # Only load non-expired entries
                    }
                logging.info(f"Loaded cache from {self._persistence_file}")
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to disk with robust serialization."""
        try:
            self.cleanup()
            serializable_cache = {}
            for key, (value, expires_at) in self._cache.items():
                try:
                    json.dumps(value)
                    serializable_value = value
                except Exception:
                    serializable_value = str(value)
                serializable_cache[key] = (serializable_value, expires_at)
            with open(self._persistence_file, 'w') as f:
                json.dump(serializable_cache, f)
            logging.info(f"Saved cache to {self._persistence_file}")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    
    def cache_decorator(self, namespace: str = "default", ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    key = f"{func.__name__}:{args}:{kwargs}"
                    result = await self.get(key, namespace)
                    if result is not None:
                        # Defensive: If result is a tuple, log and return only the first element
                        if isinstance(result, tuple):
                            logging.warning(f"[cache_decorator] Cached value for {func.__name__} is a tuple. Returning only the first element.")
                            return result[0]
                        return result
                    result = await func(*args, **kwargs)
                    # If result is a tuple, only cache and return the first element
                    if isinstance(result, tuple):
                        logging.warning(f"[cache_decorator] Function {func.__name__} returned a tuple. Caching and returning only the first element.")
                        cache_value = result[0]
                    else:
                        cache_value = result
                    await self.set(key, cache_value, namespace, ttl)
                    return cache_value
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    key = f"{func.__name__}:{args}:{kwargs}"
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(self.get(key, namespace))
                    if result is not None:
                        if isinstance(result, tuple):
                            logging.warning(f"[cache_decorator] Cached value for {func.__name__} is a tuple. Returning only the first element.")
                            return result[0]
                        return result
                    result = func(*args, **kwargs)
                    if isinstance(result, tuple):
                        logging.warning(f"[cache_decorator] Function {func.__name__} returned a tuple. Caching and returning only the first element.")
                        cache_value = result[0]
                    else:
                        cache_value = result
                    loop.run_until_complete(self.set(key, cache_value, namespace, ttl))
                    return cache_value
                return sync_wrapper
        return decorator

    async def get_cached_summary(self, text_hash):
        """Get summary from cache"""
        # This is a placeholder for future Redis integration
        # For now, use the regular cache
        return self.get(f"summary:{text_hash}")
        
    @lru_cache(maxsize=1000)
    def get_topic_keywords(self):
        """Cache topic keywords in memory"""
        return TOPICS 

    # Optionally, add a ping method for health checks
    async def ping(self):
        if self.backend == "redis":
            try:
                pong = await self.redis.ping()
                return pong
            except Exception as e:
                logging.error(f"Redis ping failed: {e}")
                return False
        return True

    # Optionally, add a clear method for testing/dev
    async def clear(self):
        if self.backend == "redis":
            await self.redis.flushdb()
        else:
            async with self._lock:
                self._cache.clear()
                # self._ttl.clear()  # Remove unused self._ttl
 