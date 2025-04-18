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

class CacheService:
    """Thread-safe caching service with persistence and TTL support."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Get the singleton ConfigService instance
        config_service = ConfigService()
        self.settings = config_service.settings
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._ttl = self.settings.CACHE_TTL
        self._max_size = self.settings.CACHE_MAX_SIZE
        self._global_lock = threading.Lock()
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
        with self._global_lock:
            if namespace not in self._locks:
                self._locks[namespace] = threading.Lock()
            return self._locks[namespace]
    
    def _maybe_cleanup(self) -> None:
        """Perform cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup()
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get a value from the cache."""
        self._maybe_cleanup()
        
        with self._get_lock(namespace):
            if namespace not in self._cache:
                return None
                
            entry = self._cache[namespace].get(key)
            if not entry:
                return None
                
            value, expiry = entry
            if expiry < time.time():
                del self._cache[namespace][key]
                return None
                
            return value
    
    def set(self, key: str, value: Any, namespace: str = "default", ttl: Optional[int] = None) -> None:
        """Set a value in the cache."""
        with self._get_lock(namespace):
            if namespace not in self._cache:
                self._cache[namespace] = {}
                
            # Use provided TTL or default
            expiry = time.time() + (ttl if ttl is not None else self._ttl)
            
            # Enforce cache size limit
            while len(self._cache[namespace]) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(
                    self._cache[namespace].keys(),
                    key=lambda k: self._cache[namespace][k][1]
                )
                del self._cache[namespace][oldest_key]
            
            self._cache[namespace][key] = (value, expiry)
    
    def delete(self, key: str, namespace: str = "default") -> None:
        """Delete a value from the cache."""
        with self._get_lock(namespace):
            if namespace in self._cache and key in self._cache[namespace]:
                del self._cache[namespace][key]
    
    def clear(self, namespace: str = "default") -> None:
        """Clear all values in a namespace."""
        with self._get_lock(namespace):
            if namespace in self._cache:
                self._cache[namespace].clear()
    
    def cleanup(self) -> int:
        """Remove expired entries and return count of removed items."""
        removed = 0
        current_time = time.time()
        
        with self._global_lock:
            for namespace in list(self._cache.keys()):
                with self._get_lock(namespace):
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
        """Save cache to disk."""
        try:
            # Clean up expired entries before saving
            self.cleanup()
            
            with open(self._persistence_file, 'w') as f:
                json.dump(self._cache, f)
            logging.info(f"Saved cache to {self._persistence_file}")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    
    def cache_decorator(self, namespace: str = "default", ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try to get from cache
                result = self.get(key, namespace)
                if result is not None:
                    return result
                
                # If not in cache, call function and cache result
                try:
                    result = await func(*args, **kwargs)
                    self.set(key, result, namespace, ttl)
                    return result
                except Exception as e:
                    logging.error(f"Error in cached function {func.__name__}: {e}")
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try to get from cache
                result = self.get(key, namespace)
                if result is not None:
                    return result
                
                # If not in cache, call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, namespace, ttl)
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
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