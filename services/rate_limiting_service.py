"""Rate limiting service for the application."""
import asyncio
import time
from typing import Dict, Optional
import threading
from datetime import datetime, timedelta
import logging
from services.config_service import ConfigService
from functools import wraps

class RateLimitingService:
    """Thread-safe rate limiting service with support for multiple rate limits."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RateLimitingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = ConfigService()
        
        # Check if get_rate_limit_config exists, otherwise use default values
        if hasattr(self.config, 'get_rate_limit_config'):
            rate_limit_config = self.config.get_rate_limit_config()
        else:
            # Default rate limit configuration
            rate_limit_config = {
                'default_calls': 10,
                'default_period': 1.0,
                'delay': 0.1  # Default delay in seconds
            }
            logging.warning("ConfigService.get_rate_limit_config() not found, using default rate limit configuration")
        
        self._default_delay = rate_limit_config['delay']
        self._limits: Dict[str, Dict] = {}
        self._last_call: Dict[str, float] = {}
        self._tokens: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        
        self._initialized = True
    
    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a rate limit key."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]
    
    def configure_limit(
        self,
        key: str,
        calls: int,
        period: int,
        burst: Optional[int] = None
    ) -> None:
        """Configure a rate limit for a specific key.
        
        Args:
            key: Unique identifier for this rate limit
            calls: Number of calls allowed
            period: Time period in seconds
            burst: Optional burst size (allows temporary exceeding of rate)
        """
        with self._get_lock(key):
            self._limits[key] = {
                'calls': calls,
                'period': period,
                'burst': burst or calls,
                'rate': calls / period
            }
            self._tokens[key] = float(burst or calls)
            self._last_call[key] = time.time()
    
    def _update_tokens(self, key: str) -> None:
        """Update token bucket for a key."""
        now = time.time()
        if key in self._limits:
            elapsed = now - self._last_call[key]
            self._tokens[key] = min(
                self._limits[key]['burst'],
                self._tokens[key] + elapsed * self._limits[key]['rate']
            )
            self._last_call[key] = now
    
    async def acquire(self, key: str = "default") -> None:
        """Acquire a rate limit token, waiting if necessary."""
        with self._get_lock(key):
            # If no specific limit is configured, use default delay
            if key not in self._limits:
                if self._default_delay > 0:
                    await asyncio.sleep(self._default_delay)
                return
            
            self._update_tokens(key)
            
            # Wait until we have at least one token
            while self._tokens[key] < 1:
                wait_time = (1 - self._tokens[key]) / self._limits[key]['rate']
                await asyncio.sleep(wait_time)
                self._update_tokens(key)
            
            self._tokens[key] -= 1
    
    def try_acquire(self, key: str = "default") -> bool:
        """Try to acquire a rate limit token without waiting."""
        with self._get_lock(key):
            # If no specific limit is configured, use default delay
            if key not in self._limits:
                return True
            
            self._update_tokens(key)
            
            if self._tokens[key] >= 1:
                self._tokens[key] -= 1
                return True
            
            return False
    
    def get_remaining(self, key: str = "default") -> float:
        """Get remaining tokens for a rate limit."""
        with self._get_lock(key):
            if key not in self._limits:
                return float('inf')
            
            self._update_tokens(key)
            return self._tokens[key]
    
    def reset(self, key: str = "default") -> None:
        """Reset a rate limit to its initial state."""
        with self._get_lock(key):
            if key in self._limits:
                self._tokens[key] = float(self._limits[key]['burst'])
                self._last_call[key] = time.time()
    
    def rate_limit_decorator(self, key: str = "default"):
        """Decorator for rate limiting async function calls in Quart."""
        def decorator(func):
            if not asyncio.iscoroutinefunction(func):
                raise TypeError("rate_limit_decorator can only be applied to async functions in Quart.")
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.acquire(key)
                return await func(*args, **kwargs)
            return async_wrapper
        return decorator 