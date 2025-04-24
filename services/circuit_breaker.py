"""Circuit breaker pattern implementation for handling service failures."""
import threading
import time
from functools import wraps
from typing import Callable, Any
import logging
from werkzeug.exceptions import ServiceUnavailable

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: int = 60):
        self.name = name
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
        
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    logging.info(f"Circuit {self.name} transitioning to half-open")
                    self.state = "half-open"
                    return True
                return False
            
            return self.state == "half-open"
            
    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logging.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
                self.state = "open"
                
    def record_success(self):
        with self._lock:
            if self.state == "half-open":
                logging.info(f"Circuit {self.name} closing after successful execution")
                self.state = "closed"
            self.failure_count = 0
            
class CircuitBreakerService:
    """Service to manage circuit breakers for different operations"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CircuitBreakerService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.breakers = {
            'update': CircuitBreaker('update'),
            'summary': CircuitBreaker('summary'),
            'rag': CircuitBreaker('rag'),
            'database': CircuitBreaker('database', failure_threshold=3, reset_timeout=30)
        }
        self._initialized = True
        
    def get_breaker(self, name: str) -> CircuitBreaker:
        return self.breakers.get(name)
        
    def circuit_breaker(self, name: str) -> Callable:
        """Decorator to apply circuit breaker pattern to a function"""
        breaker = self.get_breaker(name)
        if not breaker:
            raise ValueError(f"No circuit breaker found for {name}")
            
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            async def wrapped(*args, **kwargs) -> Any:
                if not breaker.can_execute():
                    raise ServiceUnavailable(
                        f"Service {name} temporarily unavailable. Please try again later."
                    )
                try:
                    result = await f(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as e:
                    breaker.record_failure()
                    raise
            return wrapped
        return decorator 