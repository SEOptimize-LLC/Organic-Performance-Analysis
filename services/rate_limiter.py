"""
Rate limiting service for API requests.
Implements token bucket algorithm with per-API configuration.
"""

import time
import threading
from functools import wraps
from typing import Callable, Dict
from collections import defaultdict

from config.settings import settings
from utils.logger import logger


class TokenBucket:
    """
    Token bucket rate limiter implementation.
    Allows burst requests while maintaining average rate.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second to add
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            blocking: Whether to block until tokens available
            
        Returns:
            True if tokens acquired, False if non-blocking and unavailable
        """
        with self.lock:
            while True:
                # Add tokens based on time elapsed
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                if not blocking:
                    return False
                
                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(min(wait_time, 1.0))
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds (0 if available now)
        """
        with self.lock:
            # Update tokens
            now = time.time()
            elapsed = now - self.last_update
            current_tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            
            if current_tokens >= tokens:
                return 0.0
            
            return (tokens - current_tokens) / self.rate


class RateLimiter:
    """
    Rate limiter service with per-API buckets.
    Provides decorators for rate-limited functions.
    """
    
    def __init__(self):
        """Initialize rate limiter with configured buckets"""
        self.buckets: Dict[str, TokenBucket] = {}
        self._init_buckets()
        self.request_counts: Dict[str, int] = defaultdict(int)
    
    def _init_buckets(self):
        """Initialize token buckets for each API"""
        # GSC: 200 requests per 100 seconds = 2 per second
        self.buckets["gsc"] = TokenBucket(
            rate=settings.gsc_requests_per_minute / 60,
            capacity=10  # Allow small burst
        )
        
        # DataForSEO: 60 requests per minute
        self.buckets["dataforseo"] = TokenBucket(
            rate=settings.dataforseo_requests_per_minute / 60,
            capacity=5
        )
        
        # OpenRouter: 30 requests per minute
        self.buckets["openrouter"] = TokenBucket(
            rate=settings.openrouter_requests_per_minute / 60,
            capacity=3
        )
    
    def acquire(self, api: str, tokens: int = 1) -> bool:
        """
        Acquire rate limit tokens for an API.
        
        Args:
            api: API identifier (gsc, dataforseo, openrouter)
            tokens: Number of tokens to acquire
            
        Returns:
            True when tokens acquired
        """
        bucket = self.buckets.get(api)
        if bucket:
            result = bucket.acquire(tokens)
            if result:
                self.request_counts[api] += 1
            return result
        return True  # No rate limit for unknown APIs
    
    def get_wait_time(self, api: str, tokens: int = 1) -> float:
        """
        Get wait time for an API.
        
        Args:
            api: API identifier
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        bucket = self.buckets.get(api)
        if bucket:
            return bucket.get_wait_time(tokens)
        return 0.0
    
    def limit(self, api: str, tokens: int = 1):
        """
        Decorator to rate limit a function.
        
        Args:
            api: API identifier
            tokens: Tokens per call
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                wait_time = self.get_wait_time(api, tokens)
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {api}")
                
                self.acquire(api, tokens)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def limit_gsc(self, func: Callable) -> Callable:
        """Decorator for GSC rate limiting"""
        return self.limit("gsc")(func)
    
    def limit_dataforseo(self, func: Callable) -> Callable:
        """Decorator for DataForSEO rate limiting"""
        return self.limit("dataforseo")(func)
    
    def limit_openrouter(self, func: Callable) -> Callable:
        """Decorator for OpenRouter rate limiting"""
        return self.limit("openrouter")(func)
    
    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dict with request counts and bucket states
        """
        stats = {
            "request_counts": dict(self.request_counts),
            "buckets": {}
        }
        
        for api, bucket in self.buckets.items():
            stats["buckets"][api] = {
                "tokens_available": bucket.tokens,
                "capacity": bucket.capacity,
                "rate": bucket.rate
            }
        
        return stats
    
    def reset_counts(self):
        """Reset request counts"""
        self.request_counts.clear()


class RetryHandler:
    """
    Retry handler for rate limit errors with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ):
        """
        Execute function with retry on failure.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = any(
                    indicator in error_str
                    for indicator in ['rate limit', '429', 'too many requests']
                )
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay
                    )
                    
                    if is_rate_limit:
                        delay *= 2  # Double delay for rate limits
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} "
                        f"after {delay:.1f}s: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} retries failed: {str(e)}"
                    )
        
        raise last_exception


# Singleton instances
rate_limiter = RateLimiter()
retry_handler = RetryHandler()