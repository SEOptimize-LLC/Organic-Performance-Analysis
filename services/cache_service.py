"""
Cache service for storing API responses.
Uses diskcache for persistence with configurable TTL.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import diskcache

from config.settings import settings
from utils.logger import logger


class CacheService:
    """
    Disk-based caching service for API responses.
    Supports different TTLs for different data types.
    """
    
    def __init__(self):
        """Initialize cache with disk storage"""
        self.cache_dir = settings.cache_dir
        self.cache = diskcache.Cache(str(self.cache_dir))
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.
        
        Args:
            prefix: Key prefix (e.g., 'gsc', 'dataforseo')
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            Unique cache key string
        """
        # Create a string from all arguments
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        key_string = ":".join(key_parts)
        
        # Hash long keys
        if len(key_string) > 200:
            hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:16]
            key_string = f"{prefix}:{hash_suffix}"
        
        return key_string
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit: {key}")
            return value
        except Exception as e:
            logger.warning(f"Cache get error: {str(e)}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
            
        Returns:
            True if successful
        """
        try:
            if ttl is None:
                ttl = settings.get_cache_ttl_seconds()
            
            self.cache.set(key, value, expire=ttl)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            self.cache.delete(key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {str(e)}")
            return False
    
    # GSC-specific methods
    
    def get_gsc_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        data_type: str = "query_page"
    ) -> Optional[Dict]:
        """
        Get cached GSC data.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            data_type: Type of data (query_page, queries, pages, etc.)
            
        Returns:
            Cached data or None
        """
        key = self._generate_key(
            "gsc",
            property_url,
            start_date,
            end_date,
            data_type=data_type
        )
        return self.get(key)
    
    def set_gsc_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        data: Dict,
        data_type: str = "query_page"
    ) -> bool:
        """
        Cache GSC data.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            data: Data to cache
            data_type: Type of data
            
        Returns:
            True if successful
        """
        key = self._generate_key(
            "gsc",
            property_url,
            start_date,
            end_date,
            data_type=data_type
        )
        ttl = settings.get_cache_ttl_seconds("gsc")
        return self.set(key, data, ttl)
    
    # DataForSEO-specific methods
    
    def get_dataforseo_data(
        self,
        endpoint: str,
        target: str,
        **params
    ) -> Optional[Dict]:
        """
        Get cached DataForSEO data.
        
        Args:
            endpoint: API endpoint name
            target: Target domain or URL
            **params: Additional parameters
            
        Returns:
            Cached data or None
        """
        key = self._generate_key("dataforseo", endpoint, target, **params)
        return self.get(key)
    
    def set_dataforseo_data(
        self,
        endpoint: str,
        target: str,
        data: Dict,
        **params
    ) -> bool:
        """
        Cache DataForSEO data.
        
        Args:
            endpoint: API endpoint name
            target: Target domain or URL
            data: Data to cache
            **params: Additional parameters
            
        Returns:
            True if successful
        """
        key = self._generate_key("dataforseo", endpoint, target, **params)
        ttl = settings.get_cache_ttl_seconds("dataforseo")
        return self.set(key, data, ttl)
    
    def get_search_volumes_batch(
        self,
        keywords: List[str]
    ) -> Dict[str, int]:
        """
        Get cached search volumes for multiple keywords.
        
        Args:
            keywords: List of keywords
            
        Returns:
            Dict of keyword -> volume for cached keywords
        """
        cached = {}
        
        for kw in keywords:
            kw_lower = kw.lower().strip()
            key = self._generate_key("search_volume", kw_lower)
            volume = self.get(key)
            if volume is not None:
                cached[kw_lower] = volume
        
        return cached
    
    def set_search_volumes_batch(
        self,
        volumes: Dict[str, int]
    ) -> bool:
        """
        Cache search volumes for multiple keywords.
        
        Args:
            volumes: Dict of keyword -> volume
            
        Returns:
            True if all successful
        """
        ttl = settings.get_cache_ttl_seconds("dataforseo")
        success = True
        
        for kw, volume in volumes.items():
            kw_lower = kw.lower().strip()
            key = self._generate_key("search_volume", kw_lower)
            if not self.set(key, volume, ttl):
                success = False
        
        return success
    
    # Cache management
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear cache entries matching a pattern prefix.
        
        Args:
            pattern: Key prefix pattern
            
        Returns:
            Number of entries cleared
        """
        count = 0
        try:
            for key in self.cache:
                if key.startswith(pattern):
                    self.cache.delete(key)
                    count += 1
            logger.info(f"Cleared {count} cache entries matching '{pattern}'")
        except Exception as e:
            logger.warning(f"Error clearing cache pattern: {str(e)}")
        return count
    
    def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        try:
            self.cache.clear()
            logger.info("All cache entries cleared")
            return True
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        try:
            return {
                "size": len(self.cache),
                "volume": self.cache.volume(),
                "directory": str(self.cache_dir)
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {str(e)}")
            return {}
    
    def close(self):
        """Close the cache connection"""
        try:
            self.cache.close()
        except Exception:
            pass


# Singleton instance
cache_service = CacheService()