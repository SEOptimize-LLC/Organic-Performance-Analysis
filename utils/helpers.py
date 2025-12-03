"""
Helper functions for data processing and formatting.
"""

import re
from typing import List, Any, Optional, Generator
from datetime import datetime, timedelta


def batch_list(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Split a list into batches of specified size.
    
    Args:
        items: List to split
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_keyword(keyword: str) -> str:
    """
    Normalize a keyword for consistent matching.
    
    Args:
        keyword: Raw keyword string
        
    Returns:
        Normalized keyword (lowercase, stripped, single spaces)
    """
    if not keyword:
        return ""
    
    # Lowercase
    keyword = keyword.lower()
    
    # Strip whitespace
    keyword = keyword.strip()
    
    # Replace multiple spaces with single space
    keyword = re.sub(r'\s+', ' ', keyword)
    
    return keyword


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value
    """
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def format_number(number: float, decimals: int = 0) -> str:
    """
    Format a number with thousands separators.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if decimals > 0:
        return f"{number:,.{decimals}f}"
    return f"{number:,.0f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal as percentage string.
    
    Args:
        value: Decimal value (0.15 = 15%)
        decimals: Number of decimal places
        
    Returns:
        Percentage string with % symbol
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_percentage_change(
    old_value: float,
    new_value: float
) -> Optional[float]:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change or None if calculation not possible
    """
    if old_value == 0:
        if new_value == 0:
            return 0.0
        return None  # Can't calculate change from zero
    
    return ((new_value - old_value) / old_value) * 100


def get_date_range(
    days: int,
    end_offset: int = 3
) -> tuple:
    """
    Calculate date range for GSC queries.
    
    Args:
        days: Number of days to look back
        end_offset: Days to subtract from today (GSC data delay)
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now() - timedelta(days=end_offset)
    start_date = end_date - timedelta(days=days)
    
    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def get_yoy_date_range(
    days: int,
    end_offset: int = 3
) -> tuple:
    """
    Calculate year-over-year date range for comparison.
    
    Args:
        days: Number of days to look back
        end_offset: Days to subtract from today
        
    Returns:
        Tuple of (start_date, end_date) for same period last year
    """
    end_date = datetime.now() - timedelta(days=end_offset) - timedelta(days=365)
    start_date = end_date - timedelta(days=days)
    
    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def extract_domain(url: str) -> str:
    """
    Extract domain from a URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain without protocol or path
    """
    if not url:
        return ""
    
    # Remove protocol
    domain = re.sub(r'^https?://', '', url)
    
    # Remove path and query string
    domain = domain.split('/')[0]
    
    # Remove port
    domain = domain.split(':')[0]
    
    return domain.lower()


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def deduplicate_keywords(keywords: List[str]) -> List[str]:
    """
    Remove duplicate keywords while preserving order.
    
    Args:
        keywords: List of keywords
        
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    
    for kw in keywords:
        normalized = normalize_keyword(kw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(kw)
    
    return result


def classify_url_type(url: str) -> str:
    """
    Classify URL type based on common patterns.
    
    Args:
        url: URL to classify
        
    Returns:
        URL type (blog, product, category, etc.)
    """
    url_lower = url.lower()
    
    # Blog patterns
    if any(p in url_lower for p in ['/blog/', '/article/', '/post/', '/news/']):
        return 'blog'
    
    # Product patterns
    if any(p in url_lower for p in ['/product/', '/item/', '/p/', '/pdp/']):
        return 'product'
    
    # Category patterns
    if any(p in url_lower for p in ['/category/', '/c/', '/collection/']):
        return 'category'
    
    # Service patterns
    if any(p in url_lower for p in ['/service/', '/services/']):
        return 'service'
    
    # About/Contact patterns
    if any(p in url_lower for p in ['/about', '/contact', '/team']):
        return 'about'
    
    # Homepage
    if url_lower.rstrip('/').count('/') <= 2:
        return 'homepage'
    
    return 'other'