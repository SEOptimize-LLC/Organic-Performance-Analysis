"""
Input validation utilities.
"""

import re
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import validators as url_validators


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"
    
    # Check if it's a valid URL
    if not url_validators.url(url):
        return False, "Invalid URL format"
    
    return True, None


def validate_domain(domain: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a domain name.
    
    Args:
        domain: Domain to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not domain:
        return False, "Domain cannot be empty"
    
    # Remove protocol if present
    domain = re.sub(r'^https?://', '', domain)
    
    # Remove trailing slash and path
    domain = domain.split('/')[0]
    
    # Check domain format
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.'
    domain_pattern += r'[a-zA-Z]{2,}$'
    
    if not re.match(domain_pattern, domain):
        # Check if it's a valid domain with subdomain
        if not url_validators.domain(domain):
            return False, "Invalid domain format"
    
    return True, None


def validate_date_range(
    start_date: str,
    end_date: str,
    max_days: int = 540
) -> Tuple[bool, Optional[str]]:
    """
    Validate a date range for GSC queries.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_days: Maximum allowed range in days
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    date_format = '%Y-%m-%d'
    
    try:
        start = datetime.strptime(start_date, date_format)
        end = datetime.strptime(end_date, date_format)
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"
    
    # Check start is before end
    if start > end:
        return False, "Start date must be before end date"
    
    # Check range doesn't exceed maximum
    delta = (end - start).days
    if delta > max_days:
        return False, f"Date range exceeds maximum of {max_days} days"
    
    # Check dates are not in the future
    today = datetime.now()
    if end > today:
        return False, "End date cannot be in the future"
    
    # GSC data has a 3-day delay
    min_end = today - timedelta(days=3)
    if end > min_end:
        return False, "GSC data has a 3-day delay"
    
    return True, None


def validate_brand_terms(terms: str) -> List[str]:
    """
    Validate and parse brand terms input.
    
    Args:
        terms: Comma or newline separated brand terms
        
    Returns:
        List of cleaned brand terms
    """
    if not terms:
        return []
    
    # Split by comma or newline
    parts = re.split(r'[,\n]', terms)
    
    # Clean each term
    cleaned = []
    for part in parts:
        term = part.strip().lower()
        if term and len(term) >= 2:
            cleaned.append(term)
    
    return cleaned


def validate_competitor_domains(domains: str) -> List[str]:
    """
    Validate and parse competitor domain input.
    
    Args:
        domains: Comma or newline separated domains
        
    Returns:
        List of valid domains
    """
    if not domains:
        return []
    
    # Split by comma or newline
    parts = re.split(r'[,\n]', domains)
    
    # Validate each domain
    valid_domains = []
    for part in parts:
        domain = part.strip().lower()
        
        # Remove protocol if present
        domain = re.sub(r'^https?://', '', domain)
        
        # Remove trailing slash and path
        domain = domain.split('/')[0]
        
        if domain:
            is_valid, _ = validate_domain(domain)
            if is_valid:
                valid_domains.append(domain)
    
    return valid_domains


def validate_gsc_property(property_url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a GSC property URL.
    
    Args:
        property_url: GSC property URL
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not property_url:
        return False, "Property URL cannot be empty"
    
    # GSC properties can be:
    # - Domain property: sc-domain:example.com
    # - URL prefix: https://example.com/
    
    if property_url.startswith('sc-domain:'):
        # Validate domain property
        domain = property_url.replace('sc-domain:', '')
        return validate_domain(domain)
    else:
        # Validate URL prefix
        return validate_url(property_url)


def validate_min_impressions(value: int) -> Tuple[bool, Optional[str]]:
    """
    Validate minimum impressions threshold.
    
    Args:
        value: Minimum impressions value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < 0:
        return False, "Minimum impressions cannot be negative"
    
    if value > 10000:
        return False, "Minimum impressions seems too high (max 10,000)"
    
    return True, None


def validate_position_range(
    min_pos: float,
    max_pos: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate a position range.
    
    Args:
        min_pos: Minimum position
        max_pos: Maximum position
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if min_pos < 1:
        return False, "Minimum position must be at least 1"
    
    if max_pos > 100:
        return False, "Maximum position cannot exceed 100"
    
    if min_pos > max_pos:
        return False, "Minimum position must be less than maximum"
    
    return True, None