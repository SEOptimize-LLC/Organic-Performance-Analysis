"""
Google Search Console data collector.
Handles comprehensive data collection with multi-window analysis and YoY comparison.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import time

from services.auth_service import AuthService
from services.cache_service import cache_service
from services.rate_limiter import rate_limiter, retry_handler
from config.settings import settings
from utils.logger import logger
from utils.helpers import safe_float, safe_int


class GSCCollector:
    """
    Comprehensive GSC data collection with multi-window support.
    Handles pagination, caching, and error recovery.
    """
    
    # Rows per request (GSC API limit)
    ROWS_PER_REQUEST = 25000
    
    # Available dimensions
    DIMENSIONS = {
        'queries': ['query'],
        'pages': ['page'],
        'query_page': ['query', 'page'],
        'page_device': ['page', 'device'],
        'page_country': ['page', 'country'],
        'page_trends': ['date', 'page'],
        'query_trends': ['date', 'query'],
        'search_appearance': ['searchAppearance'],
        'full': ['query', 'page', 'device', 'country']
    }
    
    def __init__(self, auth_service: Optional[AuthService] = None):
        """
        Initialize GSC collector.
        
        Args:
            auth_service: Optional auth service instance
        """
        self.auth_service = auth_service or AuthService()
        self.service = None
    
    def _get_service(self):
        """Get or create GSC service"""
        if self.service:
            return self.service
        
        if st.session_state.get('gsc_service'):
            self.service = st.session_state.gsc_service
            return self.service
        
        self.service = self.auth_service.get_gsc_service()
        return self.service
    
    def list_properties(self) -> List[str]:
        """
        List all available GSC properties.
        
        Returns:
            List of property URLs
        """
        service = self._get_service()
        if not service:
            logger.error("GSC service not available")
            return []
        
        try:
            response = service.sites().list().execute()
            properties = response.get('siteEntry', [])
            return [prop['siteUrl'] for prop in properties]
        except HttpError as e:
            logger.error(f"Error listing properties: {str(e)}")
            return []
    
    def validate_property_access(self, property_url: str) -> bool:
        """
        Validate access to a specific property.
        
        Args:
            property_url: GSC property URL
            
        Returns:
            True if access is valid
        """
        properties = self.list_properties()
        return property_url in properties
    
    def get_date_range(
        self,
        days: int,
        end_offset: int = None
    ) -> Tuple[str, str]:
        """
        Calculate date range for queries.
        
        Args:
            days: Number of days to look back
            end_offset: Days to subtract from today (default: GSC delay)
            
        Returns:
            Tuple of (start_date, end_date) as strings
        """
        if end_offset is None:
            end_offset = settings.gsc_data_freshness_delay
        
        end_date = datetime.now() - timedelta(days=end_offset)
        start_date = end_date - timedelta(days=days)
        
        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    def get_yoy_date_range(
        self,
        days: int,
        end_offset: int = None
    ) -> Tuple[str, str]:
        """
        Calculate year-over-year date range.
        
        Args:
            days: Number of days
            end_offset: Days offset
            
        Returns:
            Tuple of (start_date, end_date) for same period last year
        """
        if end_offset is None:
            end_offset = settings.gsc_data_freshness_delay
        
        end_date = datetime.now() - timedelta(days=end_offset + 365)
        start_date = end_date - timedelta(days=days)
        
        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    @rate_limiter.limit_gsc
    def _execute_query(
        self,
        property_url: str,
        body: Dict
    ) -> Optional[Dict]:
        """
        Execute a GSC query with rate limiting.
        
        Args:
            property_url: GSC property URL
            body: Request body
            
        Returns:
            API response or None
        """
        service = self._get_service()
        if not service:
            return None
        
        try:
            return service.searchanalytics().query(
                siteUrl=property_url,
                body=body
            ).execute()
        except HttpError as e:
            if e.resp.status == 429:
                logger.warning("Rate limit hit, waiting...")
                time.sleep(10)
                return self._execute_query(property_url, body)
            logger.error(f"GSC query error: {str(e)}")
            raise
    
    def get_search_analytics(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        dimensions: List[str],
        row_limit: int = 25000,
        filters: Optional[List[Dict]] = None,
        data_type: str = "web"
    ) -> pd.DataFrame:
        """
        Get search analytics data with pagination.
        
        Args:
            property_url: GSC property URL
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dimensions: List of dimensions
            row_limit: Maximum rows to retrieve
            filters: Optional dimension filters
            data_type: Search type (web, image, video)
            
        Returns:
            DataFrame with search analytics data
        """
        # Check cache first
        cache_key_parts = f"{dimensions}_{data_type}"
        cached = cache_service.get_gsc_data(
            property_url, start_date, end_date, cache_key_parts
        )
        if cached is not None:
            logger.info("Using cached GSC data")
            return pd.DataFrame(cached)
        
        all_rows = []
        start_row = 0
        
        while len(all_rows) < row_limit:
            # Build request body
            body = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': dimensions,
                'rowLimit': min(self.ROWS_PER_REQUEST, row_limit - len(all_rows)),
                'startRow': start_row,
                'type': data_type
            }
            
            if filters:
                body['dimensionFilterGroups'] = [{'filters': filters}]
            
            # Execute request
            try:
                response = retry_handler.execute_with_retry(
                    self._execute_query,
                    property_url,
                    body
                )
            except Exception as e:
                logger.error(f"Failed to fetch GSC data: {str(e)}")
                break
            
            if not response:
                break
            
            # Process response
            rows = response.get('rows', [])
            if not rows:
                break
            
            all_rows.extend(rows)
            
            logger.info(f"Fetched {len(all_rows)} rows from GSC")
            
            # Check if we have all data
            if len(rows) < self.ROWS_PER_REQUEST:
                break
            
            start_row += len(rows)
        
        # Convert to DataFrame
        if not all_rows:
            return pd.DataFrame()
        
        df = self._process_rows(all_rows, dimensions)
        
        # Cache results
        cache_service.set_gsc_data(
            property_url, start_date, end_date,
            df.to_dict('records'), cache_key_parts
        )
        
        return df
    
    def _process_rows(
        self,
        rows: List[Dict],
        dimensions: List[str]
    ) -> pd.DataFrame:
        """
        Process GSC API rows into DataFrame.
        
        Args:
            rows: List of API response rows
            dimensions: List of dimensions
            
        Returns:
            Processed DataFrame
        """
        processed = []
        
        for row in rows:
            try:
                keys = row.get('keys', [])
                record = {
                    'clicks': safe_int(row.get('clicks', 0)),
                    'impressions': safe_int(row.get('impressions', 0)),
                    'ctr': safe_float(row.get('ctr', 0)),
                    'position': safe_float(row.get('position', 0))
                }
                
                # Add dimension values
                for i, dim in enumerate(dimensions):
                    if i < len(keys):
                        record[dim] = keys[i]
                
                processed.append(record)
            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue
        
        return pd.DataFrame(processed)
    
    def get_query_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        min_impressions: int = None
    ) -> pd.DataFrame:
        """
        Get query-level data.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            min_impressions: Minimum impressions filter
            
        Returns:
            DataFrame with query data
        """
        df = self.get_search_analytics(
            property_url,
            start_date,
            end_date,
            dimensions=['query'],
            row_limit=25000
        )
        
        if min_impressions and not df.empty:
            df = df[df['impressions'] >= min_impressions]
        
        return df
    
    def get_page_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        min_impressions: int = None
    ) -> pd.DataFrame:
        """
        Get page-level data.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            min_impressions: Minimum impressions filter
            
        Returns:
            DataFrame with page data
        """
        df = self.get_search_analytics(
            property_url,
            start_date,
            end_date,
            dimensions=['page'],
            row_limit=25000
        )
        
        if min_impressions and not df.empty:
            df = df[df['impressions'] >= min_impressions]
        
        return df
    
    def get_query_page_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str,
        min_impressions: int = None
    ) -> pd.DataFrame:
        """
        Get query + page combination data.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            min_impressions: Minimum impressions filter
            
        Returns:
            DataFrame with query-page data
        """
        df = self.get_search_analytics(
            property_url,
            start_date,
            end_date,
            dimensions=['query', 'page'],
            row_limit=50000
        )
        
        if min_impressions and not df.empty:
            df = df[df['impressions'] >= min_impressions]
        
        return df
    
    def get_device_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get page + device data for mobile vs desktop analysis.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with device data
        """
        return self.get_search_analytics(
            property_url,
            start_date,
            end_date,
            dimensions=['page', 'device'],
            row_limit=25000
        )
    
    def get_country_data(
        self,
        property_url: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get page + country data for geo analysis.
        
        Args:
            property_url: GSC property URL
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with country data
        """
        return self.get_search_analytics(
            property_url,
            start_date,
            end_date,
            dimensions=['page', 'country'],
            row_limit=25000
        )
    
    def get_multi_window_data(
        self,
        property_url: str,
        windows: List[str] = None,
        include_yoy: bool = True,
        min_impressions: int = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data across multiple time windows.
        
        Args:
            property_url: GSC property URL
            windows: List of window keys (28d, 90d, etc.)
            include_yoy: Include year-over-year data
            min_impressions: Minimum impressions filter
            
        Returns:
            Nested dict: {window: {data_type: DataFrame}}
        """
        if windows is None:
            windows = list(settings.date_windows.keys())
        
        if min_impressions is None:
            min_impressions = settings.min_impressions_default
        
        results = {}
        
        for window in windows:
            window_config = settings.date_windows.get(window)
            if not window_config:
                continue
            
            days = window_config['days']
            
            logger.info(f"Collecting data for {window_config['label']}")
            
            # Current period
            start_date, end_date = self.get_date_range(days)
            
            results[window] = {
                'queries': self.get_query_data(
                    property_url, start_date, end_date, min_impressions
                ),
                'pages': self.get_page_data(
                    property_url, start_date, end_date, min_impressions
                ),
                'query_page': self.get_query_page_data(
                    property_url, start_date, end_date, min_impressions
                ),
                'device': self.get_device_data(
                    property_url, start_date, end_date
                ),
                'country': self.get_country_data(
                    property_url, start_date, end_date
                )
            }
            
            # Year-over-year data
            if include_yoy:
                yoy_start, yoy_end = self.get_yoy_date_range(days)
                
                results[f"{window}_yoy"] = {
                    'queries': self.get_query_data(
                        property_url, yoy_start, yoy_end, min_impressions
                    ),
                    'pages': self.get_page_data(
                        property_url, yoy_start, yoy_end, min_impressions
                    ),
                    'query_page': self.get_query_page_data(
                        property_url, yoy_start, yoy_end, min_impressions
                    )
                }
        
        return results
    
    def get_comprehensive_data(
        self,
        property_url: str,
        days: int = 90,
        min_impressions: int = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive data for a single time window.
        
        Args:
            property_url: GSC property URL
            days: Number of days
            min_impressions: Minimum impressions filter
            
        Returns:
            Dict with all data types
        """
        start_date, end_date = self.get_date_range(days)
        
        if min_impressions is None:
            min_impressions = settings.min_impressions_default
        
        return {
            'property_url': property_url,
            'date_range': {
                'start': start_date,
                'end': end_date,
                'days': days
            },
            'queries': self.get_query_data(
                property_url, start_date, end_date, min_impressions
            ),
            'pages': self.get_page_data(
                property_url, start_date, end_date, min_impressions
            ),
            'query_page': self.get_query_page_data(
                property_url, start_date, end_date, min_impressions
            ),
            'device': self.get_device_data(
                property_url, start_date, end_date
            ),
            'country': self.get_country_data(
                property_url, start_date, end_date
            )
        }