"""
Data normalization and joining module.
Combines GSC and DataForSEO data into unified views.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

from config.settings import settings
from utils.logger import logger
from utils.helpers import safe_float, safe_int, normalize_keyword


class DataNormalizer:
    """
    Normalizes and joins data from GSC and DataForSEO.
    Creates unified datasets for analysis.
    """
    
    def __init__(self, brand_terms: List[str] = None):
        """
        Initialize normalizer.
        
        Args:
            brand_terms: List of brand term patterns
        """
        self.brand_terms = brand_terms or []
        self.brand_pattern = self._compile_brand_pattern()
    
    def _compile_brand_pattern(self) -> Optional[re.Pattern]:
        """Compile brand terms into regex pattern."""
        if not self.brand_terms:
            return None
        
        # Escape special regex chars and join with OR
        escaped = [re.escape(term.lower()) for term in self.brand_terms]
        pattern = '|'.join(escaped)
        return re.compile(pattern, re.IGNORECASE)
    
    def set_brand_terms(self, brand_terms: List[str]):
        """Update brand terms."""
        self.brand_terms = brand_terms
        self.brand_pattern = self._compile_brand_pattern()
    
    def is_brand_query(self, query: str) -> bool:
        """Check if query contains brand terms."""
        if not self.brand_pattern:
            return False
        return bool(self.brand_pattern.search(query.lower()))
    
    def normalize_gsc_data(
        self,
        gsc_data: pd.DataFrame,
        data_type: str = 'queries'
    ) -> pd.DataFrame:
        """
        Normalize GSC data for consistent analysis.
        
        Args:
            gsc_data: Raw GSC DataFrame
            data_type: Type of data (queries, pages, etc.)
            
        Returns:
            Normalized DataFrame
        """
        if gsc_data.empty:
            return gsc_data
        
        df = gsc_data.copy()
        
        # Normalize query column
        if 'query' in df.columns:
            df['query_normalized'] = df['query'].apply(normalize_keyword)
            df['is_brand'] = df['query'].apply(self.is_brand_query)
            df['query_word_count'] = df['query'].str.split().str.len()
        
        # Normalize URL column
        if 'page' in df.columns:
            df['page_normalized'] = df['page'].apply(self._normalize_url)
            df['page_path'] = df['page'].apply(self._extract_path)
        
        # Ensure numeric types
        numeric_cols = ['clicks', 'impressions', 'ctr', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Round CTR to 4 decimals
        if 'ctr' in df.columns:
            df['ctr'] = df['ctr'].round(4)
        
        # Round position to 2 decimals
        if 'position' in df.columns:
            df['position'] = df['position'].round(2)
        
        return df
    
    def normalize_dataforseo_keywords(
        self,
        df_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize DataForSEO keyword data.
        
        Args:
            df_data: Raw DataForSEO DataFrame
            
        Returns:
            Normalized DataFrame
        """
        if df_data.empty:
            return df_data
        
        df = df_data.copy()
        
        # Normalize keyword column
        if 'keyword' in df.columns:
            df['keyword_normalized'] = df['keyword'].apply(normalize_keyword)
            df['is_brand'] = df['keyword'].apply(self.is_brand_query)
        
        # Normalize URL column
        if 'url' in df.columns:
            df['url_normalized'] = df['url'].apply(self._normalize_url)
            df['url_path'] = df['url'].apply(self._extract_path)
        
        # Ensure numeric types
        numeric_cols = [
            'search_volume', 'cpc', 'competition',
            'traffic', 'position', 'keyword_difficulty'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url:
            return ''
        
        try:
            parsed = urlparse(url)
            # Lowercase domain, remove www, trailing slash
            domain = parsed.netloc.lower().replace('www.', '')
            path = parsed.path.rstrip('/')
            return f"{domain}{path}"
        except Exception:
            return url.lower()
    
    def _extract_path(self, url: str) -> str:
        """Extract path from URL."""
        if not url:
            return ''
        
        try:
            parsed = urlparse(url)
            return parsed.path.rstrip('/')
        except Exception:
            return ''
    
    def join_gsc_dataforseo(
        self,
        gsc_queries: pd.DataFrame,
        dataforseo_keywords: pd.DataFrame,
        join_type: str = 'outer'
    ) -> pd.DataFrame:
        """
        Join GSC and DataForSEO data on normalized keywords.
        
        Args:
            gsc_queries: Normalized GSC query data
            dataforseo_keywords: Normalized DataForSEO data
            join_type: Type of join (outer, inner, left, right)
            
        Returns:
            Joined DataFrame
        """
        if gsc_queries.empty and dataforseo_keywords.empty:
            return pd.DataFrame()
        
        if gsc_queries.empty:
            return dataforseo_keywords
        
        if dataforseo_keywords.empty:
            return gsc_queries
        
        # Ensure normalized columns exist
        gsc = self.normalize_gsc_data(gsc_queries)
        dfs = self.normalize_dataforseo_keywords(dataforseo_keywords)
        
        # Prepare columns for merge
        gsc_cols = ['query', 'query_normalized', 'clicks', 'impressions',
                    'ctr', 'position', 'is_brand']
        gsc_subset = gsc[[c for c in gsc_cols if c in gsc.columns]].copy()
        gsc_subset = gsc_subset.add_prefix('gsc_')
        gsc_subset = gsc_subset.rename(
            columns={'gsc_query_normalized': 'keyword_normalized'}
        )
        
        dfs_cols = ['keyword', 'keyword_normalized', 'search_volume',
                    'cpc', 'competition', 'traffic', 'position', 'url']
        dfs_subset = dfs[[c for c in dfs_cols if c in dfs.columns]].copy()
        dfs_subset = dfs_subset.add_prefix('dfs_')
        dfs_subset = dfs_subset.rename(
            columns={'dfs_keyword_normalized': 'keyword_normalized'}
        )
        
        # Merge on normalized keyword
        merged = pd.merge(
            gsc_subset,
            dfs_subset,
            on='keyword_normalized',
            how=join_type
        )
        
        # Calculate enrichment metrics
        merged = self._calculate_enrichment_metrics(merged)
        
        return merged
    
    def _calculate_enrichment_metrics(
        self,
        merged: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate additional metrics from merged data."""
        if merged.empty:
            return merged
        
        df = merged.copy()
        
        # Position comparison
        if 'gsc_position' in df.columns and 'dfs_position' in df.columns:
            df['position_diff'] = df['dfs_position'] - df['gsc_position']
        
        # CTR performance ratio
        if 'gsc_ctr' in df.columns and 'gsc_position' in df.columns:
            df['expected_ctr'] = df['gsc_position'].apply(
                lambda p: settings.expected_ctr_by_position.get(
                    int(min(p, 20)), 0.01
                ) if p > 0 else 0
            )
            df['ctr_performance'] = df.apply(
                lambda r: r['gsc_ctr'] / r['expected_ctr']
                if r['expected_ctr'] > 0 else 0,
                axis=1
            )
        
        # Traffic potential
        if ('dfs_search_volume' in df.columns and
                'gsc_ctr' in df.columns):
            df['traffic_potential'] = (
                df['dfs_search_volume'] * df['gsc_ctr']
            )
        
        # Keyword opportunity score components
        if 'dfs_search_volume' in df.columns:
            df['volume_bucket'] = pd.cut(
                df['dfs_search_volume'],
                bins=[0, 100, 500, 1000, 5000, 10000, float('inf')],
                labels=['very_low', 'low', 'medium', 'high',
                        'very_high', 'exceptional']
            )
        
        return df
    
    def join_page_query_data(
        self,
        page_data: pd.DataFrame,
        query_page_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create page portfolios with aggregated query metrics.
        
        Args:
            page_data: Page-level GSC data
            query_page_data: Query+page GSC data
            
        Returns:
            DataFrame with page portfolios
        """
        if page_data.empty or query_page_data.empty:
            return page_data
        
        # Aggregate queries per page
        query_agg = query_page_data.groupby('page').agg({
            'query': 'count',
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        query_agg.columns = ['page', 'query_count', 'query_clicks',
                             'query_impressions', 'avg_query_position']
        
        # Merge with page data
        merged = pd.merge(
            page_data,
            query_agg,
            on='page',
            how='left'
        )
        
        return merged
    
    def calculate_yoy_changes(
        self,
        current_data: pd.DataFrame,
        yoy_data: pd.DataFrame,
        key_column: str = 'query'
    ) -> pd.DataFrame:
        """
        Calculate year-over-year changes.
        
        Args:
            current_data: Current period data
            yoy_data: Same period last year data
            key_column: Column to join on
            
        Returns:
            DataFrame with YoY metrics
        """
        if current_data.empty or yoy_data.empty:
            return current_data
        
        # Prepare YoY data with suffix
        yoy_subset = yoy_data[[
            key_column, 'clicks', 'impressions', 'position', 'ctr'
        ]].copy()
        yoy_subset.columns = [
            key_column, 'yoy_clicks', 'yoy_impressions',
            'yoy_position', 'yoy_ctr'
        ]
        
        # Merge
        merged = pd.merge(
            current_data,
            yoy_subset,
            on=key_column,
            how='left'
        )
        
        # Calculate changes
        merged['clicks_change'] = (
            merged['clicks'] - merged['yoy_clicks']
        )
        merged['clicks_change_pct'] = merged.apply(
            lambda r: (r['clicks_change'] / r['yoy_clicks'] * 100)
            if r['yoy_clicks'] > 0 else 0,
            axis=1
        )
        
        merged['impressions_change'] = (
            merged['impressions'] - merged['yoy_impressions']
        )
        merged['impressions_change_pct'] = merged.apply(
            lambda r: (r['impressions_change'] / r['yoy_impressions'] * 100)
            if r['yoy_impressions'] > 0 else 0,
            axis=1
        )
        
        merged['position_change'] = (
            merged['yoy_position'] - merged['position']
        )  # Positive = improved
        
        merged['ctr_change'] = merged['ctr'] - merged['yoy_ctr']
        
        return merged
    
    def segment_by_device(
        self,
        device_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment data by device type.
        
        Args:
            device_data: GSC device data
            
        Returns:
            Dict with device-segmented DataFrames
        """
        if device_data.empty or 'device' not in device_data.columns:
            return {'all': device_data}
        
        segments = {}
        for device in device_data['device'].unique():
            segments[device.lower()] = device_data[
                device_data['device'] == device
            ].copy()
        
        return segments
    
    def segment_by_country(
        self,
        country_data: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment data by country.
        
        Args:
            country_data: GSC country data
            top_n: Number of top countries to return
            
        Returns:
            Dict with country-segmented DataFrames
        """
        if country_data.empty or 'country' not in country_data.columns:
            return {'all': country_data}
        
        # Get top countries by clicks
        top_countries = (
            country_data.groupby('country')['clicks']
            .sum()
            .nlargest(top_n)
            .index.tolist()
        )
        
        segments = {}
        for country in top_countries:
            segments[country] = country_data[
                country_data['country'] == country
            ].copy()
        
        return segments
    
    def create_unified_dataset(
        self,
        gsc_data: Dict[str, pd.DataFrame],
        dataforseo_data: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """
        Create unified dataset from all sources.
        
        Args:
            gsc_data: Dict of GSC DataFrames
            dataforseo_data: Dict of DataForSEO data
            
        Returns:
            Unified dataset dict
        """
        unified = {}
        
        # Normalize GSC queries
        if 'queries' in gsc_data:
            unified['gsc_queries'] = self.normalize_gsc_data(
                gsc_data['queries'], 'queries'
            )
        
        # Normalize GSC pages
        if 'pages' in gsc_data:
            unified['gsc_pages'] = self.normalize_gsc_data(
                gsc_data['pages'], 'pages'
            )
        
        # Normalize DataForSEO keywords
        if 'ranked_keywords' in dataforseo_data:
            rk = dataforseo_data['ranked_keywords']
            if isinstance(rk, pd.DataFrame):
                unified['dfs_keywords'] = self.normalize_dataforseo_keywords(
                    rk
                )
        
        # Create joined keyword dataset
        if 'gsc_queries' in unified and 'dfs_keywords' in unified:
            unified['keywords_joined'] = self.join_gsc_dataforseo(
                unified['gsc_queries'],
                unified['dfs_keywords']
            )
        
        # Add competitors if available
        if 'competitors' in dataforseo_data:
            comp = dataforseo_data['competitors']
            if isinstance(comp, pd.DataFrame):
                unified['competitors'] = comp
        
        return unified