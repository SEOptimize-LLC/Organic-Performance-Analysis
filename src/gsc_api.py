"""
Google Search Console API Integration Module

Handles OAuth authentication and data fetching from Google Search Console.
Supports query-level, page-level, and search appearance data exports.
"""

import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Optional, List, Dict, Any
import json


# OAuth scopes required for GSC
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']


class GSCClient:
    """Google Search Console API Client"""

    def __init__(self, credentials: Credentials):
        """Initialize the GSC client with credentials."""
        self.credentials = credentials
        self.service = build('searchconsole', 'v1', credentials=credentials)

    def get_sites(self) -> List[Dict[str, str]]:
        """Get list of sites the user has access to."""
        try:
            site_list = self.service.sites().list().execute()
            return site_list.get('siteEntry', [])
        except Exception as e:
            st.error(f"Error fetching sites: {str(e)}")
            return []

    def get_search_analytics(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        dimensions: List[str],
        row_limit: int = 25000,
        start_row: int = 0,
        dimension_filter_groups: Optional[List[Dict]] = None,
        search_type: str = 'web'
    ) -> pd.DataFrame:
        """
        Fetch search analytics data from GSC.

        Args:
            site_url: The site URL (property) to query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            dimensions: List of dimensions (query, page, date, country, device, searchAppearance)
            row_limit: Maximum rows to return per request
            start_row: Starting row for pagination
            dimension_filter_groups: Optional filters
            search_type: Type of search (web, image, video, news)

        Returns:
            DataFrame with search analytics data
        """
        request_body = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions,
            'rowLimit': row_limit,
            'startRow': start_row,
            'searchType': search_type
        }

        if dimension_filter_groups:
            request_body['dimensionFilterGroups'] = dimension_filter_groups

        all_rows = []
        current_row = start_row

        while True:
            request_body['startRow'] = current_row

            try:
                response = self.service.searchanalytics().query(
                    siteUrl=site_url,
                    body=request_body
                ).execute()

                rows = response.get('rows', [])
                if not rows:
                    break

                all_rows.extend(rows)
                current_row += len(rows)

                # Stop if we got fewer rows than requested (end of data)
                if len(rows) < row_limit:
                    break

            except Exception as e:
                st.error(f"Error fetching search analytics: {str(e)}")
                break

        return self._parse_response(all_rows, dimensions)

    def _parse_response(self, rows: List[Dict], dimensions: List[str]) -> pd.DataFrame:
        """Parse GSC API response into a DataFrame."""
        if not rows:
            return pd.DataFrame()

        data = []
        for row in rows:
            row_data = {}
            keys = row.get('keys', [])
            for i, dim in enumerate(dimensions):
                if i < len(keys):
                    row_data[dim] = keys[i]

            row_data['clicks'] = row.get('clicks', 0)
            row_data['impressions'] = row.get('impressions', 0)
            row_data['ctr'] = row.get('ctr', 0)
            row_data['position'] = row.get('position', 0)

            data.append(row_data)

        return pd.DataFrame(data)

    def get_query_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        min_impressions: int = 10,
        device: Optional[str] = None,
        country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get query-level data with optional device and country filters.

        Args:
            site_url: Site URL to query
            start_date: Start date
            end_date: End date
            min_impressions: Minimum impressions threshold
            device: Optional device filter (MOBILE, DESKTOP, TABLET)
            country: Optional country filter (3-letter country code)

        Returns:
            DataFrame with query data
        """
        dimensions = ['query', 'page', 'device', 'country']

        filters = []
        if device:
            filters.append({
                'dimension': 'device',
                'operator': 'equals',
                'expression': device
            })
        if country:
            filters.append({
                'dimension': 'country',
                'operator': 'equals',
                'expression': country
            })

        dimension_filter_groups = None
        if filters:
            dimension_filter_groups = [{'filters': filters}]

        df = self.get_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            dimension_filter_groups=dimension_filter_groups
        )

        if not df.empty and min_impressions > 0:
            df = df[df['impressions'] >= min_impressions]

        return df

    def get_page_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        min_impressions: int = 10,
        device: Optional[str] = None,
        country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get page-level aggregated data.

        Args:
            site_url: Site URL to query
            start_date: Start date
            end_date: End date
            min_impressions: Minimum impressions threshold
            device: Optional device filter
            country: Optional country filter

        Returns:
            DataFrame with page-level data
        """
        dimensions = ['page']

        filters = []
        if device:
            filters.append({
                'dimension': 'device',
                'operator': 'equals',
                'expression': device
            })
        if country:
            filters.append({
                'dimension': 'country',
                'operator': 'equals',
                'expression': country
            })

        dimension_filter_groups = None
        if filters:
            dimension_filter_groups = [{'filters': filters}]

        df = self.get_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            dimension_filter_groups=dimension_filter_groups
        )

        if not df.empty and min_impressions > 0:
            df = df[df['impressions'] >= min_impressions]

        return df

    def get_search_appearance_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get data broken down by search appearance (rich results, etc.)."""
        dimensions = ['searchAppearance', 'query']

        return self.get_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions
        )

    def get_date_range_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get daily data for trend analysis."""
        dimensions = ['date']

        return self.get_search_analytics(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions
        )


def get_date_ranges() -> Dict[str, tuple]:
    """
    Get predefined date ranges for analysis.

    Returns:
        Dictionary with date range names and (start_date, end_date) tuples
    """
    today = datetime.now()

    return {
        'last_28_days': (
            (today - timedelta(days=28)).strftime('%Y-%m-%d'),
            (today - timedelta(days=1)).strftime('%Y-%m-%d')
        ),
        'last_3_months': (
            (today - relativedelta(months=3)).strftime('%Y-%m-%d'),
            (today - timedelta(days=1)).strftime('%Y-%m-%d')
        ),
        'last_6_months': (
            (today - relativedelta(months=6)).strftime('%Y-%m-%d'),
            (today - timedelta(days=1)).strftime('%Y-%m-%d')
        ),
        'last_12_months': (
            (today - relativedelta(months=12)).strftime('%Y-%m-%d'),
            (today - timedelta(days=1)).strftime('%Y-%m-%d')
        ),
        'previous_year_3_months': (
            (today - relativedelta(months=15)).strftime('%Y-%m-%d'),
            (today - relativedelta(months=12) - timedelta(days=1)).strftime('%Y-%m-%d')
        ),
        'previous_year_same_period': (
            (today - relativedelta(years=1) - timedelta(days=28)).strftime('%Y-%m-%d'),
            (today - relativedelta(years=1) - timedelta(days=1)).strftime('%Y-%m-%d')
        )
    }


def create_oauth_flow(client_id: str, client_secret: str, redirect_uri: str) -> Flow:
    """
    Create OAuth flow for GSC authentication.

    Args:
        client_id: Google OAuth client ID
        client_secret: Google OAuth client secret
        redirect_uri: Redirect URI for OAuth callback

    Returns:
        OAuth Flow object
    """
    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri]
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )

    return flow


def credentials_from_dict(creds_dict: Dict[str, Any]) -> Credentials:
    """Create Credentials object from dictionary."""
    return Credentials(
        token=creds_dict.get('token'),
        refresh_token=creds_dict.get('refresh_token'),
        token_uri=creds_dict.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=creds_dict.get('client_id'),
        client_secret=creds_dict.get('client_secret'),
        scopes=creds_dict.get('scopes', SCOPES)
    )


def credentials_to_dict(credentials: Credentials) -> Dict[str, Any]:
    """Convert Credentials object to dictionary for storage."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': list(credentials.scopes) if credentials.scopes else SCOPES
    }


def classify_brand_queries(
    df: pd.DataFrame,
    brand_terms: List[str],
    query_column: str = 'query'
) -> pd.DataFrame:
    """
    Classify queries as brand or non-brand.

    Args:
        df: DataFrame with query data
        brand_terms: List of brand terms/patterns
        query_column: Name of the query column

    Returns:
        DataFrame with 'is_brand' column added
    """
    if df.empty or not brand_terms:
        df['is_brand'] = False
        return df

    # Create regex pattern from brand terms
    brand_pattern = '|'.join([term.lower() for term in brand_terms])

    df['is_brand'] = df[query_column].str.lower().str.contains(
        brand_pattern,
        regex=True,
        na=False
    )

    return df
