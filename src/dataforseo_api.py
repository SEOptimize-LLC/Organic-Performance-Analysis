"""
DataForSEO API Integration Module

Handles all DataForSEO API calls for SEO analysis including:
- Ranked keywords
- Domain metrics
- Competitor analysis
- SERP data
- Keyword suggestions
"""

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from typing import Optional, List, Dict, Any
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import time


class DataForSEOClient:
    """DataForSEO API Client"""

    BASE_URL = "https://api.dataforseo.com/v3"

    def __init__(self, login: str, password: str):
        """
        Initialize the DataForSEO client.

        Args:
            login: DataForSEO API login (email)
            password: DataForSEO API password
        """
        self.auth = HTTPBasicAuth(login, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request to DataForSEO.

        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request payload

        Returns:
            API response as dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method == 'POST':
                response = self.session.post(url, json=data)
            else:
                response = self.session.get(url)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"DataForSEO API error: {str(e)}")
            return {'status_code': 0, 'tasks': []}

    def get_ranked_keywords(
        self,
        target: str,
        location_code: int = 2840,  # United States
        language_code: str = "en",
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[List[str]] = None,
        filters: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Get ranked keywords for a domain.

        Args:
            target: Domain to analyze (e.g., "example.com")
            location_code: Location code (default: 2840 for US)
            language_code: Language code (default: "en")
            limit: Maximum results to return
            offset: Starting offset for pagination
            order_by: Fields to order by
            filters: Optional filters

        Returns:
            DataFrame with ranked keywords data
        """
        endpoint = "dataforseo_labs/google/ranked_keywords/live"

        payload = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "offset": offset
        }]

        if order_by:
            payload[0]["order_by"] = order_by
        else:
            payload[0]["order_by"] = ["keyword_data.keyword_info.search_volume,desc"]

        if filters:
            payload[0]["filters"] = filters

        response = self._make_request(endpoint, data=payload)
        return self._parse_ranked_keywords_response(response)

    def _parse_ranked_keywords_response(self, response: Dict) -> pd.DataFrame:
        """Parse ranked keywords API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                keyword_data = item.get('keyword_data', {})
                keyword_info = keyword_data.get('keyword_info', {})
                serp_info = keyword_data.get('serp_info', {})
                ranked_serp = item.get('ranked_serp_element', {})
                serp_item = ranked_serp.get('serp_item', {})

                data.append({
                    'keyword': keyword_data.get('keyword', ''),
                    'search_volume': keyword_info.get('search_volume', 0),
                    'cpc': keyword_info.get('cpc', 0),
                    'competition': keyword_info.get('competition', 0),
                    'competition_level': keyword_info.get('competition_level', ''),
                    'position': serp_item.get('rank_absolute', 0),
                    'url': serp_item.get('url', ''),
                    'etv': ranked_serp.get('etv', 0),
                    'is_featured_snippet': serp_item.get('is_featured_snippet', False),
                    'serp_type': serp_info.get('serp_type', ''),
                    'check_url': serp_info.get('check_url', '')
                })

        return pd.DataFrame(data)

    def get_domain_overview(
        self,
        target: str,
        location_code: int = 2840,
        language_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Get domain overview metrics.

        Args:
            target: Domain to analyze
            location_code: Location code
            language_code: Language code

        Returns:
            Dictionary with domain metrics
        """
        endpoint = "dataforseo_labs/google/domain_rank_overview/live"

        payload = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code
        }]

        response = self._make_request(endpoint, data=payload)

        if response.get('status_code') != 20000:
            return {}

        tasks = response.get('tasks', [])
        if not tasks or tasks[0].get('status_code') != 20000:
            return {}

        result = tasks[0].get('result', [])
        if not result:
            return {}

        return result[0].get('items', [{}])[0] if result[0].get('items') else {}

    def get_competitors(
        self,
        target: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get domain competitors based on SERP overlaps.

        Args:
            target: Domain to analyze
            location_code: Location code
            language_code: Language code
            limit: Maximum competitors to return

        Returns:
            DataFrame with competitor data
        """
        endpoint = "dataforseo_labs/google/competitors_domain/live"

        payload = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "order_by": ["intersections,desc"]
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_competitors_response(response)

    def _parse_competitors_response(self, response: Dict) -> pd.DataFrame:
        """Parse competitors API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                data.append({
                    'domain': item.get('domain', ''),
                    'intersections': item.get('intersections', 0),
                    'organic_etv': item.get('metrics', {}).get('organic', {}).get('etv', 0),
                    'organic_count': item.get('metrics', {}).get('organic', {}).get('count', 0),
                    'organic_pos_1': item.get('metrics', {}).get('organic', {}).get('pos_1', 0),
                    'organic_pos_2_3': item.get('metrics', {}).get('organic', {}).get('pos_2_3', 0),
                    'organic_pos_4_10': item.get('metrics', {}).get('organic', {}).get('pos_4_10', 0),
                    'organic_pos_11_20': item.get('metrics', {}).get('organic', {}).get('pos_11_20', 0),
                    'paid_count': item.get('metrics', {}).get('paid', {}).get('count', 0)
                })

        return pd.DataFrame(data)

    def get_keyword_suggestions(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 100,
        include_serp_info: bool = True
    ) -> pd.DataFrame:
        """
        Get keyword suggestions/ideas based on a seed keyword.

        Args:
            keyword: Seed keyword
            location_code: Location code
            language_code: Language code
            limit: Maximum suggestions to return
            include_serp_info: Include SERP analysis

        Returns:
            DataFrame with keyword suggestions
        """
        endpoint = "dataforseo_labs/google/keyword_suggestions/live"

        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "include_serp_info": include_serp_info
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_keyword_suggestions_response(response)

    def _parse_keyword_suggestions_response(self, response: Dict) -> pd.DataFrame:
        """Parse keyword suggestions API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                keyword_data = item.get('keyword_data', {})
                keyword_info = keyword_data.get('keyword_info', {})
                serp_info = keyword_data.get('serp_info', {})

                data.append({
                    'keyword': keyword_data.get('keyword', ''),
                    'search_volume': keyword_info.get('search_volume', 0),
                    'cpc': keyword_info.get('cpc', 0),
                    'competition': keyword_info.get('competition', 0),
                    'competition_level': keyword_info.get('competition_level', ''),
                    'serp_type': serp_info.get('serp_type', ''),
                    'se_results_count': serp_info.get('se_results_count', 0)
                })

        return pd.DataFrame(data)

    def get_related_keywords(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get related keywords for topic expansion.

        Args:
            keyword: Seed keyword
            location_code: Location code
            language_code: Language code
            limit: Maximum results

        Returns:
            DataFrame with related keywords
        """
        endpoint = "dataforseo_labs/google/related_keywords/live"

        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_keyword_suggestions_response(response)

    def get_serp_organic(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        device: str = "desktop",
        depth: int = 100
    ) -> pd.DataFrame:
        """
        Get organic SERP results for a keyword.

        Args:
            keyword: Keyword to analyze
            location_code: Location code
            language_code: Language code
            device: Device type (desktop/mobile)
            depth: Number of results to fetch

        Returns:
            DataFrame with SERP results
        """
        endpoint = "serp/google/organic/live/regular"

        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": device,
            "depth": depth
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_serp_response(response)

    def _parse_serp_response(self, response: Dict) -> pd.DataFrame:
        """Parse SERP API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                if item.get('type') == 'organic':
                    data.append({
                        'position': item.get('rank_absolute', 0),
                        'rank_group': item.get('rank_group', 0),
                        'domain': item.get('domain', ''),
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'is_featured_snippet': item.get('is_featured_snippet', False),
                        'is_malicious': item.get('is_malicious', False),
                        'breadcrumb': item.get('breadcrumb', ''),
                        'type': item.get('type', '')
                    })
                elif item.get('type') in ['featured_snippet', 'people_also_ask', 'local_pack', 'video', 'images']:
                    data.append({
                        'position': item.get('rank_absolute', 0),
                        'rank_group': item.get('rank_group', 0),
                        'domain': item.get('domain', '') if item.get('domain') else 'N/A',
                        'url': item.get('url', '') if item.get('url') else 'N/A',
                        'title': item.get('title', '') if item.get('title') else f"SERP Feature: {item.get('type')}",
                        'description': '',
                        'is_featured_snippet': item.get('type') == 'featured_snippet',
                        'is_malicious': False,
                        'breadcrumb': '',
                        'type': item.get('type', '')
                    })

        return pd.DataFrame(data)

    def get_categories_for_domain(
        self,
        target: str,
        location_code: int = 2840,
        language_code: str = "en"
    ) -> pd.DataFrame:
        """
        Get category distribution for a domain.

        Args:
            target: Domain to analyze
            location_code: Location code
            language_code: Language code

        Returns:
            DataFrame with category data
        """
        endpoint = "dataforseo_labs/google/categories_for_domain/live"

        payload = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_categories_response(response)

    def _parse_categories_response(self, response: Dict) -> pd.DataFrame:
        """Parse categories API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                metrics = item.get('metrics', {}).get('organic', {})
                data.append({
                    'category': item.get('category', ''),
                    'category_code': item.get('category_code', 0),
                    'keyword_count': metrics.get('count', 0),
                    'etv': metrics.get('etv', 0),
                    'impressions_etv': metrics.get('impressions_etv', 0),
                    'is_up': metrics.get('is_up', 0),
                    'is_down': metrics.get('is_down', 0),
                    'is_new': metrics.get('is_new', 0),
                    'is_lost': metrics.get('is_lost', 0)
                })

        return pd.DataFrame(data)

    def get_keyword_gap(
        self,
        target: str,
        competitors: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get keyword gap analysis between target and competitors.

        Args:
            target: Target domain
            competitors: List of competitor domains
            location_code: Location code
            language_code: Language code
            limit: Maximum results

        Returns:
            DataFrame with keyword gap data
        """
        endpoint = "dataforseo_labs/google/domain_intersection/live"

        # Include target in the domains list
        all_domains = [target] + competitors[:2]  # Max 3 domains

        payload = [{
            "targets": {str(i+1): {"target": d, "target_type": "domain"} for i, d in enumerate(all_domains)},
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "order_by": ["keyword_data.keyword_info.search_volume,desc"]
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_keyword_gap_response(response, target)

    def _parse_keyword_gap_response(self, response: Dict, target: str) -> pd.DataFrame:
        """Parse keyword gap API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                keyword_data = item.get('keyword_data', {})
                keyword_info = keyword_data.get('keyword_info', {})

                # Get positions for each domain
                intersection_result = item.get('intersection_result', {})

                target_pos = None
                competitor_positions = {}

                for key, value in intersection_result.items():
                    if value and len(value) > 0:
                        serp_item = value[0].get('serp_item', {})
                        pos = serp_item.get('rank_absolute', 0)
                        domain = serp_item.get('domain', '')

                        if target.lower() in domain.lower():
                            target_pos = pos
                        else:
                            competitor_positions[domain] = pos

                data.append({
                    'keyword': keyword_data.get('keyword', ''),
                    'search_volume': keyword_info.get('search_volume', 0),
                    'cpc': keyword_info.get('cpc', 0),
                    'competition': keyword_info.get('competition', 0),
                    'target_position': target_pos,
                    'competitor_positions': str(competitor_positions),
                    'is_gap': target_pos is None and len(competitor_positions) > 0
                })

        return pd.DataFrame(data)

    def get_historical_rank_data(
        self,
        target: str,
        location_code: int = 2840,
        language_code: str = "en"
    ) -> pd.DataFrame:
        """
        Get historical ranking data for trend analysis.

        Args:
            target: Domain to analyze
            location_code: Location code
            language_code: Language code

        Returns:
            DataFrame with historical data
        """
        endpoint = "dataforseo_labs/google/historical_rank_overview/live"

        payload = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code
        }]

        response = self._make_request(endpoint, data=payload)
        return self._parse_historical_response(response)

    def _parse_historical_response(self, response: Dict) -> pd.DataFrame:
        """Parse historical rank API response."""
        data = []

        if response.get('status_code') != 20000:
            return pd.DataFrame()

        tasks = response.get('tasks', [])
        for task in tasks:
            if task.get('status_code') != 20000:
                continue

            result = task.get('result', [])
            if not result:
                continue

            items = result[0].get('items', [])
            for item in items:
                organic = item.get('metrics', {}).get('organic', {})
                data.append({
                    'date': item.get('month', ''),
                    'etv': organic.get('etv', 0),
                    'count': organic.get('count', 0),
                    'impressions_etv': organic.get('impressions_etv', 0),
                    'pos_1': organic.get('pos_1', 0),
                    'pos_2_3': organic.get('pos_2_3', 0),
                    'pos_4_10': organic.get('pos_4_10', 0),
                    'pos_11_20': organic.get('pos_11_20', 0),
                    'pos_21_30': organic.get('pos_21_30', 0),
                    'is_new': organic.get('is_new', 0),
                    'is_up': organic.get('is_up', 0),
                    'is_down': organic.get('is_down', 0),
                    'is_lost': organic.get('is_lost', 0)
                })

        return pd.DataFrame(data)


def get_location_codes() -> Dict[str, int]:
    """Return common location codes for DataForSEO."""
    return {
        "United States": 2840,
        "United Kingdom": 2826,
        "Canada": 2124,
        "Australia": 2036,
        "Germany": 2276,
        "France": 2250,
        "Spain": 2724,
        "Italy": 2380,
        "Netherlands": 2528,
        "Brazil": 2076,
        "Mexico": 2484,
        "India": 2356,
        "Japan": 2392
    }


def get_language_codes() -> Dict[str, str]:
    """Return common language codes for DataForSEO."""
    return {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Japanese": "ja",
        "Hindi": "hi"
    }
