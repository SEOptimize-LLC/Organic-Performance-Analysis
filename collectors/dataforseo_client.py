"""
DataForSEO API client.
Implements all required endpoints for organic performance analysis.
"""

import requests
import base64
import pandas as pd
from typing import List, Dict, Optional, Any
import time

from config.api_config import api_config
from services.cache_service import cache_service
from services.rate_limiter import rate_limiter, retry_handler
from utils.logger import logger
from utils.helpers import safe_float, safe_int


class DataForSEOClient:
    """
    Comprehensive DataForSEO client for organic analysis.
    Implements: Ranked Keywords, Competitor Analysis,
    Keyword Suggestions, and SERP Analysis endpoints.
    """
    
    def __init__(self):
        """Initialize DataForSEO client with credentials."""
        # Get auth tuple (login, password)
        auth = api_config.get_dataforseo_auth()
        if not auth:
            raise ValueError("DataForSEO credentials not configured")
        
        self.login, self.password = auth
        self.base_url = api_config.dataforseo_base_url
        
        # Encode credentials
        creds = f"{self.login}:{self.password}"
        self.auth_header = base64.b64encode(creds.encode()).decode()
        
        self.headers = {
            'Authorization': f'Basic {self.auth_header}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(
        self,
        endpoint: str,
        data: List[Dict],
        method: str = 'POST'
    ) -> Optional[Dict]:
        """
        Make authenticated API request.
        
        Args:
            endpoint: API endpoint path
            data: Request payload
            method: HTTP method
            
        Returns:
            API response or None
        """
        url = f"{self.base_url}{endpoint}"
        
        logger.info(f"DataForSEO request: {endpoint}")
        logger.debug(f"Payload: {data}")
        
        try:
            if method == 'POST':
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=data,
                    timeout=120
                )
            else:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=120
                )
            
            logger.info(f"DataForSEO HTTP status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors
            api_status = result.get('status_code', 0)
            api_message = result.get('status_message', 'No message')
            
            logger.info(f"DataForSEO API status: {api_status} - {api_message}")
            
            if api_status != 20000:
                logger.error(f"DataForSEO API error: {api_message}")
                # Log task-level errors if available
                tasks = result.get('tasks', [])
                for task in tasks:
                    task_status = task.get('status_code', 0)
                    task_msg = task.get('status_message', '')
                    if task_status != 20000:
                        logger.error(f"Task error: {task_status} - {task_msg}")
                return None
            
            # Log task results count
            tasks = result.get('tasks', [])
            for task in tasks:
                task_result = task.get('result', [])
                if task_result:
                    items_count = len(task_result[0].get('items', []))
                    logger.info(f"DataForSEO returned {items_count} items")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {response.status_code}: {str(e)}")
            # Try to get error details from response
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except Exception:
                pass
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None
    
    # =========================================================================
    # RANKED KEYWORDS ENDPOINT
    # =========================================================================
    
    @rate_limiter.limit_dataforseo
    def get_ranked_keywords(
        self,
        domain: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 1000,
        offset: int = 0,
        order_by: str = None,
        filters: List = None
    ) -> pd.DataFrame:
        """
        Get keywords ranking for a domain.
        
        Args:
            domain: Target domain
            location_code: Location (default: US 2840)
            language_code: Language code
            limit: Max results
            offset: Pagination offset
            order_by: Sort field
            filters: Optional filters
            
        Returns:
            DataFrame with ranked keywords
        """
        # Check cache
        cache_key = f"ranked_kw_{domain}_{location_code}"
        cached = cache_service.get_dataforseo_data(cache_key, limit)
        if cached is not None:
            logger.info("Using cached ranked keywords data")
            return pd.DataFrame(cached)
        
        endpoint = "/v3/dataforseo_labs/google/ranked_keywords/live"
        
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "offset": offset
        }]
        
        if order_by:
            payload[0]["order_by"] = [order_by]
        
        if filters:
            payload[0]["filters"] = filters
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return pd.DataFrame()
        
        # Process results
        df = self._process_ranked_keywords(response)
        
        # Cache results
        if not df.empty:
            cache_service.set_dataforseo_data(
                cache_key, df.to_dict('records'), limit
            )
        
        return df
    
    def _process_ranked_keywords(self, response: Dict) -> pd.DataFrame:
        """Process ranked keywords response into DataFrame."""
        records = []
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                for item in items:
                    kw_data = item.get('keyword_data', {})
                    records.append({
                        'keyword': kw_data.get('keyword', ''),
                        'search_volume': safe_int(
                            kw_data.get('keyword_info', {}).get(
                                'search_volume', 0
                            )
                        ),
                        'cpc': safe_float(
                            kw_data.get('keyword_info', {}).get('cpc', 0)
                        ),
                        'competition': safe_float(
                            kw_data.get('keyword_info', {}).get(
                                'competition', 0
                            )
                        ),
                        'traffic': safe_float(
                            item.get('ranked_serp_element', {}).get(
                                'etv', 0
                            )
                        ),
                        'position': safe_int(
                            item.get('ranked_serp_element', {}).get(
                                'serp_item', {}
                            ).get('rank_absolute', 0)
                        ),
                        'url': item.get(
                            'ranked_serp_element', {}
                        ).get('serp_item', {}).get('url', ''),
                        'is_featured_snippet': item.get(
                            'ranked_serp_element', {}
                        ).get('serp_item', {}).get(
                            'is_featured_snippet', False
                        ),
                        'serp_type': item.get(
                            'ranked_serp_element', {}
                        ).get('serp_item', {}).get('type', '')
                    })
        except Exception as e:
            logger.error(f"Error processing ranked keywords: {str(e)}")
        
        return pd.DataFrame(records)
    
    def get_all_ranked_keywords(
        self,
        domain: str,
        location_code: int = 2840,
        language_code: str = "en",
        max_keywords: int = 5000
    ) -> pd.DataFrame:
        """
        Get all ranked keywords with pagination.
        
        Args:
            domain: Target domain
            location_code: Location code
            language_code: Language
            max_keywords: Maximum keywords to fetch
            
        Returns:
            Complete DataFrame of ranked keywords
        """
        all_keywords = []
        offset = 0
        batch_size = 1000
        
        while len(all_keywords) < max_keywords:
            df = self.get_ranked_keywords(
                domain=domain,
                location_code=location_code,
                language_code=language_code,
                limit=batch_size,
                offset=offset
            )
            
            if df.empty:
                break
            
            all_keywords.extend(df.to_dict('records'))
            
            if len(df) < batch_size:
                break
            
            offset += batch_size
            time.sleep(0.5)  # Rate limiting pause
        
        return pd.DataFrame(all_keywords)
    
    # =========================================================================
    # COMPETITOR ANALYSIS ENDPOINT
    # =========================================================================
    
    @rate_limiter.limit_dataforseo
    def get_competitors(
        self,
        domain: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Auto-discover competitors based on SERP overlap.
        
        Args:
            domain: Target domain
            location_code: Location code
            language_code: Language
            limit: Number of competitors
            
        Returns:
            DataFrame with competitor analysis
        """
        # Check cache
        cache_key = f"competitors_{domain}_{location_code}"
        cached = cache_service.get_dataforseo_data(cache_key, limit)
        if cached is not None:
            logger.info("Using cached competitors data")
            return pd.DataFrame(cached)
        
        endpoint = "/v3/dataforseo_labs/google/competitors_domain/live"
        
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "exclude_top_domains": True
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return pd.DataFrame()
        
        # Process results
        df = self._process_competitors(response)
        
        # Cache results
        if not df.empty:
            cache_service.set_dataforseo_data(
                cache_key, df.to_dict('records'), limit
            )
        
        return df
    
    def _process_competitors(self, response: Dict) -> pd.DataFrame:
        """Process competitors response into DataFrame."""
        records = []
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                for item in items:
                    records.append({
                        'competitor_domain': item.get('domain', ''),
                        'avg_position': safe_float(
                            item.get('avg_position', 0)
                        ),
                        'sum_position': safe_int(
                            item.get('sum_position', 0)
                        ),
                        'intersections': safe_int(
                            item.get('intersections', 0)
                        ),
                        'full_domain_metrics': item.get(
                            'full_domain_metrics', {}
                        ),
                        'organic_etv': safe_float(
                            item.get('full_domain_metrics', {}).get(
                                'organic', {}
                            ).get('etv', 0)
                        ),
                        'organic_count': safe_int(
                            item.get('full_domain_metrics', {}).get(
                                'organic', {}
                            ).get('count', 0)
                        )
                    })
        except Exception as e:
            logger.error(f"Error processing competitors: {str(e)}")
        
        return pd.DataFrame(records)
    
    @rate_limiter.limit_dataforseo
    def get_competitor_keywords(
        self,
        target_domain: str,
        competitor_domain: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get keyword gaps between target and competitor.
        
        Args:
            target_domain: Your domain
            competitor_domain: Competitor domain
            location_code: Location
            language_code: Language
            limit: Max results
            
        Returns:
            DataFrame with keyword gap analysis
        """
        endpoint = "/v3/dataforseo_labs/google/domain_intersection/live"
        
        payload = [{
            "target1": target_domain,
            "target2": competitor_domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "include_subdomains": True,
            "intersections": False  # Keywords competitor has, you don't
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return pd.DataFrame()
        
        return self._process_keyword_intersection(response)
    
    def _process_keyword_intersection(self, response: Dict) -> pd.DataFrame:
        """Process keyword intersection response."""
        records = []
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                for item in items:
                    kw_data = item.get('keyword_data', {})
                    kw_info = kw_data.get('keyword_info', {})
                    
                    records.append({
                        'keyword': kw_data.get('keyword', ''),
                        'search_volume': safe_int(
                            kw_info.get('search_volume', 0)
                        ),
                        'cpc': safe_float(kw_info.get('cpc', 0)),
                        'competition': safe_float(
                            kw_info.get('competition', 0)
                        ),
                        'target1_position': safe_int(
                            item.get('first_domain_serp_element', {}).get(
                                'serp_item', {}
                            ).get('rank_absolute', 0)
                        ),
                        'target2_position': safe_int(
                            item.get('second_domain_serp_element', {}).get(
                                'serp_item', {}
                            ).get('rank_absolute', 0)
                        )
                    })
        except Exception as e:
            logger.error(f"Error processing intersection: {str(e)}")
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # KEYWORD SUGGESTIONS ENDPOINT
    # =========================================================================
    
    @rate_limiter.limit_dataforseo
    def get_keyword_suggestions(
        self,
        seed_keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 500,
        include_seed: bool = True
    ) -> pd.DataFrame:
        """
        Get keyword suggestions based on seed keyword.
        
        Args:
            seed_keyword: Seed keyword
            location_code: Location
            language_code: Language
            limit: Max results
            include_seed: Include seed in results
            
        Returns:
            DataFrame with keyword suggestions
        """
        endpoint = "/v3/dataforseo_labs/google/keyword_suggestions/live"
        
        payload = [{
            "keyword": seed_keyword,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "include_seed_keyword": include_seed,
            "include_serp_info": True
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return pd.DataFrame()
        
        return self._process_keyword_suggestions(response)
    
    def _process_keyword_suggestions(self, response: Dict) -> pd.DataFrame:
        """Process keyword suggestions response."""
        records = []
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                for item in items:
                    kw_info = item.get('keyword_info', {})
                    serp_info = item.get('serp_info', {})
                    
                    records.append({
                        'keyword': item.get('keyword', ''),
                        'search_volume': safe_int(
                            kw_info.get('search_volume', 0)
                        ),
                        'cpc': safe_float(kw_info.get('cpc', 0)),
                        'competition': safe_float(
                            kw_info.get('competition', 0)
                        ),
                        'keyword_difficulty': safe_float(
                            item.get('keyword_properties', {}).get(
                                'keyword_difficulty', 0
                            )
                        ),
                        'serp_featured_snippet': serp_info.get(
                            'featured_snippet', False
                        ),
                        'serp_people_also_ask': len(
                            serp_info.get('item_types', [])
                        ) if 'people_also_ask' in str(serp_info) else 0
                    })
        except Exception as e:
            logger.error(f"Error processing suggestions: {str(e)}")
        
        return pd.DataFrame(records)
    
    @rate_limiter.limit_dataforseo
    def get_related_keywords(
        self,
        seed_keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get semantically related keywords.
        
        Args:
            seed_keyword: Seed keyword
            location_code: Location
            language_code: Language
            limit: Max results
            
        Returns:
            DataFrame with related keywords
        """
        endpoint = "/v3/dataforseo_labs/google/related_keywords/live"
        
        payload = [{
            "keyword": seed_keyword,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return pd.DataFrame()
        
        return self._process_related_keywords(response)
    
    def _process_related_keywords(self, response: Dict) -> pd.DataFrame:
        """Process related keywords response."""
        records = []
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                for item in items:
                    kw_data = item.get('keyword_data', {})
                    kw_info = kw_data.get('keyword_info', {})
                    
                    records.append({
                        'keyword': kw_data.get('keyword', ''),
                        'search_volume': safe_int(
                            kw_info.get('search_volume', 0)
                        ),
                        'cpc': safe_float(kw_info.get('cpc', 0)),
                        'competition': safe_float(
                            kw_info.get('competition', 0)
                        ),
                        'depth': safe_int(item.get('depth', 0)),
                        'related_count': safe_int(
                            item.get('related_result_count', 0)
                        )
                    })
        except Exception as e:
            logger.error(f"Error processing related keywords: {str(e)}")
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # SERP ANALYSIS ENDPOINT
    # =========================================================================
    
    @rate_limiter.limit_dataforseo
    def get_serp_results(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        device: str = "desktop",
        depth: int = 100
    ) -> Dict[str, Any]:
        """
        Get SERP analysis for a keyword.
        
        Args:
            keyword: Target keyword
            location_code: Location
            language_code: Language
            device: desktop or mobile
            depth: Number of results
            
        Returns:
            Dict with SERP data and features
        """
        endpoint = "/v3/serp/google/organic/live/advanced"
        
        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": device,
            "depth": depth
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return {}
        
        return self._process_serp_results(response)
    
    def _process_serp_results(self, response: Dict) -> Dict[str, Any]:
        """Process SERP results into structured data."""
        result_data = {
            'organic_results': [],
            'serp_features': {},
            'featured_snippet': None,
            'people_also_ask': [],
            'title': '',
            'check_url': ''
        }
        
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                r = result[0]
                result_data['check_url'] = r.get('check_url', '')
                
                items = r.get('items', [])
                for item in items:
                    item_type = item.get('type', '')
                    
                    if item_type == 'organic':
                        result_data['organic_results'].append({
                            'position': safe_int(
                                item.get('rank_absolute', 0)
                            ),
                            'url': item.get('url', ''),
                            'title': item.get('title', ''),
                            'description': item.get('description', ''),
                            'domain': item.get('domain', ''),
                            'breadcrumb': item.get('breadcrumb', ''),
                            'is_featured': item.get(
                                'is_featured_snippet', False
                            )
                        })
                    
                    elif item_type == 'featured_snippet':
                        result_data['featured_snippet'] = {
                            'url': item.get('url', ''),
                            'title': item.get('title', ''),
                            'description': item.get('description', ''),
                            'snippet_type': item.get('featured_title', '')
                        }
                    
                    elif item_type == 'people_also_ask':
                        items_list = item.get('items', [])
                        for paa in items_list:
                            result_data['people_also_ask'].append({
                                'question': paa.get('title', ''),
                                'url': paa.get('url', '')
                            })
                    
                    # Track all SERP features
                    if item_type not in result_data['serp_features']:
                        result_data['serp_features'][item_type] = 0
                    result_data['serp_features'][item_type] += 1
                    
        except Exception as e:
            logger.error(f"Error processing SERP results: {str(e)}")
        
        return result_data
    
    def analyze_serp_landscape(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en"
    ) -> pd.DataFrame:
        """
        Analyze SERP landscape for multiple keywords.
        
        Args:
            keywords: List of keywords to analyze
            location_code: Location
            language_code: Language
            
        Returns:
            DataFrame with SERP feature analysis
        """
        records = []
        
        for keyword in keywords[:50]:  # Limit to prevent API overuse
            serp_data = self.get_serp_results(
                keyword=keyword,
                location_code=location_code,
                language_code=language_code
            )
            
            if serp_data:
                records.append({
                    'keyword': keyword,
                    'has_featured_snippet': bool(
                        serp_data.get('featured_snippet')
                    ),
                    'paa_count': len(serp_data.get('people_also_ask', [])),
                    'organic_count': len(serp_data.get('organic_results', [])),
                    'serp_features': list(
                        serp_data.get('serp_features', {}).keys()
                    ),
                    'top_3_domains': [
                        r['domain'] for r in serp_data.get(
                            'organic_results', []
                        )[:3]
                    ]
                })
            
            time.sleep(0.3)  # Rate limiting
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # DOMAIN OVERVIEW ENDPOINT
    # =========================================================================
    
    @rate_limiter.limit_dataforseo
    def get_domain_overview(
        self,
        domain: str,
        location_code: int = 2840,
        language_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Get comprehensive domain overview.
        
        Args:
            domain: Target domain
            location_code: Location
            language_code: Language
            
        Returns:
            Dict with domain metrics
        """
        endpoint = "/v3/dataforseo_labs/google/domain_rank_overview/live"
        
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code
        }]
        
        response = retry_handler.execute_with_retry(
            self._make_request,
            endpoint,
            payload
        )
        
        if not response:
            return {}
        
        return self._process_domain_overview(response)
    
    def _process_domain_overview(self, response: Dict) -> Dict[str, Any]:
        """Process domain overview response."""
        try:
            tasks = response.get('tasks', [])
            for task in tasks:
                result = task.get('result', [])
                if not result:
                    continue
                
                items = result[0].get('items', [])
                if items:
                    item = items[0]
                    return {
                        'domain': item.get('target', ''),
                        'organic_etv': safe_float(
                            item.get('metrics', {}).get('organic', {}).get(
                                'etv', 0
                            )
                        ),
                        'organic_count': safe_int(
                            item.get('metrics', {}).get('organic', {}).get(
                                'count', 0
                            )
                        ),
                        'organic_is_new': safe_int(
                            item.get('metrics', {}).get('organic', {}).get(
                                'is_new', 0
                            )
                        ),
                        'organic_is_up': safe_int(
                            item.get('metrics', {}).get('organic', {}).get(
                                'is_up', 0
                            )
                        ),
                        'organic_is_down': safe_int(
                            item.get('metrics', {}).get('organic', {}).get(
                                'is_down', 0
                            )
                        ),
                        'organic_is_lost': safe_int(
                            item.get('metrics', {}).get('organic', {}).get(
                                'is_lost', 0
                            )
                        )
                    }
        except Exception as e:
            logger.error(f"Error processing domain overview: {str(e)}")
        
        return {}
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    def get_comprehensive_domain_data(
        self,
        domain: str,
        location_code: int = 2840,
        language_code: str = "en",
        max_keywords: int = 2000
    ) -> Dict[str, Any]:
        """
        Get comprehensive data for a domain.
        
        Args:
            domain: Target domain
            location_code: Location
            language_code: Language
            max_keywords: Max keywords to fetch
            
        Returns:
            Dict with all domain data
        """
        logger.info(f"Fetching comprehensive data for {domain}")
        
        return {
            'domain': domain,
            'overview': self.get_domain_overview(
                domain, location_code, language_code
            ),
            'ranked_keywords': self.get_all_ranked_keywords(
                domain, location_code, language_code, max_keywords
            ),
            'competitors': self.get_competitors(
                domain, location_code, language_code
            )
        }