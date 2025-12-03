"""
Brand vs non-brand query classifier.
Segments traffic for strategic analysis.
"""

import pandas as pd
import re
from typing import Dict, List, Optional


class BrandClassifier:
    """
    Classifies queries as brand or non-brand.
    Provides traffic segmentation analysis.
    """
    
    def __init__(self, brand_terms: List[str] = None):
        """
        Initialize classifier.
        
        Args:
            brand_terms: List of brand term patterns
        """
        self.brand_terms = brand_terms or []
        self.brand_pattern = None
        if self.brand_terms:
            self._compile_pattern()
    
    def _compile_pattern(self):
        """Compile brand terms into regex pattern."""
        # Escape special chars and create pattern
        escaped = [re.escape(term.lower().strip()) for term in self.brand_terms]
        pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
        self.brand_pattern = re.compile(pattern_str, re.IGNORECASE)
    
    def set_brand_terms(self, brand_terms: List[str]):
        """
        Update brand terms.
        
        Args:
            brand_terms: New list of brand terms
        """
        self.brand_terms = [t.strip() for t in brand_terms if t.strip()]
        if self.brand_terms:
            self._compile_pattern()
        else:
            self.brand_pattern = None
    
    def is_brand_query(self, query: str) -> bool:
        """
        Check if query is a brand query.
        
        Args:
            query: Search query
            
        Returns:
            True if brand query
        """
        if not self.brand_pattern or not query:
            return False
        return bool(self.brand_pattern.search(query.lower()))
    
    def classify_queries(
        self,
        queries_df: pd.DataFrame,
        query_column: str = 'query'
    ) -> pd.DataFrame:
        """
        Classify all queries in a DataFrame.
        
        Args:
            queries_df: DataFrame with queries
            query_column: Name of query column
            
        Returns:
            DataFrame with brand classification
        """
        if queries_df.empty or query_column not in queries_df.columns:
            return queries_df
        
        df = queries_df.copy()
        df['is_brand'] = df[query_column].apply(self.is_brand_query)
        df['query_type'] = df['is_brand'].apply(
            lambda x: 'brand' if x else 'non-brand'
        )
        
        return df
    
    def segment_traffic(
        self,
        queries_df: pd.DataFrame,
        query_column: str = 'query'
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment queries into brand and non-brand.
        
        Args:
            queries_df: DataFrame with queries
            query_column: Name of query column
            
        Returns:
            Dict with segmented DataFrames
        """
        classified = self.classify_queries(queries_df, query_column)
        
        if classified.empty:
            return {'brand': pd.DataFrame(), 'non_brand': pd.DataFrame()}
        
        return {
            'brand': classified[classified['is_brand']].copy(),
            'non_brand': classified[~classified['is_brand']].copy()
        }
    
    def calculate_brand_metrics(
        self,
        queries_df: pd.DataFrame,
        query_column: str = 'query'
    ) -> Dict:
        """
        Calculate brand vs non-brand metrics.
        
        Args:
            queries_df: DataFrame with queries
            query_column: Query column name
            
        Returns:
            Dict with brand/non-brand metrics
        """
        segments = self.segment_traffic(queries_df, query_column)
        
        brand_df = segments['brand']
        non_brand_df = segments['non_brand']
        
        total_clicks = queries_df['clicks'].sum() if 'clicks' in queries_df.columns else 0  # noqa: E501
        total_impressions = queries_df['impressions'].sum() if 'impressions' in queries_df.columns else 0  # noqa: E501
        
        brand_clicks = brand_df['clicks'].sum() if not brand_df.empty and 'clicks' in brand_df.columns else 0  # noqa: E501
        brand_impressions = brand_df['impressions'].sum() if not brand_df.empty and 'impressions' in brand_df.columns else 0  # noqa: E501
        
        non_brand_clicks = non_brand_df['clicks'].sum() if not non_brand_df.empty and 'clicks' in non_brand_df.columns else 0  # noqa: E501
        non_brand_impressions = non_brand_df['impressions'].sum() if not non_brand_df.empty and 'impressions' in non_brand_df.columns else 0  # noqa: E501
        
        return {
            'total': {
                'queries': len(queries_df),
                'clicks': int(total_clicks),
                'impressions': int(total_impressions)
            },
            'brand': {
                'queries': len(brand_df),
                'clicks': int(brand_clicks),
                'impressions': int(brand_impressions),
                'click_share': round(
                    brand_clicks / total_clicks * 100, 1
                ) if total_clicks > 0 else 0,
                'impression_share': round(
                    brand_impressions / total_impressions * 100, 1
                ) if total_impressions > 0 else 0,
                'avg_position': round(
                    brand_df['position'].mean(), 1
                ) if not brand_df.empty and 'position' in brand_df.columns else 0,  # noqa: E501
                'avg_ctr': round(
                    brand_df['ctr'].mean() * 100, 2
                ) if not brand_df.empty and 'ctr' in brand_df.columns else 0
            },
            'non_brand': {
                'queries': len(non_brand_df),
                'clicks': int(non_brand_clicks),
                'impressions': int(non_brand_impressions),
                'click_share': round(
                    non_brand_clicks / total_clicks * 100, 1
                ) if total_clicks > 0 else 0,
                'impression_share': round(
                    non_brand_impressions / total_impressions * 100, 1
                ) if total_impressions > 0 else 0,
                'avg_position': round(
                    non_brand_df['position'].mean(), 1
                ) if not non_brand_df.empty and 'position' in non_brand_df.columns else 0,  # noqa: E501
                'avg_ctr': round(
                    non_brand_df['ctr'].mean() * 100, 2
                ) if not non_brand_df.empty and 'ctr' in non_brand_df.columns else 0  # noqa: E501
            },
            'dependency_score': round(
                brand_clicks / total_clicks * 100, 1
            ) if total_clicks > 0 else 0
        }
    
    def get_non_brand_opportunities(
        self,
        queries_df: pd.DataFrame,
        min_impressions: int = 100,
        max_position: float = 20.0
    ) -> pd.DataFrame:
        """
        Get non-brand opportunities for growth.
        
        Args:
            queries_df: Query DataFrame
            min_impressions: Minimum impressions
            max_position: Maximum position to consider
            
        Returns:
            DataFrame with non-brand opportunities
        """
        segments = self.segment_traffic(queries_df)
        non_brand = segments['non_brand']
        
        if non_brand.empty:
            return pd.DataFrame()
        
        # Filter for opportunities
        opps = non_brand[
            (non_brand['impressions'] >= min_impressions) &
            (non_brand['position'] <= max_position)
        ].copy()
        
        # Sort by potential (impressions * CTR gap)
        if 'ctr' in opps.columns and 'position' in opps.columns:
            opps['ctr_potential'] = opps.apply(
                lambda r: max(0, 0.10 - r['ctr'])
                if r['position'] <= 5 else max(0, 0.05 - r['ctr']),
                axis=1
            )
            opps['opportunity_value'] = (
                opps['impressions'] * opps['ctr_potential']
            )
            opps = opps.sort_values('opportunity_value', ascending=False)
        
        return opps
    
    def analyze_brand_dependency(
        self,
        current_metrics: Dict,
        previous_metrics: Dict = None
    ) -> Dict:
        """
        Analyze brand dependency and trends.
        
        Args:
            current_metrics: Current brand metrics
            previous_metrics: Previous period metrics
            
        Returns:
            Analysis dict
        """
        dependency = current_metrics.get('dependency_score', 0)
        
        analysis = {
            'dependency_level': 'low',
            'risk_assessment': '',
            'recommendations': [],
            'trend': 'stable'
        }
        
        # Classify dependency level
        if dependency >= 70:
            analysis['dependency_level'] = 'critical'
            analysis['risk_assessment'] = (
                'Extremely high brand dependency. '
                'Non-brand growth is crucial for sustainability.'
            )
            analysis['recommendations'] = [
                'Prioritize non-brand keyword expansion',
                'Build content for informational queries',
                'Diversify traffic sources urgently',
                'Invest in topical authority building'
            ]
        elif dependency >= 50:
            analysis['dependency_level'] = 'high'
            analysis['risk_assessment'] = (
                'High brand dependency. '
                'Significant growth potential in non-brand.'
            )
            analysis['recommendations'] = [
                'Expand non-brand content strategy',
                'Target mid-funnel informational content',
                'Build topic clusters around core services',
                'Improve internal linking to non-brand pages'
            ]
        elif dependency >= 30:
            analysis['dependency_level'] = 'moderate'
            analysis['risk_assessment'] = (
                'Balanced traffic mix with room for improvement.'
            )
            analysis['recommendations'] = [
                'Continue non-brand optimization',
                'Focus on converting non-brand traffic',
                'Expand into adjacent topic areas',
                'Optimize existing non-brand pages'
            ]
        else:
            analysis['dependency_level'] = 'low'
            analysis['risk_assessment'] = (
                'Healthy non-brand traffic. '
                'Good organic growth foundation.'
            )
            analysis['recommendations'] = [
                'Maintain current strategy',
                'Protect brand positions',
                'Expand into new topic areas',
                'Focus on conversion optimization'
            ]
        
        # Calculate trend if previous data available
        if previous_metrics:
            prev_dep = previous_metrics.get('dependency_score', 0)
            change = dependency - prev_dep
            
            if change > 5:
                analysis['trend'] = 'increasing'
                analysis['trend_note'] = (
                    f'Brand dependency increased by {change:.1f}%'
                )
            elif change < -5:
                analysis['trend'] = 'decreasing'
                analysis['trend_note'] = (
                    f'Brand dependency decreased by {abs(change):.1f}% (good)'
                )
            else:
                analysis['trend'] = 'stable'
                analysis['trend_note'] = 'Brand dependency is stable'
        
        return analysis
    
    def get_summary_stats(
        self,
        queries_df: pd.DataFrame
    ) -> Dict:
        """
        Get summary statistics for brand analysis.
        
        Args:
            queries_df: Query DataFrame
            
        Returns:
            Summary statistics
        """
        metrics = self.calculate_brand_metrics(queries_df)
        analysis = self.analyze_brand_dependency(metrics)
        
        return {
            'metrics': metrics,
            'analysis': analysis,
            'top_brand_queries': [],
            'top_non_brand_queries': []
        }