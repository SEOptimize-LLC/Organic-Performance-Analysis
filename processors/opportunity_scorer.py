"""
Opportunity scoring model.
Calculates composite scores for prioritizing SEO opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from config.settings import settings
from utils.logger import logger


class OpportunityScorer:
    """
    Calculates opportunity scores for keywords and pages.
    Uses composite scoring based on volume, position, CTR gap,
    commercial value, and trend direction.
    """
    
    def __init__(self, custom_weights: Dict[str, float] = None):
        """
        Initialize scorer with configurable weights.
        
        Args:
            custom_weights: Optional custom scoring weights
        """
        self.weights = custom_weights or settings.scoring_weights
        self.ctr_benchmarks = settings.expected_ctr_by_position
    
    def get_expected_ctr(self, position: float) -> float:
        """
        Get expected CTR for a position.
        
        Args:
            position: Average position
            
        Returns:
            Expected CTR value
        """
        # Handle NaN, None, or invalid values
        if position is None or pd.isna(position) or position <= 0:
            return 0.01  # Default CTR for unknown position
        
        pos_int = int(min(float(position), 20))
        return self.ctr_benchmarks.get(pos_int, 0.01)
    
    def calculate_ctr_gap(
        self,
        actual_ctr: float,
        position: float
    ) -> float:
        """
        Calculate CTR gap vs benchmark.
        
        Args:
            actual_ctr: Actual CTR
            position: Average position
            
        Returns:
            CTR gap score (positive = underperforming)
        """
        expected = self.get_expected_ctr(position)
        if expected == 0:
            return 0
        
        # Gap is how much actual CTR differs from expected
        # Positive = underperforming (opportunity)
        gap = (expected - actual_ctr) / expected
        return max(0, min(1, gap))  # Normalize to 0-1
    
    def calculate_position_potential(self, position: float) -> float:
        """
        Calculate position improvement potential.
        
        Args:
            position: Current position
            
        Returns:
            Position potential score (0-1)
        """
        # Handle NaN, None, or invalid values
        if position is None or pd.isna(position) or position <= 0:
            return 0
        
        position = float(position)
        
        # Best opportunity: positions 4-15 (striking distance)
        # Can improve significantly with optimization
        if 4 <= position <= 10:
            return 1.0
        elif 11 <= position <= 15:
            return 0.9
        elif 16 <= position <= 20:
            return 0.7
        elif 21 <= position <= 30:
            return 0.5
        elif 1 <= position < 4:
            return 0.3  # Already ranking well
        else:
            return 0.2  # Too far back
    
    def calculate_volume_score(
        self,
        search_volume: int,
        max_volume: int = None
    ) -> float:
        """
        Calculate normalized search volume score.
        
        Args:
            search_volume: Monthly search volume
            max_volume: Maximum volume for normalization
            
        Returns:
            Volume score (0-1)
        """
        # Handle NaN or invalid values
        if search_volume is None or pd.isna(search_volume) or search_volume <= 0:
            return 0
        
        search_volume = float(search_volume)
        
        if max_volume is None:
            max_volume = 100000  # Default max
        
        # Log scale normalization (volumes span orders of magnitude)
        log_vol = np.log1p(search_volume)
        log_max = np.log1p(max_volume)
        
        return min(1, log_vol / log_max)
    
    def calculate_commercial_score(
        self,
        cpc: float,
        competition: float = 0,
        max_cpc: float = None
    ) -> float:
        """
        Calculate commercial value score.
        
        Args:
            cpc: Cost per click
            competition: Competition level (0-1)
            max_cpc: Maximum CPC for normalization
            
        Returns:
            Commercial score (0-1)
        """
        # Handle NaN values
        if cpc is None or pd.isna(cpc):
            cpc = 0.0
        if competition is None or pd.isna(competition):
            competition = 0.0
        
        cpc = float(cpc)
        competition = float(competition)
        
        if max_cpc is None:
            max_cpc = 50.0  # Default max CPC
        
        # Normalize CPC
        cpc_score = min(1, cpc / max_cpc) if max_cpc > 0 else 0
        
        # Competition as modifier (high competition = high value)
        comp_modifier = 0.5 + (competition * 0.5)
        
        return cpc_score * comp_modifier
    
    def calculate_trend_score(
        self,
        current_clicks: int,
        previous_clicks: int,
        current_position: float,
        previous_position: float
    ) -> float:
        """
        Calculate trend direction score.
        
        Args:
            current_clicks: Current period clicks
            previous_clicks: Previous period clicks
            current_position: Current position
            previous_position: Previous position
            
        Returns:
            Trend score (-1 to 1, negative = declining)
        """
        # Handle NaN values
        if current_clicks is None or pd.isna(current_clicks):
            current_clicks = 0
        if previous_clicks is None or pd.isna(previous_clicks):
            previous_clicks = 0
        if current_position is None or pd.isna(current_position):
            current_position = 0
        if previous_position is None or pd.isna(previous_position):
            previous_position = 0
        
        current_clicks = int(current_clicks)
        previous_clicks = int(previous_clicks)
        current_position = float(current_position)
        previous_position = float(previous_position)
        
        # Click trend
        if previous_clicks > 0:
            click_change = (current_clicks - previous_clicks) / previous_clicks
        else:
            click_change = 0.5 if current_clicks > 0 else 0
        
        # Position trend (improvement = positive)
        if previous_position > 0:
            pos_change = (previous_position - current_position) / 10
        else:
            pos_change = 0
        
        # Combined trend score
        trend = (click_change * 0.6) + (pos_change * 0.4)
        
        # Normalize to -1 to 1
        return max(-1, min(1, trend))
    
    def calculate_opportunity_score(
        self,
        search_volume: int = 0,
        position: float = 0,
        ctr: float = 0,
        cpc: float = 0,
        competition: float = 0,
        current_clicks: int = 0,
        previous_clicks: int = 0,
        current_position: float = 0,
        previous_position: float = 0,
        max_volume: int = None,
        max_cpc: float = None
    ) -> Dict[str, float]:
        """
        Calculate composite opportunity score.
        
        Returns:
            Dict with component scores and total
        """
        # Sanitize all input values - handle NaN
        def safe_val(val, default=0):
            if val is None or pd.isna(val):
                return default
            return val
        
        search_volume = safe_val(search_volume, 0)
        position = safe_val(position, 0)
        ctr = safe_val(ctr, 0)
        cpc = safe_val(cpc, 0)
        competition = safe_val(competition, 0)
        current_clicks = safe_val(current_clicks, 0)
        previous_clicks = safe_val(previous_clicks, 0)
        current_position = safe_val(current_position, 0)
        previous_position = safe_val(previous_position, 0)
        
        # Calculate component scores
        volume_score = self.calculate_volume_score(search_volume, max_volume)
        position_score = self.calculate_position_potential(position)
        ctr_gap_score = self.calculate_ctr_gap(ctr, position)
        commercial_score = self.calculate_commercial_score(
            cpc, competition, max_cpc
        )
        trend_score = self.calculate_trend_score(
            current_clicks, previous_clicks,
            current_position, previous_position
        )
        
        # Normalize trend to 0-1 for weighted sum
        trend_normalized = (trend_score + 1) / 2
        
        # Weighted sum
        total = (
            volume_score * self.weights['search_volume'] +
            position_score * self.weights['position_potential'] +
            ctr_gap_score * self.weights['ctr_gap'] +
            commercial_score * self.weights['commercial_value'] +
            trend_normalized * self.weights['trend_direction']
        )
        
        return {
            'volume_score': round(volume_score, 3),
            'position_score': round(position_score, 3),
            'ctr_gap_score': round(ctr_gap_score, 3),
            'commercial_score': round(commercial_score, 3),
            'trend_score': round(trend_score, 3),
            'opportunity_score': round(total * 100, 1)
        }
    
    def score_keywords(
        self,
        keywords_df: pd.DataFrame,
        yoy_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Score all keywords in a DataFrame.
        
        Args:
            keywords_df: Keyword data with GSC and/or DFS metrics
            yoy_df: Year-over-year comparison data
            
        Returns:
            DataFrame with opportunity scores
        """
        if keywords_df.empty:
            return keywords_df
        
        df = keywords_df.copy()
        
        # Get max values for normalization
        max_volume = df['dfs_search_volume'].max() if 'dfs_search_volume' in df.columns else 10000  # noqa: E501
        max_cpc = df['dfs_cpc'].max() if 'dfs_cpc' in df.columns else 10.0
        
        # Prepare YoY data if available
        yoy_lookup = {}
        if yoy_df is not None and not yoy_df.empty:
            key_col = 'query' if 'query' in yoy_df.columns else 'keyword'
            yoy_lookup = yoy_df.set_index(key_col).to_dict('index')
        
        # Score each keyword
        scores = []
        for _, row in df.iterrows():
            # Get keyword identifier
            keyword = row.get(
                'gsc_query', row.get('keyword', row.get('keyword_normalized'))
            )
            
            # Get YoY data
            yoy_data = yoy_lookup.get(keyword, {})
            
            score = self.calculate_opportunity_score(
                search_volume=row.get('dfs_search_volume', 0),
                position=row.get('gsc_position', row.get('position', 0)),
                ctr=row.get('gsc_ctr', row.get('ctr', 0)),
                cpc=row.get('dfs_cpc', row.get('cpc', 0)),
                competition=row.get('dfs_competition', row.get('competition', 0)),  # noqa: E501
                current_clicks=row.get('gsc_clicks', row.get('clicks', 0)),
                previous_clicks=yoy_data.get('clicks', 0),
                current_position=row.get('gsc_position', row.get('position', 0)),  # noqa: E501
                previous_position=yoy_data.get('position', 0),
                max_volume=max_volume,
                max_cpc=max_cpc
            )
            scores.append(score)
        
        # Add scores to DataFrame
        scores_df = pd.DataFrame(scores)
        result = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
        
        # Sort by opportunity score
        result = result.sort_values('opportunity_score', ascending=False)
        
        return result
    
    def classify_opportunities(
        self,
        scored_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Classify keywords into opportunity buckets.
        
        Args:
            scored_df: DataFrame with opportunity scores
            
        Returns:
            Dict with categorized opportunities
        """
        if scored_df.empty or 'opportunity_score' not in scored_df.columns:
            return {}
        
        df = scored_df.copy()
        
        # Define thresholds
        high_threshold = 70
        medium_threshold = 50
        
        # Quick wins: high score + position 5-15
        pos_col = 'gsc_position' if 'gsc_position' in df.columns else 'position'
        quick_wins = df[
            (df['opportunity_score'] >= high_threshold) &
            (df[pos_col] >= 5) &
            (df[pos_col] <= 15)
        ]
        
        # CTR opportunities: high ctr_gap + good position
        ctr_opps = df[
            (df['ctr_gap_score'] >= 0.3) &
            (df[pos_col] <= 10) &
            (df['opportunity_score'] >= medium_threshold)
        ]
        
        # Scaling opportunities: high volume + position 2-5
        volume_col = 'dfs_search_volume' if 'dfs_search_volume' in df.columns else 'search_volume'  # noqa: E501
        scaling = df[
            (df[volume_col] >= df[volume_col].quantile(0.75)) &
            (df[pos_col] >= 2) &
            (df[pos_col] <= 5)
        ]
        
        # Declining: negative trend + was getting traffic
        clicks_col = 'gsc_clicks' if 'gsc_clicks' in df.columns else 'clicks'
        declining = df[
            (df['trend_score'] < -0.2) &
            (df[clicks_col] >= 10)
        ]
        
        # New opportunities: high score but low current clicks
        new_opps = df[
            (df['opportunity_score'] >= high_threshold) &
            (df[clicks_col] < 10) &
            (df[volume_col] >= 100)
        ]
        
        return {
            'quick_wins': quick_wins.head(50),
            'ctr_opportunities': ctr_opps.head(50),
            'scaling_opportunities': scaling.head(50),
            'declining_keywords': declining.head(50),
            'new_opportunities': new_opps.head(50)
        }
    
    def score_pages(
        self,
        pages_df: pd.DataFrame,
        query_page_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Score pages based on their keyword portfolios.
        
        Args:
            pages_df: Page-level data
            query_page_df: Query+page data for portfolio analysis
            
        Returns:
            DataFrame with page opportunity scores
        """
        if pages_df.empty:
            return pages_df
        
        df = pages_df.copy()
        
        # Aggregate query metrics per page if available
        if query_page_df is not None and not query_page_df.empty:
            page_agg = query_page_df.groupby('page').agg({
                'query': 'count',
                'clicks': 'sum',
                'impressions': 'sum',
                'position': ['mean', 'std', lambda x: (x < 10).sum()]
            }).reset_index()
            
            page_agg.columns = [
                'page', 'query_count', 'total_clicks', 'total_impressions',
                'avg_position', 'position_std', 'top_10_queries'
            ]
            
            df = pd.merge(df, page_agg, on='page', how='left')
        
        # Calculate page scores
        scores = []
        for _, row in df.iterrows():
            # Position-based score
            pos = row.get('position', row.get('avg_position', 50))
            position_score = self.calculate_position_potential(pos)
            
            # Query diversity score
            query_count = row.get('query_count', 1)
            diversity_score = min(1, np.log1p(query_count) / 5)
            
            # CTR opportunity
            ctr = row.get('ctr', 0)
            ctr_gap = self.calculate_ctr_gap(ctr, pos)
            
            # Traffic score
            clicks = row.get('clicks', row.get('total_clicks', 0))
            traffic_score = min(1, np.log1p(clicks) / 10)
            
            # Top 10 ratio
            top_10 = row.get('top_10_queries', 0)
            top_10_ratio = top_10 / query_count if query_count > 0 else 0
            
            # Combined page score
            page_score = (
                position_score * 0.25 +
                diversity_score * 0.20 +
                ctr_gap * 0.20 +
                traffic_score * 0.20 +
                top_10_ratio * 0.15
            )
            
            scores.append({
                'page_score': round(page_score * 100, 1),
                'position_potential': round(position_score, 3),
                'query_diversity': round(diversity_score, 3),
                'ctr_opportunity': round(ctr_gap, 3),
                'top_10_ratio': round(top_10_ratio, 3)
            })
        
        scores_df = pd.DataFrame(scores)
        result = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
        
        return result.sort_values('page_score', ascending=False)
    
    def get_top_opportunities(
        self,
        scored_df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get top N opportunities.
        
        Args:
            scored_df: Scored DataFrame
            top_n: Number of top opportunities
            
        Returns:
            Top opportunities
        """
        score_col = 'opportunity_score'
        if score_col not in scored_df.columns:
            score_col = 'page_score'
        
        if score_col not in scored_df.columns:
            return scored_df.head(top_n)
        
        return scored_df.nlargest(top_n, score_col)