"""
Content decay and performance trend detector.
Identifies declining keywords and pages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

from config.settings import settings
from utils.logger import logger


class DecayType(Enum):
    """Types of content decay."""
    POSITION_DROP = "position_drop"
    IMPRESSIONS_DROP = "impressions_drop"
    CLICKS_DROP = "clicks_drop"
    CTR_DROP = "ctr_drop"
    FULL_DECAY = "full_decay"
    DEMAND_SHIFT = "demand_shift"
    COMPETITION_LOSS = "competition_loss"


class DecayDetector:
    """
    Detects content decay patterns across time windows.
    Classifies decay types for targeted recovery strategies.
    """
    
    # Thresholds for decay detection
    POSITION_DECAY_THRESHOLD = 3.0  # Positions dropped
    IMPRESSIONS_DECAY_PCT = -0.20  # 20% drop
    CLICKS_DECAY_PCT = -0.25  # 25% drop
    CTR_DECAY_PCT = -0.15  # 15% drop
    
    def __init__(self, custom_thresholds: Dict = None):
        """
        Initialize decay detector.
        
        Args:
            custom_thresholds: Optional custom decay thresholds
        """
        if custom_thresholds:
            self.POSITION_DECAY_THRESHOLD = custom_thresholds.get(
                'position', self.POSITION_DECAY_THRESHOLD
            )
            self.IMPRESSIONS_DECAY_PCT = custom_thresholds.get(
                'impressions', self.IMPRESSIONS_DECAY_PCT
            )
            self.CLICKS_DECAY_PCT = custom_thresholds.get(
                'clicks', self.CLICKS_DECAY_PCT
            )
            self.CTR_DECAY_PCT = custom_thresholds.get(
                'ctr', self.CTR_DECAY_PCT
            )
    
    def calculate_change_metrics(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        key_column: str = 'query'
    ) -> pd.DataFrame:
        """
        Calculate change metrics between time periods.
        
        Args:
            current_df: Current period data
            previous_df: Previous period data
            key_column: Column to join on
            
        Returns:
            DataFrame with change metrics
        """
        if current_df.empty or previous_df.empty:
            return pd.DataFrame()
        
        # Prepare previous data with suffix
        prev_cols = [
            key_column, 'clicks', 'impressions', 'position', 'ctr'
        ]
        available_cols = [c for c in prev_cols if c in previous_df.columns]
        prev_subset = previous_df[available_cols].copy()
        
        rename_map = {
            'clicks': 'prev_clicks',
            'impressions': 'prev_impressions',
            'position': 'prev_position',
            'ctr': 'prev_ctr'
        }
        prev_subset = prev_subset.rename(columns=rename_map)
        
        # Merge datasets
        merged = pd.merge(
            current_df,
            prev_subset,
            on=key_column,
            how='outer'
        )
        
        # Fill NaN with 0 for calculations
        for col in ['clicks', 'impressions', 'prev_clicks', 'prev_impressions']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Calculate changes
        merged['clicks_change'] = merged['clicks'] - merged['prev_clicks']
        merged['clicks_change_pct'] = merged.apply(
            lambda r: (r['clicks_change'] / r['prev_clicks'])
            if r['prev_clicks'] > 0 else 0,
            axis=1
        )
        
        merged['impressions_change'] = (
            merged['impressions'] - merged['prev_impressions']
        )
        merged['impressions_change_pct'] = merged.apply(
            lambda r: (r['impressions_change'] / r['prev_impressions'])
            if r['prev_impressions'] > 0 else 0,
            axis=1
        )
        
        if 'position' in merged.columns and 'prev_position' in merged.columns:
            merged['position_change'] = (
                merged['prev_position'] - merged['position']
            )  # Positive = improved
        
        if 'ctr' in merged.columns and 'prev_ctr' in merged.columns:
            merged['ctr_change'] = merged['ctr'] - merged['prev_ctr']
            merged['ctr_change_pct'] = merged.apply(
                lambda r: (r['ctr_change'] / r['prev_ctr'])
                if r['prev_ctr'] > 0 else 0,
                axis=1
            )
        
        return merged
    
    def classify_decay(
        self,
        row: pd.Series
    ) -> Tuple[bool, List[str]]:
        """
        Classify decay type for a single row.
        
        Args:
            row: Data row with change metrics
            
        Returns:
            Tuple of (is_decaying, list of decay types)
        """
        decay_types = []
        
        # Check position decay
        pos_change = row.get('position_change', 0)
        if pos_change < -self.POSITION_DECAY_THRESHOLD:
            decay_types.append(DecayType.POSITION_DROP.value)
        
        # Check impressions decay
        imp_change = row.get('impressions_change_pct', 0)
        if imp_change < self.IMPRESSIONS_DECAY_PCT:
            decay_types.append(DecayType.IMPRESSIONS_DROP.value)
        
        # Check clicks decay
        clicks_change = row.get('clicks_change_pct', 0)
        if clicks_change < self.CLICKS_DECAY_PCT:
            decay_types.append(DecayType.CLICKS_DROP.value)
        
        # Check CTR decay
        ctr_change = row.get('ctr_change_pct', 0)
        if ctr_change < self.CTR_DECAY_PCT:
            decay_types.append(DecayType.CTR_DROP.value)
        
        # Classify combined patterns
        if len(decay_types) >= 3:
            decay_types.append(DecayType.FULL_DECAY.value)
        
        # Demand shift: impressions dropped but position stable
        if (DecayType.IMPRESSIONS_DROP.value in decay_types and
                DecayType.POSITION_DROP.value not in decay_types):
            decay_types.append(DecayType.DEMAND_SHIFT.value)
        
        # Competition loss: position dropped but impressions stable
        if (DecayType.POSITION_DROP.value in decay_types and
                DecayType.IMPRESSIONS_DROP.value not in decay_types):
            decay_types.append(DecayType.COMPETITION_LOSS.value)
        
        is_decaying = len(decay_types) > 0
        return is_decaying, decay_types
    
    def detect_decaying_keywords(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        min_prev_clicks: int = 5
    ) -> pd.DataFrame:
        """
        Detect decaying keywords.
        
        Args:
            current_df: Current period data
            previous_df: Previous period data
            min_prev_clicks: Minimum previous clicks to consider
            
        Returns:
            DataFrame with decaying keywords
        """
        # Calculate changes
        changes = self.calculate_change_metrics(
            current_df, previous_df, 'query'
        )
        
        if changes.empty:
            return pd.DataFrame()
        
        # Filter to meaningful keywords
        changes = changes[changes['prev_clicks'] >= min_prev_clicks]
        
        # Classify decay for each keyword
        decay_results = []
        for idx, row in changes.iterrows():
            is_decaying, decay_types = self.classify_decay(row)
            if is_decaying:
                decay_results.append({
                    'query': row.get('query', ''),
                    'current_clicks': row.get('clicks', 0),
                    'prev_clicks': row.get('prev_clicks', 0),
                    'clicks_change_pct': round(
                        row.get('clicks_change_pct', 0) * 100, 1
                    ),
                    'current_position': row.get('position', 0),
                    'prev_position': row.get('prev_position', 0),
                    'position_change': round(row.get('position_change', 0), 1),
                    'impressions_change_pct': round(
                        row.get('impressions_change_pct', 0) * 100, 1
                    ),
                    'decay_types': decay_types,
                    'primary_decay': decay_types[0] if decay_types else None,
                    'severity': len(decay_types)
                })
        
        result = pd.DataFrame(decay_results)
        
        if not result.empty:
            result = result.sort_values('severity', ascending=False)
        
        return result
    
    def detect_decaying_pages(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        min_prev_clicks: int = 10
    ) -> pd.DataFrame:
        """
        Detect decaying pages.
        
        Args:
            current_df: Current period page data
            previous_df: Previous period page data
            min_prev_clicks: Minimum previous clicks
            
        Returns:
            DataFrame with decaying pages
        """
        # Calculate changes
        changes = self.calculate_change_metrics(
            current_df, previous_df, 'page'
        )
        
        if changes.empty:
            return pd.DataFrame()
        
        # Filter to meaningful pages
        changes = changes[changes['prev_clicks'] >= min_prev_clicks]
        
        # Classify decay
        decay_results = []
        for idx, row in changes.iterrows():
            is_decaying, decay_types = self.classify_decay(row)
            if is_decaying:
                decay_results.append({
                    'page': row.get('page', ''),
                    'current_clicks': row.get('clicks', 0),
                    'prev_clicks': row.get('prev_clicks', 0),
                    'clicks_change_pct': round(
                        row.get('clicks_change_pct', 0) * 100, 1
                    ),
                    'current_impressions': row.get('impressions', 0),
                    'prev_impressions': row.get('prev_impressions', 0),
                    'current_position': row.get('position', 0),
                    'position_change': round(row.get('position_change', 0), 1),
                    'decay_types': decay_types,
                    'primary_decay': decay_types[0] if decay_types else None,
                    'severity': len(decay_types)
                })
        
        result = pd.DataFrame(decay_results)
        
        if not result.empty:
            result = result.sort_values('severity', ascending=False)
        
        return result
    
    def get_recovery_recommendations(
        self,
        decay_type: str
    ) -> Dict[str, str]:
        """
        Get recovery recommendations based on decay type.
        
        Args:
            decay_type: Type of decay detected
            
        Returns:
            Dict with recommendations
        """
        recommendations = {
            DecayType.POSITION_DROP.value: {
                'diagnosis': 'Position dropped while impressions stable',
                'likely_cause': 'Competitors improved or algorithm change',
                'actions': [
                    'Audit competing pages for content gaps',
                    'Update content with fresh information',
                    'Improve internal linking to the page',
                    'Add structured data if missing',
                    'Check for technical issues (page speed, mobile)'
                ]
            },
            DecayType.IMPRESSIONS_DROP.value: {
                'diagnosis': 'Impressions dropped',
                'likely_cause': 'Reduced search demand or lost visibility',
                'actions': [
                    'Check if search demand decreased (trends)',
                    'Verify page is still indexed',
                    'Look for cannibalization with other pages',
                    'Expand keyword targeting',
                    'Consider seasonality factors'
                ]
            },
            DecayType.CLICKS_DROP.value: {
                'diagnosis': 'Clicks dropped despite impressions',
                'likely_cause': 'CTR issue - snippets less compelling',
                'actions': [
                    'Optimize title tag for CTR',
                    'Improve meta description',
                    'Add structured data for rich snippets',
                    'Check SERP for new features taking clicks',
                    'A/B test different titles'
                ]
            },
            DecayType.CTR_DROP.value: {
                'diagnosis': 'CTR specifically dropped',
                'likely_cause': 'SERP changes or less compelling listing',
                'actions': [
                    'Analyze SERP changes for key queries',
                    'Update title with current year/benefits',
                    'Add FAQ schema for more SERP space',
                    'Check for featured snippets to target',
                    'Analyze competitor snippets'
                ]
            },
            DecayType.FULL_DECAY.value: {
                'diagnosis': 'Multiple metrics declining together',
                'likely_cause': 'Content becoming outdated or major issue',
                'actions': [
                    'Full content refresh needed',
                    'Check for technical/indexation issues',
                    'Consider consolidating with other pages',
                    'Evaluate if topic is still relevant',
                    'Major update or rebuild may be required'
                ]
            },
            DecayType.DEMAND_SHIFT.value: {
                'diagnosis': 'Search demand has decreased',
                'likely_cause': 'Market or seasonal change',
                'actions': [
                    'Research evolving search intent',
                    'Find related growing topics',
                    'Update content for new angles',
                    'Consider pivoting content focus',
                    'Add related subtopics'
                ]
            },
            DecayType.COMPETITION_LOSS.value: {
                'diagnosis': 'Lost ground to competitors',
                'likely_cause': 'Competitors improved their content',
                'actions': [
                    'Analyze what competitors are doing better',
                    'Update and expand content depth',
                    'Improve page experience metrics',
                    'Build more relevant backlinks',
                    'Add unique data or insights'
                ]
            }
        }
        
        return recommendations.get(decay_type, {
            'diagnosis': 'Unknown decay pattern',
            'likely_cause': 'Requires manual investigation',
            'actions': ['Conduct detailed manual audit']
        })
    
    def summarize_decay(
        self,
        decaying_keywords: pd.DataFrame,
        decaying_pages: pd.DataFrame
    ) -> Dict:
        """
        Summarize decay findings.
        
        Args:
            decaying_keywords: Decaying keywords DataFrame
            decaying_pages: Decaying pages DataFrame
            
        Returns:
            Summary dict
        """
        summary = {
            'keywords': {
                'total_decaying': len(decaying_keywords),
                'by_type': {},
                'total_clicks_lost': 0,
                'avg_severity': 0
            },
            'pages': {
                'total_decaying': len(decaying_pages),
                'by_type': {},
                'total_clicks_lost': 0,
                'avg_severity': 0
            }
        }
        
        if not decaying_keywords.empty:
            summary['keywords']['total_clicks_lost'] = int(
                decaying_keywords['clicks_change_pct'].apply(
                    lambda x: abs(x) if x < 0 else 0
                ).sum()
            )
            summary['keywords']['avg_severity'] = round(
                decaying_keywords['severity'].mean(), 1
            )
            
            # Count by type
            all_types = []
            for types in decaying_keywords['decay_types']:
                all_types.extend(types)
            for t in set(all_types):
                summary['keywords']['by_type'][t] = all_types.count(t)
        
        if not decaying_pages.empty:
            summary['pages']['total_clicks_lost'] = int(
                decaying_pages['clicks_change_pct'].apply(
                    lambda x: abs(x) if x < 0 else 0
                ).sum()
            )
            summary['pages']['avg_severity'] = round(
                decaying_pages['severity'].mean(), 1
            )
            
            all_types = []
            for types in decaying_pages['decay_types']:
                all_types.extend(types)
            for t in set(all_types):
                summary['pages']['by_type'][t] = all_types.count(t)
        
        return summary