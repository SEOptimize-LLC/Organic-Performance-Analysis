"""
Metric card components for Streamlit dashboard.
Displays KPIs and summary statistics.
"""

import streamlit as st
from typing import Dict, Optional, Union
import pandas as pd


class MetricCards:
    """
    Creates metric card displays for KPIs.
    Uses Streamlit native components.
    """
    
    @staticmethod
    def format_number(value: Union[int, float], decimals: int = 0) -> str:
        """Format number with thousands separator."""
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, float):
            if decimals > 0:
                return f"{value:,.{decimals}f}"
            return f"{int(value):,}"
        return f"{value:,}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format as percentage."""
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_delta(value: float, is_percentage: bool = False) -> str:
        """Format delta value with sign."""
        if pd.isna(value):
            return None
        
        sign = "+" if value > 0 else ""
        if is_percentage:
            return f"{sign}{value:.1f}%"
        return f"{sign}{value:,.0f}"
    
    @classmethod
    def overview_metrics(
        cls,
        metrics: Dict,
        comparison: Dict = None
    ):
        """
        Display overview metric cards.
        
        Args:
            metrics: Current metrics dict
            comparison: Previous period for delta calculation
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = None
            if comparison and 'clicks' in comparison:
                delta = cls.format_delta(
                    metrics.get('clicks', 0) - comparison.get('clicks', 0)
                )
            st.metric(
                label="Total Clicks",
                value=cls.format_number(metrics.get('clicks', 0)),
                delta=delta
            )
        
        with col2:
            delta = None
            if comparison and 'impressions' in comparison:
                delta = cls.format_delta(
                    metrics.get('impressions', 0) - comparison.get('impressions', 0)  # noqa: E501
                )
            st.metric(
                label="Total Impressions",
                value=cls.format_number(metrics.get('impressions', 0)),
                delta=delta
            )
        
        with col3:
            current_ctr = metrics.get('ctr', 0) * 100
            delta = None
            if comparison and 'ctr' in comparison:
                prev_ctr = comparison.get('ctr', 0) * 100
                delta = cls.format_delta(current_ctr - prev_ctr, True)
            st.metric(
                label="Average CTR",
                value=cls.format_percentage(current_ctr),
                delta=delta
            )
        
        with col4:
            current_pos = metrics.get('position', 0)
            delta = None
            if comparison and 'position' in comparison:
                # For position, lower is better
                pos_change = comparison.get('position', 0) - current_pos
                delta = cls.format_delta(pos_change)
            st.metric(
                label="Avg Position",
                value=f"{current_pos:.1f}",
                delta=delta,
                delta_color="inverse"  # Green for lower position
            )
    
    @classmethod
    def brand_metrics(cls, brand_data: Dict):
        """
        Display brand vs non-brand metrics.
        
        Args:
            brand_data: Brand analysis dict
        """
        brand = brand_data.get('brand', {})
        non_brand = brand_data.get('non_brand', {})
        
        st.subheader("Brand vs Non-Brand")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Brand Click Share",
                value=cls.format_percentage(brand.get('click_share', 0)),
                help="Percentage of clicks from brand queries"
            )
        
        with col2:
            st.metric(
                label="Non-Brand Clicks",
                value=cls.format_number(non_brand.get('clicks', 0)),
                help="Total clicks from non-brand queries"
            )
        
        with col3:
            dependency = brand_data.get('dependency_score', 0)
            st.metric(
                label="Brand Dependency",
                value=cls.format_percentage(dependency),
                help="Higher = more reliant on brand traffic"
            )
        
        # Additional row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Brand Queries",
                value=cls.format_number(brand.get('queries', 0))
            )
        
        with col2:
            st.metric(
                label="Non-Brand Queries",
                value=cls.format_number(non_brand.get('queries', 0))
            )
        
        with col3:
            st.metric(
                label="Non-Brand Avg CTR",
                value=cls.format_percentage(non_brand.get('avg_ctr', 0))
            )
    
    @classmethod
    def opportunity_summary(cls, opportunities: Dict[str, pd.DataFrame]):
        """
        Display opportunity count summary.
        
        Args:
            opportunities: Dict of opportunity DataFrames
        """
        st.subheader("Opportunities Identified")
        
        cols = st.columns(5)
        
        labels = {
            'quick_wins': '🎯 Quick Wins',
            'ctr_opportunities': '📈 CTR Opps',
            'scaling_opportunities': '🚀 Scale Opps',
            'declining_keywords': '📉 Declining',
            'new_opportunities': '✨ New Opps'
        }
        
        for i, (key, label) in enumerate(labels.items()):
            with cols[i]:
                count = len(opportunities.get(key, []))
                st.metric(label=label, value=count)
    
    @classmethod
    def decay_summary(cls, decay_data: Dict):
        """
        Display decay summary metrics.
        
        Args:
            decay_data: Decay summary dict
        """
        st.subheader("Content Decay Status")
        
        kw_data = decay_data.get('keywords', {})
        page_data = decay_data.get('pages', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Decaying Keywords",
                value=cls.format_number(kw_data.get('total_decaying', 0)),
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Decaying Pages",
                value=cls.format_number(page_data.get('total_decaying', 0)),
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="KW Severity",
                value=f"{kw_data.get('avg_severity', 0):.1f}",
                help="Average decay indicators per keyword"
            )
        
        with col4:
            st.metric(
                label="Page Severity",
                value=f"{page_data.get('avg_severity', 0):.1f}",
                help="Average decay indicators per page"
            )
    
    @classmethod
    def competitor_summary(
        cls,
        competitor_df: pd.DataFrame,
        domain_overview: Dict
    ):
        """
        Display competitor summary.
        
        Args:
            competitor_df: Competitors DataFrame
            domain_overview: Domain metrics
        """
        st.subheader("Competitive Landscape")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Competitors Found",
                value=cls.format_number(len(competitor_df))
            )
        
        with col2:
            avg_overlap = competitor_df['intersections'].mean() if not competitor_df.empty else 0  # noqa: E501
            st.metric(
                label="Avg Keyword Overlap",
                value=cls.format_number(avg_overlap)
            )
        
        with col3:
            st.metric(
                label="Your Organic ETV",
                value=cls.format_number(
                    domain_overview.get('organic_etv', 0)
                )
            )
        
        with col4:
            st.metric(
                label="Ranking Keywords",
                value=cls.format_number(
                    domain_overview.get('organic_count', 0)
                )
            )
    
    @classmethod
    def device_comparison(cls, device_metrics: Dict):
        """
        Display device comparison metrics.
        
        Args:
            device_metrics: Device performance dict
        """
        st.subheader("Device Performance")
        
        devices = list(device_metrics.keys())
        cols = st.columns(len(devices))
        
        for i, device in enumerate(devices):
            with cols[i]:
                data = device_metrics[device]
                st.markdown(f"**{device.title()}**")
                st.metric(
                    label="Clicks",
                    value=cls.format_number(data.get('clicks', 0))
                )
                st.metric(
                    label="Avg Position",
                    value=f"{data.get('position', 0):.1f}"
                )
    
    @classmethod
    def scoring_breakdown(cls, score_data: Dict):
        """
        Display opportunity score breakdown.
        
        Args:
            score_data: Score components dict
        """
        st.subheader("Score Components")
        
        components = [
            ('volume_score', 'Search Volume', '📊'),
            ('position_score', 'Position Potential', '📍'),
            ('ctr_gap_score', 'CTR Opportunity', '🖱️'),
            ('commercial_score', 'Commercial Value', '💰'),
            ('trend_score', 'Trend Direction', '📈')
        ]
        
        cols = st.columns(5)
        
        for i, (key, label, emoji) in enumerate(components):
            with cols[i]:
                value = score_data.get(key, 0)
                st.metric(
                    label=f"{emoji} {label}",
                    value=f"{value:.2f}"
                )
    
    @classmethod
    def data_quality(
        cls,
        gsc_rows: int,
        dfs_rows: int,
        date_range: str
    ):
        """
        Display data quality indicators.
        
        Args:
            gsc_rows: Number of GSC data rows
            dfs_rows: Number of DataForSEO rows
            date_range: Analysis date range
        """
        st.subheader("Data Quality")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="GSC Data Points",
                value=cls.format_number(gsc_rows)
            )
        
        with col2:
            st.metric(
                label="DataForSEO Keywords",
                value=cls.format_number(dfs_rows)
            )
        
        with col3:
            st.metric(
                label="Analysis Period",
                value=date_range
            )