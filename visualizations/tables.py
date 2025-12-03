"""
Data table components for Streamlit dashboard.
Creates interactive and styled data tables.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


class DataTables:
    """
    Creates styled data tables for analysis display.
    Supports filtering, sorting, and export.
    """
    
    @staticmethod
    def style_dataframe(
        df: pd.DataFrame,
        highlight_cols: List[str] = None,
        format_rules: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Apply basic formatting to DataFrame.
        
        Args:
            df: Source DataFrame
            highlight_cols: Columns to highlight
            format_rules: Column format specifications
            
        Returns:
            Formatted DataFrame
        """
        styled = df.copy()
        
        if format_rules:
            for col, fmt in format_rules.items():
                if col in styled.columns:
                    if fmt == 'percent':
                        styled[col] = styled[col].apply(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                    elif fmt == 'number':
                        styled[col] = styled[col].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) else ""
                        )
                    elif fmt == 'decimal':
                        styled[col] = styled[col].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
        
        return styled
    
    @classmethod
    def quick_wins_table(
        cls,
        df: pd.DataFrame,
        max_rows: int = 50
    ):
        """
        Display quick wins opportunities table.
        
        Args:
            df: Quick wins DataFrame
            max_rows: Maximum rows to display
        """
        if df.empty:
            st.info("No quick win opportunities found")
            return
        
        # Select and rename columns
        display_cols = {
            'query': 'Keyword',
            'gsc_query': 'Keyword',
            'keyword': 'Keyword',
            'gsc_position': 'Position',
            'position': 'Position',
            'gsc_impressions': 'Impressions',
            'impressions': 'Impressions',
            'gsc_clicks': 'Clicks',
            'clicks': 'Clicks',
            'gsc_ctr': 'CTR',
            'ctr': 'CTR',
            'dfs_search_volume': 'Search Vol',
            'search_volume': 'Search Vol',
            'opportunity_score': 'Opp Score',
            'ctr_gap_score': 'CTR Gap'
        }
        
        available = [c for c in display_cols.keys() if c in df.columns]
        display_df = df[available].head(max_rows).copy()
        display_df = display_df.rename(
            columns={c: display_cols[c] for c in available}
        )
        
        # Remove duplicate column names (keep first)
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]
        
        # Format
        format_rules = {
            'CTR': 'percent',
            'Position': 'decimal',
            'Impressions': 'number',
            'Clicks': 'number',
            'Search Vol': 'number',
            'Opp Score': 'decimal',
            'CTR Gap': 'decimal'
        }
        
        display_df = cls.style_dataframe(display_df, format_rules=format_rules)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def decaying_keywords_table(
        cls,
        df: pd.DataFrame,
        max_rows: int = 30
    ):
        """
        Display decaying keywords table.
        
        Args:
            df: Decaying keywords DataFrame
            max_rows: Maximum rows
        """
        if df.empty:
            st.info("No decaying keywords detected")
            return
        
        display_cols = [
            'query', 'current_clicks', 'prev_clicks',
            'clicks_change_pct', 'current_position',
            'position_change', 'primary_decay', 'severity'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].head(max_rows).copy()
        
        # Rename for display
        rename_map = {
            'query': 'Keyword',
            'current_clicks': 'Clicks Now',
            'prev_clicks': 'Clicks Before',
            'clicks_change_pct': 'Change %',
            'current_position': 'Position',
            'position_change': 'Pos Change',
            'primary_decay': 'Decay Type',
            'severity': 'Severity'
        }
        display_df = display_df.rename(columns=rename_map)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def competitor_table(
        cls,
        df: pd.DataFrame
    ):
        """
        Display competitors table.
        
        Args:
            df: Competitors DataFrame
        """
        if df.empty:
            st.info("No competitor data available")
            return
        
        display_cols = [
            'competitor_domain', 'intersections', 'avg_position',
            'organic_etv', 'organic_count'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].copy()
        
        rename_map = {
            'competitor_domain': 'Competitor',
            'intersections': 'Keyword Overlap',
            'avg_position': 'Avg Position',
            'organic_etv': 'Est. Traffic Value',
            'organic_count': 'Ranking Keywords'
        }
        display_df = display_df.rename(columns=rename_map)
        
        format_rules = {
            'Keyword Overlap': 'number',
            'Avg Position': 'decimal',
            'Est. Traffic Value': 'number',
            'Ranking Keywords': 'number'
        }
        
        display_df = cls.style_dataframe(display_df, format_rules=format_rules)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def keyword_gaps_table(
        cls,
        df: pd.DataFrame,
        max_rows: int = 50
    ):
        """
        Display keyword gaps table.
        
        Args:
            df: Keyword gaps DataFrame
            max_rows: Maximum rows
        """
        if df.empty:
            st.info("No keyword gaps found")
            return
        
        display_cols = [
            'keyword', 'search_volume', 'cpc',
            'target1_position', 'target2_position', 'competition'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].head(max_rows).copy()
        
        rename_map = {
            'keyword': 'Keyword',
            'search_volume': 'Search Volume',
            'cpc': 'CPC',
            'target1_position': 'Your Position',
            'target2_position': 'Competitor Position',
            'competition': 'Competition'
        }
        display_df = display_df.rename(columns=rename_map)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def page_performance_table(
        cls,
        df: pd.DataFrame,
        max_rows: int = 30
    ):
        """
        Display page performance table.
        
        Args:
            df: Page data DataFrame
            max_rows: Maximum rows
        """
        if df.empty:
            st.info("No page data available")
            return
        
        display_cols = [
            'page', 'clicks', 'impressions', 'ctr', 'position',
            'page_score', 'query_count'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].head(max_rows).copy()
        
        # Truncate URLs for display
        if 'page' in display_df.columns:
            display_df['page'] = display_df['page'].apply(
                lambda x: x[-60:] if len(str(x)) > 60 else x
            )
        
        rename_map = {
            'page': 'Page',
            'clicks': 'Clicks',
            'impressions': 'Impressions',
            'ctr': 'CTR',
            'position': 'Position',
            'page_score': 'Score',
            'query_count': 'Keywords'
        }
        display_df = display_df.rename(columns=rename_map)
        
        format_rules = {
            'Clicks': 'number',
            'Impressions': 'number',
            'CTR': 'percent',
            'Position': 'decimal',
            'Score': 'decimal',
            'Keywords': 'number'
        }
        
        display_df = cls.style_dataframe(display_df, format_rules=format_rules)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def ranked_keywords_table(
        cls,
        df: pd.DataFrame,
        max_rows: int = 50
    ):
        """
        Display ranked keywords table.
        
        Args:
            df: Ranked keywords DataFrame
            max_rows: Maximum rows
        """
        if df.empty:
            st.info("No ranked keywords data")
            return
        
        display_cols = [
            'keyword', 'position', 'search_volume',
            'traffic', 'cpc', 'url'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].head(max_rows).copy()
        
        # Truncate URLs
        if 'url' in display_df.columns:
            display_df['url'] = display_df['url'].apply(
                lambda x: str(x)[-50:] if len(str(x)) > 50 else x
            )
        
        rename_map = {
            'keyword': 'Keyword',
            'position': 'Position',
            'search_volume': 'Search Volume',
            'traffic': 'Est. Traffic',
            'cpc': 'CPC',
            'url': 'Ranking URL'
        }
        display_df = display_df.rename(columns=rename_map)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def brand_queries_table(
        cls,
        brand_df: pd.DataFrame,
        non_brand_df: pd.DataFrame,
        max_rows: int = 20
    ):
        """
        Display brand/non-brand query tables.
        
        Args:
            brand_df: Brand queries DataFrame
            non_brand_df: Non-brand queries DataFrame
            max_rows: Maximum rows each
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Brand Queries**")
            if brand_df.empty:
                st.info("No brand queries")
            else:
                display = brand_df.head(max_rows)[
                    ['query', 'clicks', 'impressions']
                ].copy()
                display.columns = ['Query', 'Clicks', 'Impressions']
                st.dataframe(display, hide_index=True)
        
        with col2:
            st.markdown("**Top Non-Brand Queries**")
            if non_brand_df.empty:
                st.info("No non-brand queries")
            else:
                display = non_brand_df.head(max_rows)[
                    ['query', 'clicks', 'impressions']
                ].copy()
                display.columns = ['Query', 'Clicks', 'Impressions']
                st.dataframe(display, hide_index=True)
    
    @classmethod
    def serp_features_table(
        cls,
        df: pd.DataFrame
    ):
        """
        Display SERP features analysis table.
        
        Args:
            df: SERP features DataFrame
        """
        if df.empty:
            st.info("No SERP feature data")
            return
        
        display_cols = [
            'keyword', 'has_featured_snippet', 'paa_count',
            'organic_count', 'top_3_domains'
        ]
        
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].copy()
        
        rename_map = {
            'keyword': 'Keyword',
            'has_featured_snippet': 'Featured Snippet',
            'paa_count': 'PAA Count',
            'organic_count': 'Organic Results',
            'top_3_domains': 'Top 3 Domains'
        }
        display_df = display_df.rename(columns=rename_map)
        
        # Format top 3 domains
        if 'Top 3 Domains' in display_df.columns:
            display_df['Top 3 Domains'] = display_df['Top 3 Domains'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def custom_table(
        cls,
        df: pd.DataFrame,
        columns: List[str] = None,
        rename: Dict[str, str] = None,
        max_rows: int = 50,
        title: str = None
    ):
        """
        Display custom table with specified columns.
        
        Args:
            df: Source DataFrame
            columns: Columns to display
            rename: Column rename mapping
            max_rows: Maximum rows
            title: Optional title
        """
        if df.empty:
            st.info("No data available")
            return
        
        if title:
            st.markdown(f"**{title}**")
        
        display_df = df.copy()
        
        if columns:
            available = [c for c in columns if c in df.columns]
            display_df = display_df[available]
        
        display_df = display_df.head(max_rows)
        
        if rename:
            display_df = display_df.rename(columns=rename)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    @classmethod
    def download_button(
        cls,
        df: pd.DataFrame,
        filename: str,
        label: str = "Download CSV"
    ):
        """
        Add download button for DataFrame.
        
        Args:
            df: DataFrame to download
            filename: Download filename
            label: Button label
        """
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )