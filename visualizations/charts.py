"""
Chart visualization components using Plotly.
Creates interactive charts for organic performance analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional


class ChartBuilder:
    """
    Builds interactive charts for SEO analysis visualization.
    Uses Plotly for Streamlit compatibility.
    """
    
    # Color scheme
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffbb00',
        'info': '#17becf',
        'brand': '#9467bd',
        'non_brand': '#8c564b'
    }
    
    @classmethod
    def opportunity_scatter(
        cls,
        df: pd.DataFrame,
        x_col: str = 'position',
        y_col: str = 'impressions',
        size_col: str = 'opportunity_score',
        color_col: str = 'ctr_gap_score',
        hover_col: str = 'query',
        title: str = 'Keyword Opportunities'
    ) -> go.Figure:
        """
        Create scatter plot of keyword opportunities.
        
        Args:
            df: DataFrame with opportunity data
            x_col: X-axis column (position)
            y_col: Y-axis column (impressions)
            size_col: Bubble size column
            color_col: Color intensity column
            hover_col: Hover label column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if df.empty:
            return cls._empty_chart("No opportunity data")
        
        # Prepare data
        plot_df = df.copy()
        
        # Handle column name variations
        x = cls._get_column(plot_df, [x_col, f'gsc_{x_col}'])
        y = cls._get_column(plot_df, [y_col, f'gsc_{y_col}'])
        size = cls._get_column(plot_df, [size_col])
        color = cls._get_column(plot_df, [color_col])
        hover = cls._get_column(plot_df, [hover_col, 'keyword'])
        
        if x is None or y is None:
            return cls._empty_chart("Missing required columns")
        
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            size=size if size else None,
            color=color if color else None,
            hover_name=hover,
            title=title,
            color_continuous_scale='RdYlGn_r',
            labels={
                x: 'Position',
                y: 'Impressions',
                size: 'Opportunity Score' if size else '',
                color: 'CTR Gap' if color else ''
            }
        )
        
        fig.update_layout(
            xaxis_title="Position",
            yaxis_title="Impressions",
            xaxis=dict(autorange='reversed'),  # Position 1 on right
            height=500
        )
        
        return fig
    
    @classmethod
    def position_distribution(
        cls,
        df: pd.DataFrame,
        position_col: str = 'position',
        title: str = 'Position Distribution'
    ) -> go.Figure:
        """
        Create histogram of position distribution.
        
        Args:
            df: DataFrame with position data
            position_col: Position column name
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if df.empty:
            return cls._empty_chart("No position data")
        
        pos = cls._get_column(df, [position_col, 'gsc_position'])
        if pos is None:
            return cls._empty_chart("Position column not found")
        
        # Create position buckets
        bins = [0, 3, 10, 20, 50, 100]
        labels = ['1-3', '4-10', '11-20', '21-50', '50+']
        
        plot_df = df.copy()
        plot_df['position_bucket'] = pd.cut(
            plot_df[pos],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        bucket_counts = plot_df['position_bucket'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=bucket_counts.index.astype(str),
                y=bucket_counts.values,
                marker_color=[
                    cls.COLORS['success'],
                    cls.COLORS['primary'],
                    cls.COLORS['warning'],
                    cls.COLORS['secondary'],
                    cls.COLORS['danger']
                ]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Position Range",
            yaxis_title="Number of Keywords",
            height=400
        )
        
        return fig
    
    @classmethod
    def brand_vs_non_brand(
        cls,
        brand_metrics: Dict,
        chart_type: str = 'pie'
    ) -> go.Figure:
        """
        Create brand vs non-brand comparison chart.
        
        Args:
            brand_metrics: Brand metrics dict
            chart_type: 'pie' or 'bar'
            
        Returns:
            Plotly figure
        """
        brand = brand_metrics.get('brand', {})
        non_brand = brand_metrics.get('non_brand', {})
        
        if chart_type == 'pie':
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Brand', 'Non-Brand'],
                    values=[
                        brand.get('clicks', 0),
                        non_brand.get('clicks', 0)
                    ],
                    marker_colors=[
                        cls.COLORS['brand'],
                        cls.COLORS['non_brand']
                    ],
                    hole=0.4
                )
            ])
            fig.update_layout(
                title='Click Distribution: Brand vs Non-Brand',
                height=400
            )
        else:
            categories = ['Clicks', 'Impressions', 'Queries']
            brand_vals = [
                brand.get('clicks', 0),
                brand.get('impressions', 0),
                brand.get('queries', 0)
            ]
            non_brand_vals = [
                non_brand.get('clicks', 0),
                non_brand.get('impressions', 0),
                non_brand.get('queries', 0)
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Brand',
                    x=categories,
                    y=brand_vals,
                    marker_color=cls.COLORS['brand']
                ),
                go.Bar(
                    name='Non-Brand',
                    x=categories,
                    y=non_brand_vals,
                    marker_color=cls.COLORS['non_brand']
                )
            ])
            fig.update_layout(
                barmode='group',
                title='Brand vs Non-Brand Metrics',
                height=400
            )
        
        return fig
    
    @classmethod
    def trend_comparison(
        cls,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        metric: str = 'clicks',
        top_n: int = 10,
        key_col: str = 'query'
    ) -> go.Figure:
        """
        Create trend comparison chart.
        
        Args:
            current_df: Current period data
            previous_df: Previous period data
            metric: Metric to compare
            top_n: Number of items to show
            key_col: Key column for comparison
            
        Returns:
            Plotly figure
        """
        if current_df.empty or previous_df.empty:
            return cls._empty_chart("Insufficient data for comparison")
        
        # Get top items from current period
        top_current = current_df.nlargest(top_n, metric)
        
        # Merge with previous
        merged = pd.merge(
            top_current[[key_col, metric]],
            previous_df[[key_col, metric]],
            on=key_col,
            how='left',
            suffixes=('_current', '_previous')
        )
        merged = merged.fillna(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Previous',
            x=merged[key_col],
            y=merged[f'{metric}_previous'],
            marker_color=cls.COLORS['secondary']
        ))
        
        fig.add_trace(go.Bar(
            name='Current',
            x=merged[key_col],
            y=merged[f'{metric}_current'],
            marker_color=cls.COLORS['primary']
        ))
        
        fig.update_layout(
            barmode='group',
            title=f'Top {top_n} {key_col.title()}s: {metric.title()} Comparison',
            xaxis_title=key_col.title(),
            yaxis_title=metric.title(),
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @classmethod
    def decay_waterfall(
        cls,
        decay_df: pd.DataFrame,
        metric: str = 'clicks_change_pct',
        top_n: int = 15
    ) -> go.Figure:
        """
        Create waterfall chart showing decay impact.
        
        Args:
            decay_df: Decaying items DataFrame
            metric: Change metric column
            top_n: Number of items
            
        Returns:
            Plotly figure
        """
        if decay_df.empty:
            return cls._empty_chart("No decay data")
        
        # Get top decaying items
        plot_df = decay_df.nsmallest(top_n, metric)
        
        key_col = 'query' if 'query' in plot_df.columns else 'page'
        
        fig = go.Figure(go.Waterfall(
            name="Decay",
            orientation="v",
            x=plot_df[key_col].str[:30],  # Truncate labels
            y=plot_df[metric],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": cls.COLORS['danger']}},
            increasing={"marker": {"color": cls.COLORS['success']}}
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Declining {key_col.title()}s",
            yaxis_title="Change %",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @classmethod
    def competitor_comparison(
        cls,
        competitor_df: pd.DataFrame,
        metric: str = 'intersections'
    ) -> go.Figure:
        """
        Create competitor comparison chart.
        
        Args:
            competitor_df: Competitor data
            metric: Comparison metric
            
        Returns:
            Plotly figure
        """
        if competitor_df.empty:
            return cls._empty_chart("No competitor data")
        
        domain_col = 'competitor_domain'
        if domain_col not in competitor_df.columns:
            domain_col = 'domain'
        
        fig = px.bar(
            competitor_df.head(10),
            x=domain_col,
            y=metric,
            title=f'Top Competitors by {metric.title()}',
            color=metric,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title="Competitor",
            yaxis_title=metric.replace('_', ' ').title(),
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @classmethod
    def device_performance(
        cls,
        device_data: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """
        Create device performance comparison.
        
        Args:
            device_data: Dict with device-segmented data
            
        Returns:
            Plotly figure
        """
        metrics = []
        for device, df in device_data.items():
            if not df.empty:
                metrics.append({
                    'device': device.title(),
                    'clicks': df['clicks'].sum(),
                    'impressions': df['impressions'].sum(),
                    'avg_position': df['position'].mean(),
                    'avg_ctr': df['ctr'].mean() * 100
                })
        
        if not metrics:
            return cls._empty_chart("No device data")
        
        metrics_df = pd.DataFrame(metrics)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Clicks by Device', 'Avg Position by Device')
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['device'],
                y=metrics_df['clicks'],
                name='Clicks',
                marker_color=cls.COLORS['primary']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['device'],
                y=metrics_df['avg_position'],
                name='Avg Position',
                marker_color=cls.COLORS['secondary']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Device Performance Comparison',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @classmethod
    def opportunity_heatmap(
        cls,
        scored_df: pd.DataFrame,
        top_n: int = 20
    ) -> go.Figure:
        """
        Create opportunity score heatmap.
        
        Args:
            scored_df: Scored keywords DataFrame
            top_n: Number of keywords
            
        Returns:
            Plotly figure
        """
        if scored_df.empty:
            return cls._empty_chart("No scored data")
        
        plot_df = scored_df.head(top_n)
        
        key_col = 'query' if 'query' in plot_df.columns else 'keyword'
        if 'gsc_query' in plot_df.columns:
            key_col = 'gsc_query'
        
        score_cols = [
            'volume_score', 'position_score', 'ctr_gap_score',
            'commercial_score', 'trend_score'
        ]
        available_scores = [c for c in score_cols if c in plot_df.columns]
        
        if not available_scores:
            return cls._empty_chart("No score columns found")
        
        z_data = plot_df[available_scores].values
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=available_scores,
            y=plot_df[key_col].str[:40],
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title='Opportunity Score Components',
            xaxis_title='Score Component',
            yaxis_title='Keyword',
            height=600
        )
        
        return fig
    
    @classmethod
    def _get_column(
        cls,
        df: pd.DataFrame,
        candidates: List[str]
    ) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    @classmethod
    def _empty_chart(cls, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )
        return fig