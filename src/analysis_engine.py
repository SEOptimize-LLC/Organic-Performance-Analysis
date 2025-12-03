"""
SEO Analysis Engine

Handles data processing, normalization, and analysis logic for
organic performance optimization.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re


@dataclass
class AnalysisConfig:
    """Configuration for SEO analysis."""
    brand_terms: List[str]
    min_impressions: int = 10
    ctr_benchmark_enabled: bool = True
    position_threshold_low: int = 3
    position_threshold_high: int = 15
    decay_threshold_percent: float = 20.0
    competitor_domains: List[str] = None

    def __post_init__(self):
        if self.competitor_domains is None:
            self.competitor_domains = []


class SEOAnalysisEngine:
    """Core analysis engine for SEO data processing."""

    # Average CTR by position benchmark (Google organic)
    CTR_BENCHMARKS = {
        1: 0.316,
        2: 0.147,
        3: 0.085,
        4: 0.059,
        5: 0.042,
        6: 0.032,
        7: 0.025,
        8: 0.021,
        9: 0.018,
        10: 0.016,
        11: 0.010,
        12: 0.009,
        13: 0.008,
        14: 0.007,
        15: 0.006,
        16: 0.005,
        17: 0.004,
        18: 0.004,
        19: 0.003,
        20: 0.003
    }

    def __init__(self, config: AnalysisConfig):
        """Initialize the analysis engine with configuration."""
        self.config = config

    def normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for matching across data sources."""
        if not keyword:
            return ""
        # Lowercase, remove extra whitespace, strip punctuation
        keyword = keyword.lower().strip()
        keyword = re.sub(r'\s+', ' ', keyword)
        keyword = re.sub(r'[^\w\s]', '', keyword)
        return keyword

    def classify_brand_queries(self, df: pd.DataFrame, query_col: str = 'query') -> pd.DataFrame:
        """Classify queries as brand or non-brand."""
        if df.empty or not self.config.brand_terms:
            df['is_brand'] = False
            return df

        brand_pattern = '|'.join([
            re.escape(term.lower()) for term in self.config.brand_terms
        ])

        df['is_brand'] = df[query_col].str.lower().str.contains(
            brand_pattern,
            regex=True,
            na=False
        )
        return df

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent.

        Categories:
        - navigational: Brand/site-specific queries
        - informational: How, what, why, guide, tutorial
        - commercial: Reviews, comparison, best, top
        - transactional: Buy, price, order, discount, near me
        """
        query = query.lower()

        # Transactional signals
        transactional_patterns = [
            r'\bbuy\b', r'\bprice\b', r'\bcost\b', r'\border\b',
            r'\bpurchase\b', r'\bdiscount\b', r'\bcoupon\b', r'\bdeal\b',
            r'\bcheap\b', r'\bnear me\b', r'\bfor sale\b', r'\bshipping\b'
        ]
        for pattern in transactional_patterns:
            if re.search(pattern, query):
                return 'transactional'

        # Commercial investigation
        commercial_patterns = [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompar',
            r'\bvs\b', r'\balternative\b', r'\brating\b'
        ]
        for pattern in commercial_patterns:
            if re.search(pattern, query):
                return 'commercial'

        # Informational
        informational_patterns = [
            r'^how\b', r'^what\b', r'^why\b', r'^when\b', r'^where\b',
            r'^who\b', r'\bguide\b', r'\btutorial\b', r'\bexplain\b',
            r'\bdefinition\b', r'\bmeaning\b', r'^is\s', r'^are\s',
            r'^can\s', r'^does\s'
        ]
        for pattern in informational_patterns:
            if re.search(pattern, query):
                return 'informational'

        # Default to informational for unclear queries
        return 'informational'

    def get_expected_ctr(self, position: float) -> float:
        """Get expected CTR based on position benchmark."""
        pos_int = max(1, min(20, int(round(position))))
        return self.CTR_BENCHMARKS.get(pos_int, 0.003)

    def calculate_ctr_gap(self, row: pd.Series) -> float:
        """Calculate CTR gap from benchmark."""
        expected = self.get_expected_ctr(row.get('position', 100))
        actual = row.get('ctr', 0)
        return actual - expected

    def calculate_opportunity_score(
        self,
        row: pd.Series,
        has_search_volume: bool = False
    ) -> float:
        """
        Calculate opportunity score for a query-page pair.

        Factors:
        - Search volume / impressions (scale factor)
        - Position (closer to page 1 = higher opportunity)
        - CTR gap (underperforming vs benchmark = higher opportunity)
        - Commercial value (CPC proxy)
        - Trend direction (decaying = higher priority)
        """
        score = 0.0

        # Base score from impressions or search volume
        impressions = row.get('impressions', 0)
        search_volume = row.get('search_volume', impressions)
        volume_factor = np.log1p(max(search_volume, impressions)) / 10
        score += volume_factor * 30

        # Position factor (positions 3-15 have highest opportunity)
        position = row.get('position', 100)
        if 3 <= position <= 15:
            position_score = 40 - (position - 3) * 2  # Max 40 at position 3
        elif position < 3:
            position_score = 15  # Already ranking well
        elif 15 < position <= 30:
            position_score = 25 - (position - 15)
        else:
            position_score = 5

        score += position_score

        # CTR gap factor
        if self.config.ctr_benchmark_enabled:
            ctr_gap = self.calculate_ctr_gap(row)
            if ctr_gap < 0:
                # Underperforming CTR = opportunity
                score += abs(ctr_gap) * 100  # Max ~15 points

        # Commercial value factor (CPC)
        cpc = row.get('cpc', 0)
        if cpc > 0:
            cpc_score = min(cpc * 5, 15)  # Cap at 15 points
            score += cpc_score

        # Decay factor (if available)
        trend = row.get('trend', 0)
        if trend < 0:
            score += min(abs(trend) * 2, 10)  # Decaying content priority

        return round(score, 2)

    def identify_quick_wins(
        self,
        gsc_data: pd.DataFrame,
        dataforseo_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Identify quick-win opportunities.

        Criteria:
        - High impressions, low CTR
        - Position 3-15 (page 1 or top of page 2)
        - Underperforming vs CTR benchmark
        """
        if gsc_data.empty:
            return pd.DataFrame()

        df = gsc_data.copy()

        # Filter to quick-win position range
        df = df[
            (df['position'] >= self.config.position_threshold_low) &
            (df['position'] <= self.config.position_threshold_high) &
            (df['impressions'] >= self.config.min_impressions)
        ]

        if df.empty:
            return df

        # Calculate expected CTR and gap
        df['expected_ctr'] = df['position'].apply(self.get_expected_ctr)
        df['ctr_gap'] = df['ctr'] - df['expected_ctr']
        df['ctr_gap_pct'] = (df['ctr_gap'] / df['expected_ctr'] * 100).round(1)

        # Identify underperforming CTR
        df = df[df['ctr_gap'] < 0]

        # Merge with DataForSEO data if available
        if dataforseo_data is not None and not dataforseo_data.empty:
            df['keyword_normalized'] = df['query'].apply(self.normalize_keyword)
            dataforseo_data = dataforseo_data.copy()
            dataforseo_data['keyword_normalized'] = dataforseo_data['keyword'].apply(
                self.normalize_keyword
            )

            df = df.merge(
                dataforseo_data[['keyword_normalized', 'search_volume', 'cpc', 'competition']],
                on='keyword_normalized',
                how='left'
            )
            df = df.drop(columns=['keyword_normalized'])
            df['search_volume'] = df['search_volume'].fillna(0)
            df['cpc'] = df['cpc'].fillna(0)

        # Calculate opportunity score
        df['opportunity_score'] = df.apply(
            lambda x: self.calculate_opportunity_score(x, 'search_volume' in df.columns),
            axis=1
        )

        # Sort by opportunity score
        df = df.sort_values('opportunity_score', ascending=False)

        # Classify intent
        df['intent'] = df['query'].apply(self.classify_intent)

        return df

    def identify_content_decay(
        self,
        current_data: pd.DataFrame,
        previous_data: pd.DataFrame,
        comparison_type: str = 'period'
    ) -> pd.DataFrame:
        """
        Identify decaying content by comparing current vs previous period.

        Args:
            current_data: Current period GSC data
            previous_data: Previous period GSC data
            comparison_type: 'period' or 'year_over_year'

        Returns:
            DataFrame with decay analysis
        """
        if current_data.empty or previous_data.empty:
            return pd.DataFrame()

        # Aggregate by query if needed
        current_agg = current_data.groupby('query').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean',
            'page': 'first'
        }).reset_index()

        previous_agg = previous_data.groupby('query').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).reset_index()

        # Merge current and previous
        merged = current_agg.merge(
            previous_agg,
            on='query',
            suffixes=('_current', '_previous'),
            how='outer'
        )

        # Fill NaN with 0 for missing data
        for col in ['clicks_current', 'impressions_current', 'clicks_previous', 'impressions_previous']:
            merged[col] = merged[col].fillna(0)

        # Calculate changes
        merged['clicks_change'] = merged['clicks_current'] - merged['clicks_previous']
        merged['clicks_change_pct'] = np.where(
            merged['clicks_previous'] > 0,
            (merged['clicks_change'] / merged['clicks_previous'] * 100).round(1),
            0
        )

        merged['impressions_change'] = merged['impressions_current'] - merged['impressions_previous']
        merged['impressions_change_pct'] = np.where(
            merged['impressions_previous'] > 0,
            (merged['impressions_change'] / merged['impressions_previous'] * 100).round(1),
            0
        )

        merged['position_change'] = merged['position_current'] - merged['position_previous']

        # Classify decay type
        def classify_decay(row):
            pos_drop = row['position_change'] > 1  # Position got worse (higher number)
            imp_drop = row['impressions_change_pct'] < -self.config.decay_threshold_percent
            click_drop = row['clicks_change_pct'] < -self.config.decay_threshold_percent

            if pos_drop and not imp_drop:
                return 'competition_serp_change'
            elif imp_drop and not pos_drop:
                return 'demand_decline'
            elif pos_drop and imp_drop:
                return 'major_decline'
            elif click_drop and not (pos_drop or imp_drop):
                return 'ctr_decline'
            else:
                return 'stable'

        merged['decay_type'] = merged.apply(classify_decay, axis=1)

        # Filter to decaying content
        decaying = merged[merged['decay_type'] != 'stable']

        # Sort by click loss magnitude
        decaying = decaying.sort_values('clicks_change', ascending=True)

        return decaying

    def identify_keyword_gaps(
        self,
        site_keywords: pd.DataFrame,
        competitor_keywords: pd.DataFrame,
        gsc_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Identify keyword gaps where competitors rank but target site doesn't.

        Args:
            site_keywords: DataForSEO ranked keywords for target site
            competitor_keywords: DataForSEO ranked keywords for competitors
            gsc_data: Optional GSC data to filter out existing rankings

        Returns:
            DataFrame with keyword gaps
        """
        if competitor_keywords.empty:
            return pd.DataFrame()

        # Get set of site keywords
        site_kw_set = set()
        if not site_keywords.empty:
            site_kw_set = set(
                site_keywords['keyword'].apply(self.normalize_keyword).tolist()
            )

        # Add GSC queries to known keywords
        if gsc_data is not None and not gsc_data.empty:
            gsc_kw_set = set(
                gsc_data['query'].apply(self.normalize_keyword).tolist()
            )
            site_kw_set = site_kw_set.union(gsc_kw_set)

        # Filter competitor keywords to gaps
        competitor_keywords = competitor_keywords.copy()
        competitor_keywords['keyword_normalized'] = competitor_keywords['keyword'].apply(
            self.normalize_keyword
        )

        gaps = competitor_keywords[
            ~competitor_keywords['keyword_normalized'].isin(site_kw_set)
        ].copy()

        if gaps.empty:
            return gaps

        # Calculate opportunity score for gaps
        gaps['opportunity_score'] = gaps.apply(
            lambda x: self._calculate_gap_score(x),
            axis=1
        )

        # Classify intent
        gaps['intent'] = gaps['keyword'].apply(self.classify_intent)

        # Sort by opportunity
        gaps = gaps.sort_values('opportunity_score', ascending=False)

        # Drop normalized column
        gaps = gaps.drop(columns=['keyword_normalized'])

        return gaps

    def _calculate_gap_score(self, row: pd.Series) -> float:
        """Calculate opportunity score for keyword gaps."""
        score = 0.0

        # Search volume factor
        search_volume = row.get('search_volume', 0)
        score += np.log1p(search_volume) * 5

        # CPC factor (commercial value)
        cpc = row.get('cpc', 0)
        score += min(cpc * 10, 30)

        # Competitor position factor (easier if competitors rank lower)
        position = row.get('position', 50)
        if position > 10:
            score += 10
        elif position > 5:
            score += 5

        return round(score, 2)

    def cluster_keywords(
        self,
        keywords: pd.DataFrame,
        method: str = 'stem'
    ) -> pd.DataFrame:
        """
        Cluster keywords into topic groups.

        Args:
            keywords: DataFrame with keyword data
            method: Clustering method ('stem' for simple word-based)

        Returns:
            DataFrame with cluster assignments
        """
        if keywords.empty:
            return keywords

        keywords = keywords.copy()

        # Simple word-based clustering
        def extract_cluster_seed(kw):
            words = kw.lower().split()
            # Remove common words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was',
                        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                        'does', 'did', 'will', 'would', 'could', 'should', 'may',
                        'might', 'can', 'how', 'what', 'why', 'when', 'where', 'who'}
            meaningful = [w for w in words if w not in stopwords and len(w) > 2]
            return meaningful[0] if meaningful else words[0] if words else 'other'

        keywords['cluster'] = keywords['keyword'].apply(extract_cluster_seed)

        # Group by cluster and aggregate metrics
        cluster_stats = keywords.groupby('cluster').agg({
            'keyword': 'count',
            'search_volume': 'sum',
            'cpc': 'mean'
        }).reset_index()

        cluster_stats.columns = ['cluster', 'keyword_count', 'total_volume', 'avg_cpc']

        # Merge stats back
        keywords = keywords.merge(cluster_stats, on='cluster', how='left')

        return keywords

    def analyze_brand_dependency(
        self,
        gsc_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze brand vs non-brand traffic dependency.

        Returns metrics on brand/non-brand split and growth opportunities.
        """
        if gsc_data.empty:
            return {
                'brand_clicks': 0,
                'non_brand_clicks': 0,
                'brand_share': 0,
                'non_brand_share': 0,
                'brand_impressions': 0,
                'non_brand_impressions': 0
            }

        df = self.classify_brand_queries(gsc_data.copy())

        brand_data = df[df['is_brand']]
        non_brand_data = df[~df['is_brand']]

        total_clicks = df['clicks'].sum()
        total_impressions = df['impressions'].sum()

        return {
            'brand_clicks': int(brand_data['clicks'].sum()),
            'non_brand_clicks': int(non_brand_data['clicks'].sum()),
            'brand_share': round(brand_data['clicks'].sum() / max(total_clicks, 1) * 100, 1),
            'non_brand_share': round(non_brand_data['clicks'].sum() / max(total_clicks, 1) * 100, 1),
            'brand_impressions': int(brand_data['impressions'].sum()),
            'non_brand_impressions': int(non_brand_data['impressions'].sum()),
            'brand_avg_position': round(brand_data['position'].mean(), 1) if not brand_data.empty else 0,
            'non_brand_avg_position': round(non_brand_data['position'].mean(), 1) if not non_brand_data.empty else 0,
            'brand_avg_ctr': round(brand_data['ctr'].mean() * 100, 2) if not brand_data.empty else 0,
            'non_brand_avg_ctr': round(non_brand_data['ctr'].mean() * 100, 2) if not non_brand_data.empty else 0
        }

    def analyze_device_performance(
        self,
        gsc_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze performance by device type."""
        if gsc_data.empty or 'device' not in gsc_data.columns:
            return pd.DataFrame()

        device_stats = gsc_data.groupby('device').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).reset_index()

        device_stats['ctr_pct'] = (device_stats['ctr'] * 100).round(2)
        device_stats['position'] = device_stats['position'].round(1)

        return device_stats

    def analyze_country_performance(
        self,
        gsc_data: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Analyze performance by country."""
        if gsc_data.empty or 'country' not in gsc_data.columns:
            return pd.DataFrame()

        country_stats = gsc_data.groupby('country').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).reset_index()

        country_stats['ctr_pct'] = (country_stats['ctr'] * 100).round(2)
        country_stats['position'] = country_stats['position'].round(1)

        # Sort by clicks and get top N
        country_stats = country_stats.sort_values('clicks', ascending=False).head(top_n)

        return country_stats

    def generate_analysis_summary(
        self,
        gsc_data: pd.DataFrame,
        dataforseo_data: Optional[pd.DataFrame] = None,
        competitor_data: Optional[pd.DataFrame] = None,
        quick_wins: Optional[pd.DataFrame] = None,
        decay_data: Optional[pd.DataFrame] = None,
        keyword_gaps: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a comprehensive text summary of the analysis for LLM processing.
        """
        sections = []

        # Overall metrics
        if not gsc_data.empty:
            sections.append("## OVERALL PERFORMANCE METRICS")
            sections.append(f"- Total Clicks: {gsc_data['clicks'].sum():,}")
            sections.append(f"- Total Impressions: {gsc_data['impressions'].sum():,}")
            sections.append(f"- Average CTR: {gsc_data['ctr'].mean() * 100:.2f}%")
            sections.append(f"- Average Position: {gsc_data['position'].mean():.1f}")
            sections.append(f"- Unique Queries: {gsc_data['query'].nunique():,}")
            sections.append(f"- Unique Pages: {gsc_data['page'].nunique() if 'page' in gsc_data.columns else 'N/A'}")

            # Brand dependency
            brand_stats = self.analyze_brand_dependency(gsc_data)
            sections.append("\n## BRAND VS NON-BRAND ANALYSIS")
            sections.append(f"- Brand Traffic Share: {brand_stats['brand_share']}%")
            sections.append(f"- Non-Brand Traffic Share: {brand_stats['non_brand_share']}%")
            sections.append(f"- Brand Clicks: {brand_stats['brand_clicks']:,}")
            sections.append(f"- Non-Brand Clicks: {brand_stats['non_brand_clicks']:,}")
            sections.append(f"- Brand Avg Position: {brand_stats['brand_avg_position']}")
            sections.append(f"- Non-Brand Avg Position: {brand_stats['non_brand_avg_position']}")

            # Device analysis
            device_stats = self.analyze_device_performance(gsc_data)
            if not device_stats.empty:
                sections.append("\n## DEVICE PERFORMANCE")
                for _, row in device_stats.iterrows():
                    sections.append(f"- {row['device']}: {row['clicks']:,} clicks, {row['ctr_pct']}% CTR, Pos {row['position']}")

            # Country analysis
            country_stats = self.analyze_country_performance(gsc_data)
            if not country_stats.empty:
                sections.append("\n## TOP COUNTRIES BY CLICKS")
                for _, row in country_stats.iterrows():
                    sections.append(f"- {row['country']}: {row['clicks']:,} clicks, {row['ctr_pct']}% CTR, Pos {row['position']}")

        # DataForSEO domain metrics
        if dataforseo_data is not None and not dataforseo_data.empty:
            sections.append("\n## DATAFORSEO RANKING METRICS")
            sections.append(f"- Total Ranked Keywords: {len(dataforseo_data):,}")
            sections.append(f"- Estimated Traffic Value: {dataforseo_data['etv'].sum():,.0f}")
            sections.append(f"- Keywords in Top 3: {len(dataforseo_data[dataforseo_data['position'] <= 3]):,}")
            sections.append(f"- Keywords in Top 10: {len(dataforseo_data[dataforseo_data['position'] <= 10]):,}")
            sections.append(f"- Keywords on Page 2: {len(dataforseo_data[(dataforseo_data['position'] > 10) & (dataforseo_data['position'] <= 20)]):,}")

        # Quick wins
        if quick_wins is not None and not quick_wins.empty:
            sections.append("\n## TOP QUICK WIN OPPORTUNITIES")
            sections.append("(High impressions, underperforming CTR, positions 3-15)")
            for idx, row in quick_wins.head(20).iterrows():
                ctr_pct = row['ctr'] * 100
                expected_ctr = row.get('expected_ctr', 0) * 100
                sections.append(
                    f"- Query: \"{row['query']}\" | "
                    f"Pos: {row['position']:.1f} | "
                    f"Impr: {row['impressions']:,} | "
                    f"CTR: {ctr_pct:.2f}% (expected: {expected_ctr:.2f}%) | "
                    f"Score: {row['opportunity_score']}"
                )
                if 'page' in row:
                    sections.append(f"  Page: {row['page']}")

        # Content decay
        if decay_data is not None and not decay_data.empty:
            sections.append("\n## CONTENT DECAY ANALYSIS")
            for idx, row in decay_data.head(20).iterrows():
                sections.append(
                    f"- Query: \"{row['query']}\" | "
                    f"Decay Type: {row['decay_type']} | "
                    f"Clicks Change: {row['clicks_change']:+,} ({row['clicks_change_pct']:+.1f}%) | "
                    f"Impr Change: {row['impressions_change']:+,} ({row['impressions_change_pct']:+.1f}%) | "
                    f"Pos Change: {row['position_change']:+.1f}"
                )

        # Keyword gaps
        if keyword_gaps is not None and not keyword_gaps.empty:
            sections.append("\n## KEYWORD GAP OPPORTUNITIES")
            sections.append("(Keywords competitors rank for but target site doesn't)")
            for idx, row in keyword_gaps.head(30).iterrows():
                sections.append(
                    f"- \"{row['keyword']}\" | "
                    f"Volume: {row['search_volume']:,} | "
                    f"CPC: ${row['cpc']:.2f} | "
                    f"Competitor Pos: {row['position']} | "
                    f"Intent: {row['intent']} | "
                    f"Score: {row['opportunity_score']}"
                )

        # Competitor overview
        if competitor_data is not None and not competitor_data.empty:
            sections.append("\n## COMPETITOR OVERVIEW")
            for idx, row in competitor_data.head(10).iterrows():
                sections.append(
                    f"- {row['domain']}: "
                    f"{row['intersections']:,} shared keywords | "
                    f"ETV: {row['organic_etv']:,.0f} | "
                    f"Top 10 Keywords: {row.get('organic_pos_1', 0) + row.get('organic_pos_2_3', 0) + row.get('organic_pos_4_10', 0):,}"
                )

        return "\n".join(sections)


def create_analysis_config(
    brand_terms: List[str],
    min_impressions: int = 10,
    competitors: Optional[List[str]] = None
) -> AnalysisConfig:
    """Factory function to create analysis configuration."""
    return AnalysisConfig(
        brand_terms=brand_terms,
        min_impressions=min_impressions,
        competitor_domains=competitors or []
    )
