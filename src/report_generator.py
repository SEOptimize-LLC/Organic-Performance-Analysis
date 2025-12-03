"""
Report Generator Module

Generates formatted reports from SEO analysis results.
Supports multiple output formats: HTML, Markdown, PDF (via HTML).
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class ReportGenerator:
    """Generate SEO analysis reports in various formats."""

    def __init__(
        self,
        domain: str,
        analysis_date: Optional[datetime] = None
    ):
        """
        Initialize the report generator.

        Args:
            domain: The analyzed domain
            analysis_date: Date of analysis (defaults to now)
        """
        self.domain = domain
        self.analysis_date = analysis_date or datetime.now()

    def generate_markdown_report(
        self,
        gsc_data: Dict[str, pd.DataFrame],
        dataforseo_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        llm_analysis: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive markdown report.

        Args:
            gsc_data: Google Search Console data
            dataforseo_data: DataForSEO data
            analysis_results: Analysis engine results
            llm_analysis: LLM-generated analysis text

        Returns:
            Markdown formatted report string
        """
        sections = []

        # Header
        sections.append(f"# Organic Performance Analysis Report")
        sections.append(f"\n**Domain:** {self.domain}")
        sections.append(f"**Generated:** {self.analysis_date.strftime('%Y-%m-%d %H:%M')}")
        sections.append("\n---\n")

        # Executive Summary
        sections.append("## Executive Summary\n")
        sections.append(self._generate_executive_summary(gsc_data, dataforseo_data, analysis_results))

        # Performance Overview
        sections.append("\n## Performance Overview\n")
        sections.append(self._generate_performance_overview(gsc_data, dataforseo_data))

        # Brand Analysis
        brand_analysis = analysis_results.get('brand_analysis', {})
        if brand_analysis:
            sections.append("\n## Brand vs Non-Brand Analysis\n")
            sections.append(self._generate_brand_section(brand_analysis))

        # Quick Wins
        quick_wins = analysis_results.get('quick_wins', pd.DataFrame())
        if not quick_wins.empty:
            sections.append("\n## Quick Win Opportunities\n")
            sections.append(self._generate_quick_wins_section(quick_wins))

        # Content Decay
        decay = analysis_results.get('decay', pd.DataFrame())
        if not decay.empty:
            sections.append("\n## Content Decay Analysis\n")
            sections.append(self._generate_decay_section(decay))

        # Keyword Gaps
        gaps = analysis_results.get('keyword_gaps', pd.DataFrame())
        if not gaps.empty:
            sections.append("\n## Keyword Gap Opportunities\n")
            sections.append(self._generate_gaps_section(gaps))

        # Competitor Analysis
        competitors = dataforseo_data.get('competitors', pd.DataFrame())
        if not competitors.empty:
            sections.append("\n## Competitor Analysis\n")
            sections.append(self._generate_competitor_section(competitors))

        # LLM Analysis
        if llm_analysis:
            sections.append("\n## AI Strategic Analysis\n")
            sections.append(llm_analysis)

        # Action Items
        sections.append("\n## Prioritized Action Items\n")
        sections.append(self._generate_action_items(analysis_results))

        return "\n".join(sections)

    def _generate_executive_summary(
        self,
        gsc_data: Dict,
        dataforseo_data: Dict,
        analysis_results: Dict
    ) -> str:
        """Generate executive summary section."""
        lines = []

        query_data = gsc_data.get('query_data', pd.DataFrame())
        ranked_kw = dataforseo_data.get('ranked_keywords', pd.DataFrame())

        if not query_data.empty:
            total_clicks = int(query_data['clicks'].sum())
            total_impressions = int(query_data['impressions'].sum())
            avg_ctr = query_data['ctr'].mean() * 100
            avg_pos = query_data['position'].mean()

            lines.append(f"- **Total Organic Clicks:** {total_clicks:,}")
            lines.append(f"- **Total Impressions:** {total_impressions:,}")
            lines.append(f"- **Average CTR:** {avg_ctr:.2f}%")
            lines.append(f"- **Average Position:** {avg_pos:.1f}")

        if not ranked_kw.empty:
            lines.append(f"- **Ranked Keywords:** {len(ranked_kw):,}")
            lines.append(f"- **Top 10 Rankings:** {len(ranked_kw[ranked_kw['position'] <= 10]):,}")
            lines.append(f"- **Estimated Traffic Value:** {int(ranked_kw['etv'].sum()):,}")

        # Key findings
        quick_wins = analysis_results.get('quick_wins', pd.DataFrame())
        decay = analysis_results.get('decay', pd.DataFrame())
        gaps = analysis_results.get('keyword_gaps', pd.DataFrame())

        lines.append("\n### Key Findings:")
        lines.append(f"- **{len(quick_wins)}** quick-win opportunities identified")
        lines.append(f"- **{len(decay)}** queries showing content decay")
        lines.append(f"- **{len(gaps)}** keyword gaps from competitors")

        return "\n".join(lines)

    def _generate_performance_overview(
        self,
        gsc_data: Dict,
        dataforseo_data: Dict
    ) -> str:
        """Generate performance overview section."""
        lines = []

        query_data = gsc_data.get('query_data', pd.DataFrame())

        if not query_data.empty:
            lines.append("### Google Search Console Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Total Clicks | {int(query_data['clicks'].sum()):,} |")
            lines.append(f"| Total Impressions | {int(query_data['impressions'].sum()):,} |")
            lines.append(f"| Average CTR | {query_data['ctr'].mean() * 100:.2f}% |")
            lines.append(f"| Average Position | {query_data['position'].mean():.1f} |")
            lines.append(f"| Unique Queries | {query_data['query'].nunique():,} |")

        ranked_kw = dataforseo_data.get('ranked_keywords', pd.DataFrame())

        if not ranked_kw.empty:
            lines.append("\n### DataForSEO Ranking Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Total Keywords | {len(ranked_kw):,} |")
            lines.append(f"| Position 1 | {len(ranked_kw[ranked_kw['position'] == 1]):,} |")
            lines.append(f"| Positions 2-3 | {len(ranked_kw[(ranked_kw['position'] >= 2) & (ranked_kw['position'] <= 3)]):,} |")
            lines.append(f"| Positions 4-10 | {len(ranked_kw[(ranked_kw['position'] >= 4) & (ranked_kw['position'] <= 10)]):,} |")
            lines.append(f"| Positions 11-20 | {len(ranked_kw[(ranked_kw['position'] >= 11) & (ranked_kw['position'] <= 20)]):,} |")
            lines.append(f"| Est. Traffic Value | {int(ranked_kw['etv'].sum()):,} |")

        return "\n".join(lines)

    def _generate_brand_section(self, brand_analysis: Dict) -> str:
        """Generate brand analysis section."""
        lines = []

        lines.append("| Metric | Brand | Non-Brand |")
        lines.append("|--------|-------|-----------|")
        lines.append(f"| Clicks | {brand_analysis.get('brand_clicks', 0):,} | {brand_analysis.get('non_brand_clicks', 0):,} |")
        lines.append(f"| Share | {brand_analysis.get('brand_share', 0)}% | {brand_analysis.get('non_brand_share', 0)}% |")
        lines.append(f"| Avg Position | {brand_analysis.get('brand_avg_position', 0)} | {brand_analysis.get('non_brand_avg_position', 0)} |")
        lines.append(f"| Avg CTR | {brand_analysis.get('brand_avg_ctr', 0)}% | {brand_analysis.get('non_brand_avg_ctr', 0)}% |")

        # Insight
        brand_share = brand_analysis.get('brand_share', 0)
        if brand_share > 50:
            lines.append(f"\n> **Insight:** High brand dependency ({brand_share}%). Focus on growing non-brand visibility.")
        elif brand_share < 20:
            lines.append(f"\n> **Insight:** Low brand traffic ({brand_share}%). Consider brand awareness campaigns.")

        return "\n".join(lines)

    def _generate_quick_wins_section(self, quick_wins: pd.DataFrame) -> str:
        """Generate quick wins section."""
        lines = []

        lines.append(f"Identified **{len(quick_wins)}** opportunities with high impressions but underperforming CTR.\n")

        # Top opportunities table
        lines.append("### Top 15 Quick Win Opportunities\n")
        lines.append("| Query | Position | Impressions | CTR | Expected CTR | Gap | Score |")
        lines.append("|-------|----------|-------------|-----|--------------|-----|-------|")

        for _, row in quick_wins.head(15).iterrows():
            query = row['query'][:40] + "..." if len(row['query']) > 40 else row['query']
            lines.append(
                f"| {query} | {row['position']:.1f} | {row['impressions']:,} | "
                f"{row['ctr']*100:.2f}% | {row.get('expected_ctr', 0)*100:.2f}% | "
                f"{row.get('ctr_gap_pct', 0):.1f}% | {row['opportunity_score']} |"
            )

        # Intent breakdown
        if 'intent' in quick_wins.columns:
            lines.append("\n### Intent Distribution\n")
            intent_counts = quick_wins['intent'].value_counts()
            for intent, count in intent_counts.items():
                lines.append(f"- **{intent.title()}:** {count} opportunities")

        return "\n".join(lines)

    def _generate_decay_section(self, decay: pd.DataFrame) -> str:
        """Generate decay analysis section."""
        lines = []

        # Summary by type
        decay_summary = decay['decay_type'].value_counts()

        lines.append("### Decay Pattern Summary\n")
        lines.append("| Pattern | Count | Description |")
        lines.append("|---------|-------|-------------|")

        pattern_desc = {
            'competition_serp_change': 'Position dropped, impressions stable (competitors overtook)',
            'demand_decline': 'Impressions dropped, position stable (market interest decreased)',
            'major_decline': 'Both position and impressions dropped (requires immediate attention)',
            'ctr_decline': 'CTR dropped despite stable position/impressions (snippet issues)'
        }

        for pattern, count in decay_summary.items():
            desc = pattern_desc.get(pattern, 'Other decay pattern')
            lines.append(f"| {pattern.replace('_', ' ').title()} | {count} | {desc} |")

        # Top decaying content
        lines.append("\n### Top Decaying Queries\n")
        lines.append("| Query | Clicks Change | Impressions Change | Position Change | Type |")
        lines.append("|-------|---------------|--------------------|--------------------|------|")

        for _, row in decay.head(15).iterrows():
            query = row['query'][:35] + "..." if len(row['query']) > 35 else row['query']
            lines.append(
                f"| {query} | {row['clicks_change']:+,} ({row['clicks_change_pct']:+.1f}%) | "
                f"{row['impressions_change']:+,} ({row['impressions_change_pct']:+.1f}%) | "
                f"{row['position_change']:+.1f} | {row['decay_type'].replace('_', ' ')} |"
            )

        return "\n".join(lines)

    def _generate_gaps_section(self, gaps: pd.DataFrame) -> str:
        """Generate keyword gaps section."""
        lines = []

        lines.append(f"Found **{len(gaps)}** keyword opportunities from competitor analysis.\n")

        # Summary stats
        total_volume = int(gaps['search_volume'].sum())
        avg_cpc = gaps['cpc'].mean()
        high_intent = len(gaps[gaps['intent'].isin(['transactional', 'commercial'])])

        lines.append(f"- **Total Search Volume:** {total_volume:,}")
        lines.append(f"- **Average CPC:** ${avg_cpc:.2f}")
        lines.append(f"- **High Intent Keywords:** {high_intent:,}")

        # Top gaps table
        lines.append("\n### Top 20 Keyword Gaps\n")
        lines.append("| Keyword | Volume | CPC | Competitor Pos | Intent | Score |")
        lines.append("|---------|--------|-----|----------------|--------|-------|")

        for _, row in gaps.head(20).iterrows():
            keyword = row['keyword'][:35] + "..." if len(row['keyword']) > 35 else row['keyword']
            lines.append(
                f"| {keyword} | {row['search_volume']:,} | ${row['cpc']:.2f} | "
                f"{row['position']} | {row['intent']} | {row['opportunity_score']} |"
            )

        return "\n".join(lines)

    def _generate_competitor_section(self, competitors: pd.DataFrame) -> str:
        """Generate competitor analysis section."""
        lines = []

        lines.append("### Top Competitors by Keyword Overlap\n")
        lines.append("| Domain | Shared Keywords | Est. Traffic | Top 10 Rankings |")
        lines.append("|--------|-----------------|--------------|-----------------|")

        for _, row in competitors.head(10).iterrows():
            top_10 = row.get('organic_pos_1', 0) + row.get('organic_pos_2_3', 0) + row.get('organic_pos_4_10', 0)
            lines.append(
                f"| {row['domain']} | {row['intersections']:,} | "
                f"{int(row['organic_etv']):,} | {int(top_10):,} |"
            )

        return "\n".join(lines)

    def _generate_action_items(self, analysis_results: Dict) -> str:
        """Generate prioritized action items."""
        lines = []

        lines.append("### Tier 1: Immediate Actions (High Impact, Low Effort)\n")

        quick_wins = analysis_results.get('quick_wins', pd.DataFrame())
        if not quick_wins.empty:
            top_wins = quick_wins.head(5)
            for idx, row in top_wins.iterrows():
                lines.append(f"1. **Optimize CTR for:** \"{row['query']}\"")
                lines.append(f"   - Current position: {row['position']:.1f}, CTR: {row['ctr']*100:.2f}%")
                lines.append(f"   - Action: Rewrite title/meta for better click appeal")

        lines.append("\n### Tier 2: Recovery Actions (Critical Content)\n")

        decay = analysis_results.get('decay', pd.DataFrame())
        if not decay.empty:
            major_decay = decay[decay['decay_type'] == 'major_decline'].head(3)
            for idx, row in major_decay.iterrows():
                lines.append(f"1. **Recover:** \"{row['query']}\"")
                lines.append(f"   - Clicks dropped: {row['clicks_change']:+,}")
                lines.append(f"   - Action: Content refresh, re-optimization, internal linking boost")

        lines.append("\n### Tier 3: Growth Opportunities (New Content)\n")

        gaps = analysis_results.get('keyword_gaps', pd.DataFrame())
        if not gaps.empty:
            top_gaps = gaps[gaps['intent'].isin(['transactional', 'commercial'])].head(5)
            for idx, row in top_gaps.iterrows():
                lines.append(f"1. **Target:** \"{row['keyword']}\"")
                lines.append(f"   - Volume: {row['search_volume']:,}, CPC: ${row['cpc']:.2f}")
                lines.append(f"   - Action: Create dedicated landing page or service page")

        return "\n".join(lines)

    def generate_html_report(
        self,
        gsc_data: Dict[str, pd.DataFrame],
        dataforseo_data: Dict[str, Any],
        analysis_results: Dict[str, Any],
        llm_analysis: Optional[str] = None
    ) -> str:
        """
        Generate an HTML report.

        Args:
            gsc_data: Google Search Console data
            dataforseo_data: DataForSEO data
            analysis_results: Analysis results
            llm_analysis: LLM-generated analysis

        Returns:
            HTML formatted report string
        """
        # Generate markdown first, then convert to HTML
        markdown_content = self.generate_markdown_report(
            gsc_data, dataforseo_data, analysis_results, llm_analysis
        )

        # Simple markdown to HTML conversion for basic elements
        html = self._markdown_to_html(markdown_content)

        # Wrap in HTML template
        template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Organic Performance Analysis - {self.domain}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            color: #1f2937;
        }}
        h1 {{ color: #111827; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }}
        h2 {{ color: #1f2937; margin-top: 2rem; }}
        h3 {{ color: #374151; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 0.75rem; text-align: left; }}
        th {{ background-color: #f3f4f6; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f9fafb; }}
        blockquote {{
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background-color: #eff6ff;
        }}
        code {{ background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 0.25rem; }}
        .metric {{ font-weight: 600; color: #3b82f6; }}
        ul {{ padding-left: 1.5rem; }}
        li {{ margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
        return template

    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        import re

        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Tables (basic conversion)
        lines = html.split('\n')
        new_lines = []
        in_table = False

        for line in lines:
            if line.startswith('|') and line.endswith('|'):
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if all(c.replace('-', '') == '' for c in cells):
                    # Skip separator row
                    continue
                if not in_table:
                    new_lines.append('<table>')
                    in_table = True
                    tag = 'th'
                else:
                    tag = 'td'
                row = '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'
                new_lines.append(row)
            else:
                if in_table:
                    new_lines.append('</table>')
                    in_table = False
                new_lines.append(line)

        if in_table:
            new_lines.append('</table>')

        html = '\n'.join(new_lines)

        # Blockquotes
        html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

        # Lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Paragraphs (wrap loose text)
        html = re.sub(r'^(?!<[htp]|<li|<table|<tr|<block)(.+)$', r'<p>\1</p>', html, flags=re.MULTILINE)

        # Wrap lists
        html = re.sub(r'(<li>.+</li>\n)+', lambda m: '<ul>' + m.group(0) + '</ul>', html)

        # Line breaks for remaining
        html = html.replace('\n\n', '\n')

        return html

    def export_to_json(
        self,
        gsc_data: Dict[str, pd.DataFrame],
        dataforseo_data: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> str:
        """
        Export analysis data to JSON format.

        Args:
            gsc_data: Google Search Console data
            dataforseo_data: DataForSEO data
            analysis_results: Analysis results

        Returns:
            JSON string
        """
        export_data = {
            'domain': self.domain,
            'analysis_date': self.analysis_date.isoformat(),
            'gsc_summary': {},
            'dataforseo_summary': {},
            'analysis_results': {}
        }

        # GSC summary
        query_data = gsc_data.get('query_data', pd.DataFrame())
        if not query_data.empty:
            export_data['gsc_summary'] = {
                'total_clicks': int(query_data['clicks'].sum()),
                'total_impressions': int(query_data['impressions'].sum()),
                'avg_ctr': round(query_data['ctr'].mean() * 100, 2),
                'avg_position': round(query_data['position'].mean(), 1),
                'unique_queries': query_data['query'].nunique()
            }

        # DataForSEO summary
        ranked_kw = dataforseo_data.get('ranked_keywords', pd.DataFrame())
        if not ranked_kw.empty:
            export_data['dataforseo_summary'] = {
                'total_keywords': len(ranked_kw),
                'top_10_keywords': len(ranked_kw[ranked_kw['position'] <= 10]),
                'estimated_traffic_value': int(ranked_kw['etv'].sum())
            }

        # Analysis results
        quick_wins = analysis_results.get('quick_wins', pd.DataFrame())
        if not quick_wins.empty:
            export_data['analysis_results']['quick_wins_count'] = len(quick_wins)
            export_data['analysis_results']['top_quick_wins'] = quick_wins.head(10).to_dict('records')

        decay = analysis_results.get('decay', pd.DataFrame())
        if not decay.empty:
            export_data['analysis_results']['decay_count'] = len(decay)
            export_data['analysis_results']['decay_by_type'] = decay['decay_type'].value_counts().to_dict()

        gaps = analysis_results.get('keyword_gaps', pd.DataFrame())
        if not gaps.empty:
            export_data['analysis_results']['keyword_gaps_count'] = len(gaps)
            export_data['analysis_results']['top_gaps'] = gaps.head(10).to_dict('records')

        brand = analysis_results.get('brand_analysis', {})
        if brand:
            export_data['analysis_results']['brand_analysis'] = brand

        return json.dumps(export_data, indent=2, default=str)
