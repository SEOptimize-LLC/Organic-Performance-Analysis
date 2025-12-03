"""
PDF export functionality.
Creates formatted PDF reports using FPDF.
"""

import io
from fpdf import FPDF
import pandas as pd
from typing import Dict, Any
from datetime import datetime

from utils.logger import logger


class PDFExporter:
    """
    Creates PDF reports with formatted sections.
    Uses FPDF for PDF generation.
    """
    
    def __init__(self):
        """Initialize PDF exporter."""
        self.pdf = None
    
    def _init_pdf(self):
        """Initialize PDF document."""
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font('Helvetica', '', 10)
    
    def _add_title_page(
        self,
        domain: str,
        analysis_date: str
    ):
        """Add title page."""
        self.pdf.add_page()
        
        # Title
        self.pdf.set_font('Helvetica', 'B', 24)
        self.pdf.set_text_color(31, 119, 180)  # Primary blue
        self.pdf.cell(0, 40, '', ln=True)  # Spacing
        self.pdf.cell(0, 15, 'Organic Performance', ln=True, align='C')
        self.pdf.cell(0, 15, 'Analysis Report', ln=True, align='C')
        
        # Domain
        self.pdf.set_font('Helvetica', '', 16)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 30, '', ln=True)  # Spacing
        self.pdf.cell(0, 10, f'Domain: {domain}', ln=True, align='C')
        
        # Date
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.cell(0, 10, f'Generated: {analysis_date}', ln=True, align='C')
        
        # Footer note
        self.pdf.set_y(-50)
        self.pdf.set_font('Helvetica', 'I', 10)
        self.pdf.cell(
            0, 10,
            'Powered by Organic Performance Analyzer',
            ln=True, align='C'
        )
    
    def _add_section_header(self, title: str):
        """Add section header."""
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.set_text_color(31, 119, 180)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font('Helvetica', '', 10)
    
    def _add_subsection_header(self, title: str):
        """Add subsection header."""
        self.pdf.set_font('Helvetica', 'B', 11)
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.cell(0, 8, title, ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font('Helvetica', '', 10)
    
    def _add_text(self, text: str, indent: int = 0):
        """Add paragraph text."""
        self.pdf.set_font('Helvetica', '', 10)
        
        # Handle long text with word wrap
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                # Add indentation
                if indent > 0:
                    self.pdf.cell(indent, 5, '', ln=False)
                self.pdf.multi_cell(0, 5, line.strip())
            else:
                self.pdf.ln(3)
    
    def _add_metric_row(self, label: str, value: str):
        """Add metric label-value pair."""
        self.pdf.set_font('Helvetica', 'B', 10)
        self.pdf.cell(60, 7, label, ln=False)
        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.cell(0, 7, str(value), ln=True)
    
    def _add_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 20,
        col_widths: Dict[str, int] = None
    ):
        """Add data table."""
        if df.empty:
            self.pdf.cell(0, 7, 'No data available', ln=True)
            return
        
        display_df = df.head(max_rows)
        cols = display_df.columns.tolist()
        
        # Calculate column widths
        if col_widths is None:
            page_width = self.pdf.w - 20  # Margins
            col_width = page_width / len(cols)
            widths = [col_width] * len(cols)
        else:
            widths = [col_widths.get(c, 30) for c in cols]
        
        # Header
        self.pdf.set_font('Helvetica', 'B', 8)
        self.pdf.set_fill_color(31, 119, 180)
        self.pdf.set_text_color(255, 255, 255)
        
        for idx, col in enumerate(cols):
            self.pdf.cell(widths[idx], 7, str(col)[:15], border=1, fill=True)
        self.pdf.ln()
        
        # Data rows
        self.pdf.set_font('Helvetica', '', 8)
        self.pdf.set_text_color(0, 0, 0)
        
        for _, row in display_df.iterrows():
            for idx, col in enumerate(cols):
                value = row[col]
                if pd.isna(value):
                    value = ''
                elif isinstance(value, float):
                    if value < 1:
                        value = f'{value:.2%}'
                    else:
                        value = f'{value:.1f}'
                else:
                    value = str(value)[:20]
                
                self.pdf.cell(widths[idx], 6, value, border=1)
            self.pdf.ln()
    
    def _add_overview_section(self, data: Dict[str, Any]):
        """Add overview metrics section."""
        self.pdf.add_page()
        self._add_section_header('Executive Summary')
        
        if 'overview_metrics' in data:
            self._add_subsection_header('Key Metrics')
            metrics = data['overview_metrics']
            
            self._add_metric_row('Total Clicks', f"{metrics.get('clicks', 0):,}")
            self._add_metric_row(
                'Total Impressions',
                f"{metrics.get('impressions', 0):,}"
            )
            self._add_metric_row(
                'Average CTR',
                f"{metrics.get('ctr', 0) * 100:.2f}%"
            )
            self._add_metric_row(
                'Average Position',
                f"{metrics.get('position', 0):.1f}"
            )
            self.pdf.ln(5)
        
        if 'brand_metrics' in data:
            self._add_subsection_header('Brand Analysis')
            brand = data['brand_metrics']
            
            self._add_metric_row(
                'Brand Dependency',
                f"{brand.get('dependency_score', 0):.1f}%"
            )
            self._add_metric_row(
                'Brand Clicks',
                f"{brand.get('brand', {}).get('clicks', 0):,}"
            )
            self._add_metric_row(
                'Non-Brand Clicks',
                f"{brand.get('non_brand', {}).get('clicks', 0):,}"
            )
            self.pdf.ln(5)
        
        # Opportunity counts
        self._add_subsection_header('Opportunities Identified')
        self._add_metric_row(
            'Quick Wins',
            len(data.get('quick_wins', []))
        )
        self._add_metric_row(
            'Decaying Keywords',
            len(data.get('decaying_keywords', []))
        )
        self._add_metric_row(
            'Keyword Gaps',
            len(data.get('keyword_gaps', []))
        )
    
    def _add_quick_wins_section(self, df: pd.DataFrame):
        """Add quick wins section."""
        self.pdf.add_page()
        self._add_section_header('Quick Win Opportunities')
        
        self._add_text(
            'These keywords have high impressions but underperforming CTR. '
            'Optimizing titles and meta descriptions can yield immediate gains.'
        )
        self.pdf.ln(5)
        
        if df.empty:
            self._add_text('No quick wins identified in this analysis period.')
            return
        
        # Select key columns
        display_cols = ['query', 'position', 'impressions', 'ctr', 'opportunity_score']  # noqa: E501
        available = [c for c in display_cols if c in df.columns]
        
        # Also check prefixed columns
        if 'gsc_query' in df.columns:
            available = ['gsc_query', 'gsc_position', 'gsc_impressions',
                         'gsc_ctr', 'opportunity_score']
            available = [c for c in available if c in df.columns]
        
        if available:
            self._add_table(df[available].head(15))
    
    def _add_decay_section(
        self,
        keywords_df: pd.DataFrame,
        pages_df: pd.DataFrame
    ):
        """Add content decay section."""
        self.pdf.add_page()
        self._add_section_header('Content Decay Analysis')
        
        self._add_text(
            'These items show declining performance and require attention. '
            'Prioritize based on severity and traffic impact.'
        )
        self.pdf.ln(5)
        
        # Decaying keywords
        self._add_subsection_header('Decaying Keywords')
        if keywords_df.empty:
            self._add_text('No significant keyword decay detected.')
        else:
            cols = ['query', 'clicks_change_pct', 'position_change',
                    'primary_decay']
            available = [c for c in cols if c in keywords_df.columns]
            if available:
                self._add_table(keywords_df[available].head(10))
        
        self.pdf.ln(10)
        
        # Decaying pages
        self._add_subsection_header('Decaying Pages')
        if pages_df.empty:
            self._add_text('No significant page decay detected.')
        else:
            cols = ['page', 'clicks_change_pct', 'primary_decay']
            available = [c for c in cols if c in pages_df.columns]
            if available:
                self._add_table(pages_df[available].head(10))
    
    def _add_competitor_section(
        self,
        competitors_df: pd.DataFrame,
        gaps_df: pd.DataFrame
    ):
        """Add competitor analysis section."""
        self.pdf.add_page()
        self._add_section_header('Competitive Analysis')
        
        # Competitors
        self._add_subsection_header('Top Competitors')
        if competitors_df.empty:
            self._add_text('No competitor data available.')
        else:
            cols = ['competitor_domain', 'intersections', 'avg_position']
            available = [c for c in cols if c in competitors_df.columns]
            if available:
                self._add_table(competitors_df[available].head(10))
        
        self.pdf.ln(10)
        
        # Keyword gaps
        self._add_subsection_header('Keyword Gaps')
        if gaps_df.empty:
            self._add_text('No keyword gaps identified.')
        else:
            cols = ['keyword', 'search_volume', 'target2_position']
            available = [c for c in cols if c in gaps_df.columns]
            if available:
                self._add_table(gaps_df[available].head(15))
    
    def _add_ai_analysis_section(self, analysis: Dict[str, str]):
        """Add AI analysis section."""
        self.pdf.add_page()
        self._add_section_header('AI-Generated Insights')
        
        # Comprehensive analysis
        if 'comprehensive' in analysis:
            self._add_subsection_header('Strategic Overview')
            self._add_text(analysis['comprehensive'][:3000])
        
        # Add other sections on new pages if they exist
        sections = [
            ('quick_wins', 'Quick Wins Recommendations'),
            ('decay', 'Recovery Recommendations'),
            ('competitors', 'Competitive Recommendations'),
            ('brand', 'Brand Strategy Recommendations')
        ]
        
        for key, title in sections:
            if key in analysis and analysis[key]:
                self.pdf.add_page()
                self._add_section_header(title)
                self._add_text(analysis[key][:2500])
    
    def create_report(
        self,
        domain: str,
        analysis_date: str,
        data: Dict[str, Any]
    ) -> bytes:
        """
        Create complete PDF report.
        
        Args:
            domain: Analyzed domain
            analysis_date: Analysis date
            data: Analysis data
            
        Returns:
            PDF file bytes
        """
        self._init_pdf()
        
        try:
            # Title page
            self._add_title_page(domain, analysis_date)
            
            # Overview
            self._add_overview_section(data)
            
            # Quick wins
            if 'quick_wins' in data:
                self._add_quick_wins_section(
                    data['quick_wins'] if isinstance(data['quick_wins'], pd.DataFrame)  # noqa: E501
                    else pd.DataFrame()
                )
            
            # Decay analysis
            self._add_decay_section(
                data.get('decaying_keywords', pd.DataFrame()),
                data.get('decaying_pages', pd.DataFrame())
            )
            
            # Competitors
            self._add_competitor_section(
                data.get('competitors', pd.DataFrame()),
                data.get('keyword_gaps', pd.DataFrame())
            )
            
            # AI Analysis
            if 'ai_analysis' in data:
                self._add_ai_analysis_section(data['ai_analysis'])
            
            # Output
            output = io.BytesIO()
            pdf_str = self.pdf.output(dest='S').encode('latin-1')
            output.write(pdf_str)
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {str(e)}")
            raise
    
    def export_analysis_only(
        self,
        domain: str,
        analysis: Dict[str, str]
    ) -> bytes:
        """
        Export just the AI analysis as PDF.
        
        Args:
            domain: Domain name
            analysis: AI analysis dict
            
        Returns:
            PDF bytes
        """
        self._init_pdf()
        
        try:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            self._add_title_page(domain, analysis_date)
            self._add_ai_analysis_section(analysis)
            
            output = io.BytesIO()
            pdf_str = self.pdf.output(dest='S').encode('latin-1')
            output.write(pdf_str)
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            raise