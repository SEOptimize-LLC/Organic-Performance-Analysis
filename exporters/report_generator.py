"""
Report generator orchestrator.
Coordinates data collection and export.
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from exporters.excel_exporter import ExcelExporter
from exporters.pdf_exporter import PDFExporter
from utils.logger import logger


class ReportGenerator:
    """
    Orchestrates report generation combining all analysis data.
    Supports Excel and PDF output formats.
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.excel_exporter = ExcelExporter()
        self.pdf_exporter = PDFExporter()
    
    def prepare_report_data(
        self,
        domain: str,
        gsc_data: Dict[str, pd.DataFrame],
        dataforseo_data: Dict[str, Any],
        opportunities: Dict[str, pd.DataFrame],
        decay_data: Dict[str, pd.DataFrame],
        brand_metrics: Dict,
        ai_analysis: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Prepare all data for report generation.
        
        Args:
            domain: Analyzed domain
            gsc_data: GSC DataFrames
            dataforseo_data: DataForSEO data
            opportunities: Classified opportunities
            decay_data: Decay analysis data
            brand_metrics: Brand metrics
            ai_analysis: AI-generated analysis
            
        Returns:
            Consolidated report data dict
        """
        # Calculate overview metrics
        overview = self._calculate_overview(gsc_data)
        
        report_data = {
            'domain': domain,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'overview_metrics': overview,
            'brand_metrics': brand_metrics,
            
            # Opportunities
            'quick_wins': opportunities.get('quick_wins', pd.DataFrame()),
            'ctr_opportunities': opportunities.get(
                'ctr_opportunities', pd.DataFrame()
            ),
            'scaling_opportunities': opportunities.get(
                'scaling_opportunities', pd.DataFrame()
            ),
            'new_opportunities': opportunities.get(
                'new_opportunities', pd.DataFrame()
            ),
            
            # Decay
            'decaying_keywords': decay_data.get(
                'decaying_keywords', pd.DataFrame()
            ),
            'decaying_pages': decay_data.get('decaying_pages', pd.DataFrame()),
            
            # DataForSEO data
            'competitors': dataforseo_data.get('competitors', pd.DataFrame()),
            'keyword_gaps': dataforseo_data.get('keyword_gaps', pd.DataFrame()),
            'ranked_keywords': dataforseo_data.get(
                'ranked_keywords', pd.DataFrame()
            ),
            
            # Raw GSC data
            'all_keywords': gsc_data.get('queries', pd.DataFrame()),
            'pages': gsc_data.get('pages', pd.DataFrame()),
            
            # AI analysis
            'ai_analysis': ai_analysis or {}
        }
        
        return report_data
    
    def _calculate_overview(
        self,
        gsc_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Calculate overview metrics from GSC data."""
        queries_df = gsc_data.get('queries', pd.DataFrame())
        
        if queries_df.empty:
            return {
                'clicks': 0,
                'impressions': 0,
                'ctr': 0,
                'position': 0,
                'query_count': 0
            }
        
        return {
            'clicks': int(queries_df['clicks'].sum()),
            'impressions': int(queries_df['impressions'].sum()),
            'ctr': float(queries_df['ctr'].mean()),
            'position': float(queries_df['position'].mean()),
            'query_count': len(queries_df)
        }
    
    def generate_excel_report(
        self,
        report_data: Dict[str, Any]
    ) -> bytes:
        """
        Generate Excel report.
        
        Args:
            report_data: Prepared report data
            
        Returns:
            Excel file bytes
        """
        logger.info("Generating Excel report...")
        
        return self.excel_exporter.create_report(
            domain=report_data['domain'],
            analysis_date=report_data['generated_at'],
            data=report_data
        )
    
    def generate_pdf_report(
        self,
        report_data: Dict[str, Any]
    ) -> bytes:
        """
        Generate PDF report.
        
        Args:
            report_data: Prepared report data
            
        Returns:
            PDF file bytes
        """
        logger.info("Generating PDF report...")
        
        return self.pdf_exporter.create_report(
            domain=report_data['domain'],
            analysis_date=report_data['generated_at'],
            data=report_data
        )
    
    def generate_all_reports(
        self,
        report_data: Dict[str, Any]
    ) -> Dict[str, bytes]:
        """
        Generate both Excel and PDF reports.
        
        Args:
            report_data: Prepared report data
            
        Returns:
            Dict with 'excel' and 'pdf' bytes
        """
        return {
            'excel': self.generate_excel_report(report_data),
            'pdf': self.generate_pdf_report(report_data)
        }
    
    def get_report_filename(
        self,
        domain: str,
        extension: str
    ) -> str:
        """
        Generate report filename.
        
        Args:
            domain: Domain name
            extension: File extension
            
        Returns:
            Formatted filename
        """
        # Clean domain for filename
        clean_domain = domain.replace('https://', '').replace(
            'http://', ''
        ).replace('/', '_').replace('.', '_')
        
        date_str = datetime.now().strftime('%Y%m%d')
        return f"organic_analysis_{clean_domain}_{date_str}.{extension}"
    
    def export_dataframe(
        self,
        df: pd.DataFrame,
        filename: str
    ) -> bytes:
        """
        Export single DataFrame to Excel.
        
        Args:
            df: DataFrame to export
            filename: Desired filename
            
        Returns:
            Excel bytes
        """
        return self.excel_exporter.export_single_sheet(df, 'Data')
    
    def generate_quick_report(
        self,
        domain: str,
        quick_wins: pd.DataFrame,
        decaying: pd.DataFrame,
        ai_summary: str
    ) -> bytes:
        """
        Generate quick summary report (PDF).
        
        Args:
            domain: Domain
            quick_wins: Quick wins DataFrame
            decaying: Decaying items DataFrame
            ai_summary: AI-generated summary
            
        Returns:
            PDF bytes
        """
        report_data = {
            'domain': domain,
            'generated_at': datetime.now().strftime('%Y-%m-%d'),
            'overview_metrics': {},
            'brand_metrics': {},
            'quick_wins': quick_wins,
            'decaying_keywords': decaying,
            'decaying_pages': pd.DataFrame(),
            'competitors': pd.DataFrame(),
            'keyword_gaps': pd.DataFrame(),
            'ai_analysis': {'comprehensive': ai_summary}
        }
        
        return self.pdf_exporter.create_report(
            domain=domain,
            analysis_date=report_data['generated_at'],
            data=report_data
        )