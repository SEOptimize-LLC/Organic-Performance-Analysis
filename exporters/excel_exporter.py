"""
Excel export functionality.
Creates comprehensive Excel reports with multiple sheets.
"""

import io
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import xlsxwriter

from utils.logger import logger


class ExcelExporter:
    """
    Creates Excel reports with multiple worksheets.
    Includes formatting, charts, and summary data.
    """
    
    def __init__(self):
        """Initialize exporter."""
        self.workbook = None
        self.output = None
        self.formats = {}
    
    def _init_workbook(self):
        """Initialize workbook and formats."""
        self.output = io.BytesIO()
        self.workbook = xlsxwriter.Workbook(self.output, {'in_memory': True})
        self._create_formats()
    
    def _create_formats(self):
        """Create cell formats."""
        self.formats = {
            'header': self.workbook.add_format({
                'bold': True,
                'bg_color': '#1f77b4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            }),
            'title': self.workbook.add_format({
                'bold': True,
                'font_size': 16,
                'font_color': '#1f77b4'
            }),
            'subtitle': self.workbook.add_format({
                'bold': True,
                'font_size': 12,
                'font_color': '#666666'
            }),
            'number': self.workbook.add_format({
                'num_format': '#,##0',
                'align': 'right'
            }),
            'decimal': self.workbook.add_format({
                'num_format': '#,##0.00',
                'align': 'right'
            }),
            'percent': self.workbook.add_format({
                'num_format': '0.00%',
                'align': 'right'
            }),
            'positive': self.workbook.add_format({
                'num_format': '+#,##0;-#,##0',
                'font_color': 'green',
                'align': 'right'
            }),
            'negative': self.workbook.add_format({
                'num_format': '+#,##0;-#,##0',
                'font_color': 'red',
                'align': 'right'
            }),
            'text': self.workbook.add_format({
                'align': 'left',
                'text_wrap': True
            }),
            'date': self.workbook.add_format({
                'num_format': 'yyyy-mm-dd',
                'align': 'center'
            })
        }
    
    def _write_dataframe(
        self,
        worksheet,
        df: pd.DataFrame,
        start_row: int = 0,
        start_col: int = 0,
        include_header: bool = True
    ) -> int:
        """
        Write DataFrame to worksheet.
        
        Args:
            worksheet: Target worksheet
            df: DataFrame to write
            start_row: Starting row
            start_col: Starting column
            include_header: Include column headers
            
        Returns:
            Next available row
        """
        if df.empty:
            return start_row
        
        current_row = start_row
        
        # Write headers
        if include_header:
            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(
                    current_row, start_col + col_idx,
                    str(col_name), self.formats['header']
                )
            current_row += 1
        
        # Write data
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                # Determine format
                fmt = None
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        value = ''
                    elif isinstance(value, float):
                        if value < 1:  # Likely percentage
                            fmt = self.formats['percent']
                        else:
                            fmt = self.formats['decimal']
                    else:
                        fmt = self.formats['number']
                
                if fmt:
                    worksheet.write(
                        current_row, start_col + col_idx, value, fmt
                    )
                else:
                    worksheet.write(
                        current_row, start_col + col_idx, value
                    )
            current_row += 1
        
        return current_row
    
    def _auto_fit_columns(self, worksheet, df: pd.DataFrame, start_col: int = 0):
        """Auto-fit column widths."""
        for idx, col in enumerate(df.columns):
            max_len = max(
                len(str(col)),
                df[col].astype(str).str.len().max()
            )
            worksheet.set_column(
                start_col + idx,
                start_col + idx,
                min(max_len + 2, 50)
            )
    
    def create_report(
        self,
        domain: str,
        analysis_date: str,
        data: Dict[str, Any]
    ) -> bytes:
        """
        Create complete Excel report.
        
        Args:
            domain: Analyzed domain
            analysis_date: Analysis date
            data: All analysis data
            
        Returns:
            Excel file bytes
        """
        self._init_workbook()
        
        try:
            # Summary sheet
            self._create_summary_sheet(domain, analysis_date, data)
            
            # Quick wins sheet
            if 'quick_wins' in data and not data['quick_wins'].empty:
                self._create_data_sheet(
                    'Quick Wins',
                    data['quick_wins']
                )
            
            # Decaying keywords sheet
            if 'decaying_keywords' in data and not data['decaying_keywords'].empty:  # noqa: E501
                self._create_data_sheet(
                    'Decaying Keywords',
                    data['decaying_keywords']
                )
            
            # Decaying pages sheet
            if 'decaying_pages' in data and not data['decaying_pages'].empty:
                self._create_data_sheet(
                    'Decaying Pages',
                    data['decaying_pages']
                )
            
            # Competitors sheet
            if 'competitors' in data and not data['competitors'].empty:
                self._create_data_sheet(
                    'Competitors',
                    data['competitors']
                )
            
            # Keyword gaps sheet
            if 'keyword_gaps' in data and not data['keyword_gaps'].empty:
                self._create_data_sheet(
                    'Keyword Gaps',
                    data['keyword_gaps']
                )
            
            # All keywords sheet
            if 'all_keywords' in data and not data['all_keywords'].empty:
                self._create_data_sheet(
                    'All Keywords',
                    data['all_keywords'].head(5000)  # Limit rows
                )
            
            # Pages sheet
            if 'pages' in data and not data['pages'].empty:
                self._create_data_sheet(
                    'Pages',
                    data['pages']
                )
            
            # AI Analysis sheet
            if 'ai_analysis' in data:
                self._create_analysis_sheet(data['ai_analysis'])
            
            self.workbook.close()
            self.output.seek(0)
            return self.output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            if self.workbook:
                self.workbook.close()
            raise
    
    def _create_summary_sheet(
        self,
        domain: str,
        analysis_date: str,
        data: Dict[str, Any]
    ):
        """Create summary worksheet."""
        ws = self.workbook.add_worksheet('Summary')
        
        # Title
        ws.write(0, 0, 'Organic Performance Analysis Report', self.formats['title'])  # noqa: E501
        ws.write(1, 0, f'Domain: {domain}', self.formats['subtitle'])
        ws.write(2, 0, f'Generated: {analysis_date}', self.formats['subtitle'])
        
        row = 4
        
        # Overview metrics
        if 'overview_metrics' in data:
            ws.write(row, 0, 'Overview Metrics', self.formats['subtitle'])
            row += 1
            
            metrics = data['overview_metrics']
            for key, value in metrics.items():
                ws.write(row, 0, key.replace('_', ' ').title())
                if isinstance(value, (int, float)):
                    ws.write(row, 1, value, self.formats['number'])
                else:
                    ws.write(row, 1, str(value))
                row += 1
            row += 1
        
        # Brand metrics
        if 'brand_metrics' in data:
            ws.write(row, 0, 'Brand Analysis', self.formats['subtitle'])
            row += 1
            
            brand = data['brand_metrics']
            brand_data = brand.get('brand', {})
            non_brand = brand.get('non_brand', {})
            
            ws.write(row, 0, 'Brand Click Share')
            ws.write(row, 1, brand_data.get('click_share', 0) / 100,
                     self.formats['percent'])
            row += 1
            
            ws.write(row, 0, 'Non-Brand Clicks')
            ws.write(row, 1, non_brand.get('clicks', 0),
                     self.formats['number'])
            row += 1
            
            ws.write(row, 0, 'Brand Dependency Score')
            ws.write(row, 1, brand.get('dependency_score', 0) / 100,
                     self.formats['percent'])
            row += 2
        
        # Opportunity counts
        ws.write(row, 0, 'Opportunities Identified', self.formats['subtitle'])
        row += 1
        
        opp_counts = [
            ('Quick Wins', len(data.get('quick_wins', []))),
            ('Decaying Keywords', len(data.get('decaying_keywords', []))),
            ('Decaying Pages', len(data.get('decaying_pages', []))),
            ('Keyword Gaps', len(data.get('keyword_gaps', [])))
        ]
        
        for label, count in opp_counts:
            ws.write(row, 0, label)
            ws.write(row, 1, count, self.formats['number'])
            row += 1
        
        ws.set_column(0, 0, 30)
        ws.set_column(1, 1, 20)
    
    def _create_data_sheet(
        self,
        sheet_name: str,
        df: pd.DataFrame
    ):
        """Create data worksheet."""
        # Sanitize sheet name
        safe_name = sheet_name[:31]  # Excel limit
        ws = self.workbook.add_worksheet(safe_name)
        
        self._write_dataframe(ws, df)
        self._auto_fit_columns(ws, df)
    
    def _create_analysis_sheet(self, analysis: Dict[str, str]):
        """Create AI analysis worksheet."""
        ws = self.workbook.add_worksheet('AI Analysis')
        
        row = 0
        ws.write(row, 0, 'AI-Generated Analysis', self.formats['title'])
        row += 2
        
        sections = [
            ('comprehensive', 'Comprehensive Analysis'),
            ('quick_wins', 'Quick Wins Analysis'),
            ('decay', 'Decay Analysis'),
            ('brand', 'Brand Analysis'),
            ('competitors', 'Competitor Analysis'),
            ('pages', 'Page Analysis')
        ]
        
        for key, title in sections:
            if key in analysis:
                ws.write(row, 0, title, self.formats['subtitle'])
                row += 1
                
                # Write text in wrapped format
                ws.set_row(row, 200)  # Tall row for text
                ws.write(row, 0, analysis[key], self.formats['text'])
                row += 2
        
        ws.set_column(0, 0, 100)
    
    def export_single_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str = 'Data'
    ) -> bytes:
        """
        Export single DataFrame to Excel.
        
        Args:
            df: DataFrame to export
            sheet_name: Worksheet name
            
        Returns:
            Excel file bytes
        """
        self._init_workbook()
        
        try:
            self._create_data_sheet(sheet_name, df)
            self.workbook.close()
            self.output.seek(0)
            return self.output.getvalue()
        except Exception as e:
            logger.error(f"Error exporting Excel: {str(e)}")
            if self.workbook:
                self.workbook.close()
            raise