"""
AI Analysis Agent using OpenRouter API.
Generates actionable SEO insights from organic performance data.
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any

from config.api_config import api_config
from config.settings import settings
from services.rate_limiter import rate_limiter
from agents.prompts import AnalysisPrompts
from utils.logger import logger


class AnalysisAgent:
    """
    AI-powered analysis agent for organic performance.
    Uses OpenRouter to access multiple LLM models.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize analysis agent.
        
        Args:
            model: OpenRouter model identifier
        """
        # Get API key from config
        if api_config.openrouter_api_key:
            self.api_key = api_config.openrouter_api_key.get_secret_value()
        else:
            raise ValueError("OpenRouter API key not configured")
        
        self.base_url = api_config.openrouter_base_url
        self.model = model or settings.available_llm_models[0]
        self.system_prompt = AnalysisPrompts.get_system_prompt()
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://organic-performance-analysis.streamlit.app',
            'X-Title': 'Organic Performance Analyzer'
        }
    
    def set_model(self, model: str):
        """
        Change the LLM model.
        
        Args:
            model: Model identifier from available_models
        """
        if model in settings.available_models:
            self.model = model
            logger.info(f"Model changed to: {model}")
        else:
            logger.warning(f"Unknown model: {model}, keeping current")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return settings.available_llm_models
    
    @rate_limiter.limit_openrouter
    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Call OpenRouter API.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum response tokens
            temperature: Response creativity (0-1)
            
        Returns:
            LLM response text or None
        """
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            logger.error("OpenRouter request timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter request error: {str(e)}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenRouter response: {str(e)}")
            return None
    
    def _prepare_dataframe_summary(
        self,
        df: pd.DataFrame,
        max_rows: int = 50,
        columns: List[str] = None
    ) -> str:
        """
        Prepare DataFrame for prompt insertion.
        
        Args:
            df: Source DataFrame
            max_rows: Maximum rows to include
            columns: Specific columns to include
            
        Returns:
            Formatted string representation
        """
        if df.empty:
            return "No data available"
        
        subset = df.head(max_rows)
        
        if columns:
            available = [c for c in columns if c in subset.columns]
            subset = subset[available]
        
        return subset.to_string(index=False)
    
    def analyze_quick_wins(
        self,
        quick_wins_df: pd.DataFrame
    ) -> str:
        """
        Analyze quick-win opportunities.
        
        Args:
            quick_wins_df: DataFrame with quick-win keywords
            
        Returns:
            Analysis text
        """
        data = {
            'quick_wins_data': self._prepare_dataframe_summary(
                quick_wins_df,
                max_rows=30,
                columns=[
                    'query', 'gsc_query', 'keyword',
                    'position', 'gsc_position',
                    'ctr', 'gsc_ctr',
                    'impressions', 'gsc_impressions',
                    'clicks', 'gsc_clicks',
                    'search_volume', 'dfs_search_volume',
                    'opportunity_score'
                ]
            )
        }
        
        prompt = AnalysisPrompts.get_prompt('quick_wins', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def analyze_decay(
        self,
        decaying_keywords: pd.DataFrame,
        decaying_pages: pd.DataFrame,
        decay_summary: Dict
    ) -> str:
        """
        Analyze content decay patterns.
        
        Args:
            decaying_keywords: Decaying keywords DataFrame
            decaying_pages: Decaying pages DataFrame
            decay_summary: Summary statistics
            
        Returns:
            Analysis text
        """
        # Combine keyword and page data
        decay_data_parts = []
        
        if not decaying_keywords.empty:
            decay_data_parts.append("DECAYING KEYWORDS:\n")
            decay_data_parts.append(self._prepare_dataframe_summary(
                decaying_keywords, max_rows=20
            ))
        
        if not decaying_pages.empty:
            decay_data_parts.append("\n\nDECAYING PAGES:\n")
            decay_data_parts.append(self._prepare_dataframe_summary(
                decaying_pages, max_rows=15
            ))
        
        data = {
            'decay_data': '\n'.join(decay_data_parts),
            'decay_summary': json.dumps(decay_summary, indent=2)
        }
        
        prompt = AnalysisPrompts.get_prompt('decay', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def analyze_competitors(
        self,
        competitor_df: pd.DataFrame,
        keyword_gaps: pd.DataFrame
    ) -> str:
        """
        Analyze competitive landscape.
        
        Args:
            competitor_df: Competitor analysis DataFrame
            keyword_gaps: Keyword gap DataFrame
            
        Returns:
            Analysis text
        """
        data = {
            'competitor_data': self._prepare_dataframe_summary(
                competitor_df, max_rows=20
            ),
            'keyword_gap_data': self._prepare_dataframe_summary(
                keyword_gaps, max_rows=30
            )
        }
        
        prompt = AnalysisPrompts.get_prompt('competitor', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def analyze_brand_performance(
        self,
        brand_metrics: Dict,
        non_brand_opportunities: pd.DataFrame
    ) -> str:
        """
        Analyze brand vs non-brand performance.
        
        Args:
            brand_metrics: Brand metrics dict
            non_brand_opportunities: Non-brand opportunities DataFrame
            
        Returns:
            Analysis text
        """
        data = {
            'brand_metrics': json.dumps(brand_metrics, indent=2),
            'non_brand_opportunities': self._prepare_dataframe_summary(
                non_brand_opportunities, max_rows=25
            )
        }
        
        prompt = AnalysisPrompts.get_prompt('brand', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def analyze_pages(
        self,
        page_data: pd.DataFrame,
        query_portfolio: pd.DataFrame,
        page_scores: pd.DataFrame
    ) -> str:
        """
        Analyze pages for optimization.
        
        Args:
            page_data: Page-level data
            query_portfolio: Query-page combinations
            page_scores: Page opportunity scores
            
        Returns:
            Analysis text
        """
        data = {
            'page_data': self._prepare_dataframe_summary(
                page_data, max_rows=20
            ),
            'query_portfolio': self._prepare_dataframe_summary(
                query_portfolio, max_rows=40
            ),
            'page_scores': self._prepare_dataframe_summary(
                page_scores, max_rows=20
            )
        }
        
        prompt = AnalysisPrompts.get_prompt('page_optimization', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def generate_topic_clusters(
        self,
        ranked_keywords: pd.DataFrame,
        keyword_suggestions: pd.DataFrame,
        related_keywords: pd.DataFrame
    ) -> str:
        """
        Generate topic cluster recommendations.
        
        Args:
            ranked_keywords: Currently ranking keywords
            keyword_suggestions: Suggested keywords
            related_keywords: Related keywords
            
        Returns:
            Analysis text
        """
        data = {
            'ranked_keywords': self._prepare_dataframe_summary(
                ranked_keywords, max_rows=30
            ),
            'keyword_suggestions': self._prepare_dataframe_summary(
                keyword_suggestions, max_rows=30
            ),
            'related_keywords': self._prepare_dataframe_summary(
                related_keywords, max_rows=30
            )
        }
        
        prompt = AnalysisPrompts.get_prompt('topic_cluster', data)
        return self._call_llm(prompt) or "Analysis could not be generated."
    
    def generate_comprehensive_report(
        self,
        domain: str,
        period: str,
        overview_metrics: Dict,
        quick_wins_summary: str,
        decay_summary: str,
        brand_summary: str,
        competitor_summary: str,
        serp_features: str
    ) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            domain: Analyzed domain
            period: Analysis period
            overview_metrics: Overall metrics dict
            quick_wins_summary: Quick wins analysis
            decay_summary: Decay analysis
            brand_summary: Brand analysis
            competitor_summary: Competitor analysis
            serp_features: SERP feature analysis
            
        Returns:
            Complete report text
        """
        data = {
            'domain': domain,
            'period': period,
            'overview_metrics': json.dumps(overview_metrics, indent=2),
            'quick_wins_summary': quick_wins_summary[:2000],
            'decay_summary': decay_summary[:2000],
            'brand_summary': brand_summary[:1500],
            'competitor_summary': competitor_summary[:1500],
            'serp_features': serp_features[:1000]
        }
        
        prompt = AnalysisPrompts.get_prompt('comprehensive', data)
        return self._call_llm(
            prompt, max_tokens=6000
        ) or "Report could not be generated."
    
    def run_full_analysis(
        self,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Run complete analysis pipeline.
        
        Args:
            analysis_data: Dict containing all analysis data:
                - quick_wins: DataFrame
                - decaying_keywords: DataFrame
                - decaying_pages: DataFrame
                - decay_summary: Dict
                - competitors: DataFrame
                - keyword_gaps: DataFrame
                - brand_metrics: Dict
                - non_brand_opportunities: DataFrame
                - page_data: DataFrame
                - query_portfolio: DataFrame
                - page_scores: DataFrame
                - domain: str
                - period: str
                - overview_metrics: Dict
            
        Returns:
            Dict with all analysis sections
        """
        results = {}
        
        # Quick wins analysis
        if 'quick_wins' in analysis_data:
            logger.info("Analyzing quick wins...")
            results['quick_wins'] = self.analyze_quick_wins(
                analysis_data['quick_wins']
            )
        
        # Decay analysis
        if 'decaying_keywords' in analysis_data:
            logger.info("Analyzing content decay...")
            results['decay'] = self.analyze_decay(
                analysis_data.get('decaying_keywords', pd.DataFrame()),
                analysis_data.get('decaying_pages', pd.DataFrame()),
                analysis_data.get('decay_summary', {})
            )
        
        # Competitor analysis
        if 'competitors' in analysis_data:
            logger.info("Analyzing competitors...")
            results['competitors'] = self.analyze_competitors(
                analysis_data.get('competitors', pd.DataFrame()),
                analysis_data.get('keyword_gaps', pd.DataFrame())
            )
        
        # Brand analysis
        if 'brand_metrics' in analysis_data:
            logger.info("Analyzing brand performance...")
            results['brand'] = self.analyze_brand_performance(
                analysis_data.get('brand_metrics', {}),
                analysis_data.get('non_brand_opportunities', pd.DataFrame())
            )
        
        # Page analysis
        if 'page_data' in analysis_data:
            logger.info("Analyzing pages...")
            results['pages'] = self.analyze_pages(
                analysis_data.get('page_data', pd.DataFrame()),
                analysis_data.get('query_portfolio', pd.DataFrame()),
                analysis_data.get('page_scores', pd.DataFrame())
            )
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        results['comprehensive'] = self.generate_comprehensive_report(
            domain=analysis_data.get('domain', 'Unknown'),
            period=analysis_data.get('period', 'N/A'),
            overview_metrics=analysis_data.get('overview_metrics', {}),
            quick_wins_summary=results.get('quick_wins', 'N/A'),
            decay_summary=results.get('decay', 'N/A'),
            brand_summary=results.get('brand', 'N/A'),
            competitor_summary=results.get('competitors', 'N/A'),
            serp_features=analysis_data.get('serp_features', 'N/A')
        )
        
        return results