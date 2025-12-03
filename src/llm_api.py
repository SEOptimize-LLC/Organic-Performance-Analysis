"""
OpenRouter LLM API Integration Module

Handles LLM-powered analysis using OpenRouter API.
Supports multiple reasoning models for SEO analysis.
"""

import requests
from typing import Optional, List, Dict, Any, Generator
import streamlit as st
import json


# Available models for SEO analysis
AVAILABLE_MODELS = {
    "openai/gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "provider": "OpenAI",
        "description": "Fast and efficient for analysis tasks"
    },
    "openai/gpt-5-mini": {
        "name": "GPT-5 Mini",
        "provider": "OpenAI",
        "description": "Latest mini model with advanced reasoning"
    },
    "anthropic/claude-haiku-4.5": {
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "description": "Fast, compact model for quick analysis"
    },
    "anthropic/claude-sonnet-4.5": {
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "description": "Balanced performance and capability"
    },
    "google/gemini-3-pro-preview": {
        "name": "Gemini 3 Pro Preview",
        "provider": "Google",
        "description": "Advanced multimodal capabilities"
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "name": "Gemini 2.5 Flash Preview",
        "provider": "Google",
        "description": "Fast inference with strong reasoning"
    },
    "x-ai/grok-4-fast": {
        "name": "Grok 4 Fast",
        "provider": "xAI",
        "description": "High-speed reasoning model"
    },
    "deepseek/deepseek-r1-0528-qwen3-8b": {
        "name": "DeepSeek R1 Qwen3 8B",
        "provider": "DeepSeek",
        "description": "Efficient reasoning model"
    }
}


class OpenRouterClient:
    """OpenRouter API Client for LLM-powered analysis."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
            site_url: Optional site URL for ranking
            site_name: Optional site name for display
        """
        self.api_key = api_key
        self.site_url = site_url or "https://organic-performance-analysis.streamlit.app"
        self.site_name = site_name or "Organic Performance Analysis"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-sonnet-4.5",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response

        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"OpenRouter API error: {str(e)}")
            return {"error": str(e)}

    def _handle_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue

    def analyze_seo_data(
        self,
        data_summary: str,
        analysis_type: str,
        model: str = "anthropic/claude-sonnet-4.5",
        additional_context: Optional[str] = None
    ) -> str:
        """
        Analyze SEO data using LLM.

        Args:
            data_summary: Summary of the data to analyze
            analysis_type: Type of analysis to perform
            model: Model to use
            additional_context: Additional context for analysis

        Returns:
            Analysis text
        """
        system_prompt = self._get_seo_analysis_prompt(analysis_type)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_summary}
        ]

        if additional_context:
            messages.append({
                "role": "user",
                "content": f"Additional context: {additional_context}"
            })

        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=8192
        )

        if 'error' in response:
            return f"Error in analysis: {response['error']}"

        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']

        return "Unable to generate analysis."

    def _get_seo_analysis_prompt(self, analysis_type: str) -> str:
        """Get the system prompt for SEO analysis based on type."""

        base_prompt = """You are an expert SEO analyst specializing in organic performance optimization.
Your role is to analyze data and provide actionable, concrete recommendations that can be immediately implemented.

CRITICAL GUIDELINES:
1. NO FLUFF - Every statement must be actionable or directly support an action
2. NO VANITY METRICS - Focus on opportunities for growth, not celebrating existing traffic
3. PRIORITIZE BY ROI - Order recommendations by expected impact vs effort
4. BE SPECIFIC - Include specific URLs, queries, and numbers where available
5. EXPLAIN THE WHY - For each recommendation, explain the data-driven rationale
6. AVOID GENERIC ADVICE - Every recommendation should be based on the specific data provided

Output Format:
- Use clear headers and bullet points
- Lead with the highest-impact opportunities
- Include specific metrics to track success
- Provide implementation complexity estimates (Low/Medium/High)"""

        type_specific = {
            "quick_wins": """
FOCUS: Quick Win Opportunities Analysis

Analyze the data to identify:
1. High-impression, low-CTR queries (positions 3-15) that need title/meta optimization
2. Pages with multiple queries clustering around positions 5-20
3. Transactional queries with strong positions but underperforming clicks
4. SERP feature opportunities (featured snippets, PAA, rich results)

For each opportunity, provide:
- The specific query/page combination
- Current metrics (position, impressions, CTR, clicks)
- Expected improvement potential
- Specific optimization action to take
- Priority tier (1-3)""",

            "content_decay": """
FOCUS: Content Decay & Recovery Analysis

Analyze the data to identify:
1. Queries/pages with declining impressions, clicks, or position over time
2. Classification of decay patterns:
   - Position drop + stable impressions = Competition/SERP changes
   - Impressions drop + stable position = Demand decline/coverage loss
   - Both dropping = Technical/indexation issues or content obsolescence
3. High-value decaying content (based on search volume and CPC)

For each decaying asset, provide:
- Specific URL and queries affected
- Decay timeline and magnitude
- Root cause hypothesis based on data patterns
- Recovery strategy with specific actions
- Urgency level (Critical/High/Medium)""",

            "keyword_gaps": """
FOCUS: Keyword Gap & Content Opportunity Analysis

Analyze the data to identify:
1. Keywords competitors rank for that the target site doesn't
2. Topics with high search volume where the site has weak or no presence
3. Clusters suitable for new money pages vs supporting content vs content hubs
4. High-commercial-value gaps (based on CPC and search volume)

For each opportunity cluster, provide:
- Keyword cluster theme and top keywords
- Total addressable search volume
- Commercial value indicators (CPC, intent)
- Recommended content type (money page/blog/hub)
- Content brief outline""",

            "strategic_overview": """
FOCUS: Strategic Organic Performance Overview

Analyze the data to provide:
1. Brand vs non-brand traffic dependency analysis
2. Category-level strengths and weaknesses
3. Device and geo performance discrepancies
4. Competitive positioning assessment
5. Key growth levers and strategic priorities

Structure the output as:
- Executive Summary (3-5 key findings)
- Strengths to Leverage
- Critical Gaps to Address
- Strategic Recommendations (prioritized)
- Resource Allocation Guidance""",

            "full_analysis": """
FOCUS: Comprehensive Organic Performance Analysis

Provide a complete analysis covering:

1. STRATEGIC OVERVIEW
   - Brand vs non-brand dependency
   - Category strengths/weaknesses
   - Competitive positioning

2. QUICK WINS (Immediate Opportunities)
   - CTR optimization opportunities
   - Low-hanging fruit by position/impression gaps
   - SERP feature opportunities

3. CONTENT DECAY & RECOVERY
   - Declining assets requiring attention
   - Decay pattern classification
   - Recovery priorities

4. GROWTH OPPORTUNITIES
   - Keyword gaps to fill
   - New content opportunities
   - Topic clusters to develop

5. STRUCTURAL RECOMMENDATIONS
   - Internal linking improvements
   - Content architecture changes
   - Technical considerations based on patterns

6. PRIORITIZED ACTION PLAN
   - Tier 1: High impact, low effort (do immediately)
   - Tier 2: High impact, medium effort (next 30 days)
   - Tier 3: Strategic initiatives (ongoing)

Each section must include specific queries, pages, and metrics.
Avoid any fluff or celebratory language - focus purely on actions and opportunities."""
        }

        return base_prompt + type_specific.get(analysis_type, type_specific["full_analysis"])

    def stream_analysis(
        self,
        data_summary: str,
        analysis_type: str,
        model: str = "anthropic/claude-sonnet-4.5",
        additional_context: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream SEO analysis using LLM.

        Args:
            data_summary: Summary of the data to analyze
            analysis_type: Type of analysis to perform
            model: Model to use
            additional_context: Additional context

        Yields:
            Analysis text chunks
        """
        system_prompt = self._get_seo_analysis_prompt(analysis_type)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_summary}
        ]

        if additional_context:
            messages.append({
                "role": "user",
                "content": f"Additional context: {additional_context}"
            })

        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 8192,
            "stream": True
        }

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            yield f"\n\nError in streaming analysis: {str(e)}"


def get_model_options() -> List[str]:
    """Get list of available model IDs."""
    return list(AVAILABLE_MODELS.keys())


def get_model_info(model_id: str) -> Dict[str, str]:
    """Get information about a specific model."""
    return AVAILABLE_MODELS.get(model_id, {
        "name": model_id,
        "provider": "Unknown",
        "description": "No description available"
    })


def format_model_display(model_id: str) -> str:
    """Format model ID for display in dropdown."""
    info = get_model_info(model_id)
    return f"{info['name']} ({info['provider']})"
