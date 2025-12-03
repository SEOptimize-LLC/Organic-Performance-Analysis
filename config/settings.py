"""
Application settings and configuration.
Contains thresholds, scoring weights, and other configurable parameters.
"""

import os
from pathlib import Path
from typing import Dict, List
from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings management with Pydantic validation"""
    
    # Application
    app_name: str = "Organic Performance Analyzer"
    app_version: str = "1.0.0"
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    cache_dir: Path = base_dir / ".cache"
    
    # Cache TTL (in hours)
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "168"))  # 7 days
    gsc_cache_ttl_hours: int = int(os.getenv("GSC_CACHE_TTL_HOURS", "24"))
    dataforseo_cache_ttl_hours: int = int(
        os.getenv("DATAFORSEO_CACHE_TTL_HOURS", "168")
    )
    
    # Rate limiting
    gsc_requests_per_minute: int = 200
    dataforseo_requests_per_minute: int = 60
    openrouter_requests_per_minute: int = 30
    
    # GSC Data Freshness Delay (days)
    gsc_data_freshness_delay: int = 3
    
    # Date Windows Configuration
    date_windows: Dict[str, Dict] = {
        "28d": {"days": 28, "label": "Last 28 Days"},
        "90d": {"days": 90, "label": "Last 3 Months"},
        "180d": {"days": 180, "label": "Last 6 Months"},
        "365d": {"days": 365, "label": "Last 12 Months"},
    }
    
    # Position Thresholds for Opportunity Classification
    quick_win_position_min: int = 4
    quick_win_position_max: int = 10
    striking_distance_position_min: int = 11
    striking_distance_position_max: int = 20
    strategic_position_min: int = 21
    strategic_position_max: int = 30
    
    # Minimum Thresholds
    min_impressions_default: int = 50
    min_clicks_default: int = 1
    min_search_volume: int = 10
    
    # Decay Detection Thresholds
    decay_threshold_position: int = 3  # Position drop > 3 = decay
    decay_threshold_impressions_pct: float = -20.0  # -20% = decay
    decay_threshold_clicks_pct: float = -20.0  # -20% = decay
    
    # CTR Benchmarks by Position (based on industry studies)
    expected_ctr_by_position: Dict[int, float] = {
        1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07,
        6: 0.05, 7: 0.04, 8: 0.03, 9: 0.03, 10: 0.02,
        11: 0.01, 12: 0.009, 13: 0.008, 14: 0.007, 15: 0.006,
        16: 0.005, 17: 0.004, 18: 0.004, 19: 0.003, 20: 0.003
    }
    
    # Opportunity Scoring Weights
    scoring_weights: Dict[str, float] = {
        "search_volume": 0.25,
        "position_potential": 0.20,
        "ctr_gap": 0.20,
        "commercial_value": 0.20,
        "trend_direction": 0.15
    }
    
    # Cannibalization Settings
    cannibalization_min_pages: int = 2
    cannibalization_min_impressions: int = 100
    
    # DataForSEO Settings
    dataforseo_batch_size: int = 1000
    dataforseo_default_location: int = 2840  # United States
    dataforseo_default_language: str = "en"
    
    # OpenRouter Models
    available_llm_models: List[str] = [
        "openai/gpt-4.1-mini",
        "openai/gpt-5-mini",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
        "google/gemini-3-pro-preview",
        "google/gemini-2.5-flash-preview-09-2025",
        "x-ai/grok-4-fast",
        "deepseek/deepseek-r1-0528-qwen3-8b"
    ]
    default_llm_model: str = "anthropic/claude-sonnet-4.5"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7
    
    # Report Settings
    max_quick_wins: int = 50
    max_decay_items: int = 50
    max_gap_keywords: int = 100
    max_cannibalization_cases: int = 25
    
    @field_validator("cache_dir")
    @classmethod
    def create_cache_dir(cls, v: Path) -> Path:
        """Ensure cache directory exists"""
        v.mkdir(exist_ok=True, parents=True)
        return v
    
    def get_cache_ttl_seconds(self, cache_type: str = "default") -> int:
        """Get cache TTL in seconds for specific cache type"""
        ttl_map = {
            "gsc": self.gsc_cache_ttl_hours,
            "dataforseo": self.dataforseo_cache_ttl_hours,
            "default": self.cache_ttl_hours
        }
        return ttl_map.get(cache_type, self.cache_ttl_hours) * 3600
    
    def get_expected_ctr(self, position: float) -> float:
        """Get expected CTR for a given position"""
        pos_int = min(int(round(position)), 20)
        if pos_int < 1:
            pos_int = 1
        return self.expected_ctr_by_position.get(pos_int, 0.002)
    
    def classify_position_tier(self, position: float) -> str:
        """Classify position into opportunity tiers"""
        if position <= 3:
            return "top3"
        elif position <= self.quick_win_position_max:
            return "quick_win"
        elif position <= self.striking_distance_position_max:
            return "striking_distance"
        elif position <= self.strategic_position_max:
            return "strategic"
        else:
            return "long_term"
    
    class Config:
        case_sensitive = False


# Singleton instance
settings = Settings()