"""
API credentials and endpoints configuration.
Handles credentials from Streamlit secrets and environment variables.
"""

import os
from typing import Optional, Tuple, Dict, Any
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_streamlit_secret(
    section: str,
    key: str,
    default: str = "",
    flat_key: str = None
) -> str:
    """
    Get secret from Streamlit secrets or environment variable.
    Supports both nested (st.secrets["section"]["key"]) and flat formats.
    Streamlit secrets take priority over environment variables.
    
    Args:
        section: Section name in secrets (e.g., "gsc", "dataforseo")
        key: Key name within section
        default: Default value if not found
        flat_key: Optional flat environment variable name
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            # Try nested format first: st.secrets["section"]["key"]
            if section in st.secrets:
                section_data = st.secrets[section]
                if isinstance(section_data, dict) and key in section_data:
                    return section_data[key]
                elif hasattr(section_data, key):
                    return getattr(section_data, key)
            # Try flat format: st.secrets["FLAT_KEY"]
            if flat_key and flat_key in st.secrets:
                return st.secrets[flat_key]
    except Exception:
        pass
    
    # Fall back to environment variable
    env_key = flat_key if flat_key else f"{section.upper()}_{key.upper()}"
    return os.getenv(env_key, default)


class APIConfig(BaseSettings):
    """API credentials and endpoints configuration"""
    
    # Google Search Console OAuth
    google_client_id: Optional[str] = None
    google_client_secret: Optional[SecretStr] = None
    google_redirect_uri: str = "http://localhost:8501"
    
    # DataForSEO Credentials
    dataforseo_login: Optional[str] = None
    dataforseo_password: Optional[SecretStr] = None
    
    # OpenRouter API
    openrouter_api_key: Optional[SecretStr] = None
    
    # API Base URLs
    dataforseo_base_url: str = "https://api.dataforseo.com"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # DataForSEO API Endpoints
    # Labs endpoints
    dataforseo_ranked_keywords_endpoint: str = (
        "/v3/dataforseo_labs/google/ranked_keywords/live"
    )
    dataforseo_domain_metrics_endpoint: str = (
        "/v3/dataforseo_labs/google/domain_metrics_by_categories/live"
    )
    dataforseo_competitors_endpoint: str = (
        "/v3/dataforseo_labs/google/competitors_domain/live"
    )
    dataforseo_keyword_suggestions_endpoint: str = (
        "/v3/dataforseo_labs/google/keyword_suggestions/live"
    )
    dataforseo_search_volume_endpoint: str = (
        "/v3/keywords_data/clickstream_data/dataforseo_search_volume/live"
    )
    
    # SERP endpoints
    dataforseo_serp_organic_endpoint: str = (
        "/v3/serp/google/organic/live/regular"
    )
    
    # OAuth2 Settings
    oauth2_scopes: list = [
        'https://www.googleapis.com/auth/webmasters.readonly'
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_credentials()
    
    def _load_credentials(self):
        """Load credentials from Streamlit secrets or environment"""
        # Google credentials - supports [gsc] section or flat keys
        self.google_client_id = get_streamlit_secret(
            section="gsc",
            key="client_id",
            flat_key="GOOGLE_CLIENT_ID"
        )
        
        google_secret = get_streamlit_secret(
            section="gsc",
            key="client_secret",
            flat_key="GOOGLE_CLIENT_SECRET"
        )
        if google_secret:
            self.google_client_secret = SecretStr(google_secret)
        
        # Google Redirect URI (important for Streamlit Cloud)
        self.google_redirect_uri = get_streamlit_secret(
            section="gsc",
            key="redirect_uri",
            default="http://localhost:8501",
            flat_key="GOOGLE_REDIRECT_URI"
        )
        
        # DataForSEO credentials - supports [dataforseo] section
        self.dataforseo_login = get_streamlit_secret(
            section="dataforseo",
            key="login",
            flat_key="DATAFORSEO_LOGIN"
        )
        
        dataforseo_pass = get_streamlit_secret(
            section="dataforseo",
            key="password",
            flat_key="DATAFORSEO_PASSWORD"
        )
        if dataforseo_pass:
            self.dataforseo_password = SecretStr(dataforseo_pass)
        
        # OpenRouter API Key - supports [openrouter] section
        openrouter_key = get_streamlit_secret(
            section="openrouter",
            key="api_key",
            flat_key="OPENROUTER_API_KEY"
        )
        if openrouter_key:
            self.openrouter_api_key = SecretStr(openrouter_key)
    
    def get_dataforseo_auth(self) -> Optional[Tuple[str, str]]:
        """Get DataForSEO authentication tuple for requests"""
        if self.dataforseo_login and self.dataforseo_password:
            return (
                self.dataforseo_login,
                self.dataforseo_password.get_secret_value()
            )
        return None
    
    def get_openrouter_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.openrouter_api_key:
            headers["Authorization"] = (
                f"Bearer {self.openrouter_api_key.get_secret_value()}"
            )
        return headers
    
    def get_google_client_config(self) -> Dict[str, Any]:
        """Get Google OAuth2 client configuration"""
        if not self.google_client_id or not self.google_client_secret:
            return {}
        
        return {
            "web": {
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret.get_secret_value(),
                "redirect_uris": [self.google_redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        }
    
    def has_gsc_credentials(self) -> bool:
        """Check if GSC credentials are configured"""
        return bool(self.google_client_id and self.google_client_secret)
    
    def has_dataforseo_credentials(self) -> bool:
        """Check if DataForSEO credentials are configured"""
        return bool(self.dataforseo_login and self.dataforseo_password)
    
    def has_openrouter_credentials(self) -> bool:
        """Check if OpenRouter API key is configured"""
        return bool(self.openrouter_api_key)
    
    def get_credentials_status(self) -> Dict[str, bool]:
        """Get status of all API credentials"""
        return {
            "gsc": self.has_gsc_credentials(),
            "dataforseo": self.has_dataforseo_credentials(),
            "openrouter": self.has_openrouter_credentials()
        }
    
    class Config:
        case_sensitive = False


# Singleton instance
api_config = APIConfig()