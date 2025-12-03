"""
Authentication service for Google Search Console API.
Handles OAuth2 flow with proper state management and token persistence.
"""

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from typing import Optional, Tuple, Dict, Any
import pickle
from pathlib import Path
import os
from datetime import datetime, timedelta
import secrets
import hashlib

from config.api_config import api_config
from utils.logger import logger


class AuthService:
    """
    Production-ready Google OAuth2 authentication with proper state management.
    Handles token refresh, persistence, and Streamlit Cloud compatibility.
    """
    
    def __init__(self):
        """Initialize authentication service"""
        self._init_session_state()
        
        # Set token file path based on environment
        if self._is_streamlit_cloud():
            self.token_file = Path('/tmp/.organic_analyzer/token.pickle')
        else:
            self.token_file = Path.home() / '.organic_analyzer' / 'token.pickle'
        
        self.token_file.parent.mkdir(exist_ok=True, parents=True)
    
    def _init_session_state(self):
        """Initialize required session state keys"""
        defaults = {
            'auth_state': None,
            'auth_initiated_at': None,
            'auth_code_used': set(),
            'credentials_cache': None,
            'gsc_service': None,
            'authenticated': False
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def _is_streamlit_cloud(self) -> bool:
        """Check if running on Streamlit Cloud"""
        return os.environ.get('STREAMLIT_SHARING_MODE') == 'private'
    
    def generate_state_token(self) -> str:
        """Generate a secure state token for CSRF protection"""
        state = secrets.token_urlsafe(32)
        st.session_state.auth_state = state
        st.session_state.auth_initiated_at = datetime.now()
        logger.info(f"Generated new state token: {state[:8]}...")
        return state
    
    def verify_state_token(self, state: str) -> bool:
        """Verify the state token is valid and not expired"""
        if not st.session_state.auth_state:
            logger.error("No auth state found in session")
            return False
        
        if state != st.session_state.auth_state:
            logger.error("State mismatch in OAuth callback")
            return False
        
        # Check if state is expired (5 minutes timeout)
        if st.session_state.auth_initiated_at:
            elapsed = datetime.now() - st.session_state.auth_initiated_at
            if elapsed > timedelta(minutes=5):
                logger.error("Auth state expired")
                return False
        
        return True
    
    def get_auth_url(self, login_hint: str = None) -> str:
        """
        Generate OAuth2 authorization URL with proper state management.
        
        Args:
            login_hint: Optional email to pre-select in Google account picker
        """
        if not api_config.has_gsc_credentials():
            raise ValueError("Google OAuth credentials not configured")
        
        try:
            # Clear any previous auth attempts
            self.clear_auth_state()
            
            # Generate new state token
            state = self.generate_state_token()
            
            # Get client config and redirect URI
            client_config = api_config.get_google_client_config()
            
            # Extract redirect_uri from config (installed type)
            config_key = "installed" if "installed" in client_config else "web"
            redirect_uri = client_config[config_key]["redirect_uris"][0]
            
            logger.info(f"Using redirect_uri: '{redirect_uri}'")
            logger.info(f"Using config type: '{config_key}'")
            
            # Create OAuth flow
            flow = Flow.from_client_config(
                client_config,
                scopes=api_config.oauth2_scopes,
                redirect_uri=redirect_uri,
                state=state
            )
            
            # Store flow in session for later token exchange
            st.session_state.auth_flow = flow
            
            # Build authorization URL with optional login hint
            auth_params = {
                'access_type': 'offline',
                'include_granted_scopes': 'true',
                'prompt': 'consent'
            }
            
            # Add login_hint if provided to pre-select account
            if login_hint:
                auth_params['login_hint'] = login_hint
                logger.info(f"Using login_hint: {login_hint}")
            
            auth_url, _ = flow.authorization_url(**auth_params)
            
            logger.info(f"Generated auth URL with redirect: {redirect_uri}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Error creating auth URL: {str(e)}")
            raise
    
    def handle_callback(
        self, 
        query_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Handle OAuth2 callback with comprehensive error handling"""
        try:
            # Extract parameters - handle both list and string values
            code = query_params.get('code')
            if isinstance(code, list):
                code = code[0] if code else None
            
            state = query_params.get('state')
            if isinstance(state, list):
                state = state[0] if state else None
            
            error = query_params.get('error')
            if isinstance(error, list):
                error = error[0] if error else None
            
            # Check for OAuth errors
            if error:
                error_desc = query_params.get('error_description', 'Unknown error')
                if isinstance(error_desc, list):
                    error_desc = error_desc[0]
                logger.error(f"OAuth error: {error} - {error_desc}")
                return False, f"Authorization failed: {error_desc}"
            
            if not code:
                return False, "No authorization code received"
            
            if not state:
                return False, "No state parameter received"
            
            # Verify state to prevent CSRF
            if not self.verify_state_token(state):
                return False, "Invalid or expired state token. Please try again."
            
            # Check if code was already used
            if code in st.session_state.auth_code_used:
                return False, "Authorization code already used. Please start over."
            
            # Mark code as used immediately
            st.session_state.auth_code_used.add(code)
            
            # Get flow from session or create new one
            client_config = api_config.get_google_client_config()
            config_key = "installed" if "installed" in client_config else "web"
            redirect_uri = client_config[config_key]["redirect_uris"][0]
            
            # Check if we have flow in session
            if hasattr(st.session_state, 'auth_flow') and st.session_state.auth_flow:
                flow = st.session_state.auth_flow
            else:
                flow = Flow.from_client_config(
                    client_config,
                    scopes=api_config.oauth2_scopes,
                    redirect_uri=redirect_uri,
                    state=state
                )
            
            # Build authorization response URL
            auth_response = f"{redirect_uri}?code={code}&state={state}"
            
            try:
                # Exchange code for token
                logger.info("Exchanging authorization code for token...")
                flow.fetch_token(authorization_response=auth_response)
                
                # Get credentials
                credentials = flow.credentials
                if not credentials or not credentials.token:
                    return False, "Failed to obtain valid credentials"
                
                # Save credentials
                self._save_credentials(credentials)
                
                # Build and cache GSC service
                service = build('searchconsole', 'v1', credentials=credentials)
                st.session_state.gsc_service = service
                st.session_state.authenticated = True
                
                # Clear auth state
                self.clear_auth_state()
                
                logger.info("Successfully authenticated with Google")
                return True, None
                
            except Exception as token_error:
                error_msg = str(token_error)
                logger.error(f"Token exchange failed: {error_msg}")
                
                # Provide user-friendly error messages
                if "invalid_grant" in error_msg:
                    return False, (
                        "The authorization code is invalid or expired. "
                        "Please try again."
                    )
                elif "redirect_uri_mismatch" in error_msg:
                    return False, (
                        "Configuration error: Redirect URI mismatch. "
                        "Please check your Google Cloud Console settings."
                    )
                elif "invalid_client" in error_msg:
                    return False, (
                        "Configuration error: Invalid client credentials. "
                        "Please verify your client ID and secret."
                    )
                else:
                    return False, f"Authentication failed: {error_msg}"
                
        except Exception as e:
            logger.error(f"Callback handling failed: {str(e)}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"
    
    def clear_auth_state(self):
        """Clear authentication state"""
        st.session_state.auth_state = None
        st.session_state.auth_initiated_at = None
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get stored credentials with proper caching and refresh handling"""
        # Check memory cache first
        if st.session_state.credentials_cache:
            creds = st.session_state.credentials_cache
            
            # Check if refresh is needed
            if creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired credentials from cache")
                    creds.refresh(Request())
                    self._save_credentials(creds)
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {str(e)}")
                    self.revoke_credentials()
                    return None
            
            if creds.valid:
                return creds
        
        # Try loading from file
        if self.token_file.exists():
            try:
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                
                # Verify credentials are valid
                if not hasattr(creds, 'token'):
                    logger.error("Invalid credentials format")
                    self.revoke_credentials()
                    return None
                
                # Refresh if needed
                if creds.expired and creds.refresh_token:
                    try:
                        logger.info("Refreshing expired credentials from file")
                        creds.refresh(Request())
                        self._save_credentials(creds)
                    except Exception as e:
                        logger.error(f"Failed to refresh credentials: {str(e)}")
                        self.revoke_credentials()
                        return None
                
                # Cache in memory
                st.session_state.credentials_cache = creds
                return creds if creds.valid else None
                
            except Exception as e:
                logger.error(f"Failed to load credentials from file: {str(e)}")
                self.revoke_credentials()
                return None
        
        return None
    
    def _save_credentials(self, credentials: Credentials):
        """Save credentials with proper error handling"""
        try:
            # Save to memory cache
            st.session_state.credentials_cache = credentials
            
            # Save to file
            try:
                with open(self.token_file, 'wb') as token:
                    pickle.dump(credentials, token)
                logger.info("Credentials saved successfully")
            except Exception as e:
                # File save might fail on Streamlit Cloud, but that's okay
                logger.warning(f"Could not save credentials to file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to save credentials: {str(e)}")
            raise
    
    def revoke_credentials(self):
        """Revoke stored credentials and clean up"""
        try:
            # Clear file
            if self.token_file.exists():
                self.token_file.unlink()
                logger.info("Credentials file deleted")
        except Exception as e:
            logger.warning(f"Could not delete token file: {str(e)}")
        
        # Clear session state
        st.session_state.credentials_cache = None
        st.session_state.gsc_service = None
        st.session_state.authenticated = False
        
        self.clear_auth_state()
        st.session_state.auth_code_used.clear()
        
        logger.info("All credentials and auth state cleared")
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated with valid credentials"""
        creds = self.get_credentials()
        return creds is not None and creds.valid
    
    def get_gsc_service(self):
        """Get or create GSC service"""
        if st.session_state.gsc_service:
            return st.session_state.gsc_service
        
        creds = self.get_credentials()
        if creds:
            service = build('searchconsole', 'v1', credentials=creds)
            st.session_state.gsc_service = service
            return service
        
        return None
    
    def test_credentials(self) -> Tuple[bool, Optional[str]]:
        """Test if credentials are working by making a simple API call"""
        creds = self.get_credentials()
        if not creds:
            return False, "No credentials found"
        
        try:
            service = build('searchconsole', 'v1', credentials=creds)
            # Try to list sites (minimal API call)
            service.sites().list().execute()
            return True, None
        except Exception as e:
            return False, str(e)


def get_query_params() -> Dict[str, Any]:
    """Get query parameters in a version-safe way"""
    try:
        # Try new API (Streamlit >= 1.28)
        return dict(st.query_params)
    except AttributeError:
        try:
            # Try experimental API (older versions)
            params = st.experimental_get_query_params()
            return {
                k: v[0] if isinstance(v, list) and v else v 
                for k, v in params.items()
            }
        except Exception:
            return {}


def clear_query_params():
    """Clear query parameters in a version-safe way"""
    try:
        st.query_params.clear()
    except AttributeError:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass