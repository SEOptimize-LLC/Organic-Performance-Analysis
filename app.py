"""
Organic Performance Analysis Tool

A comprehensive SEO analysis dashboard that leverages Google Search Console,
DataForSEO, and LLM-powered insights to provide actionable recommendations.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import time

# Import custom modules
from src.gsc_api import (
    GSCClient, create_oauth_flow, credentials_from_dict,
    credentials_to_dict, get_date_ranges, classify_brand_queries
)
from src.dataforseo_api import (
    DataForSEOClient, get_location_codes, get_language_codes
)
from src.llm_api import (
    OpenRouterClient, AVAILABLE_MODELS, get_model_options,
    get_model_info, format_model_display
)
from src.analysis_engine import (
    SEOAnalysisEngine, AnalysisConfig, create_analysis_config
)

# Page configuration
st.set_page_config(
    page_title="Organic Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f9fafb;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #e5e7eb;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'gsc_authenticated': False,
        'gsc_credentials': None,
        'gsc_sites': [],
        'selected_site': None,
        'analysis_data': {},
        'analysis_complete': False,
        'llm_response': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_secrets():
    """Check if required secrets are configured."""
    required = {
        'dataforseo': ['login', 'password'],
        'openrouter': ['api_key'],
        'google': ['client_id', 'client_secret']
    }

    missing = []
    for section, keys in required.items():
        if section not in st.secrets:
            missing.append(f"Section: [{section}]")
        else:
            for key in keys:
                if key not in st.secrets[section]:
                    missing.append(f"{section}.{key}")

    return missing


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.markdown("## Configuration")

    # Check secrets
    missing_secrets = check_secrets()
    if missing_secrets:
        st.sidebar.error("Missing secrets configuration:")
        for secret in missing_secrets:
            st.sidebar.write(f"- {secret}")
        st.sidebar.info(
            "Please configure secrets in Streamlit Cloud settings "
            "or `.streamlit/secrets.toml` for local development."
        )
        return None

    # GSC Authentication
    st.sidebar.markdown("### Google Search Console")

    if not st.session_state.gsc_authenticated:
        if st.sidebar.button("Connect to GSC", type="primary"):
            handle_gsc_auth()
    else:
        st.sidebar.success("GSC Connected")

        # Site selector
        if st.session_state.gsc_sites:
            site_options = [site['siteUrl'] for site in st.session_state.gsc_sites]
            selected = st.sidebar.selectbox(
                "Select Property",
                options=site_options,
                key="site_selector"
            )
            st.session_state.selected_site = selected

        if st.sidebar.button("Disconnect"):
            st.session_state.gsc_authenticated = False
            st.session_state.gsc_credentials = None
            st.session_state.gsc_sites = []
            st.rerun()

    st.sidebar.markdown("---")

    # Analysis Parameters
    st.sidebar.markdown("### Analysis Parameters")

    # Date range
    date_range = st.sidebar.selectbox(
        "Date Range",
        options=[
            "Last 28 days",
            "Last 3 months",
            "Last 6 months",
            "Last 12 months"
        ],
        index=1
    )

    # Location
    location = st.sidebar.selectbox(
        "Location",
        options=list(get_location_codes().keys()),
        index=0
    )

    # Language
    language = st.sidebar.selectbox(
        "Language",
        options=list(get_language_codes().keys()),
        index=0
    )

    # Brand terms
    brand_terms = st.sidebar.text_area(
        "Brand Terms (one per line)",
        placeholder="company name\nbrand\nabbreviation"
    )

    # Competitors
    competitors = st.sidebar.text_area(
        "Competitor Domains (one per line)",
        placeholder="competitor1.com\ncompetitor2.com"
    )

    # Minimum impressions
    min_impressions = st.sidebar.slider(
        "Minimum Impressions",
        min_value=1,
        max_value=100,
        value=10
    )

    st.sidebar.markdown("---")

    # LLM Configuration
    st.sidebar.markdown("### AI Analysis Model")

    model_options = get_model_options()
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        format_func=format_model_display,
        index=3  # Default to Claude Sonnet 4.5
    )

    model_info = get_model_info(selected_model)
    st.sidebar.caption(f"Provider: {model_info['provider']}")
    st.sidebar.caption(model_info['description'])

    return {
        'date_range': date_range,
        'location': location,
        'language': language,
        'brand_terms': [t.strip() for t in brand_terms.split('\n') if t.strip()],
        'competitors': [c.strip() for c in competitors.split('\n') if c.strip()],
        'min_impressions': min_impressions,
        'selected_model': selected_model
    }


def handle_gsc_auth():
    """Handle GSC OAuth authentication flow."""
    try:
        client_id = st.secrets.google.client_id
        client_secret = st.secrets.google.client_secret

        # For Streamlit Cloud, use the app URL
        # For local development, use localhost
        redirect_uri = st.secrets.google.get(
            'redirect_uri',
            'http://localhost:8501'
        )

        flow = create_oauth_flow(client_id, client_secret, redirect_uri)
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )

        st.sidebar.markdown(f"[Click here to authorize]({auth_url})")
        st.sidebar.info("After authorizing, paste the code below:")

        auth_code = st.sidebar.text_input("Authorization Code", type="password")

        if auth_code and st.sidebar.button("Complete Authentication"):
            try:
                flow.fetch_token(code=auth_code)
                credentials = flow.credentials

                st.session_state.gsc_credentials = credentials_to_dict(credentials)
                st.session_state.gsc_credentials['client_id'] = client_id
                st.session_state.gsc_credentials['client_secret'] = client_secret

                # Get list of sites
                creds = credentials_from_dict(st.session_state.gsc_credentials)
                gsc_client = GSCClient(creds)
                sites = gsc_client.get_sites()

                st.session_state.gsc_sites = sites
                st.session_state.gsc_authenticated = True

                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Authentication failed: {str(e)}")

    except Exception as e:
        st.sidebar.error(f"Error setting up authentication: {str(e)}")


def get_date_range_values(date_range: str) -> tuple:
    """Convert date range string to actual dates."""
    today = datetime.now()

    ranges = {
        "Last 28 days": (
            today - timedelta(days=28),
            today - timedelta(days=1)
        ),
        "Last 3 months": (
            today - relativedelta(months=3),
            today - timedelta(days=1)
        ),
        "Last 6 months": (
            today - relativedelta(months=6),
            today - timedelta(days=1)
        ),
        "Last 12 months": (
            today - relativedelta(months=12),
            today - timedelta(days=1)
        )
    }

    start, end = ranges.get(date_range, ranges["Last 3 months"])
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def fetch_gsc_data(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Fetch data from Google Search Console."""
    if not st.session_state.gsc_authenticated:
        return {}

    try:
        creds = credentials_from_dict(st.session_state.gsc_credentials)
        gsc_client = GSCClient(creds)

        site_url = st.session_state.selected_site
        start_date, end_date = get_date_range_values(config['date_range'])

        data = {}

        with st.spinner("Fetching query data from GSC..."):
            data['query_data'] = gsc_client.get_query_data(
                site_url=site_url,
                start_date=start_date,
                end_date=end_date,
                min_impressions=config['min_impressions']
            )

        with st.spinner("Fetching page data from GSC..."):
            data['page_data'] = gsc_client.get_page_data(
                site_url=site_url,
                start_date=start_date,
                end_date=end_date,
                min_impressions=config['min_impressions']
            )

        # Get comparison period data
        with st.spinner("Fetching comparison period data..."):
            # Calculate previous period
            days_diff = (datetime.strptime(end_date, '%Y-%m-%d') -
                        datetime.strptime(start_date, '%Y-%m-%d')).days

            prev_end = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)
            prev_start = prev_end - timedelta(days=days_diff)

            data['previous_query_data'] = gsc_client.get_query_data(
                site_url=site_url,
                start_date=prev_start.strftime('%Y-%m-%d'),
                end_date=prev_end.strftime('%Y-%m-%d'),
                min_impressions=config['min_impressions']
            )

        # Get search appearance data
        with st.spinner("Fetching search appearance data..."):
            data['search_appearance'] = gsc_client.get_search_appearance_data(
                site_url=site_url,
                start_date=start_date,
                end_date=end_date
            )

        return data

    except Exception as e:
        st.error(f"Error fetching GSC data: {str(e)}")
        return {}


def fetch_dataforseo_data(domain: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch data from DataForSEO."""
    try:
        client = DataForSEOClient(
            login=st.secrets.dataforseo.login,
            password=st.secrets.dataforseo.password
        )

        location_code = get_location_codes()[config['location']]
        language_code = get_language_codes()[config['language']]

        data = {}

        with st.spinner("Fetching ranked keywords from DataForSEO..."):
            data['ranked_keywords'] = client.get_ranked_keywords(
                target=domain,
                location_code=location_code,
                language_code=language_code,
                limit=1000
            )

        with st.spinner("Fetching domain overview..."):
            data['domain_overview'] = client.get_domain_overview(
                target=domain,
                location_code=location_code,
                language_code=language_code
            )

        with st.spinner("Identifying competitors..."):
            data['competitors'] = client.get_competitors(
                target=domain,
                location_code=location_code,
                language_code=language_code
            )

        with st.spinner("Fetching category data..."):
            data['categories'] = client.get_categories_for_domain(
                target=domain,
                location_code=location_code,
                language_code=language_code
            )

        # Fetch competitor keyword data if competitors specified
        if config['competitors']:
            with st.spinner("Analyzing competitor keywords..."):
                competitor_keywords = []
                for comp in config['competitors'][:3]:  # Limit to 3 competitors
                    comp_kw = client.get_ranked_keywords(
                        target=comp,
                        location_code=location_code,
                        language_code=language_code,
                        limit=500
                    )
                    if not comp_kw.empty:
                        comp_kw['competitor'] = comp
                        competitor_keywords.append(comp_kw)

                if competitor_keywords:
                    data['competitor_keywords'] = pd.concat(competitor_keywords, ignore_index=True)
                else:
                    data['competitor_keywords'] = pd.DataFrame()

        with st.spinner("Fetching historical data..."):
            data['historical'] = client.get_historical_rank_data(
                target=domain,
                location_code=location_code,
                language_code=language_code
            )

        return data

    except Exception as e:
        st.error(f"Error fetching DataForSEO data: {str(e)}")
        return {}


def run_analysis(gsc_data: Dict, dataforseo_data: Dict, config: Dict) -> Dict:
    """Run the SEO analysis."""
    analysis_config = create_analysis_config(
        brand_terms=config['brand_terms'],
        min_impressions=config['min_impressions'],
        competitors=config['competitors']
    )

    engine = SEOAnalysisEngine(analysis_config)

    results = {}

    # Quick wins analysis
    with st.spinner("Identifying quick wins..."):
        results['quick_wins'] = engine.identify_quick_wins(
            gsc_data=gsc_data.get('query_data', pd.DataFrame()),
            dataforseo_data=dataforseo_data.get('ranked_keywords', pd.DataFrame())
        )

    # Content decay analysis
    with st.spinner("Analyzing content decay..."):
        results['decay'] = engine.identify_content_decay(
            current_data=gsc_data.get('query_data', pd.DataFrame()),
            previous_data=gsc_data.get('previous_query_data', pd.DataFrame())
        )

    # Keyword gaps
    if 'competitor_keywords' in dataforseo_data:
        with st.spinner("Identifying keyword gaps..."):
            results['keyword_gaps'] = engine.identify_keyword_gaps(
                site_keywords=dataforseo_data.get('ranked_keywords', pd.DataFrame()),
                competitor_keywords=dataforseo_data.get('competitor_keywords', pd.DataFrame()),
                gsc_data=gsc_data.get('query_data')
            )

    # Brand analysis
    with st.spinner("Analyzing brand dependency..."):
        results['brand_analysis'] = engine.analyze_brand_dependency(
            gsc_data.get('query_data', pd.DataFrame())
        )

    # Device analysis
    with st.spinner("Analyzing device performance..."):
        results['device_analysis'] = engine.analyze_device_performance(
            gsc_data.get('query_data', pd.DataFrame())
        )

    # Country analysis
    with st.spinner("Analyzing country performance..."):
        results['country_analysis'] = engine.analyze_country_performance(
            gsc_data.get('query_data', pd.DataFrame())
        )

    # Generate summary for LLM
    with st.spinner("Generating analysis summary..."):
        results['summary'] = engine.generate_analysis_summary(
            gsc_data=gsc_data.get('query_data', pd.DataFrame()),
            dataforseo_data=dataforseo_data.get('ranked_keywords'),
            competitor_data=dataforseo_data.get('competitors'),
            quick_wins=results.get('quick_wins'),
            decay_data=results.get('decay'),
            keyword_gaps=results.get('keyword_gaps')
        )

    return results


def render_metrics_overview(gsc_data: Dict, dataforseo_data: Dict, analysis_results: Dict):
    """Render the metrics overview section."""
    st.markdown("## Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    query_data = gsc_data.get('query_data', pd.DataFrame())

    with col1:
        clicks = int(query_data['clicks'].sum()) if not query_data.empty else 0
        st.metric("Total Clicks", f"{clicks:,}")

    with col2:
        impressions = int(query_data['impressions'].sum()) if not query_data.empty else 0
        st.metric("Total Impressions", f"{impressions:,}")

    with col3:
        avg_ctr = query_data['ctr'].mean() * 100 if not query_data.empty else 0
        st.metric("Avg CTR", f"{avg_ctr:.2f}%")

    with col4:
        avg_pos = query_data['position'].mean() if not query_data.empty else 0
        st.metric("Avg Position", f"{avg_pos:.1f}")

    # Second row with DataForSEO metrics
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    ranked_kw = dataforseo_data.get('ranked_keywords', pd.DataFrame())

    with col1:
        total_kw = len(ranked_kw) if not ranked_kw.empty else 0
        st.metric("Ranked Keywords", f"{total_kw:,}")

    with col2:
        top_10 = len(ranked_kw[ranked_kw['position'] <= 10]) if not ranked_kw.empty else 0
        st.metric("Top 10 Keywords", f"{top_10:,}")

    with col3:
        etv = int(ranked_kw['etv'].sum()) if not ranked_kw.empty else 0
        st.metric("Est. Traffic Value", f"{etv:,}")

    with col4:
        brand = analysis_results.get('brand_analysis', {})
        non_brand_share = brand.get('non_brand_share', 0)
        st.metric("Non-Brand Share", f"{non_brand_share}%")


def render_quick_wins(quick_wins: pd.DataFrame):
    """Render quick wins section."""
    st.markdown("## Quick Win Opportunities")

    if quick_wins.empty:
        st.info("No quick win opportunities identified with current filters.")
        return

    st.markdown(f"Found **{len(quick_wins):,}** opportunities with underperforming CTR")

    # Show top opportunities
    display_cols = ['query', 'page', 'position', 'impressions', 'clicks',
                   'ctr', 'expected_ctr', 'ctr_gap_pct', 'opportunity_score', 'intent']
    available_cols = [c for c in display_cols if c in quick_wins.columns]

    # Format for display
    display_df = quick_wins[available_cols].head(50).copy()
    if 'ctr' in display_df.columns:
        display_df['ctr'] = (display_df['ctr'] * 100).round(2).astype(str) + '%'
    if 'expected_ctr' in display_df.columns:
        display_df['expected_ctr'] = (display_df['expected_ctr'] * 100).round(2).astype(str) + '%'
    if 'position' in display_df.columns:
        display_df['position'] = display_df['position'].round(1)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Visualization
    if not quick_wins.empty:
        fig = px.scatter(
            quick_wins.head(100),
            x='impressions',
            y='ctr_gap_pct',
            size='opportunity_score',
            color='intent',
            hover_data=['query', 'position'],
            title='Quick Win Opportunities: Impressions vs CTR Gap'
        )
        fig.update_layout(
            xaxis_title='Impressions',
            yaxis_title='CTR Gap vs Benchmark (%)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_decay_analysis(decay_data: pd.DataFrame):
    """Render content decay analysis section."""
    st.markdown("## Content Decay Analysis")

    if decay_data.empty:
        st.info("No significant content decay detected in the comparison period.")
        return

    # Summary by decay type
    decay_summary = decay_data['decay_type'].value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:
        competition = decay_summary.get('competition_serp_change', 0)
        st.metric("Competition/SERP Changes", competition)

    with col2:
        demand = decay_summary.get('demand_decline', 0)
        st.metric("Demand Decline", demand)

    with col3:
        major = decay_summary.get('major_decline', 0)
        st.metric("Major Decline", major, delta_color="inverse")

    # Show decaying content
    display_cols = ['query', 'clicks_current', 'clicks_previous', 'clicks_change_pct',
                   'impressions_change_pct', 'position_change', 'decay_type']
    available_cols = [c for c in display_cols if c in decay_data.columns]

    st.dataframe(
        decay_data[available_cols].head(30),
        use_container_width=True,
        hide_index=True
    )


def render_keyword_gaps(gaps_data: pd.DataFrame):
    """Render keyword gaps section."""
    st.markdown("## Keyword Gap Opportunities")

    if gaps_data.empty:
        st.info("No keyword gaps identified. Try adding more competitors.")
        return

    st.markdown(f"Found **{len(gaps_data):,}** keyword opportunities from competitors")

    # Summary stats
    col1, col2, col3 = st.columns(3)

    with col1:
        total_volume = int(gaps_data['search_volume'].sum())
        st.metric("Total Search Volume", f"{total_volume:,}")

    with col2:
        avg_cpc = gaps_data['cpc'].mean()
        st.metric("Avg CPC", f"${avg_cpc:.2f}")

    with col3:
        high_intent = len(gaps_data[gaps_data['intent'].isin(['transactional', 'commercial'])])
        st.metric("High Intent Keywords", f"{high_intent:,}")

    # Display gaps
    display_cols = ['keyword', 'search_volume', 'cpc', 'position', 'intent', 'opportunity_score']
    available_cols = [c for c in display_cols if c in gaps_data.columns]

    st.dataframe(
        gaps_data[available_cols].head(50),
        use_container_width=True,
        hide_index=True
    )

    # Intent distribution
    if 'intent' in gaps_data.columns:
        intent_dist = gaps_data.groupby('intent')['search_volume'].sum().reset_index()

        fig = px.pie(
            intent_dist,
            values='search_volume',
            names='intent',
            title='Keyword Gaps by Intent (Search Volume)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_llm_analysis(analysis_results: Dict, config: Dict):
    """Render LLM-powered analysis section."""
    st.markdown("## AI-Powered Strategic Analysis")

    if not analysis_results.get('summary'):
        st.warning("No data available for analysis.")
        return

    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Focus",
        options=[
            ("Full Comprehensive Analysis", "full_analysis"),
            ("Quick Wins Deep Dive", "quick_wins"),
            ("Content Decay & Recovery", "content_decay"),
            ("Keyword Gap Strategy", "keyword_gaps"),
            ("Strategic Overview", "strategic_overview")
        ],
        format_func=lambda x: x[0]
    )

    # Additional context
    additional_context = st.text_area(
        "Additional Context (optional)",
        placeholder="Add any specific business context, goals, or constraints..."
    )

    if st.button("Generate AI Analysis", type="primary"):
        try:
            openrouter = OpenRouterClient(
                api_key=st.secrets.openrouter.api_key
            )

            with st.spinner(f"Generating analysis with {format_model_display(config['selected_model'])}..."):
                # Stream the response
                response_container = st.empty()
                full_response = ""

                for chunk in openrouter.stream_analysis(
                    data_summary=analysis_results['summary'],
                    analysis_type=analysis_type[1],
                    model=config['selected_model'],
                    additional_context=additional_context if additional_context else None
                ):
                    full_response += chunk
                    response_container.markdown(full_response)

                st.session_state.llm_response = full_response

        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")

    # Display previous response if available
    elif st.session_state.llm_response:
        st.markdown(st.session_state.llm_response)


def render_competitor_analysis(dataforseo_data: Dict):
    """Render competitor analysis section."""
    st.markdown("## Competitor Analysis")

    competitors = dataforseo_data.get('competitors', pd.DataFrame())

    if competitors.empty:
        st.info("No competitor data available.")
        return

    # Display competitor table
    display_cols = ['domain', 'intersections', 'organic_etv', 'organic_count',
                   'organic_pos_1', 'organic_pos_2_3', 'organic_pos_4_10']
    available_cols = [c for c in display_cols if c in competitors.columns]

    st.dataframe(
        competitors[available_cols].head(15),
        use_container_width=True,
        hide_index=True
    )

    # Visualization
    fig = px.bar(
        competitors.head(10),
        x='domain',
        y='intersections',
        color='organic_etv',
        title='Top Competitors by Keyword Overlap'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_category_analysis(dataforseo_data: Dict):
    """Render category performance analysis."""
    st.markdown("## Category Performance")

    categories = dataforseo_data.get('categories', pd.DataFrame())

    if categories.empty:
        st.info("No category data available.")
        return

    # Display top categories
    display_cols = ['category', 'keyword_count', 'etv', 'is_up', 'is_down', 'is_new', 'is_lost']
    available_cols = [c for c in display_cols if c in categories.columns]

    st.dataframe(
        categories[available_cols].head(15),
        use_container_width=True,
        hide_index=True
    )

    # Treemap visualization
    if not categories.empty and 'etv' in categories.columns:
        fig = px.treemap(
            categories.head(20),
            path=['category'],
            values='etv',
            title='Category Distribution by Traffic Value'
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<p class="main-header">Organic Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced SEO analysis with actionable insights powered by GSC, DataForSEO, and AI</p>',
        unsafe_allow_html=True
    )

    # Render sidebar and get config
    config = render_sidebar()

    if config is None:
        st.warning("Please configure the required API secrets to continue.")
        render_setup_instructions()
        return

    # Main content area
    if not st.session_state.gsc_authenticated:
        st.info("Connect to Google Search Console to begin analysis.")
        render_setup_instructions()
        return

    if not st.session_state.selected_site:
        st.warning("Please select a property from the sidebar.")
        return

    # Extract domain from site URL
    site_url = st.session_state.selected_site
    domain = site_url.replace('https://', '').replace('http://', '').replace('sc-domain:', '').rstrip('/')

    st.markdown(f"**Analyzing:** `{domain}`")

    # Run Analysis Button
    if st.button("Run Full Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.llm_response = None

        # Fetch data
        with st.status("Fetching data...", expanded=True) as status:
            gsc_data = fetch_gsc_data(config)
            dataforseo_data = fetch_dataforseo_data(domain, config)
            status.update(label="Data fetched successfully!", state="complete")

        # Run analysis
        with st.status("Running analysis...", expanded=True) as status:
            analysis_results = run_analysis(gsc_data, dataforseo_data, config)
            status.update(label="Analysis complete!", state="complete")

        # Store results
        st.session_state.analysis_data = {
            'gsc': gsc_data,
            'dataforseo': dataforseo_data,
            'results': analysis_results
        }
        st.session_state.analysis_complete = True

    # Display results
    if st.session_state.analysis_complete:
        data = st.session_state.analysis_data

        # Metrics overview
        render_metrics_overview(
            data['gsc'],
            data['dataforseo'],
            data['results']
        )

        # Tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "AI Analysis",
            "Quick Wins",
            "Content Decay",
            "Keyword Gaps",
            "Competitors",
            "Categories"
        ])

        with tab1:
            render_llm_analysis(data['results'], config)

        with tab2:
            render_quick_wins(data['results'].get('quick_wins', pd.DataFrame()))

        with tab3:
            render_decay_analysis(data['results'].get('decay', pd.DataFrame()))

        with tab4:
            render_keyword_gaps(data['results'].get('keyword_gaps', pd.DataFrame()))

        with tab5:
            render_competitor_analysis(data['dataforseo'])

        with tab6:
            render_category_analysis(data['dataforseo'])

        # Export options
        st.markdown("---")
        st.markdown("## Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export Quick Wins CSV"):
                quick_wins = data['results'].get('quick_wins', pd.DataFrame())
                if not quick_wins.empty:
                    csv = quick_wins.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "quick_wins.csv",
                        "text/csv"
                    )

        with col2:
            if st.button("Export Decay Analysis CSV"):
                decay = data['results'].get('decay', pd.DataFrame())
                if not decay.empty:
                    csv = decay.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "content_decay.csv",
                        "text/csv"
                    )

        with col3:
            if st.button("Export Keyword Gaps CSV"):
                gaps = data['results'].get('keyword_gaps', pd.DataFrame())
                if not gaps.empty:
                    csv = gaps.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "keyword_gaps.csv",
                        "text/csv"
                    )


def render_setup_instructions():
    """Render setup instructions for new users."""
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### Configuration Steps

        1. **Configure Streamlit Secrets**

           In Streamlit Cloud, go to App Settings > Secrets and add:

           ```toml
           [google]
           client_id = "your-google-client-id"
           client_secret = "your-google-client-secret"
           redirect_uri = "https://your-app.streamlit.app"

           [dataforseo]
           login = "your-dataforseo-email"
           password = "your-dataforseo-api-password"

           [openrouter]
           api_key = "your-openrouter-api-key"
           ```

        2. **Google Search Console API Setup**
           - Go to [Google Cloud Console](https://console.cloud.google.com)
           - Create a new project or select existing
           - Enable the Search Console API
           - Create OAuth 2.0 credentials (Web application)
           - Add your app URL to authorized redirect URIs

        3. **DataForSEO API Setup**
           - Sign up at [DataForSEO](https://dataforseo.com)
           - Get your API credentials from the dashboard

        4. **OpenRouter API Setup**
           - Sign up at [OpenRouter](https://openrouter.ai)
           - Generate an API key from your account

        ### Local Development

        For local development, create `.streamlit/secrets.toml`:

        ```toml
        [google]
        client_id = "your-client-id"
        client_secret = "your-client-secret"
        redirect_uri = "http://localhost:8501"

        [dataforseo]
        login = "your-email"
        password = "your-password"

        [openrouter]
        api_key = "your-api-key"
        ```
        """)


if __name__ == "__main__":
    main()
