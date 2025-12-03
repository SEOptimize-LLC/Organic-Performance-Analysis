"""
Organic Performance Analyzer - Main Streamlit Application
Advanced SEO analysis tool using GSC and DataForSEO data.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Organic Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from config.settings import settings
from config.api_config import api_config
from services.auth_service import AuthService
from collectors.gsc_collector import GSCCollector
from collectors.dataforseo_client import DataForSEOClient
from processors.data_normalizer import DataNormalizer
from processors.opportunity_scorer import OpportunityScorer
from processors.decay_detector import DecayDetector
from processors.brand_classifier import BrandClassifier
from agents.analysis_agent import AnalysisAgent
from visualizations.charts import ChartBuilder
from visualizations.metrics import MetricCards
from visualizations.tables import DataTables
from exporters.report_generator import ReportGenerator
from utils.logger import logger


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'authenticated': False,
        'gsc_service': None,
        'selected_property': None,
        'analysis_complete': False,
        'gsc_data': {},
        'dataforseo_data': {},
        'opportunities': {},
        'decay_data': {},
        'brand_metrics': {},
        'ai_analysis': {},
        'report_data': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.title("⚙️ Analysis Configuration")
    
    # GSC Authentication
    st.sidebar.header("1. Google Search Console")
    
    auth_service = AuthService()
    
    if not st.session_state.authenticated:
        if st.sidebar.button("🔐 Connect to GSC", use_container_width=True):
            auth_url = auth_service.get_authorization_url()
            st.sidebar.markdown(f"[Click here to authorize]({auth_url})")
        
        # Handle OAuth callback
        auth_code = st.sidebar.text_input(
            "Paste authorization code:",
            help="After authorizing, paste the code here"
        )
        if auth_code:
            if auth_service.exchange_code(auth_code):
                st.session_state.authenticated = True
                st.session_state.gsc_service = auth_service.get_gsc_service()
                st.rerun()
    else:
        st.sidebar.success("✅ Connected to GSC")
        
        # Property selection
        collector = GSCCollector(auth_service)
        properties = collector.list_properties()
        
        if properties:
            selected = st.sidebar.selectbox(
                "Select Property:",
                options=properties,
                index=0
            )
            st.session_state.selected_property = selected
        else:
            st.sidebar.warning("No properties found")
    
    st.sidebar.divider()
    
    # Analysis Parameters
    st.sidebar.header("2. Analysis Parameters")
    
    analysis_period = st.sidebar.selectbox(
        "Analysis Period:",
        options=list(settings.date_windows.keys()),
        format_func=lambda x: settings.date_windows[x]['label'],
        index=1  # Default to 90 days
    )
    
    min_impressions = st.sidebar.slider(
        "Min Impressions Filter:",
        min_value=0,
        max_value=500,
        value=settings.min_impressions_default,
        step=10
    )
    
    include_yoy = st.sidebar.checkbox(
        "Include Year-over-Year Comparison",
        value=True
    )
    
    st.sidebar.divider()
    
    # Brand Terms
    st.sidebar.header("3. Brand Configuration")
    
    brand_terms_input = st.sidebar.text_area(
        "Brand Terms (one per line):",
        help="Enter brand terms to exclude from non-brand analysis"
    )
    brand_terms = [t.strip() for t in brand_terms_input.split('\n') if t.strip()]
    
    st.sidebar.divider()
    
    # Competitors (optional)
    st.sidebar.header("4. Competitors (Optional)")
    
    auto_discover = st.sidebar.checkbox(
        "Auto-discover competitors",
        value=True
    )
    
    manual_competitors = []
    if not auto_discover:
        competitors_input = st.sidebar.text_area(
            "Manual competitors (one per line):"
        )
        manual_competitors = [
            c.strip() for c in competitors_input.split('\n') if c.strip()
        ]
    
    st.sidebar.divider()
    
    # AI Model Selection
    st.sidebar.header("5. AI Analysis Model")
    
    selected_model = st.sidebar.selectbox(
        "Select LLM Model:",
        options=settings.available_models,
        index=0,
        help="Choose the AI model for analysis generation"
    )
    
    return {
        'period': analysis_period,
        'min_impressions': min_impressions,
        'include_yoy': include_yoy,
        'brand_terms': brand_terms,
        'auto_discover_competitors': auto_discover,
        'manual_competitors': manual_competitors,
        'ai_model': selected_model
    }


def run_analysis(config: dict):
    """Run the complete analysis pipeline."""
    if not st.session_state.selected_property:
        st.error("Please select a GSC property first")
        return
    
    property_url = st.session_state.selected_property
    
    progress = st.progress(0, text="Initializing analysis...")
    
    try:
        # Initialize components
        auth_service = AuthService()
        gsc_collector = GSCCollector(auth_service)
        dfs_client = DataForSEOClient()
        normalizer = DataNormalizer(config['brand_terms'])
        scorer = OpportunityScorer()
        decay_detector = DecayDetector()
        brand_classifier = BrandClassifier(config['brand_terms'])
        ai_agent = AnalysisAgent(config['ai_model'])
        
        # Extract domain from property URL
        domain = property_url.replace('sc-domain:', '').replace(
            'https://', ''
        ).replace('http://', '').rstrip('/')
        
        # Step 1: Collect GSC data
        progress.progress(10, text="Collecting GSC data...")
        
        days = settings.date_windows[config['period']]['days']
        gsc_data = gsc_collector.get_comprehensive_data(
            property_url,
            days=days,
            min_impressions=config['min_impressions']
        )
        
        # Get YoY data if requested
        yoy_data = {}
        if config['include_yoy']:
            progress.progress(25, text="Collecting YoY comparison data...")
            yoy_start, yoy_end = gsc_collector.get_yoy_date_range(days)
            yoy_data = {
                'queries': gsc_collector.get_query_data(
                    property_url, yoy_start, yoy_end,
                    config['min_impressions']
                ),
                'pages': gsc_collector.get_page_data(
                    property_url, yoy_start, yoy_end,
                    config['min_impressions']
                )
            }
        
        st.session_state.gsc_data = gsc_data
        
        # Step 2: Collect DataForSEO data
        progress.progress(35, text="Fetching DataForSEO data...")
        
        dataforseo_data = dfs_client.get_comprehensive_domain_data(
            domain=domain,
            max_keywords=2000
        )
        
        # Get competitors
        if config['auto_discover_competitors']:
            competitors = dfs_client.get_competitors(domain)
        else:
            competitors = pd.DataFrame()
        
        dataforseo_data['competitors'] = competitors
        
        # Get keyword gaps for top competitor
        if not competitors.empty:
            top_competitor = competitors.iloc[0]['competitor_domain']
            keyword_gaps = dfs_client.get_competitor_keywords(
                domain, top_competitor
            )
            dataforseo_data['keyword_gaps'] = keyword_gaps
        
        st.session_state.dataforseo_data = dataforseo_data
        
        # Step 3: Normalize and join data
        progress.progress(50, text="Processing and normalizing data...")
        
        normalized_queries = normalizer.normalize_gsc_data(
            gsc_data.get('queries', pd.DataFrame())
        )
        
        # Join with DataForSEO
        if 'ranked_keywords' in dataforseo_data:
            rk = dataforseo_data['ranked_keywords']
            if isinstance(rk, pd.DataFrame) and not rk.empty:
                joined_keywords = normalizer.join_gsc_dataforseo(
                    normalized_queries,
                    rk
                )
            else:
                joined_keywords = normalized_queries
        else:
            joined_keywords = normalized_queries
        
        # Step 4: Calculate opportunity scores
        progress.progress(60, text="Calculating opportunity scores...")
        
        yoy_queries = yoy_data.get('queries', pd.DataFrame())
        scored_keywords = scorer.score_keywords(joined_keywords, yoy_queries)
        opportunities = scorer.classify_opportunities(scored_keywords)
        
        st.session_state.opportunities = opportunities
        
        # Step 5: Detect decay
        progress.progress(70, text="Analyzing content decay...")
        
        if not yoy_queries.empty:
            decaying_keywords = decay_detector.detect_decaying_keywords(
                normalized_queries, yoy_queries
            )
            decaying_pages = decay_detector.detect_decaying_pages(
                gsc_data.get('pages', pd.DataFrame()),
                yoy_data.get('pages', pd.DataFrame())
            )
            decay_summary = decay_detector.summarize_decay(
                decaying_keywords, decaying_pages
            )
        else:
            decaying_keywords = pd.DataFrame()
            decaying_pages = pd.DataFrame()
            decay_summary = {}
        
        st.session_state.decay_data = {
            'decaying_keywords': decaying_keywords,
            'decaying_pages': decaying_pages,
            'summary': decay_summary
        }
        
        # Step 6: Brand analysis
        progress.progress(80, text="Analyzing brand performance...")
        
        brand_metrics = brand_classifier.calculate_brand_metrics(
            normalized_queries
        )
        non_brand_opps = brand_classifier.get_non_brand_opportunities(
            normalized_queries
        )
        
        st.session_state.brand_metrics = brand_metrics
        
        # Step 7: AI Analysis
        progress.progress(90, text="Generating AI insights...")
        
        analysis_data = {
            'domain': domain,
            'period': config['period'],
            'overview_metrics': {
                'clicks': int(normalized_queries['clicks'].sum()),
                'impressions': int(normalized_queries['impressions'].sum()),
                'ctr': float(normalized_queries['ctr'].mean()),
                'position': float(normalized_queries['position'].mean())
            },
            'quick_wins': opportunities.get('quick_wins', pd.DataFrame()),
            'decaying_keywords': decaying_keywords,
            'decaying_pages': decaying_pages,
            'decay_summary': decay_summary,
            'competitors': competitors,
            'keyword_gaps': dataforseo_data.get('keyword_gaps', pd.DataFrame()),
            'brand_metrics': brand_metrics,
            'non_brand_opportunities': non_brand_opps,
            'page_data': gsc_data.get('pages', pd.DataFrame()),
            'query_portfolio': gsc_data.get('query_page', pd.DataFrame()),
            'page_scores': scorer.score_pages(
                gsc_data.get('pages', pd.DataFrame()),
                gsc_data.get('query_page', pd.DataFrame())
            ),
            'serp_features': 'SERP feature analysis not available'
        }
        
        ai_analysis = ai_agent.run_full_analysis(analysis_data)
        st.session_state.ai_analysis = ai_analysis
        
        # Complete
        progress.progress(100, text="Analysis complete!")
        st.session_state.analysis_complete = True
        
        # Prepare report data
        report_gen = ReportGenerator()
        st.session_state.report_data = report_gen.prepare_report_data(
            domain=domain,
            gsc_data=gsc_data,
            dataforseo_data=dataforseo_data,
            opportunities=opportunities,
            decay_data=st.session_state.decay_data,
            brand_metrics=brand_metrics,
            ai_analysis=ai_analysis
        )
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")
        raise


def render_results():
    """Render analysis results."""
    if not st.session_state.analysis_complete:
        return
    
    # Overview Tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎯 Quick Wins",
        "📉 Decay Analysis",
        "🏆 Competitors",
        "🤖 AI Insights",
        "📥 Export"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_quick_wins_tab()
    
    with tab3:
        render_decay_tab()
    
    with tab4:
        render_competitors_tab()
    
    with tab5:
        render_ai_insights_tab()
    
    with tab6:
        render_export_tab()


def render_overview_tab():
    """Render overview metrics and charts."""
    st.header("Overview")
    
    gsc_data = st.session_state.gsc_data
    brand_metrics = st.session_state.brand_metrics
    opportunities = st.session_state.opportunities
    
    # Key metrics
    if 'queries' in gsc_data:
        queries_df = gsc_data['queries']
        overview = {
            'clicks': queries_df['clicks'].sum(),
            'impressions': queries_df['impressions'].sum(),
            'ctr': queries_df['ctr'].mean(),
            'position': queries_df['position'].mean()
        }
        MetricCards.overview_metrics(overview)
    
    st.divider()
    
    # Brand metrics
    if brand_metrics:
        MetricCards.brand_metrics(brand_metrics)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = ChartBuilder.brand_vs_non_brand(brand_metrics, 'pie')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = ChartBuilder.brand_vs_non_brand(brand_metrics, 'bar')
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Opportunity summary
    MetricCards.opportunity_summary(opportunities)
    
    # Position distribution
    if 'queries' in gsc_data:
        fig = ChartBuilder.position_distribution(gsc_data['queries'])
        st.plotly_chart(fig, use_container_width=True)


def render_quick_wins_tab():
    """Render quick wins analysis."""
    st.header("Quick Win Opportunities")
    
    opportunities = st.session_state.opportunities
    quick_wins = opportunities.get('quick_wins', pd.DataFrame())
    
    if quick_wins.empty:
        st.info("No quick wins identified in this analysis period.")
        return
    
    st.markdown("""
    These keywords have high impressions but underperforming CTR.
    Optimizing titles and meta descriptions can yield immediate traffic gains.
    """)
    
    # Scatter plot
    fig = ChartBuilder.opportunity_scatter(quick_wins)
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("Top Quick Wins")
    DataTables.quick_wins_table(quick_wins)
    
    # Heatmap
    fig = ChartBuilder.opportunity_heatmap(quick_wins)
    st.plotly_chart(fig, use_container_width=True)


def render_decay_tab():
    """Render decay analysis."""
    st.header("Content Decay Analysis")
    
    decay_data = st.session_state.decay_data
    
    if 'summary' in decay_data:
        MetricCards.decay_summary(decay_data['summary'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decaying Keywords")
        decaying_kw = decay_data.get('decaying_keywords', pd.DataFrame())
        DataTables.decaying_keywords_table(decaying_kw)
        
        if not decaying_kw.empty:
            fig = ChartBuilder.decay_waterfall(decaying_kw)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Decaying Pages")
        decaying_pages = decay_data.get('decaying_pages', pd.DataFrame())
        if not decaying_pages.empty:
            DataTables.custom_table(
                decaying_pages,
                columns=['page', 'clicks_change_pct', 'primary_decay'],
                rename={
                    'page': 'Page',
                    'clicks_change_pct': 'Change %',
                    'primary_decay': 'Decay Type'
                }
            )


def render_competitors_tab():
    """Render competitor analysis."""
    st.header("Competitive Analysis")
    
    dataforseo_data = st.session_state.dataforseo_data
    competitors = dataforseo_data.get('competitors', pd.DataFrame())
    keyword_gaps = dataforseo_data.get('keyword_gaps', pd.DataFrame())
    
    if competitors.empty:
        st.info("No competitor data available.")
        return
    
    # Competitor table
    st.subheader("Top Competitors")
    DataTables.competitor_table(competitors)
    
    # Competitor chart
    fig = ChartBuilder.competitor_comparison(competitors)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Keyword gaps
    st.subheader("Keyword Gaps")
    DataTables.keyword_gaps_table(keyword_gaps)


def render_ai_insights_tab():
    """Render AI-generated insights."""
    st.header("AI-Generated Insights")
    
    ai_analysis = st.session_state.ai_analysis
    
    if not ai_analysis:
        st.info("AI analysis not available.")
        return
    
    # Comprehensive report
    if 'comprehensive' in ai_analysis:
        st.subheader("📋 Comprehensive Analysis")
        st.markdown(ai_analysis['comprehensive'])
    
    st.divider()
    
    # Section-specific insights
    sections = [
        ('quick_wins', '🎯 Quick Wins Recommendations'),
        ('decay', '📉 Recovery Recommendations'),
        ('competitors', '🏆 Competitive Recommendations'),
        ('brand', '🏷️ Brand Strategy'),
        ('pages', '📄 Page Optimization')
    ]
    
    for key, title in sections:
        if key in ai_analysis and ai_analysis[key]:
            with st.expander(title, expanded=False):
                st.markdown(ai_analysis[key])


def render_export_tab():
    """Render export options."""
    st.header("Export Reports")
    
    report_data = st.session_state.report_data
    
    if not report_data:
        st.info("Complete an analysis first to export reports.")
        return
    
    report_gen = ReportGenerator()
    domain = report_data.get('domain', 'unknown')
    
    st.subheader("📥 Download Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Excel Report")
        st.markdown("Comprehensive data export with multiple worksheets.")
        
        if st.button("Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                excel_bytes = report_gen.generate_excel_report(report_data)
                filename = report_gen.get_report_filename(domain, 'xlsx')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # noqa
                    use_container_width=True
                )
    
    with col2:
        st.markdown("### PDF Report")
        st.markdown("Formatted report with AI insights.")
        
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_bytes = report_gen.generate_pdf_report(report_data)
                filename = report_gen.get_report_filename(domain, 'pdf')
                
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
    
    st.divider()
    
    # Individual data exports
    st.subheader("📊 Export Individual Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'quick_wins' in report_data:
            qw = report_data['quick_wins']
            if isinstance(qw, pd.DataFrame) and not qw.empty:
                DataTables.download_button(
                    qw,
                    f"quick_wins_{domain}.csv",
                    "Download Quick Wins CSV"
                )
    
    with col2:
        if 'decaying_keywords' in report_data:
            dk = report_data['decaying_keywords']
            if isinstance(dk, pd.DataFrame) and not dk.empty:
                DataTables.download_button(
                    dk,
                    f"decaying_keywords_{domain}.csv",
                    "Download Decay CSV"
                )
    
    with col3:
        if 'keyword_gaps' in report_data:
            kg = report_data['keyword_gaps']
            if isinstance(kg, pd.DataFrame) and not kg.empty:
                DataTables.download_button(
                    kg,
                    f"keyword_gaps_{domain}.csv",
                    "Download Gaps CSV"
                )


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.title("📊 Organic Performance Analyzer")
    st.markdown("""
    Advanced SEO analysis tool combining Google Search Console data with 
    DataForSEO insights and AI-powered recommendations.
    """)
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content area
    if st.session_state.authenticated and st.session_state.selected_property:
        # Run Analysis button
        if st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        ):
            run_analysis(config)
        
        # Display results if analysis is complete
        if st.session_state.analysis_complete:
            st.divider()
            render_results()
    
    elif not st.session_state.authenticated:
        st.info("👈 Connect to Google Search Console to get started")
        
        # Show API configuration status
        st.subheader("API Configuration Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if api_config.get_gsc_credentials():
                st.success("✅ GSC credentials configured")
            else:
                st.warning("⚠️ GSC credentials missing")
        
        with col2:
            creds = api_config.get_dataforseo_credentials()
            if creds['login'] and creds['password']:
                st.success("✅ DataForSEO configured")
            else:
                st.warning("⚠️ DataForSEO credentials missing")
        
        with col3:
            if api_config.get_openrouter_key():
                st.success("✅ OpenRouter configured")
            else:
                st.warning("⚠️ OpenRouter API key missing")
        
        st.markdown("""
        ### Setup Instructions
        
        Add the following to your Streamlit secrets (`.streamlit/secrets.toml`):
        
        ```toml
        [gsc]
        client_id = "your-client-id"
        client_secret = "your-client-secret"
        redirect_uri = "your-redirect-uri"
        
        [dataforseo]
        login = "your-login"
        password = "your-password"
        
        [openrouter]
        api_key = "your-api-key"
        ```
        """)


if __name__ == "__main__":
    main()