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
    """Render sidebar with API status and GSC connection."""
    st.sidebar.title("🔌 Connection Status")
    
    # API Configuration Status (moved to sidebar)
    st.sidebar.subheader("API Configuration")
    
    if api_config.has_gsc_credentials():
        st.sidebar.success("✅ GSC configured")
    else:
        st.sidebar.warning("⚠️ GSC missing")
    
    if api_config.has_dataforseo_credentials():
        st.sidebar.success("✅ DataForSEO configured")
    else:
        st.sidebar.warning("⚠️ DataForSEO missing")
    
    if api_config.has_openrouter_credentials():
        st.sidebar.success("✅ OpenRouter configured")
    else:
        st.sidebar.warning("⚠️ OpenRouter missing")
    
    st.sidebar.divider()
    
    # GSC Authentication
    st.sidebar.subheader("Google Search Console")
    
    auth_service = AuthService()
    
    if not st.session_state.authenticated:
        # Check for OAuth callback in query params FIRST
        try:
            qp = dict(st.query_params) if hasattr(st, 'query_params') else {}
        except Exception:
            qp = {}
        
        if 'code' in qp:
            with st.sidebar:
                with st.spinner("Authenticating..."):
                    success, error = auth_service.handle_callback(qp)
                    if success:
                        st.session_state.authenticated = True
                        # Clear query params
                        try:
                            st.query_params.clear()
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error(f"Auth failed: {error}")
        
        # Generate auth URL and show link button for direct navigation
        try:
            auth_url = auth_service.get_auth_url()
            # Use link_button for direct single-click authentication
            st.sidebar.link_button(
                "🔐 Sign in with Google",
                auth_url,
                type="primary",
                use_container_width=True
            )
            
            # Debug expander (collapsed by default)
            with st.sidebar.expander("🔧 Debug Info", expanded=False):
                st.code(api_config.google_redirect_uri, language=None)
        except ValueError as e:
            st.sidebar.error(f"Config error: {str(e)}")
    else:
        st.sidebar.success("✅ Connected to GSC")
        
        # Property selection
        collector = GSCCollector(auth_service)
        properties = collector.list_properties()
        
        if properties:
            # Separate domain and URL properties for better display
            domain_props = [
                p for p in properties if p.startswith('sc-domain:')
            ]
            url_props = [
                p for p in properties if not p.startswith('sc-domain:')
            ]
            
            # Show property counts for debugging
            st.sidebar.caption(
                f"Found {len(properties)} properties "
                f"({len(domain_props)} domain, {len(url_props)} URL)"
            )
            
            # Combine with domain properties first (usually preferred)
            sorted_properties = domain_props + url_props
            
            # Calculate the correct default index
            # Keep the user's previous selection if it exists and is valid
            default_index = 0
            if st.session_state.selected_property:
                try:
                    prev_idx = sorted_properties.index(
                        st.session_state.selected_property
                    )
                    default_index = prev_idx
                except ValueError:
                    # Previous selection no longer in list
                    default_index = 0
            
            # Use a unique key and on_change callback for proper state
            selected = st.sidebar.selectbox(
                "Select Property:",
                options=sorted_properties,
                index=default_index,
                key="property_selector",
                help="Domain properties (sc-domain:) provide complete data"
            )
            
            # Update session state with current selection
            st.session_state.selected_property = selected
            
            # Show property type indicator
            if selected.startswith('sc-domain:'):
                st.sidebar.info("📊 Domain property selected")
            else:
                st.sidebar.info("🔗 URL property selected")
        else:
            st.sidebar.warning("No properties found")
            st.sidebar.caption(
                "Make sure your Google account has access to GSC properties"
            )
        
        if st.sidebar.button("🔓 Disconnect", use_container_width=True):
            auth_service.revoke_credentials()
            st.session_state.authenticated = False
            st.session_state.selected_property = None
            st.rerun()
    
    return auth_service


def render_config_panel():
    """Render configuration panel in main area."""
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Analysis Parameters
        st.markdown("**Analysis Parameters**")
        
        analysis_period = st.selectbox(
            "Analysis Period:",
            options=list(settings.date_windows.keys()),
            format_func=lambda x: settings.date_windows[x]['label'],
            index=1
        )
        
        min_impressions = st.slider(
            "Min Impressions Filter:",
            min_value=0,
            max_value=500,
            value=settings.min_impressions_default,
            step=10
        )
        
        include_yoy = st.checkbox(
            "Include Year-over-Year Comparison",
            value=True
        )
        
        # AI Model Selection
        st.markdown("**AI Model**")
        selected_model = st.selectbox(
            "Select LLM Model:",
            options=settings.available_llm_models,
            index=0,
            help="Choose the AI model for analysis generation"
        )
    
    with col2:
        # Brand Terms
        st.markdown("**Brand Configuration**")
        
        brand_terms_input = st.text_area(
            "Brand Terms (one per line):",
            help="Enter brand terms to exclude from non-brand analysis",
            height=100
        )
        brand_terms = [
            t.strip() for t in brand_terms_input.split('\n') if t.strip()
        ]
        
        # Competitors
        st.markdown("**Competitors**")
        
        auto_discover = st.checkbox(
            "Auto-discover competitors",
            value=True
        )
        
        manual_competitors = []
        if not auto_discover:
            competitors_input = st.text_area(
                "Manual competitors (one per line):",
                height=100
            )
            manual_competitors = [
                c.strip() for c in competitors_input.split('\n') if c.strip()
            ]
    
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
        # Remove protocol, www, and trailing slash
        domain = property_url.replace('sc-domain:', '').replace(
            'https://', ''
        ).replace('http://', '').replace('www.', '').rstrip('/')
        
        # Step 1: Collect GSC data
        progress.progress(10, text="Collecting GSC data...")
        
        days = settings.date_windows[config['period']]['days']
        
        # Get UNFILTERED data for accurate totals
        gsc_data_unfiltered = gsc_collector.get_comprehensive_data(
            property_url,
            days=days,
            min_impressions=0  # No filter for totals
        )
        
        # Get FILTERED data for analysis
        gsc_data = gsc_collector.get_comprehensive_data(
            property_url,
            days=days,
            min_impressions=config['min_impressions']
        )
        
        # Store unfiltered totals separately
        unfiltered_queries = gsc_data_unfiltered.get('queries', pd.DataFrame())
        if not unfiltered_queries.empty:
            gsc_data['total_metrics'] = {
                'clicks': int(unfiltered_queries['clicks'].sum()),
                'impressions': int(unfiltered_queries['impressions'].sum()),
                'ctr': float(unfiltered_queries['ctr'].mean()),
                'position': float(unfiltered_queries['position'].mean())
            }
        else:
            gsc_data['total_metrics'] = {}
        
        # Get YoY data if requested (with lower filter threshold)
        yoy_data = {}
        if config['include_yoy']:
            progress.progress(25, text="Collecting YoY comparison data...")
            yoy_start, yoy_end = gsc_collector.get_yoy_date_range(days)
            # Use lower filter for YoY to capture more decay
            yoy_min_imp = max(10, config['min_impressions'] // 2)
            yoy_data = {
                'queries': gsc_collector.get_query_data(
                    property_url, yoy_start, yoy_end, yoy_min_imp
                ),
                'pages': gsc_collector.get_page_data(
                    property_url, yoy_start, yoy_end, yoy_min_imp
                )
            }
        
        st.session_state.gsc_data = gsc_data
        
        # Step 2: Collect DataForSEO data
        progress.progress(35, text=f"Fetching DataForSEO data for {domain}...")
        
        try:
            dataforseo_data = dfs_client.get_comprehensive_domain_data(
                domain=domain,
                max_keywords=2000
            )
            
            # Log DataForSEO results for debugging
            ranked_kw = dataforseo_data.get(
                'ranked_keywords', pd.DataFrame()
            )
            overview = dataforseo_data.get('overview', {})
            if isinstance(ranked_kw, pd.DataFrame) and not ranked_kw.empty:
                kw_count = len(ranked_kw)
                logger.info(f"DataForSEO: Found {kw_count} keywords")
            else:
                logger.warning(f"DataForSEO: No keywords for {domain}")
            
            if overview:
                logger.info(f"DataForSEO overview: {overview}")
            else:
                logger.warning(f"DataForSEO: No overview for {domain}")
                
        except Exception as e:
            logger.error(f"DataForSEO error: {str(e)}")
            dataforseo_data = {
                'domain': domain,
                'overview': {},
                'ranked_keywords': pd.DataFrame()
            }
        
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
            
            # Store DataForSEO status for UI display
            rk = dataforseo_data.get('ranked_keywords', pd.DataFrame())
            rk_count = len(rk) if isinstance(rk, pd.DataFrame) else 0
            comp_count = len(competitors) if not competitors.empty else 0
            
            dataforseo_data['status'] = {
                'ranked_keywords_count': rk_count,
                'competitors_count': comp_count,
                'overview_available': bool(dataforseo_data.get('overview')),
                'domain_queried': domain
            }
            
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
        
        # Store min_impressions for UI display
        st.session_state.min_impressions = config['min_impressions']
        
        # Step 7: AI Analysis
        progress.progress(90, text="Generating AI insights...")
        
        # Use ACTUAL GSC totals for overview
        total_metrics = gsc_data.get('total_metrics', {})
        if not total_metrics:
            total_metrics = {
                'clicks': int(normalized_queries['clicks'].sum()),
                'impressions': int(normalized_queries['impressions'].sum()),
                'ctr': float(normalized_queries['ctr'].mean()),
                'position': float(normalized_queries['position'].mean())
            }
        
        analysis_data = {
            'domain': domain,
            'period': settings.date_windows[config['period']]['label'],
            'overview_metrics': total_metrics,
            'quick_wins': opportunities.get('quick_wins', pd.DataFrame()),
            'decaying_keywords': decaying_keywords,
            'decaying_pages': decaying_pages,
            'decay_summary': decay_summary,
            'competitors': competitors,
            'keyword_gaps': dataforseo_data.get(
                'keyword_gaps', pd.DataFrame()
            ),
            'brand_metrics': brand_metrics,
            'non_brand_opportunities': non_brand_opps,
            'page_data': gsc_data.get('pages', pd.DataFrame()),
            'query_portfolio': gsc_data.get('query_page', pd.DataFrame()),
            'page_scores': scorer.score_pages(
                gsc_data.get('pages', pd.DataFrame()),
                gsc_data.get('query_page', pd.DataFrame())
            ),
            'serp_features': 'SERP feature analysis pending'
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
    dataforseo_data = st.session_state.dataforseo_data
    brand_metrics = st.session_state.brand_metrics
    opportunities = st.session_state.opportunities
    
    # Key metrics - use TOTAL metrics (unfiltered)
    if 'total_metrics' in gsc_data and gsc_data['total_metrics']:
        MetricCards.overview_metrics(gsc_data['total_metrics'])
        st.caption(
            f"*Based on ALL queries. Analysis below uses queries with "
            f"≥{st.session_state.get('min_impressions', 50)} impressions.*"
        )
    elif 'queries' in gsc_data:
        queries_df = gsc_data['queries']
        overview = {
            'clicks': queries_df['clicks'].sum(),
            'impressions': queries_df['impressions'].sum(),
            'ctr': queries_df['ctr'].mean(),
            'position': queries_df['position'].mean()
        }
        MetricCards.overview_metrics(overview)
    
    st.divider()
    
    # DataForSEO Status & Metrics
    st.subheader("📊 DataForSEO Intelligence")
    dfs_status = dataforseo_data.get('status', {})
    dfs_overview = dataforseo_data.get('overview', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Ranked Keywords",
            f"{dfs_status.get('ranked_keywords_count', 0):,}",
            help="Keywords found ranking in DataForSEO database"
        )
    
    with col2:
        st.metric(
            "Est. Organic Traffic",
            f"{int(dfs_overview.get('organic_etv', 0)):,}",
            help="Estimated monthly organic traffic value"
        )
    
    with col3:
        st.metric(
            "Competitors Found",
            f"{dfs_status.get('competitors_count', 0)}",
            help="Competing domains discovered"
        )
    
    with col4:
        st.metric(
            "Domain Queried",
            dfs_status.get('domain_queried', 'N/A'),
            help="Domain sent to DataForSEO API"
        )
    
    # Show ranked keywords sample if available
    ranked_kw = dataforseo_data.get('ranked_keywords', pd.DataFrame())
    if isinstance(ranked_kw, pd.DataFrame) and not ranked_kw.empty:
        with st.expander("📋 Top Ranked Keywords (DataForSEO)"):
            display_cols = [
                'keyword', 'position', 'search_volume',
                'cpc', 'traffic', 'url'
            ]
            available_cols = [
                c for c in display_cols if c in ranked_kw.columns
            ]
            if available_cols:
                st.dataframe(
                    ranked_kw[available_cols].head(20),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info(
            "ℹ️ No DataForSEO ranked keywords available. "
            "API may have failed or domain has no indexed keywords."
        )
    
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
    
    # Sidebar - API status and GSC connection
    render_sidebar()
    
    # Main content area
    if st.session_state.authenticated and st.session_state.selected_property:
        # Configuration panel in main area
        config = render_config_panel()
        
        st.divider()
        
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
        st.info("👈 Connect to Google Search Console in the sidebar")
    
    elif not st.session_state.selected_property:
        st.info("👈 Select a GSC property in the sidebar")


if __name__ == "__main__":
    main()