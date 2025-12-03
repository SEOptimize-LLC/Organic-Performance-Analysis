# Organic Performance Analyzer

Advanced SEO analysis tool that conducts in-depth organic performance analysis combining Google Search Console (GSC) data with DataForSEO insights, powered by AI-generated actionable recommendations.

## 🎯 Overview

This tool is designed to surface concrete growth opportunities and prioritize actions for maximum ROI, not to generate vanity reports or simple status summaries. It provides:

- **Quick-win opportunities** with high-impact, low-effort optimizations
- **Content decay detection** with recovery recommendations
- **Competitor analysis** with keyword gap identification
- **Brand vs non-brand segmentation** with growth headroom analysis
- **AI-powered insights** using cutting-edge LLMs via OpenRouter

## ✨ Features

### Data Collection
- Multi-window GSC data collection (28d, 90d, 180d, 365d)
- Year-over-year comparison analysis
- Device and country segmentation
- Query and page-level metrics

### DataForSEO Integration
- Ranked keywords retrieval
- Auto-discovery of competitors
- Keyword suggestions and gaps
- SERP feature analysis

### Analysis Engine
- Opportunity scoring model with 5 components:
  - Search volume score
  - Position potential score
  - CTR gap score
  - Commercial value score
  - Trend direction score
- Content decay classification
- Brand dependency analysis

### AI-Powered Insights
Multiple LLM models available via OpenRouter:
- `openai/gpt-4.1-mini`
- `openai/gpt-5-mini`
- `anthropic/claude-haiku-4.5`
- `anthropic/claude-sonnet-4.5`
- `google/gemini-3-pro-preview`
- `google/gemini-2.5-flash-preview-09-2025`
- `x-ai/grok-4-fast`
- `deepseek/deepseek-r1-0528-qwen3-8b`

### Export Options
- **Excel reports** with multiple worksheets
- **PDF reports** with formatted sections
- **CSV exports** for individual data tables

## 🚀 Deployment

### Streamlit Cloud

1. Fork or clone this repository
2. Connect to Streamlit Cloud
3. Add secrets in the Streamlit Cloud dashboard

### Local Development

```bash
# Clone the repository
cd "Organic Performance Analyzer"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ⚙️ Configuration

### Streamlit Secrets

Create a `.streamlit/secrets.toml` file with your API credentials:

```toml
[gsc]
client_id = "your-google-client-id"
client_secret = "your-google-client-secret"
redirect_uri = "http://localhost:8501"

[dataforseo]
login = "your-dataforseo-login"
password = "your-dataforseo-password"

[openrouter]
api_key = "your-openrouter-api-key"
```

### Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Search Console API
4. Create OAuth 2.0 credentials (Web application)
5. Add authorized redirect URIs
6. Download credentials and add to secrets

### DataForSEO Setup

1. Sign up at [DataForSEO](https://dataforseo.com/)
2. Get your API login and password from the dashboard
3. Add to Streamlit secrets

### OpenRouter Setup

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Create an API key
3. Add to Streamlit secrets

## 📊 Usage

1. **Connect to GSC**: Click the authorization button and complete OAuth flow
2. **Select Property**: Choose from available GSC properties
3. **Configure Analysis**:
   - Select analysis period (28d to 365d)
   - Set minimum impressions filter
   - Enable/disable YoY comparison
   - Add brand terms for segmentation
   - Choose AI model for insights
4. **Run Analysis**: Click the "Run Analysis" button
5. **Review Results**: Navigate through the analysis tabs
6. **Export Reports**: Download Excel or PDF reports

## 📁 Project Structure

```
Organic Performance Analyzer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── config/
│   ├── __init__.py
│   ├── settings.py             # Application settings
│   └── api_config.py           # API configuration
├── services/
│   ├── __init__.py
│   ├── auth_service.py         # GSC OAuth handling
│   ├── cache_service.py        # Data caching
│   └── rate_limiter.py         # API rate limiting
├── collectors/
│   ├── __init__.py
│   ├── gsc_collector.py        # GSC data collection
│   └── dataforseo_client.py    # DataForSEO API client
├── processors/
│   ├── __init__.py
│   ├── data_normalizer.py      # Data normalization
│   ├── opportunity_scorer.py   # Opportunity scoring
│   ├── decay_detector.py       # Content decay detection
│   └── brand_classifier.py     # Brand classification
├── agents/
│   ├── __init__.py
│   ├── analysis_agent.py       # AI analysis agent
│   └── prompts.py              # Analysis prompts
├── visualizations/
│   ├── __init__.py
│   ├── charts.py               # Plotly charts
│   ├── metrics.py              # Metric cards
│   └── tables.py               # Data tables
├── exporters/
│   ├── __init__.py
│   ├── excel_exporter.py       # Excel export
│   ├── pdf_exporter.py         # PDF export
│   └── report_generator.py     # Report orchestration
└── utils/
    ├── __init__.py
    ├── logger.py               # Logging utilities
    ├── helpers.py              # Helper functions
    └── validators.py           # Input validation
```

## 🔧 Analysis Methodology

### Opportunity Scoring

Each keyword is scored using a composite model:

```
Score = (Volume × 0.25) + (Position × 0.20) + (CTR Gap × 0.20) + 
        (Commercial × 0.20) + (Trend × 0.15)
```

- **Volume Score**: Normalized search volume (log scale)
- **Position Score**: Potential based on current ranking (positions 4-15 highest)
- **CTR Gap Score**: Difference vs expected CTR for position
- **Commercial Score**: Based on CPC and competition
- **Trend Score**: Performance change vs previous period

### Decay Detection

Content decay is classified into types:
- **Position Drop**: Position dropped but impressions stable
- **Impressions Drop**: Impressions dropped
- **Clicks Drop**: Clicks dropped significantly
- **CTR Drop**: CTR declined
- **Full Decay**: Multiple metrics declining
- **Demand Shift**: Impressions dropped, position stable
- **Competition Loss**: Position dropped, impressions stable

### Quick Win Criteria

Keywords qualify as quick wins when:
- High impressions (top tier)
- Below-expected CTR for position
- Position in striking distance (3-15)
- High opportunity score

## 📈 Report Sections

### 1. Executive Summary
- Overall organic health assessment
- Key opportunities identified
- Critical actions needed

### 2. Quick Wins
- Top immediate optimization opportunities
- Specific actions with expected impact
- Grouped by optimization type

### 3. Recovery Section
- Declining keywords/pages requiring attention
- Root cause analysis
- Recovery action plan

### 4. Growth Opportunities
- Net-new keyword/topic opportunities
- Content gaps to fill
- Topic clusters to build

### 5. Strategic Recommendations
- Structural changes needed
- Priority investment areas
- Risk mitigation actions

## ⚠️ Limitations

- GSC data has 3-day delay
- DataForSEO has rate limits based on plan
- YoY comparison requires 1+ year of GSC history
- Large sites may require pagination handling

## 🛠️ Troubleshooting

### OAuth Issues
- Ensure redirect URI matches exactly
- Check that Search Console API is enabled
- Verify credentials are correct

### API Errors
- Check DataForSEO credit balance
- Verify OpenRouter API key is valid
- Review rate limit status

### Data Issues
- Ensure property has sufficient data
- Check minimum impressions filter
- Verify date range has data

## 📝 License

This project is proprietary. All rights reserved.

## 🤝 Support

For support, please contact the development team.