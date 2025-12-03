# Organic Performance Analysis Tool

A comprehensive SEO analysis dashboard that leverages Google Search Console, DataForSEO, and LLM-powered insights to provide actionable recommendations for improving organic performance.

## Features

### Data Integration
- **Google Search Console API**: Query-level and page-level performance data
- **DataForSEO API**: Ranked keywords, competitor analysis, SERP features, keyword gaps
- **OpenRouter LLM**: AI-powered strategic analysis using multiple reasoning models

### Analysis Capabilities
- **Quick Win Identification**: High-impression, underperforming CTR opportunities
- **Content Decay Detection**: Identify declining queries and pages with trend analysis
- **Keyword Gap Analysis**: Find keywords competitors rank for that you don't
- **Brand vs Non-Brand Segmentation**: Understand traffic dependency patterns
- **Device & Geographic Analysis**: Performance breakdown by device and country
- **Competitor Benchmarking**: Compare against key competitors

### AI-Powered Insights
Choose from multiple LLM models for analysis:
- OpenAI GPT-4.1 Mini / GPT-5 Mini
- Anthropic Claude Haiku 4.5 / Claude Sonnet 4.5
- Google Gemini 3 Pro / Gemini 2.5 Flash
- xAI Grok 4 Fast
- DeepSeek R1 Qwen3

## Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Project with Search Console API enabled
- DataForSEO account
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SEOptimize-LLC/Organic-Performance-Analysis.git
cd Organic-Performance-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure secrets:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API credentials
```

4. Run the application:
```bash
streamlit run app.py
```

## Configuration

### Streamlit Secrets

Create `.streamlit/secrets.toml` with the following structure:

```toml
[google]
client_id = "your-google-client-id.apps.googleusercontent.com"
client_secret = "your-google-client-secret"
redirect_uri = "http://localhost:8501"

[dataforseo]
login = "your-dataforseo-email@example.com"
password = "your-dataforseo-api-password"

[openrouter]
api_key = "sk-or-v1-your-openrouter-api-key"
```

### Google Search Console API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Enable the **Search Console API**
4. Go to **APIs & Services > Credentials**
5. Create **OAuth 2.0 Client ID** (Web application type)
6. Add authorized redirect URIs:
   - For local development: `http://localhost:8501`
   - For Streamlit Cloud: `https://your-app.streamlit.app`
7. Download the client configuration and use the client ID and secret

### DataForSEO API Setup

1. Sign up at [DataForSEO](https://dataforseo.com)
2. Go to your dashboard
3. Find your API credentials (login email and password)
4. Add to secrets configuration

### OpenRouter API Setup

1. Sign up at [OpenRouter](https://openrouter.ai)
2. Go to **Keys** in your account
3. Generate a new API key
4. Add to secrets configuration

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Create a new app and connect your repository
4. Go to **App Settings > Secrets**
5. Add your secrets in TOML format:

```toml
[google]
client_id = "your-client-id"
client_secret = "your-client-secret"
redirect_uri = "https://your-app.streamlit.app"

[dataforseo]
login = "your-email"
password = "your-password"

[openrouter]
api_key = "your-api-key"
```

## Usage Guide

### 1. Connect to Google Search Console
- Click "Connect to GSC" in the sidebar
- Follow the OAuth flow to authorize access
- Select a property from the dropdown

### 2. Configure Analysis Parameters
- **Date Range**: Select analysis period (28 days to 12 months)
- **Location**: Target country for DataForSEO data
- **Language**: Target language for keyword data
- **Brand Terms**: Enter your brand terms (one per line)
- **Competitors**: Enter competitor domains (one per line)
- **Minimum Impressions**: Filter threshold for queries

### 3. Select AI Model
Choose from available reasoning models based on your needs:
- **Fast analysis**: Claude Haiku 4.5, GPT-4.1 Mini
- **Balanced**: Claude Sonnet 4.5, Gemini 2.5 Flash
- **Deep reasoning**: GPT-5 Mini, Gemini 3 Pro

### 4. Run Analysis
Click "Run Full Analysis" to:
1. Fetch data from GSC and DataForSEO
2. Process and analyze the data
3. Identify opportunities and issues
4. Generate AI-powered insights

### 5. Review Results
Navigate through tabs to explore:
- **AI Analysis**: Strategic recommendations from LLM
- **Quick Wins**: CTR optimization opportunities
- **Content Decay**: Declining queries and recovery strategies
- **Keyword Gaps**: Competitor keyword opportunities
- **Competitors**: Competitive landscape overview
- **Categories**: Topic/category performance

### 6. Export Data
Export analysis results as CSV for further processing.

## Analysis Methodology

### Quick Win Identification
Identifies queries with:
- Positions 3-15 (page 1 or top of page 2)
- High impressions but CTR below benchmark
- Calculated opportunity score based on volume, position, CTR gap, and commercial value

### Content Decay Detection
Compares current period vs previous period to identify:
- **Competition/SERP Changes**: Position drop with stable impressions
- **Demand Decline**: Impressions drop with stable position
- **Major Decline**: Both metrics declining
- **CTR Decline**: CTR drop with stable position/impressions

### Opportunity Scoring
Each opportunity is scored based on:
- Search volume / impressions (scale factor)
- Current ranking position
- CTR gap vs benchmark
- Commercial value (CPC proxy)
- Trend direction

## Project Structure

```
Organic-Performance-Analysis/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── .streamlit/
│   ├── config.toml            # Streamlit configuration
│   └── secrets.toml.example   # Example secrets template
└── src/
    ├── __init__.py
    ├── gsc_api.py             # Google Search Console API client
    ├── dataforseo_api.py      # DataForSEO API client
    ├── llm_api.py             # OpenRouter LLM integration
    ├── analysis_engine.py     # Core analysis logic
    └── report_generator.py    # Report generation module
```

## API Rate Limits & Costs

### Google Search Console API
- Free tier available
- Subject to Google API quotas

### DataForSEO API
- Pay-per-use model
- Check [DataForSEO pricing](https://dataforseo.com/pricing)

### OpenRouter API
- Pay-per-token model
- Different rates per model
- Check [OpenRouter pricing](https://openrouter.ai/pricing)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please open an issue on GitHub.
