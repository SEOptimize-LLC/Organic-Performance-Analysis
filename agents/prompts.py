"""
Analysis prompts for organic performance analysis.
Contains system prompts and templates for AI analysis.
"""

from typing import Dict


class AnalysisPrompts:
    """
    Prompt templates for organic performance analysis.
    Designed to generate actionable, no-fluff insights.
    """
    
    SYSTEM_PROMPT = """You are an expert SEO analyst specializing in organic
performance optimization. Your role is to analyze data from Google Search
Console and DataForSEO to provide actionable, high-ROI recommendations.

## CRITICAL RULES:
1. ONLY analyze the ACTUAL data provided - NEVER invent examples or sample data
2. If data is empty or says "No data available", state that clearly and move on
3. NEVER create fake keyword examples like "best running shoes" or placeholder URLs
4. Use ONLY the real keywords, URLs, and metrics from the provided datasets
5. If a section has no data, write: "No data available for this analysis."

## Core Principles:
1. NEVER provide vanity metrics or celebratory statements
2. ALWAYS tie insights to specific, implementable actions from REAL data
3. Prioritize recommendations by estimated impact and effort
4. Focus on opportunities that drive traffic AND conversions
5. Be direct and technical - avoid fluff and generic advice
6. Use GSC impressions/clicks (actual data) NOT estimated search volume

## Analysis Framework:
- Quick Wins: High-impact, low-effort opportunities
- Recovery Actions: Decaying content that needs attention
- Growth Plays: Net-new opportunities for expansion
- Strategic Shifts: Structural changes for long-term gains

## Output Format:
- Use clear hierarchical structure
- Include specific keywords, URLs, and metrics FROM THE DATA
- Provide concrete action steps (not vague suggestions)
- Estimate impact based on ACTUAL impressions and CTR potential
- Group by priority tier (Critical, High, Medium, Low)"""
    
    QUICK_WINS_PROMPT = """Analyze the following quick-win opportunities and
provide actionable recommendations.

IMPORTANT: Only analyze the ACTUAL data below. Do NOT invent examples.
If the data shows "No data available" or is empty, state that and skip analysis.

## Quick Wins Data:
{quick_wins_data}

## Instructions:
If data is available, for each keyword in the data:
1. Reference the ACTUAL keyword from the data
2. Note its REAL position, CTR, and impressions values
3. Suggest specific title/meta improvements
4. Calculate realistic click potential based on:
   - ACTUAL impressions (not estimated search volume)
   - Realistic CTR improvement (e.g., from 1% to 3% is 2% gain)
   - Extra clicks = impressions × CTR improvement

Example calculation for a REAL keyword:
- If keyword has 1,000 impressions and 1% CTR = 10 clicks
- Improving to 3% CTR = 30 clicks = +20 additional clicks
- Do NOT claim 200+ clicks unless impressions support it

Group by action type (title fixes, meta description, content updates).

If no data is available, simply state:
"No quick win opportunities identified in the analyzed data."

DO NOT create sample data or placeholder examples."""
    
    DECAY_ANALYSIS_PROMPT = """Analyze the following decaying keywords/pages
and provide recovery recommendations.

IMPORTANT: Only analyze ACTUAL data provided. Do NOT invent examples.

## Decaying Content Data:
{decay_data}

## Decay Summary:
{decay_summary}

## Instructions:
If decay data is available:
1. Analyze ONLY the keywords/pages shown in the data above
2. Reference their ACTUAL metrics (clicks change, position change, etc.)
3. Diagnose likely causes based on the data patterns
4. Provide specific recovery actions for each

If data shows "No data available" or is empty, state:
"No keyword or page-level decay detected in the comparison period."

DO NOT create fictional examples or sample decay scenarios."""
    
    COMPETITOR_ANALYSIS_PROMPT = """Analyze the competitive landscape and 
identify keyword gaps and opportunities.

## Data Context:
- Competitor domains and their organic performance
- Keyword intersection data showing gaps
- Domain authority and traffic estimates

## Competitor Data:
{competitor_data}

## Keyword Gap Data:
{keyword_gap_data}

## Analysis Required:
1. Assess competitive positioning:
   - Where do competitors outrank us?
   - What content types do they have that we lack?
   - What SERP features do they own?

2. Identify high-value keyword gaps:
   - Keywords competitors rank for that we don't
   - Priority by search volume and commercial intent
   - Group by topic/category

3. For top 20 gap keywords, provide:
   - Competitor ranking for it
   - Our current status (if any ranking)
   - Content recommendation (new page vs optimize existing)
   - Target page type (money page, blog, resource)

4. Strategic recommendations:
   - Content gaps to fill first
   - Topics where we can realistically compete
   - Topics to avoid (too competitive/low value)

Focus on opportunities that match our site's authority level."""
    
    BRAND_ANALYSIS_PROMPT = """Analyze brand vs non-brand performance and 
provide strategic recommendations.

## Data Context:
- Traffic split between brand and non-brand queries
- Performance metrics for each segment
- Trend data showing changes over time

## Brand Metrics:
{brand_metrics}

## Non-Brand Opportunities:
{non_brand_opportunities}

## Analysis Required:
1. Assess brand dependency:
   - Current brand/non-brand split
   - Risk assessment for over-reliance
   - Comparison to industry benchmarks

2. Non-brand growth opportunities:
   - Top non-brand keywords with expansion potential
   - Topics where non-brand performance is strong
   - Topics where non-brand is underperforming

3. Strategic recommendations:
   - How to reduce brand dependency (if needed)
   - Non-brand content priorities
   - Balance between brand protection and growth

4. Action plan:
   - Immediate non-brand optimization targets
   - Content gaps in non-brand space
   - Internal linking improvements for non-brand pages

Provide specific keywords and pages to focus on."""
    
    COMPREHENSIVE_REPORT_PROMPT = """Generate a comprehensive organic 
performance analysis report based on all collected data.

## Domain Analyzed: {domain}
## Analysis Period: {period}
## Data Sources: Google Search Console, DataForSEO

## Overview Metrics:
{overview_metrics}

## Quick Wins Identified:
{quick_wins_summary}

## Decay Analysis:
{decay_summary}

## Brand Performance:
{brand_summary}

## Competitor Insights:
{competitor_summary}

## SERP Feature Opportunities:
{serp_features}

## Report Structure Required:

### 1. Executive Summary (2-3 paragraphs)
- Current organic health assessment
- Key opportunities identified
- Critical actions needed

### 2. Quick Wins Section
- Top 10 immediate optimization opportunities
- Specific action for each with expected impact
- Grouped by optimization type

### 3. Recovery Section  
- Top declining keywords/pages requiring attention
- Root cause analysis for each
- Recovery action plan with timelines

### 4. Growth Opportunities Section
- Net-new keyword/topic opportunities
- Content gaps to fill
- Topic clusters to build

### 5. Strategic Recommendations
- Structural changes needed
- Priority investment areas
- Risk mitigation actions

### 6. Implementation Roadmap
- Week 1-2: Critical fixes
- Week 3-4: Quick wins
- Month 2: Growth initiatives
- Month 3+: Strategic buildout

Every recommendation must include:
- Specific action (what to do)
- Target (which page/keyword)
- Expected impact (traffic/position estimate)
- Priority level (Critical/High/Medium/Low)

NO fluff. NO vanity metrics. ALL actionable."""
    
    PAGE_OPTIMIZATION_PROMPT = """Analyze specific pages and provide detailed 
optimization recommendations.

## Page Data:
{page_data}

## Query Portfolio for Pages:
{query_portfolio}

## Page Scores:
{page_scores}

## Analysis Required:
1. For each priority page, analyze:
   - Current ranking keywords
   - Keywords with improvement potential
   - CTR performance vs benchmarks
   - Content alignment with query intent

2. Optimization recommendations:
   - Title tag improvements (be specific)
   - Meta description optimization
   - H1/H2 structure changes
   - Content gaps to fill
   - Internal links to add

3. Technical considerations:
   - Page speed issues (if indicated by device gaps)
   - Mobile optimization needs
   - Structured data opportunities

4. Prioritization:
   - Quick fixes (< 30 min each)
   - Medium effort (2-4 hours)
   - Major updates (full rewrite/restructure)

Provide specific, actionable recommendations for each page."""
    
    TOPIC_CLUSTER_PROMPT = """Analyze keyword data and recommend topic 
cluster strategy.

## Ranked Keywords:
{ranked_keywords}

## Keyword Suggestions:
{keyword_suggestions}

## Related Keywords:
{related_keywords}

## Analysis Required:
1. Identify potential topic clusters:
   - Main pillar topics (high volume, broad)
   - Supporting cluster topics (specific, long-tail)
   - Content gaps within each cluster

2. For each cluster, provide:
   - Pillar page recommendation (topic, type)
   - 5-10 cluster content pieces
   - Internal linking structure
   - Estimated total search volume

3. Prioritization:
   - Clusters with existing content to leverage
   - Clusters aligned with business goals  
   - Clusters with manageable competition

4. Implementation order:
   - Start with clusters where we have foundation
   - Build out highest-impact clusters first
   - Timeline for each cluster buildout

Be specific about content types and topics."""
    
    @classmethod
    def get_prompt(
        cls,
        prompt_type: str,
        data: Dict
    ) -> str:
        """
        Get formatted prompt with data.
        
        Args:
            prompt_type: Type of analysis prompt
            data: Data to insert into prompt
            
        Returns:
            Formatted prompt string
        """
        prompts = {
            'quick_wins': cls.QUICK_WINS_PROMPT,
            'decay': cls.DECAY_ANALYSIS_PROMPT,
            'competitor': cls.COMPETITOR_ANALYSIS_PROMPT,
            'brand': cls.BRAND_ANALYSIS_PROMPT,
            'comprehensive': cls.COMPREHENSIVE_REPORT_PROMPT,
            'page_optimization': cls.PAGE_OPTIMIZATION_PROMPT,
            'topic_cluster': cls.TOPIC_CLUSTER_PROMPT
        }
        
        template = prompts.get(prompt_type, cls.QUICK_WINS_PROMPT)
        
        # Format with available data
        try:
            return template.format(**data)
        except KeyError as e:
            # Return template with placeholder note
            return f"{template}\n\n[Note: Missing data key: {e}]"
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt."""
        return cls.SYSTEM_PROMPT