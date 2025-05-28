import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from mention_monitor import MentionMonitorApp, MentionStats
    from scraper_modules import create_scrapers, cleanup_scrapers
    from llm_analysis import create_analyzer, MentionEnricher
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mention Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .mention-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #28a745;
    }
    
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_app():
    """ intializing monitoring application"""
    # Load API keys from environment 
    try:
        travily_api_key = os.environ.get('TRAVILY_API_KEY')
        cohere_api_key = os.environ.get('COHERE_API_KEY')
    except:
        travily_api_key = os.environ.get('TRAVILY_API_KEY')
        cohere_api_key = os.environ.get('COHERE_API_KEY')
    
    # Initialize main app
    mention_app = MentionMonitorApp()
    
    # Create and register scrapers
    scrapers = create_scrapers(travily_api_key)
    for scraper in scrapers:
        mention_app.register_scraper(scraper)
    
    # Initialize LLM analyzer
    analyzer = create_analyzer(cohere_api_key)
    enricher = None
    if analyzer:
        enricher = MentionEnricher(analyzer)
    
    return mention_app, scrapers, enricher

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Mention Monitor</h1>
        <p>Track public mentions of companies and individuals across the web</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_config():
    """configuration options in sidebar"""
    st.sidebar.header("⚙️ Configuration")
    
    # API Status
    st.sidebar.subheader("API Status")
    
    #  API keys checks
    travily_key = os.environ.get('TRAVILY_API_KEY', '')
    cohere_key = os.environ.get('COHERE_API_KEY', '')
    
    st.sidebar.write("🔍 Travily API:", "✅ Connected" if travily_key else "❌ Not configured")
    st.sidebar.write("🤖 Cohere API:", "✅ Connected" if cohere_key else "❌ Not configured")
    
    # Search Settings
    st.sidebar.subheader("Search Settings")
    max_results = st.sidebar.slider("Max Results per Source", 5, 50, 20)
    enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
    
    return {
        'max_results': max_results,
        'enable_sentiment': enable_sentiment,
        'travily_available': bool(travily_key),
        'cohere_available': bool(cohere_key)
    }

async def perform_search(app, enricher, search_term, config):
    """Perform the search operation"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        status_text.text("🔍 Searching across multiple sources...")
        progress_bar.progress(20)
        
        # Search for mentions
        mentions = await app.search_mentions(search_term)
        
        progress_bar.progress(60)
        status_text.text(f"📊 Found {len(mentions)} mentions. Analyzing...")
        
        # Enrich with LLM analysis if available
        if enricher and config['enable_sentiment'] and config['cohere_available']:
            mentions = await enricher.enrich_mentions(mentions, search_term)
        
        progress_bar.progress(90)
        status_text.text("📈 Generating analytics...")
        
        # Get analytics
        stats = app.get_stats(search_term)
        recent_mentions = app.get_recent_mentions(search_term, limit=50)
        
        progress_bar.progress(100)
        status_text.text("✅ Search completed!")
        
        return stats, recent_mentions, mentions
        
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None, [], []

def display_stats(stats):
    """Display statistics in a nice layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Mentions",
            value=stats.total_mentions,
            delta=None
        )
    
    with col2:
        st.metric(
            label="📅 Last 7 Days",
            value=stats.mentions_last_7_days,
            delta=None
        )
    
    with col3:
        st.metric(
            label="🎯 Avg Relevance",
            value=f"{stats.avg_relevance_score:.2f}",
            delta=None
        )
    
    with col4:
        # Find top source
        if stats.sources_breakdown:
            top_source = max(stats.sources_breakdown.items(), key=lambda x: x[1])
            st.metric(
                label="🥇 Top Source",
                value=top_source[0],
                delta=f"{top_source[1]} mentions"
            )
        else:
            st.metric(label="🥇 Top Source", value="N/A")

def display_charts(stats):
    """Display charts for analytics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Mentions Over Time (Last 7 Days)")
        if stats.mentions_by_day:
            # Convert to DataFrame for Plotly
            df_time = pd.DataFrame([
                {'Date': date, 'Mentions': count}
                for date, count in stats.mentions_by_day.items()
            ])
            df_time['Date'] = pd.to_datetime(df_time['Date'])
            df_time = df_time.sort_values('Date')
            
            fig_time = px.line(
                df_time, 
                x='Date', 
                y='Mentions',
                title="Daily Mention Count",
                markers=True
            )
            fig_time.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No time data available")
    
    with col2:
        st.subheader("😊 Sentiment Distribution")
        if stats.sentiment_breakdown:
            # Create pie chart for sentiment
            labels = list(stats.sentiment_breakdown.keys())
            values = list(stats.sentiment_breakdown.values())
            
            colors = {
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            }
            
            fig_sentiment = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=[colors.get(label.lower(), '#667eea') for label in labels]
            )])
            
            fig_sentiment.update_layout(
                title="Sentiment Breakdown",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("No sentiment data available")

def display_sources_chart(stats):
    """Display sources breakdown"""
    st.subheader("📰 Sources Breakdown")
    if stats.sources_breakdown:
        df_sources = pd.DataFrame([
            {'Source': source, 'Count': count}
            for source, count in stats.sources_breakdown.items()
        ])
        
        fig_sources = px.bar(
            df_sources,
            x='Source',
            y='Count',
            title="Mentions by Source",
            color='Count',
            color_continuous_scale='viridis'
        )
        fig_sources.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_sources, use_container_width=True)
    else:
        st.info("No source data available")

def display_mentions(mentions):
    """Display recent mentions"""
    st.subheader("📝 Recent Mentions")
    
    if not mentions:
        st.info("No mentions found. Try a different search term.")
        return
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Source filter
        sources = list(set([m.get('source', 'Unknown') for m in mentions]))
        selected_sources = st.multiselect("Filter by Source", sources, default=sources)
    
    with col2:
        # Sentiment filter
        sentiments = list(set([m.get('sentiment', 'neutral') for m in mentions]))
        selected_sentiments = st.multiselect("Filter by Sentiment", sentiments, default=sentiments)
    
    with col3:
        # Sort options
        sort_by = st.selectbox("Sort by", ["Date", "Relevance", "Title"])
    
    # Filter mentions
    filtered_mentions = [
        m for m in mentions
        if m.get('source', 'Unknown') in selected_sources
        and m.get('sentiment', 'neutral') in selected_sentiments
    ]
    
    # Sort mentions
    if sort_by == "Date":
        filtered_mentions.sort(key=lambda x: x.get('date_found', ''), reverse=True)
    elif sort_by == "Relevance":
        filtered_mentions.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    elif sort_by == "Title":
        filtered_mentions.sort(key=lambda x: x.get('title', ''))
    
    # Display mentions
    for i, mention in enumerate(filtered_mentions[:20]):  # Limit to 20 for performance
        with st.expander(f"📰 {mention.get('title', 'No Title')[:100]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Source:** {mention.get('source', 'Unknown')}")
                st.write(f"**URL:** {mention.get('url', '')}")
                st.write(f"**Snippet:** {mention.get('snippet', 'No snippet available')}")
                
                if mention.get('date_found'):
                    try:
                        if isinstance(mention['date_found'], str):
                            date_obj = datetime.fromisoformat(mention['date_found'])
                        else:
                            date_obj = mention['date_found']
                        st.write(f"**Found:** {date_obj.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.write(f"**Found:** {mention['date_found']}")
            
            with col2:
                # Sentiment badge
                sentiment = mention.get('sentiment', 'neutral')
                sentiment_color = {
                    'positive': 'sentiment-positive',
                    'negative': 'sentiment-negative', 
                    'neutral': 'sentiment-neutral'
                }.get(sentiment, 'sentiment-neutral')
                
                st.markdown(f'<span class="{sentiment_color}">😊 {sentiment.title()}</span>', unsafe_allow_html=True)
                
                # Relevance score
                relevance = mention.get('relevance_score', 0)
                st.write(f"🎯 Relevance: {relevance:.2f}")
                
                # External link
                if mention.get('url'):
                    st.markdown(f"[🔗 View Source]({mention['url']})")

def main():
    """Main Streamlit application"""
    try:
        app, scrapers, enricher = initialize_app()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.stop()
    
    # Display header
    display_header()
    
    # Display sidebar configuration
    config = display_sidebar_config()
    
    # Main search interface
    st.header("🔍 Search for Mentions")
    
    # Search form
    search_term = st.text_input(
        "Enter company or person name:",
        placeholder="e.g., Tesla, Elon Musk, OpenAI...",
        help="Enter the name of a company or individual to monitor"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🚀 Start Search", type="primary")
    
    # Handle search
    if search_button and search_term:
        st.header(f"📊 Results for: {search_term}")
        
        # Run search
        with st.spinner("Searching..."):
            try:
                stats, mentions, raw_mentions = asyncio.run(
                    perform_search(app, enricher, search_term, config)
                )
                
                if stats:
                    # Display results
                    display_stats(stats)
                    
                    st.markdown("---")
                    
                    # Charts
                    display_charts(stats)
                    
                    st.markdown("---")
                    
                    # Sources breakdown
                    display_sources_chart(stats)
                    
                    st.markdown("---")
                    
                    # Mentions list
                    display_mentions(mentions)
                    
                    # Export options
                    st.sidebar.subheader("📥 Export Data")
                    if mentions:
                        # Convert to DataFrame
                        df = pd.DataFrame(mentions)
                        csv = df.to_csv(index=False)
                        
                        st.sidebar.download_button(
                            label="📊 Download CSV",
                            data=csv,
                            file_name=f"mentions_{search_term}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                        
                        # JSON export
                        import json
                        json_data = json.dumps(mentions, indent=2, default=str)
                        st.sidebar.download_button(
                            label="📋 Download JSON",
                            data=json_data,
                            file_name=f"mentions_{search_term}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    elif search_button and not search_term:
        st.warning("Please enter a search term")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p>🔍 Mention Monitor - Track public mentions across the web</p>
        <p>Built with Streamlit • Data from multiple sources</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()