import streamlit as st
import pandas as pd

def show_intelligence_page(stock, articles, sentiment_score, sentiment_label, sentiment_source):
    """Displays the Market Intelligence & News section."""
    st.markdown(f"### 📰 Market Intelligence: {stock}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Sentiment Score", f"{sentiment_score:+.2f}", sentiment_label)
        st.caption(f"Powered by {sentiment_source}")
        
        # Sentiment Gauge (Simple CSS version)
        gauge_color = "#00d4aa" if sentiment_score > 0 else "#ff5c7c" if sentiment_score < 0 else "#7a8299"
        st.markdown(f"""
        <div style="background:#111520;border:1px solid #1e2538;border-radius:12px;padding:20px;text-align:center">
            <div style="font-size:12px;color:#7a8299;margin-bottom:8px">SENTIMENT POLARITY</div>
            <div style="font-size:32px;font-weight:700;color:{gauge_color}">{sentiment_label.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if not articles:
            st.info("No recent news articles found for this stock.")
        else:
            for i, article in enumerate(articles[:10]):
                with st.container():
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:16px;margin-bottom:12px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                            <span style="font-size:10px;color:#00d4aa;font-weight:600;letter-spacing:1px">{article.get('source', 'NEWS')}</span>
                            <span style="font-size:10px;color:#7a8299">{article.get('date', '')}</span>
                        </div>
                        <a href="{article.get('url', '#')}" target="_blank" style="text-decoration:none;color:#e4e8f0;font-size:16px;font-weight:600;line-height:1.4">
                            {article.get('title', 'No Title')}
                        </a>
                        <p style="font-size:13px;color:#7a8299;margin-top:8px;line-height:1.5">
                            {article.get('description', '')[:200]}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
