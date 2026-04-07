import streamlit as st
import pandas as pd
from src.ipo_predictor import predict_ipo, find_similar_ipos, SECTORS, get_current_nifty_trend

def show_ipo_page():
    """Displays the IPO Predictor section."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🚀 IPO Listing Gain Predictor
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                AI Listing Gain Consensus · Subscription Analysis · Technical Outlook
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📋 IPO Details")
        company_name = st.text_input("Company Name", placeholder="e.g. Swiggy, Hyundai India...")
        sector = st.selectbox("Sector", options=SECTORS)
        
        c1, c2 = st.columns(2)
        issue_price = c1.number_input("Issue Price (₹)", min_value=10, value=500)
        lot_size = c2.number_input("Lot Size (Qty)", min_value=1, value=30)
        
        st.markdown("---")
        st.markdown("#### 📊 Subscription Data")
        qib = st.number_input("QIB Subscription (x)", min_value=0.0, value=1.5, step=0.1)
        nii = st.number_input("NII/HNI Subscription (x)", min_value=0.0, value=2.0, step=0.1)
        retail = st.number_input("Retail Subscription (x)", min_value=0.0, value=5.0, step=0.1)
        
        st.markdown("---")
        st.markdown("#### 📉 Market Sentiment")
        gmp = st.number_input("Grey Market Premium (₹)", value=50)
        
        if st.button("🚀 Predict Listing Gains", type="primary", width="stretch"):
            with st.status("Analyzing market sentiment...", expanded=True):
                prediction = predict_ipo({
                    "issue_price": issue_price, "sector": sector, "gmp": gmp,
                    "qib": qib, "nii": nii, "retail": retail
                })
                similar = find_similar_ipos(sector, prediction['gain_pct'])
                nifty_trend = get_current_nifty_trend()
            
            st.session_state.ipo_prediction = prediction
            st.session_state.ipo_similar = similar
            st.session_state.ipo_trend = nifty_trend
            st.rerun()

    if "ipo_prediction" in st.session_state:
        with col2:
            st.markdown("### 📈 Prediction Verdict")
            prediction = st.session_state.ipo_prediction
            
            # Listing Gain Badge
            gain = prediction['gain_pct']
            gain_color = "#00d4aa" if gain > 0 else "#ff5c7c"
            st.markdown(f"""
            <div style="background:#111520;border:1px solid #1e2538;border-radius:14px;padding:30px;text-align:center">
                <div style="font-size:12px;color:#7a8299;margin-bottom:8px">EXPECTED LISTING GAIN</div>
                <div style="font-size:48px;font-weight:800;color:{gain_color}">{gain:+.2f}%</div>
                <div style="font-size:18px;color:#e4e8f0;margin-top:10px">₹{issue_price * (1 + gain/100):,.2f} per share</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Breakdown
            st.markdown("#### Multi-Factor Consensus")
            m1, m2 = st.columns(2)
            m1.metric("Subscription Strength", f"{prediction['sub_score']}/10")
            m2.metric("Market Sentiment Score", f"{prediction['sentiment_score']}/10")
            
            # Market Intelligence Insight
            nifty_trend = st.session_state.ipo_trend
            st.info(f"**Market Backdrop:** Nifty 50 is currently in a **{nifty_trend}** trend, which provides a **{'+ve' if nifty_trend == 'Bullish' else '-ve'}** tailwind for listing Day 1.")
