import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.fii_dii_analyzer import (load_fii_dii_data, detect_patterns, monthly_summary, fetch_live_fii_dii,
                                   correlation_analysis, predict_next_day)

def show_tracker_page():
    """Displays the FII/DII Tracker section."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                📡 FII/DII Institutional Flow Tracker
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Real-time institutional liquidity · Smart money movement · Multi-day accumulation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Institutional Summary")
        live_data = fetch_live_fii_dii()
        if live_data:
            st.metric("FII Net (Cr)", f"₹{live_data['fii_net']:,.2f}")
            st.metric("DII Net (Cr)", f"₹{live_data['dii_net']:,.2f}")
            st.caption(f"Last Live Trading Session: {live_data['date']}")
        
        st.markdown("---")
        st.markdown("#### 💡 AI Smart Money Verdict")
        historical_data = load_fii_dii_data()
        prediction = predict_next_day(historical_data)
        
        verdict_color = "#00d4aa" if prediction['signal'] == "Institutional Accrual" else "#ff5c7c"
        st.markdown(f"""
        <div style="background:#111520;border:1px solid #1e2538;border-radius:12px;padding:20px;text-align:center">
            <div style="font-size:12px;color:#7a8299;margin-bottom:12px">AI INSTITUTIONAL BIAS</div>
            <div style="font-size:24px;font-weight:700;color:{verdict_color}">{prediction['signal']}</div>
            <div style="font-size:14px;color:#e4e8f0;margin-top:10px">Confidence: {prediction['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📈 Cumulative Institutional Flow")
        # Plotting logic for FII/DII flows
        monthly = historical_data.resample('M', on='Date').sum()
        
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Bar(
            x=monthly.index, y=monthly['FII_Net'], name="FII Net",
            marker_color="#00d4aa", opacity=0.8
        ))
        fig_flow.add_trace(go.Bar(
            x=monthly.index, y=monthly['DII_Net'], name="DII Net",
            marker_color="#6c63ff", opacity=0.8
        ))
        fig_flow.update_layout(template="plotly_dark", barmode='group', height=400, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_flow, width="stretch")
        
        st.markdown("#### 🔄 Correlation with Nifty 50")
        corr = correlation_analysis(historical_data)
        st.info(f"The FII flow has a **{corr:.2f}** correlation with the Nifty 50 Index over the last 90 days.")
