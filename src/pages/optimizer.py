import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.portfolio import recommend_portfolio, optimize_portfolio, generate_efficient_frontier

def show_optimizer_page():
    """Displays the Portfolio Optimizer section."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                💼 Institutional Portfolio Optimizer
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Modern Portfolio Theory (MPT) · Sharpe-Ratio Optimization · Risk-Adjusted Allocation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.markdown("### 📋 Configuration")
        capital = st.number_input("Investment Capital (₹)", min_value=10000, value=100000, step=10000)
        risk_profile = st.select_slider("Risk Profile", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
        
        selected_tickers = st.multiselect(
            "Select Tickers for Portfolio",
            options=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS", "AXISBANK.NS", "LTIM.NS"],
            default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        )
        
        if st.button("🚀 Optimize Portfolio", type="primary", width="stretch"):
            with st.status("Optimizing allocation...", expanded=True):
                weights, metrics = optimize_portfolio(selected_tickers, risk_profile)
                frontier = generate_efficient_frontier(selected_tickers)
            
            st.session_state.port_weights = weights
            st.session_state.port_metrics = metrics
            st.session_state.port_frontier = frontier
            st.rerun()

    if "port_weights" in st.session_state:
        with col2:
            st.markdown("### 📊 Optimized Allocation")
            weights = st.session_state.port_weights
            metrics = st.session_state.port_metrics
            
            # Pie Chart
            fig = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.3)])
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=True)
            st.plotly_chart(fig, width="stretch")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{metrics['return']:.2f}%")
            m2.metric("Annual Volatility", f"{metrics['volatility']:.2f}%")
            m3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

        st.markdown("---")
        
        # Efficient Frontier
        if "port_frontier" in st.session_state:
            st.markdown("### 📈 Efficient Frontier")
            frontier = st.session_state.port_frontier
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=frontier['volatility'], y=frontier['return'],
                mode='markers', marker=dict(color=frontier['sharpe'], colorscale='Viridis', showscale=True),
                name="Portfolios"
            ))
            # Mark current portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[metrics['volatility']], y=[metrics['return']],
                mode='markers', marker=dict(color='red', size=12, symbol='star'),
                name="Optimized Port"
            ))
            fig_frontier.update_layout(template="plotly_dark", height=400, xaxis_title="Risk (Volatility %)", yaxis_title="Return (%)")
            st.plotly_chart(fig_frontier, width="stretch")
