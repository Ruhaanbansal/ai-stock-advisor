import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.backtest import run_backtest

def show_backtest_page(data, stock, period):
    """Displays the Backtesting section."""
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🔁 Backtesting Engine
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Strategy Validation · Historical Simulation · Risk-Adjusted Returns
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### 🛠️ Configuration")
        strategy = st.selectbox("Strategy", options=["SMA Crossover", "RSI Threshold", "MACD Signal", "AI Momentum"], index=0)
        
        initial_capital = st.number_input("Initial Capital (₹)", min_value=10000, value=100000)
        
        if st.button("🔥 Run Simulation", type="primary", width="stretch"):
            with st.status("Simulating strategy...", expanded=True):
                bt_results = run_backtest(data, strategy, initial_capital)
            
            st.session_state.bt_results = bt_results
            st.rerun()

    if "bt_results" in st.session_state:
        with col2:
            st.markdown(f"### 📈 Strategy Performance: {strategy}")
            bt_results = st.session_state.bt_results
            
            # Cumulative Return Chart
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt_results['dates'], y=bt_results['equity'],
                name="Strategy Equity", line=dict(color="#00d4aa", width=2.5)
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_results['dates'], y=bt_results['benchmark'],
                name="Buy & Hold", line=dict(color="#7a8299", dash="dot")
            ))
            fig_bt.update_layout(template="plotly_dark", height=400, xaxis_title="Date", yaxis_title="Equity (₹)")
            st.plotly_chart(fig_bt, width="stretch")
            
            # Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{bt_results['total_return']}%")
            m2.metric("Annualized Return", f"{bt_results['annual_return']}%")
            m3.metric("Max Drawdown", f"{bt_results['max_drawdown']}%")
            m4.metric("Win Rate", f"{bt_results['win_rate']}%")
