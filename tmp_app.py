# =============================================================
# app.py — NiftyMind · AI Stock Advisor
# =============================================================

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Project modules ───────────────────────────────────────────
from src.config import (
    APP_TITLE, APP_ICON, CURRENCY, STOCK_LIST, STOCK_LABELS,
    SEQUENCE_LENGTH,
)
from src.stock_search  import resolve_ticker, get_suggestions
from src.data_sources  import fetch_stock_data, fetch_portfolio_data
from src.dataloader   import load_stock_data, get_close_prices, load_portfolio_data, calculate_returns, get_stock_info
from src.model        import train_lstm_model, predict_next_price, forecast_prices
from src.features     import create_extended_features
from src.sentiment    import analyse_sentiment
from src.risk         import calculate_risk_metrics
from src.explainability  import get_technical_signals, compute_feature_importance, shap_chart
from src.ipo_predictor   import load_ipo_model, predict_ipo, get_current_nifty_trend, find_similar_ipos, SECTORS
from src.chart_analyzer   import (analyse_chart, get_api_key,
                                   rec_style, trend_icon, pattern_type_color)
from src.fii_dii_analyzer import (load_fii_dii_data, load_fii_dii_model,
                                   predict_next_day, detect_patterns,
                                   fetch_live_fii_dii, monthly_summary,
                                   correlation_analysis)
from src.portfolio    import recommend_portfolio, optimize_portfolio, generate_efficient_frontier
from src.backtest     import run_backtest
from src.evaluation   import evaluate_model
from src.advisor      import run_ai_advisor
from src.alerts       import detect_market_alerts
from src.insight      import generate_ai_insight


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0a0d14;
    --surface:     rgba(17, 21, 32, 0.7);
    --surface2:    rgba(24, 29, 46, 0.6);
    --border:      rgba(30, 37, 56, 0.5);
    --accent:      #00d4aa;
    --accent2:     #6c63ff;
    --danger:      #ff5c7c;
    --warning:     #ffb347;
    --text:        #e4e8f0;
    --muted:       #7a8299;
    --card-radius: 16px;
    --glass-blur:  blur(12px);
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar Navigation ── */
[data-testid="stSidebar"] {
    background-color: #0b0e16 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebarNav"] { padding-top: 2rem; }
[data-testid="stSidebarNav"] ul { padding: 0 12px; }
[data-testid="stSidebarNav"] li {
    margin-bottom: 4px;
    border-radius: 10px;
    transition: background 0.2s;
}
[data-testid="stSidebarNav"] li:hover { background: rgba(0, 212, 170, 0.05); }

/* ── Glass Cards (Metric, Expander, Containers) ── */
[data-testid="stMetric"], [data-testid="stExpander"], div.stButton > button {
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
}

[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--card-radius) !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 212, 170, 0.3) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 0 20px rgba(0,212,170,0.05);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 13px !important; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { color: var(--text)  !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 26px !important; font-weight: 700 !important; }

/* ── Section headers ── */
h1, h2, h3, h4 { 
    font-family: 'Space Grotesk', sans-serif !important; 
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

/* ── Styled Info Cards (Glass Containers) ── */
.glass-card {
    background: var(--surface);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border);
    border-radius: var(--card-radius);
    padding: 24px;
    margin-bottom: 20px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0d14 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(108,99,255,0.3);
}

/* ── Alert banners ── */
.alert-danger, .alert-warning, .alert-info, .alert-ok {
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    font-size: 14px;
    backdrop-filter: var(--glass-blur);
}
.alert-danger  { background: rgba(255,92,124,.1); border: 1px solid rgba(255,92,124,.2); color: #ff5c7c; }
.alert-warning { background: rgba(255,179,71,.1); border: 1px solid rgba(255,179,71,.2); color: #ffb347; }
.alert-info    { background: rgba(108,99,255,.1); border: 1px solid rgba(108,99,255,.2); color: #6c63ff; }
.alert-ok      { background: rgba(0,212,170,.1);  border: 1px solid rgba(0,212,170,.2);  color: #00d4aa; }

/* ── Rec badge ── */
.rec-badge {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px; font-weight: 800;
    padding: 12px 36px;
    border-radius: 60px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* ── Inputs ── */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: var(--surface2) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ── Tabs ── */
[data-testid="stTab"] {
    padding: 10px 20px !important;
    font-weight: 600 !important;
}

/* ── Global Symmetry ── */
.stMainBlockContainer { padding-top: 3rem !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 24px; text-align:center">
      <div style="font-size:42px; margin-bottom:12px">🧠</div>
      <span style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:800;
                   background:linear-gradient(135deg,#00d4aa,#6c63ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        NiftyMind
      </span>
      <span style="font-size:11px;color:#7a8299;display:block;margin-top:4px;letter-spacing:1px;text-transform:uppercase;font-weight:600">
        AI Stock Intelligence
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Grouped Navigation ────────────────────────────────────
    st.markdown("<div style='font-size:11px;color:#7a8299;margin-bottom:8px;font-weight:600;letter-spacing:0.5px'>CORE ANALYSIS</div>", unsafe_allow_html=True)
    
    # Using a dictionary for icons to keep it clean
    nav_icons = {
        "📊 Dashboard": "📊", "🧠 AI Prediction": "🧠", "📰 Market Intelligence": "📰",
        "💼 Portfolio Optimizer": "💼", "🔁 Backtesting": "🔁", "🔬 Model Evaluation": "🔬",
        "🚀 IPO Predictor": "🚀", "📡 FII/DII Tracker": "📡", "📸 Chart Analyzer": "📸"
    }
    
    # We'll use a radio but with categories
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🧠 AI Prediction", "📰 Market Intelligence",
         "💼 Portfolio Optimizer", "🔁 Backtesting", "🔬 Model Evaluation",
         "🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#7a8299;margin-bottom:8px;font-weight:600;letter-spacing:0.5px'>STOCK SECTOR</div>", unsafe_allow_html=True)

    # ── Stock Search ──────────────────────────────────────────
    search_query = st.text_input(
        "Search",
        placeholder="Search NSE Ticker...",
        label_visibility="collapsed",
    )

    # Initialise session state
    if "selected_stock" not in st.session_state:
        st.session_state.selected_stock      = None
        st.session_state.selected_stock_name = None

    # Live suggestions
    if search_query and len(search_query) >= 2:
        suggestions = get_suggestions(search_query, max_results=5)
        if suggestions:
            q_hash = abs(hash(search_query)) % 100000
            for idx, s in enumerate(suggestions):
                btn_label = f"{s['name']}"
                btn_key   = f"sug_{q_hash}_{idx}"
                if st.button(btn_label, key=btn_key, use_container_width=True):
                    st.session_state.selected_stock      = s["ticker"]
                    st.session_state.selected_stock_name = s["name"]
                    st.rerun()

    # ── Popular Stocks (quick-select) ─────────────────────────
    preset_cols = st.columns(2)
    for i, ticker in enumerate(STOCK_LIST):
        label = STOCK_LABELS.get(ticker, ticker).split()[0]
        btn_key = f"preset_{i}_{ticker}"
        if preset_cols[i % 2].button(label, key=btn_key, use_container_width=True):
            st.session_state.selected_stock      = ticker
            st.session_state.selected_stock_name = STOCK_LABELS.get(ticker, ticker)
            st.rerun()

    # ── Currently selected (The "Active Stock" Card) ──────────
    stock = st.session_state.get("selected_stock") or "RELIANCE.NS"
    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.2); 
                border-radius:12px; padding:12px; margin-top:20px; text-align:center'>
        <div style='font-size:10px; color:var(--accent); font-weight:700; letter-spacing:1px'>ACTIVE ASSET</div>
        <div style='font-size:15px; font-weight:700; color:white; margin:4px 0'>{st.session_state.selected_stock_name or "Select a Stock"}</div>
        <div style='font-size:11px; color:var(--muted)'>{stock}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    period = st.select_slider(
        "Lookback Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
        value="6mo",
        format_func=lambda x: {
            "1mo":"1M", "3mo":"3M", "6mo":"6M", "1y":"1Y", "2y":"2Y", "5y":"5Y", "10y":"10Y"
        }[x]
    )

    st.markdown("---")
    st.caption(f"Sync: {datetime.now().strftime('%H:%M:%S')}")


# ══════════════════════════════════════════════════════════════
# WELCOME PAGE — shown until user picks a stock
# ══════════════════════════════════════════════════════════════

_STOCK_FREE_PAGES = {"🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer", "💼 Portfolio Optimizer"}

if st.session_state.get("selected_stock") is None and page not in _STOCK_FREE_PAGES:
    # Hero Section
    st.markdown("""
    <div style='text-align:center; padding:80px 20px 40px; border-radius:30px; 
                background:radial-gradient(circle at top right, rgba(108,99,255,0.1), transparent),
                           radial-gradient(circle at bottom left, rgba(0,212,170,0.05), transparent);
                margin-bottom:40px; border:1px solid rgba(255,255,255,0.02)'>
        <div style='display:inline-block; padding:8px 16px; border-radius:40px; 
                    background:rgba(0,212,170,0.1); color:var(--accent); font-size:12px; 
                    font-weight:700; margin-bottom:24px; letter-spacing:1.5px'>
            ALPHA VERSION 2.0
        </div>
        <h1 style='font-size:4rem; line-height:1.1; margin-bottom:16px'>
            Master the <span style='color:var(--accent)'>Market</span><br>
            with AI Intelligence.
        </h1>
        <p style='color:var(--muted); font-size:1.2rem; max-width:600px; margin:0 auto 40px'>
            Professional-grade analytical tools for the Indian stock market. 
            Real-time sentiment, predictive modeling, and portfolio optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # symmetrical 3-column features
    st.subheader("Explore Powerful Capabilities")
    fcol1, fcol2, fcol3 = st.columns(3)
    
    with fcol1:
        st.markdown("""<div class='glass-card' style='height:200px'>
            <div style='font-size:32px; margin-bottom:12px'>🧠</div>
            <div style='font-weight:700; font-size:18px; margin-bottom:8px'>AI Forecasting</div>
            <div style='font-size:13px; color:var(--muted); line-height:1.4'>
                Predictive GradientBoosting engine trained on deep historical price action patterns.
            </div>
        </div>""", unsafe_allow_html=True)

    with fcol2:
        st.markdown("""<div class='glass-card' style='height:200px'>
            <div style='font-size:32px; margin-bottom:12px'>📰</div>
            <div style='font-weight:700; font-size:18px; margin-bottom:8px'>Market Intel</div>
            <div style='font-size:13px; color:var(--muted); line-height:1.4'>
                Live sentiment processing using VADER and FinBERT on thousands of daily headlines.
            </div>
        </div>""", unsafe_allow_html=True)

    with fcol3:
        st.markdown("""<div class='glass-card' style='height:200px'>
            <div style='font-size:32px; margin-bottom:12px'>💼</div>
            <div style='font-weight:700; font-size:18px; margin-bottom:8px'>Portfolio Opt</div>
            <div style='font-size:13px; color:var(--muted); line-height:1.4'>
                Institutional-grade Modern Portfolio Theory (MPT) logic for risk-adjusted returns.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Symmetrical Start Call to Action
    st.info("👈 **Select a stock from the sidebar** or search any ticker to begin your analysis.")
    
    st.stop()


# ══════════════════════════════════════════════════════════════
# DATA LOADING  (skipped for stock-free pages)
# ══════════════════════════════════════════════════════════════

data         = None
close_prices = None

if page not in _STOCK_FREE_PAGES:
    with st.spinner("Fetching market data…"):
        import logging
        logging.basicConfig(level=logging.INFO)
        data = load_stock_data(stock, period=period)

    if data is None or data.empty:
        st.error(
            f"⚠️ Could not load data for **{st.session_state.get('selected_stock_name', stock)}** "
            f"(`{stock}`) from any data source (Yahoo Finance, Stooq, Alpha Vantage). "
            "This is likely a temporary outage — please try again in a moment."
        )
        col_r1, col_r2 = st.columns(2)
        if col_r1.button("🔄 Retry", key="retry_main"):
            st.cache_data.clear()
            st.rerun()
        if col_r2.button("🔍 Search different stock", key="retry_search"):
            st.session_state.selected_stock      = "RELIANCE.NS"
            st.session_state.selected_stock_name = "Reliance Industries"
            st.rerun()
        st.stop()

    close_prices = get_close_prices(data)

    if len(close_prices) < 65:
        st.warning(
            f"Not enough historical data for **{stock}** (only {len(close_prices)} days found). "
            "Try selecting a longer period or a different stock."
        )
        st.stop()


# ══════════════════════════════════════════════════════════════
# MODEL TRAINING  (skipped for stock-free pages)
# ══════════════════════════════════════════════════════════════

model = scaler = last_sequence = None
predicted_price = current_price = None
forecast_5d = []
sentiment_score = sentiment_label = headlines = sentiment_source = None
risk_metrics = volatility = sharpe_ratio = max_drawdown = advisor = None

if page not in _STOCK_FREE_PAGES:
    with st.spinner("Loading AI model…"):
        cp_hash = hash(close_prices.values.tobytes())
        model, scaler, last_sequence, train_mae, test_mae = train_lstm_model(stock, cp_hash, close_prices)

    predicted_price = predict_next_price(model, scaler, last_sequence)
    current_price   = float(close_prices.iloc[-1])
    forecast_5d     = forecast_prices(model, scaler, last_sequence, days=5)

    with st.spinner("Analysing sentiment & risk…"):
        stock_name = st.session_state.get("selected_stock_name", stock)
        sentiment_result = analyse_sentiment(stock, company_name=stock_name)
        sentiment_score  = sentiment_result.get("score", 0.0)
        sentiment_label  = sentiment_result.get("label", "Neutral")
        headlines        = sentiment_result.get("articles", [])
        sentiment_source = sentiment_result.get("model", "VADER")
        risk_metrics = calculate_risk_metrics(close_prices, stock_ticker=stock)

    volatility   = risk_metrics.get("volatility", 0.0)
    sharpe_ratio = risk_metrics.get("sharpe", 0.0)
    max_drawdown = risk_metrics.get("max_drawdown", 0.0)

    # Compute RSI and MACD for advisor
    from src.features import calculate_rsi, calculate_macd
    _rsi_s       = calculate_rsi(close_prices)
    _rsi_val     = float(_rsi_s.dropna().iloc[-1]) if not _rsi_s.dropna().empty else None
    _macd_s, _macd_sig_s = calculate_macd(close_prices)
    _macd_val    = float(_macd_s.dropna().iloc[-1])     if not _macd_s.dropna().empty     else None
    _macd_sig_v  = float(_macd_sig_s.dropna().iloc[-1]) if not _macd_sig_s.dropna().empty else None

    advisor = run_ai_advisor(
        current_price, predicted_price, volatility, sentiment_score,
        rsi=_rsi_val, macd=_macd_val, macd_signal_val=_macd_sig_v,
        test_mae=test_mae,
    )


# ══════════════════════════════════════════════════════════════
# GLOBAL KPI STRIP  (only for stock pages)
# ══════════════════════════════════════════════════════════════

if page not in _STOCK_FREE_PAGES:
    # Header Section
    _source = st.session_state.get("data_source", "Yahoo Finance")
    _source_color = {"Yahoo Finance": "#00d4aa", "Stooq": "#6c63ff", "Alpha Vantage": "#ffb347"}.get(_source, "#7a8299")
    
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:20px'>
        <div>
            <h1 style='margin:0'>{STOCK_LABELS.get(stock, stock)}</h1>
            <div style='color:var(--muted); font-size:14px; margin-top:4px'>
                <span style='color:var(--accent)'>SYMBOL:</span> {stock} &nbsp; | &nbsp; 
                <span style='color:var(--accent2)'>SOURCE:</span> {_source} &nbsp; | &nbsp; 
                <span style='color:var(--warning)'>PERIOD:</span> {period.upper()}
            </div>
        </div>
        <div style='text-align:right'>
            <div style='font-size:11px; color:var(--muted); letter-spacing:1px'>AI CONFIDENCE</div>
            <div style='font-size:24px; font-weight:700; color:var(--accent)'>{advisor.get("confidence", 0):.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    price_delta = f"{advisor['price_change']:+.2f}%"
    
    k1.metric("Current Price",   f"₹{current_price:,.2f}")
    k2.metric("AI Forecast",    f"₹{predicted_price:,.2f}", price_delta)
    k3.metric("Sentiment",       sentiment_label)
    k4.metric("Risk Profile",    advisor["risk"])
    k5.metric("Decision",        advisor["recommendation"])

    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 📊 DASHBOARD
# ─────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.subheader("Price Chart & Technical Indicators")

    extended = create_extended_features(data)

    chart_tab, bb_tab, macd_tab = st.tabs(["Price & Base", "Bollinger Bands", "MACD"])

    # Trend-based coloring (Google Finance style)
    is_up = float(close_prices.iloc[-1]) >= float(close_prices.iloc[0])
    line_color = "#00d4aa" if is_up else "#ff5c7c"
    fill_color = "rgba(0,212,170,0.1)" if is_up else "rgba(255,92,124,0.1)"

    with chart_tab:
        fig = go.Figure()
        # Baseline line
        fig.add_hline(y=float(close_prices.iloc[0]), line_dash="dash", line_color="#7a8299", line_width=1, opacity=0.5)
        # Main area chart
        fig.add_trace(go.Scatter(
            x=close_prices.index, y=close_prices,
            name="Close", fill='tozeroy',
            line=dict(color=line_color, width=2.2),
            fillcolor=fill_color,
            hovertemplate="₹%{y:,.2f}<extra></extra>"
        ))
        if "SMA20" in extended:
            fig.add_trace(go.Scatter(x=extended.index, y=extended["SMA20"],
                                     name="SMA 20", line=dict(color="#8a84ff", width=1.5, dash="dash")))
        if "SMA50" in extended:
            fig.add_trace(go.Scatter(x=extended.index, y=extended["SMA50"],
                                     name="SMA 50", line=dict(color="#ff9f1c", width=1.5, dash="dash")))
        
        fig.update_layout(
            template="plotly_dark", height=420,
            xaxis=dict(showgrid=True, gridcolor="rgba(122, 130, 153, 0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(122, 130, 153, 0.1)", side="right"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0,r=0,t=10,b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with bb_tab:
        if "BB_Upper" in extended:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Upper"],
                                      name="Upper Band", line=dict(color="rgba(255,92,124,0.3)", dash="dot", width=1)))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Mid"],
                                      name="Middle (SMA20)", line=dict(color="#7a8299", width=1, dash="dash")))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Lower"],
                                      name="Lower Band", line=dict(color="rgba(0,212,170,0.3)", dash="dot", width=1),
                                      fill="tonexty", fillcolor="rgba(122,130,153,0.05)"))
            fig2.update_layout(
                template="plotly_dark", height=400,
                xaxis=dict(gridcolor="rgba(122,130,153,0.05)"),
                yaxis=dict(gridcolor="rgba(122,130,153,0.05)", side="right"),
                margin=dict(l=0,r=0,t=10,b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data for Bollinger Bands.")

    with macd_tab:
        if "MACD" in extended:
            fig3 = go.Figure()
            macd_hist = extended["MACD"] - extended["MACD_Signal"]
            # Modern histogram
            fig3.add_trace(go.Bar(x=extended.index, y=macd_hist, name="Momentum",
                                  marker_color=["rgba(0,212,170,0.4)" if v >= 0 else "rgba(255,92,124,0.4)" for v in macd_hist]))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD"],
                                      name="MACD", line=dict(color="#6c63ff", width=1.5)))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD_Signal"],
                                      name="Signal", line=dict(color="#ffb347", dash="dot", width=1.2)))
            fig3.update_layout(
                template="plotly_dark", height=380,
                xaxis=dict(gridcolor="rgba(122,130,153,0.05)"),
                yaxis=dict(gridcolor="rgba(122,130,153,0.05)", side="right"),
                margin=dict(l=0,r=0,t=10,b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough data for MACD.")

    # 5-day forecast strip
    st.subheader("5-Day Price Forecast")
    today  = datetime.today()
    fdates = [(today + timedelta(days=i+1)).strftime("%a %d %b") for i in range(5)]
    fcols  = st.columns(5)
    for i, (d, p) in enumerate(zip(fdates, forecast_5d)):
        delta = ((p - current_price) / current_price) * 100
        fcols[i].metric(d, f"₹{p:,.2f}", f"{delta:+.2f}%")

    # Volume chart
    if "Volume" in data.columns:
        st.subheader("Trading Volume")
        vol_fig = go.Figure(go.Bar(
            x=data.index, y=data["Volume"],
            marker_color="#6c63ff", opacity=0.7,
        ))
        vol_fig.update_layout(template="plotly_dark", height=200,
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(vol_fig, use_container_width=True)

elif page == "🧠 AI Prediction":

    # Symmetrical 2-column layout
    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        st.markdown("<div class='glass-card' style='height:100%'>", unsafe_allow_html=True)
        
        # Unified Recommendation Badge
        rec = advisor["recommendation"].upper()
        badge_class = "rec-buy" if "BUY" in rec else "rec-sell" if "SELL" in rec else "rec-neutral"
        
        st.markdown(f"""
          <div style="color:var(--muted); font-size:12px; margin-bottom:12px; letter-spacing:1px; font-weight:700">AI CORE DECISION</div>
          <div style="margin-bottom:24px"><span class="rec-badge {badge_class}">{advisor['recommendation']}</span></div>
        """, unsafe_allow_html=True)

        # Factor radar chart
        factors = advisor["factor_scores"]
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=list(factors.values()),
            theta=list(factors.keys()),
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.2)',
            line=dict(color='#00d4aa', width=2),
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(30,37,56,0.3)"),
                angularaxis=dict(gridcolor="rgba(30,37,56,0.3)", linecolor="rgba(30,37,56,0.3)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            height=300,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Risk Metric Strip
        st.markdown("<hr style='opacity:0.1; margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px; color:var(--muted); font-weight:700; margin-bottom:10px'>DECISION CONFIDENCE</div>", unsafe_allow_html=True)
        confidence = advisor.get("confidence", 85)
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.05); height:8px; border-radius:4px; overflow:hidden'>
            <div style='width:{confidence}%; height:100%; background:linear-gradient(90deg, var(--accent), var(--accent2))'></div>
        </div>
        <div style='display:flex; justify-content:space-between; margin-top:6px; font-size:11px; color:var(--muted)'>
            <span>{confidence:.0f}% MODEL CERTAINTY</span>
            <span>INSTITUTIONAL GRADE</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='glass-card' style='height:100%'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:18px; font-weight:700; margin-bottom:20px'>🧠 AI Reasoning & Model Explainability</div>", unsafe_allow_html=True)
        
        # Reasoning text area
        reasons_html = " ".join([f"<div style='margin-bottom:8px'>• {r}</div>" for r in advisor['reasons']])
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.02); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); margin-bottom:24px'>
            <div style='font-size:14px; color:white; line-height:1.6; font-style:italic'>{reasons_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Importance / SHAP Plot
        st.markdown("<div style='font-size:12px; color:var(--muted); font-weight:700; margin-bottom:15px; letter-spacing:0.5px'>FEATURE CONTRIBUTION (MODEL EXPLAINABILITY)</div>", unsafe_allow_html=True)
        
        try:
            # We already have data from the search context
            importance = compute_feature_importance(data)
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            total_imp_val = sum([abs(x[1]) for x in sorted_imp])
            
            for feat, val in sorted_imp:
                norm_val = (abs(val) / total_imp_val) * 100
                st.markdown(f"""
                <div style='margin-bottom:12px'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:5px'>
                        <span style='font-size:12px; font-weight:600; color:var(--muted)'>{feat}</span>
                        <span style='font-size:12px; font-weight:700; color:white'>{norm_val:.1f}%</span>
                    </div>
                    <div style='background:rgba(255,255,255,0.03); height:6px; border-radius:3px'>
                        <div style='width:{norm_val}%; height:100%; background:var(--accent); border-radius:3px; opacity:0.8'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("Explainability metrics are calculating...")
            
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📰 Market Intelligence":
    st.markdown("### Market Intelligence")
    
    # News Source Indicator
    from src.sentiment import has_news_api_key
    if not has_news_api_key():
        st.markdown("""
        <div class='alert-info' style='margin-bottom:24px; border:1px solid rgba(0, 212, 170, 0.2); background:rgba(0, 212, 170, 0.05)'>
            <span style='font-size:18px'>💡</span> <b>Tip</b>: You're currently using basic news feed. Add a <b>NEWS_API_KEY</b> to your .env for institutional-grade global sentiment and deeper news coverage.
        </div>
        """, unsafe_allow_html=True)
    
    col_news, col_alerts = st.columns([1.6, 1])
    
    with col_news:
        st.markdown("<div class='glass-card' style='min-height:580px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:18px; font-weight:700; margin-bottom:20px; display:flex; align-items:center'>📰 Sentiment Analysis & News Feed</div>", unsafe_allow_html=True)
        
        # Horizontal Sentiment Summary
        ss1, ss2, ss3 = st.columns(3)
        ss1.metric("Sentiment Score", f"{sentiment_score:+.3f}")
        ss2.metric("Market Outlook", sentiment_label)
        ss3.metric("Daily Headlines", str(len(headlines)))
        
        st.markdown("<hr style='opacity:0.1; margin:24px 0'>", unsafe_allow_html=True)
        
        if not headlines:
            st.info("No recent headlines found for this asset. Try a major Nifty 50 stock for more coverage.")
        else:
            for h in headlines[:8]:  # Show top 8 news items
                title  = h.get("title", "No Title")
                src = h.get("source", "Market Feed")
                url    = h.get("url", "#")
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.02); padding:16px; border-radius:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.04)'>
                    <div style='font-size:14px; font-weight:600; color:white; line-height:1.4'>{title}</div>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:10px'>
                        <span style='font-size:11px; color:var(--muted); font-weight:500'>📍 SOURCE: {src.upper()}</span>
                        <a href='{url}' target='_blank' style='font-size:11px; color:var(--accent); text-decoration:none; font-weight:700'>VIEW ARTICLE →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_alerts:
        st.markdown("<div class='glass-card' style='min-height:580px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:18px; font-weight:700; margin-bottom:20px'>🛡️ Market Risk Alerts</div>", unsafe_allow_html=True)
        
        alerts = detect_market_alerts(close_prices, sentiment_score)
        if not alerts:
            st.markdown("""
            <div style='text-align:center; padding:50px 20px'>
                <div style='font-size:48px; margin-bottom:20px'>✅</div>
                <div style='color:var(--accent); font-weight:800; font-size:18px; letter-spacing:1px'>STABLE RISK PROFILE</div>
                <div style='font-size:12px; color:var(--muted); margin-top:8px'>No critical volatility or sentiment risk signals detected in the current window.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for a in alerts:
                css_class = {"danger": "alert-danger", "warning": "alert-warning"}.get(a["level"], "alert-info")
                icon      = {"danger": "🚨", "warning": "T INTELLIGENCE
# ─────────────────────────────────────────────────────────────
elif page == "📰 Market Intelligence":
    st.subheader("Market Intelligence")

    # ── Sentiment source badge ────────────────────────────────
    from src.sentiment import has_news_api_key
    if not has_news_api_key():
        st.info(
            "💡 **Tip:** You\'re using yfinance news + VADER sentiment (no setup needed). "
            "For richer, more accurate sentiment analysis, add your free "
            "[NewsAPI key](https://newsapi.org/register) as `NEWS_API_KEY` in your `.env` file.",
            icon=None,
        )
    else:
        st.success(f"✅ Sentiment powered by: **{sentiment_source}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment Score", f"{sentiment_score:+.3f}")
    c2.metric("Sentiment Label", sentiment_label)
    c3.metric("News Articles",   str(len(headlines)))

    # Alerts
    st.markdown("#### Market Alerts")
    alerts = detect_market_alerts(close_prices, sentiment_score)

    if not alerts:
        st.markdown('<div class="alert-ok">✅ No significant risk signals detected.</div>', unsafe_allow_html=True)
    else:
        for a in alerts:
            css_class = {"danger": "alert-danger", "warning": "alert-warning"}.get(a["level"], "alert-info")
            icon      = {"danger": "🚨", "warning": "⚠️"}.get(a["level"], "ℹ️")
            st.markdown(f'<div class="{css_class}">{icon} {a["message"]}</div>', unsafe_allow_html=True)

    # News headlines
    st.markdown("#### Latest News")
    if not headlines:
        st.warning("No headlines found for this stock. This may be a lesser-known ticker — try searching a major index stock.")
    else:
        for h in headlines:
            title  = h.get("title", "No Title")
            source = h.get("source", "Unknown Source")
            url    = h.get("url", "#")
            st.markdown(f"• **{title}**  \n  <small>Source: {source} · [Read more]({url})</small>", unsafe_allow_html=True)

    # RSI chart
    st.markdown("#### RSI (14-day)")
    if "RSI" in create_extended_features(data).columns:
        ext = create_extended_features(data)
        rsi_fig = go.Figure()
        rsi_fig.add_hline(y=70, line=dict(color="#ff5c7c", dash="dot"), annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line=dict(color="#00d4aa", dash="dot"), annotation_text="Oversold")
        rsi_fig.add_trace(go.Scatter(x=ext.index, y=ext["RSI"], name="RSI",
                                     line=dict(color="#6c63ff", width=1.5)))
        rsi_fig.update_layout(template="plotly_dark", height=280,
                              yaxis=dict(range=[0,100]),
                              margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(rsi_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 💼 PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────
elif page == "💼 Portfolio Optimizer":
    st.subheader("Portfolio Optimization")

    from src.stock_search import NSE_STOCKS, resolve_ticker, get_suggestions

    # ── Build deduplicated ticker → label map ──────────────────
    _seen = {}
    for _name, _ticker in NSE_STOCKS.items():
        if _ticker not in _seen or len(_name) > len(_seen[_ticker]):
            _seen[_ticker] = _name.title()

    # ── Session state: selected portfolio stocks ───────────────
    if "pf_stocks" not in st.session_state:
        st.session_state.pf_stocks = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS":      "Tata Consultancy Services",
            "INFY.NS":     "Infosys",
        }
    if "pf_search_query" not in st.session_state:
        st.session_state.pf_search_query = ""
    if "pf_error" not in st.session_state:
        st.session_state.pf_error = ""

    # ── Unified stock picker ───────────────────────────────────
    st.markdown(
        "<div style='font-size:13px;font-weight:600;color:#7a8299;"
        "letter-spacing:.5px;margin-bottom:8px'>📦 BUILD YOUR PORTFOLIO</div>",
        unsafe_allow_html=True,
    )

    # Search row
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        pf_query = st.text_input(
            "Stock search",
            placeholder="Search any NSE stock — name or ticker (e.g. Reliance, IRCTC, HDFCBANK)…",
            value=st.session_state.pf_search_query,
            key="pf_search_input",
            label_visibility="collapsed",
        )
    with s_col2:
        add_clicked = st.button("➕ Add Stock", key="pf_add_btn", use_container_width=True)

    # Handle add button
    if add_clicked and pf_query.strip():
        with st.spinner("Searching…"):
            t, n = resolve_ticker(pf_query.strip())
        if t:
            st.session_state.pf_stocks[t] = n or _seen.get(t, t)
            st.session_state.pf_search_query = ""
            st.session_state.pf_error = ""
            st.rerun()
        else:
            st.session_state.pf_error = f"Could not find \"{pf_query}\" — try a different name or the NSE ticker directly."
            st.rerun()

    if st.session_state.pf_error:
        st.markdown(
            f'<div class="alert-danger">⚠️ {st.session_state.pf_error}</div>',
            unsafe_allow_html=True,
        )

    # Live suggestions while typing
    if pf_query and len(pf_query) >= 2:
        sugg = get_suggestions(pf_query, max_results=5)
        if sugg:
            st.markdown(
                "<div style='font-size:11px;color:#7a8299;margin:6px 0 4px'>"
                "Suggestions — click to add:</div>",
                unsafe_allow_html=True,
            )
            sugg_cols = st.columns(len(sugg))
            q_h = abs(hash(pf_query)) % 100000
            for idx, s in enumerate(sugg):
                already = s["ticker"] in st.session_state.pf_stocks
                label   = f"{'✓ ' if already else ''}{s['name'].split()[0]} ({s['ticker'].replace('.NS','')})"
                if sugg_cols[idx].button(label, key=f"pf_sugg_{q_h}_{idx}", use_container_width=True, disabled=already):
                    st.session_state.pf_stocks[s["ticker"]] = s["name"]
                    st.session_state.pf_error = ""
                    st.rerun()

    # ── Quick-add popular sectors ──────────────────────────────
    st.markdown(
        "<div style='font-size:11px;color:#7a8299;margin:14px 0 6px'>"
        "⚡ Quick add by sector:</div>",
        unsafe_allow_html=True,
    )
    SECTORS = {
        "🏦 Banking":   [("HDFCBANK.NS","HDFC Bank"),("ICICIBANK.NS","ICICI Bank"),("SBIN.NS","SBI"),("AXISBANK.NS","Axis Bank")],
        "💻 IT":        [("TCS.NS","TCS"),("INFY.NS","Infosys"),("WIPRO.NS","Wipro"),("HCLTECH.NS","HCL Tech")],
        "🛒 Consumer":  [("HINDUNILVR.NS","HUL"),("ITC.NS","ITC"),("NESTLEIND.NS","Nestle"),("BRITANNIA.NS","Britannia")],
        "💊 Pharma":    [("SUNPHARMA.NS","Sun Pharma"),("DRREDDY.NS","Dr Reddy's"),("CIPLA.NS","Cipla"),("DIVISLAB.NS","Divi's")],
        "⚡ Energy":    [("RELIANCE.NS","Reliance"),("ONGC.NS","ONGC"),("NTPC.NS","NTPC"),("POWERGRID.NS","Power Grid")],
    }
    sec_cols = st.columns(len(SECTORS))
    for col, (sector_label, stocks_in_sector) in zip(sec_cols, SECTORS.items()):
        with col:
            if col.button(sector_label, key=f"sec_{sector_label}", use_container_width=True):
                for t, n in stocks_in_sector:
                    st.session_state.pf_stocks[t] = n
                st.rerun()

    # ── Selected stocks chips ──────────────────────────────────
    if st.session_state.pf_stocks:
        st.markdown(
            "<div style='font-size:11px;color:#7a8299;margin:16px 0 8px'>"
            f"Selected stocks ({len(st.session_state.pf_stocks)}) — click ✕ to remove:</div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(min(len(st.session_state.pf_stocks), 4))
        for idx, (t, n) in enumerate(list(st.session_state.pf_stocks.items())):
            short = t.replace(".NS", "")
            with chip_cols[idx % 4]:
                st.markdown(
                    f"<div style='background:#1e2538;border:1px solid #2a3050;"
                    f"border-radius:8px;padding:8px 10px;margin-bottom:6px'>"
                    f"<div style='font-size:12px;font-weight:600;color:#e4e8f0'>{n.split()[0]}</div>"
                    f"<div style='font-size:10px;color:#7a8299'>{short}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("✕ Remove", key=f"rm_{idx}_{t.replace('.','_')}", use_container_width=True):
                    del st.session_state.pf_stocks[t]
                    st.rerun()
    else:
        st.info("No stocks selected yet. Search above or use quick-add buttons.")

    st.markdown("---")

    sel_stocks = list(st.session_state.pf_stocks.keys())

    col_a, col_b = st.columns(2)
    investment  = col_a.number_input("Investment Amount (₹)", min_value=1000, value=100_000, step=5000)
    risk_level  = col_b.selectbox("Risk Appetite", ["Low", "Medium", "High"],
                                   help="Low = stable/diversified · Medium = Sharpe optimized · High = momentum concentrated")

    if len(sel_stocks) < 2:
        st.warning("Select at least 2 stocks to build a portfolio.")
        st.stop()

    with st.spinner("Fetching portfolio data…"):
        pf_data = load_portfolio_data(tuple(sel_stocks))

    if pf_data is None or pf_data.empty:
        st.error(
            "Could not load portfolio data. This can happen when: "
            "(1) one or more tickers are invalid, or "
            "(2) Yahoo Finance is temporarily unavailable. "
            "Try removing recently-listed stocks or click Retry below."
        )
        if st.button("🔄 Retry", key="pf_retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # Show a notice if fewer stocks loaded than selected
    loaded_stocks = list(pf_data.columns)
    failed_stocks = [s for s in sel_stocks if s not in loaded_stocks]
    if failed_stocks:
        st.warning(
            f"⚠️ Could not load data for: {', '.join(failed_stocks)}. "
            "Continuing with the remaining stocks."
        )

    pf_returns = calculate_returns(pf_data)
    pf_returns = pf_returns.dropna(axis=1, how="all")

    if pf_returns.shape[1] < 2:
        st.error("Need at least 2 stocks with valid data. Please add more stocks.")
        st.stop()

    allocation  = recommend_portfolio(pf_returns, investment, risk_level)

    # ── Strategy explanation badge ─────────────────────────────
    strategy_info = {
        "Low":    "⚖️ **Inverse-Volatility Weighted** — lower-risk stocks receive higher allocation",
        "Medium": "📐 **Sharpe-Ratio Optimized** — weights chosen to maximise risk-adjusted return",
        "High":   "🚀 **Momentum Weighted** — higher recent returns receive more capital",
    }
    st.info(strategy_info[risk_level])

    # ── Pie chart ──────────────────────────────────────────────
    pie_fig = go.Figure(go.Pie(
        labels=[f"{st.session_state.pf_stocks.get(s, _seen.get(s, s)).split()[0]} ({s.replace('.NS','')})"
                for s in allocation["Stock"]],
        values=allocation["Amount (₹)"],
        hole=0.45,
        marker=dict(colors=px.colors.qualitative.Vivid),
        textinfo="label+percent",
    ))
    pie_fig.update_layout(
        template="plotly_dark", height=380,
        title=f"{risk_level}-Risk Portfolio Allocation",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with c2:
        st.markdown("**Allocation Breakdown**")
        # Rename Stock column to readable name for display
        display_alloc = allocation.copy()
        display_alloc["Stock"] = display_alloc["Stock"].apply(
            lambda s: f"{st.session_state.pf_stocks.get(s, _seen.get(s, s)).split()[0]}"
        )
        st.dataframe(
            display_alloc.style.format({
                "Weight (%)":      "{:.2f}%",
                "Amount (₹)":      "₹{:,.0f}",
                "Exp. Return (%)": "{:.2f}%",
                "Volatility (%)":  "{:.2f}%",
                "Sharpe":          "{:.2f}",
            }).background_gradient(subset=["Weight (%)"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

    # Efficient Frontier
    st.markdown("#### Efficient Frontier (Monte Carlo)")
    with st.spinner("Simulating portfolios…"):
        ef_df = generate_efficient_frontier(pf_returns, num_portfolios=1500)

    ef_fig = px.scatter(
        ef_df, x="Volatility (%)", y="Return (%)",
        color="Sharpe", color_continuous_scale="Teal",
        opacity=0.6, height=400,
        labels={"Sharpe": "Sharpe Ratio"},
    )
    ef_fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(ef_fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Matrix")
    corr = pf_returns.corr()
    heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        height=350,
    )
    heat.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 🔁 BACKTESTING
# ─────────────────────────────────────────────────────────────
elif page == "🔁 Backtesting":
    st.subheader("Strategy Backtesting")

    with st.spinner("Running backtest…"):
        comparison, metrics = run_backtest(stock)

    # Strategy chart
    colors = {"Buy & Hold": "#7a8299", "AI Strategy": "#00d4aa",
              "Momentum": "#6c63ff", "Mean Reversion": "#ffb347"}

    bt_fig = go.Figure()
    for col in comparison.columns:
        bt_fig.add_trace(go.Scatter(
            x=comparison.index, y=comparison[col],
            name=col, line=dict(color=colors.get(col, "#fff"), width=1.8),
        ))
    bt_fig.update_layout(
        template="plotly_dark", height=420,
        yaxis_title="Cumulative Return (×)",
        xaxis_title="Date",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(bt_fig, use_container_width=True)

    # Drawdown chart
    from src.backtest import build_drawdown_fig, build_monthly_heatmap
    st.plotly_chart(build_drawdown_fig(comparison), use_container_width=True)

    # Metrics table
    st.markdown("#### Strategy Performance")
    rows = []
    for strat, m in metrics.items():
        rows.append({
            "Strategy":     strat,
            "Total Return": f"{m['total_return']:+.2f}%",
            "Max Drawdown": f"{m['max_drawdown']:.2f}%",
            "Sharpe Ratio": f"{m['sharpe_ratio']:.3f}",
            "Win Rate":     f"{m['win_rate']:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    # Monthly heatmap for AI Strategy
    if "AI Strategy" in comparison.columns:
        st.markdown("#### AI Strategy Monthly Returns")
        st.plotly_chart(build_monthly_heatmap(comparison["AI Strategy"], "AI Strategy"), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 🔬 MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Model Evaluation":
    st.subheader("GradientBoosting Model Evaluation")

    with st.spinner("Evaluating model…"):
        evaluation = evaluate_model(close_prices, model, scaler)

    st.plotly_chart(evaluation["chart"], use_container_width=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Train MAE",  f"₹{evaluation.get('train_mae', 0):.2f}")
    e2.metric("Test MAE",   f"₹{evaluation.get('test_mae', 0):.2f}")
    e3.metric("MAPE",       f"{evaluation.get('mape', 0):.2f}%")
    e4.metric("Dir. Accuracy", f"{evaluation.get('dir_acc', 0):.1f}%")

    st.markdown("#### Model Confidence")
    st.progress(int(evaluation["confidence"]) / 100)

    # Residuals
    st.plotly_chart(evaluation["residual_chart"], use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 🚀 IPO PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🚀 IPO Predictor":

    st.markdown("""
    <div style='margin-bottom:24px'>
        <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                   background:linear-gradient(135deg,#00d4aa,#6c63ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            🚀 IPO Success Predictor
        </h2>
        <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
            Enter IPO details to predict listing gain, 30-day return &amp; subscription recommendation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────
    with st.spinner("Loading IPO prediction model…"):
        ipo_models = load_ipo_model()

    if ipo_models is None:
        st.error(
            "⚠️ IPO training data not found. "
            "Make sure `data/raw/ipo_historical.csv` is in the repository."
        )
        st.stop()

    st.caption(
        f"Model trained on **{ipo_models['n_train']} IPOs** (2015–2025) · "
        f"Cross-val MAE: **{ipo_models['mae_cv']:.1f}%**"
    )

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────
    st.markdown("#### 📋 IPO Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        ipo_name    = st.text_input("IPO Name", placeholder="e.g. Swiggy, Hyundai India")
        issue_price = st.number_input("Issue Price (₹)", min_value=1, max_value=50000,
                                       value=500, step=10)
        sector      = st.selectbox("Sector", sorted(SECTORS))

    with col2:
        sub_total  = st.number_input("Total Subscription (×)",
                                      min_value=0.0, max_value=500.0,
                                      value=10.0, step=0.5,
                                      help="e.g. 10.5 means 10.5× oversubscribed")
        sub_qib    = st.number_input("QIB Subscription (×)",
                                      min_value=0.0, max_value=500.0,
                                      value=15.0, step=0.5,
                                      help="Qualified Institutional Buyers")
        sub_hni    = st.number_input("HNI Subscription (×)",
                                      min_value=0.0, max_value=500.0,
                                      value=8.0, step=0.5,
                                      help="High Net Worth Individuals")

    with col3:
        sub_retail = st.number_input("Retail Subscription (×)",
                                      min_value=0.0, max_value=200.0,
                                      value=5.0, step=0.5)
        gmp        = st.number_input("GMP % (Grey Market Premium)",
                                      min_value=-50.0, max_value=200.0,
                                      value=15.0, step=1.0,
                                      help="Grey market premium as % of issue price. "
                                           "Negative = grey market discount.")
        nifty_trend = st.number_input("NIFTY 30-day Trend %",
                                       min_value=-30.0, max_value=30.0,
                                       value=get_current_nifty_trend(),
                                       step=0.5,
                                       help="Auto-filled with current NIFTY trend. "
                                            "You can override manually.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict IPO Performance", type="primary")

    # ── Prediction Output ─────────────────────────────────────
    if predict_btn:
        with st.spinner("Analysing IPO…"):
            result = predict_ipo(
                ipo_models,
                sub_total=sub_total,
                sub_qib=sub_qib,
                sub_hni=sub_hni,
                sub_retail=sub_retail,
                gmp_percent=gmp,
                nifty_trend=nifty_trend,
                issue_price=issue_price,
                sector=sector,
            )

        st.markdown("---")

        # ── Recommendation badge ──────────────────────────────
        rec   = result["recommendation"]
        conf  = result["confidence"]
        color_map = {
            "Strong Subscribe": "#00d4aa",
            "Subscribe":        "#6c63ff",
            "Neutral":          "#ffb347",
            "Avoid":            "#ff6b6b",
        }
        icon_map = {
            "Strong Subscribe": "🟢",
            "Subscribe":        "🔵",
            "Neutral":          "🟡",
            "Avoid":            "🔴",
        }
        rec_color = color_map.get(rec, "#7a8299")
        rec_icon  = icon_map.get(rec, "⚪")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{rec_color}22,{rec_color}11);
                    border:1px solid {rec_color}55;border-radius:16px;
                    padding:24px 32px;text-align:center;margin-bottom:24px'>
            <div style='font-size:3rem;margin-bottom:8px'>{rec_icon}</div>
            <div style='font-size:1.8rem;font-weight:800;color:{rec_color}'>
                {rec}
            </div>
            <div style='color:#7a8299;font-size:0.9rem;margin-top:4px'>
                Model confidence: <strong style='color:#e4e8f0'>{conf:.1f}%</strong>
            </div>
            {"<div style='color:#e4e8f0;font-size:1rem;margin-top:8px'>for <strong>" + ipo_name + "</strong></div>" if ipo_name else ""}
        </div>
        """, unsafe_allow_html=True)

        # ── KPI metrics ───────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        listing_gain = result["listing_gain"]
        return_30d   = result["return_30d"]
        total_return = listing_gain + return_30d

        k1.metric("📈 Listing Day Gain",
                  f"{listing_gain:+.1f}%",
                  delta=f"vs issue ₹{issue_price}")
        k2.metric("📅 30-Day Return",
                  f"{return_30d:+.1f}%",
                  delta="post listing")
        k3.metric("💰 Expected Price",
                  f"₹{issue_price * (1 + listing_gain/100):.0f}",
                  delta=f"on listing day")
        k4.metric("🎯 Total 30-Day",
                  f"{total_return:+.1f}%",
                  delta="from issue price")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability breakdown ─────────────────────────────
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Outcome Probabilities**")
            proba_order = ["Strong Subscribe", "Subscribe", "Neutral", "Avoid"]
            for label in proba_order:
                prob  = result["probabilities"].get(label, 0)
                color = color_map.get(label, "#7a8299")
                st.markdown(f"""
                <div style='margin-bottom:8px'>
                    <div style='display:flex;justify-content:space-between;
                                margin-bottom:3px;font-size:13px'>
                        <span style='color:#e4e8f0'>{label}</span>
                        <span style='color:{color};font-weight:600'>{prob:.1f}%</span>
                    </div>
                    <div style='background:#1e2538;border-radius:4px;height:6px'>
                        <div style='background:{color};width:{prob}%;
                                    height:6px;border-radius:4px'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**Key Prediction Drivers**")
            feat_labels = {
                "sub_total":       "Total Subscription",
                "sub_qib":         "QIB Subscription",
                "sub_hni":         "HNI Subscription",
                "sub_retail":      "Retail Subscription",
                "gmp_percent":     "Grey Market Premium",
                "nifty_trend_30d": "Market Trend (NIFTY)",
                "issue_price_log": "Issue Price",
            }
            for feat, imp in result["feature_importance"]:
                label = feat_labels.get(feat, feat.replace("sector_","").replace("_"," "))
                bar   = int(imp * 400)
                st.markdown(f"""
                <div style='margin-bottom:8px'>
                    <div style='display:flex;justify-content:space-between;
                                margin-bottom:3px;font-size:13px'>
                        <span style='color:#e4e8f0'>{label}</span>
                        <span style='color:#00d4aa;font-weight:600'>{imp:.1%}</span>
                    </div>
                    <div style='background:#1e2538;border-radius:4px;height:6px'>
                        <div style='background:#00d4aa;width:{bar}%;
                                    height:6px;border-radius:4px'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Similar past IPOs ─────────────────────────────────
        st.markdown("**📚 Similar Past IPOs for Reference**")
        similar = find_similar_ipos(sub_total, gmp, sector, n=4)

        if not similar.empty:
            sim_cols = st.columns(len(similar))
            for i, (_, row) in enumerate(similar.iterrows()):
                rec_c = color_map.get(row["recommendation"], "#7a8299")
                sim_cols[i].markdown(f"""
                <div style='background:#111520;border:1px solid #1e2538;
                            border-radius:12px;padding:14px;text-align:center'>
                    <div style='font-weight:600;color:#e4e8f0;
                                font-size:13px;margin-bottom:6px'>
                        {row["name"]}
                    </div>
                    <div style='font-size:11px;color:#7a8299;margin-bottom:8px'>
                        ₹{row["issue_price"]} · {row["sub_total"]:.0f}×
                    </div>
                    <div style='font-size:1.1rem;font-weight:700;color:{rec_c}'>
                        {row["listing_gain"]:+.1f}%
                    </div>
                    <div style='font-size:10px;color:#7a8299;margin-top:2px'>
                        listing gain
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "⚠️ **Disclaimer:** These predictions are based on historical patterns "
            "and machine learning models. IPO performance depends on many factors "
            "including market conditions, company fundamentals, and investor sentiment. "
            "This is not financial advice. Always do your own research.",
            icon=None,
        )


# ══════════════════════════════════════════════════════════════
# 📡 FII/DII TRACKER
# ══════════════════════════════════════════════════════════════
elif page == "📡 FII/DII Tracker":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("""
    <div style='margin-bottom:24px'>
        <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                   background:linear-gradient(135deg,#00d4aa,#6c63ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            📡 FII / DII Activity Tracker
        </h2>
        <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
            Live foreign & domestic institutional flows · Pattern detection · NIFTY impact prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data & model ─────────────────────────────────────
    with st.spinner("Loading FII/DII data…"):
        fii_df     = load_fii_dii_data()
        fii_models = load_fii_dii_model()

    if fii_df is None or fii_df.empty:
        st.error("⚠️ FII/DII data not found. Make sure `data/raw/fii_dii_historical.csv` is in the repo.")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Data", "📈 Trends", "🔍 Patterns", "🤖 AI Prediction"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — LIVE DATA
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### Today's FII/DII Activity")

        live = fetch_live_fii_dii()

        if live:
            st.caption(f"Latest data: **{live['date']}** · Refreshes every 30 min")

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            fii_color = "#00d4aa" if live["fii_net"] >= 0 else "#ff6b6b"
            dii_color = "#00d4aa" if live["dii_net"] >= 0 else "#ff6b6b"

            k1.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>FII NET</div>
                <div style='font-size:1.5rem;font-weight:700;color:{fii_color}'>
                    {"+" if live["fii_net"]>=0 else ""}₹{live["fii_net"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)

            k2.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>DII NET</div>
                <div style='font-size:1.5rem;font-weight:700;color:{dii_color}'>
                    {"+" if live["dii_net"]>=0 else ""}₹{live["dii_net"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)

            k3.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>FII BUY</div>
                <div style='font-size:1.5rem;font-weight:700;color:#e4e8f0'>
                    ₹{live["fii_buy"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)

            k4.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>DII BUY</div>
                <div style='font-size:1.5rem;font-weight:700;color:#e4e8f0'>
                    ₹{live["dii_buy"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)

            # Last 10 days table
            if live.get("all_days"):
                st.markdown("<br>**Last 10 Trading Days**", unsafe_allow_html=True)
                rows = []
                for d in live["all_days"]:
                    try:
                        fn = float(str(d.get("fiiNetValue","0")).replace(",",""))
                        dn = float(str(d.get("diiNetValue","0")).replace(",",""))
                        rows.append({
                            "Date":     d.get("date",""),
                            "FII Net (₹Cr)": f'{"+" if fn>=0 else ""}₹{fn:,.0f}',
                            "DII Net (₹Cr)": f'{"+" if dn>=0 else ""}₹{dn:,.0f}',
                            "Combined":      f'{"+" if fn+dn>=0 else ""}₹{fn+dn:,.0f}',
                            "Signal": "🟢 Bullish" if fn+dn > 0 else "🔴 Bearish",
                        })
                    except Exception:
                        continue
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)
        else:
            # Fallback to last 10 rows of historical data
            st.info("💡 Live NSE data temporarily unavailable — showing latest historical data. NSE publishes data after market close (3:30 PM IST) on trading days.")
            recent_hist = fii_df.dropna(subset=["fii_net","dii_net"]).tail(10).copy()
            last = recent_hist.iloc[-1]
            fii_c = "#00d4aa" if last["fii_net"] >= 0 else "#ff6b6b"
            dii_c = "#00d4aa" if last["dii_net"] >= 0 else "#ff6b6b"
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>FII NET (Latest)</div>
                <div style='font-size:1.5rem;font-weight:700;color:{fii_c}'>
                    {"+" if last["fii_net"]>=0 else ""}₹{last["fii_net"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>DII NET (Latest)</div>
                <div style='font-size:1.5rem;font-weight:700;color:{dii_c}'>
                    {"+" if last["dii_net"]>=0 else ""}₹{last["dii_net"]:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)
            combined = last["fii_net"] + last["dii_net"]
            comb_c = "#00d4aa" if combined >= 0 else "#ff6b6b"
            k3.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>COMBINED</div>
                <div style='font-size:1.5rem;font-weight:700;color:{comb_c}'>
                    {"+" if combined>=0 else ""}₹{combined:,.0f} Cr
                </div></div>""", unsafe_allow_html=True)
            k4.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:16px;text-align:center'>
                <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>AS OF DATE</div>
                <div style='font-size:1rem;font-weight:700;color:#e4e8f0'>
                    {str(last["date"])[:10]}
                </div></div>""", unsafe_allow_html=True)

            # Show last 10 days table from historical data
            st.markdown("<br>**Last 10 Trading Days (Historical)**", unsafe_allow_html=True)
            display = recent_hist[["date","fii_net","dii_net"]].copy()
            display["combined"] = display["fii_net"] + display["dii_net"]
            display["signal"]   = display["combined"].apply(
                lambda x: "🟢 Bullish" if x > 0 else "🔴 Bearish"
            )
            display.columns = ["Date","FII Net (₹Cr)","DII Net (₹Cr)","Combined (₹Cr)","Signal"]
            display = display.sort_values("Date", ascending=False)
            st.dataframe(display.style.format({
                "FII Net (₹Cr)":  "{:,.0f}",
                "DII Net (₹Cr)":  "{:,.0f}",
                "Combined (₹Cr)": "{:,.0f}",
            }), hide_index=True, use_container_width=True)

        # ── Correlation stats ─────────────────────────────────
        st.markdown("<br>**FII/DII → NIFTY Correlations**", unsafe_allow_html=True)
        corr = correlation_analysis(fii_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        def corr_color(v):
            return "#00d4aa" if v > 0.1 else ("#ff6b6b" if v < -0.1 else "#ffb347")
        for col, label, val in [
            (c1, "FII → Same Day NIFTY",  corr["fii_same_day"]),
            (c2, "FII → Next Day NIFTY",  corr["fii_next_day"]),
            (c3, "DII → Same Day NIFTY",  corr["dii_same_day"]),
            (c4, "DII → Next Day NIFTY",  corr["dii_next_day"]),
            (c5, "FII vs DII Corr",        corr["fii_dii_corr"]),
        ]:
            col.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:10px;padding:12px;text-align:center'>
                <div style='font-size:10px;color:#7a8299;margin-bottom:4px'>{label}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{corr_color(val)}'>{val:+.3f}</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — TRENDS
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### FII/DII Monthly Flow vs NIFTY Returns")

        monthly = monthly_summary(fii_df)

        # Combined bar + line chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["fii_net"] / 100,
            name="FII Net (₹100Cr)",
            marker_color=["#00d4aa" if v >= 0 else "#ff6b6b"
                          for v in monthly["fii_net"]],
            opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["dii_net"] / 100,
            name="DII Net (₹100Cr)",
            marker_color=["#6c63ff" if v >= 0 else "#ffb347"
                          for v in monthly["dii_net"]],
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["nifty_return"],
            name="NIFTY Monthly Return %",
            mode="lines+markers",
            line=dict(color="#ffffff", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            template="plotly_dark",
            barmode="group",
            height=420,
            yaxis=dict(title="Flow (₹100 Cr)"),
            yaxis2=dict(title="NIFTY Return %", overlaying="y",
                        side="right", showgrid=False),
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 30-day rolling FII net
        st.markdown("#### Rolling 30-Day FII Net Flow")
        recent = fii_df.tail(252).copy()
        recent["fii_30d_roll"] = recent["fii_net"].rolling(30).sum()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=recent["date"], y=recent["fii_30d_roll"],
            mode="lines", name="30-day rolling FII net",
            line=dict(width=2, color="#00d4aa"),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.1)",
        ))
        fig2.add_hline(y=0, line=dict(color="#7a8299", dash="dot"))
        fig2.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=40, r=20, t=20, b=30),
            yaxis_title="₹ Crores (30-day sum)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — PATTERNS
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 🔍 Statistically Proven FII/DII Patterns")
        st.caption(f"Analysed from {len(fii_df)} trading days of data")

        patterns = detect_patterns(fii_df)

        if not patterns:
            st.info("Not enough data to detect patterns.")
        else:
            for p in patterns:
                sig_color = ("#00d4aa" if p["signal"] == "Bullish"
                             else "#ff6b6b" if p["signal"] == "Bearish"
                             else "#ffb347")
                sig_icon  = ("🟢" if p["signal"] == "Bullish"
                             else "🔴" if p["signal"] == "Bearish" else "🟡")

                with st.expander(
                    f"{sig_icon} **{p['name']}** · "
                    f"Win Rate: {p['win_rate']:.0f}% · "
                    f"Avg Next-Day: {p['avg_return']:+.2f}%",
                    expanded=False,
                ):
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Signal",       p["signal"])
                    pc2.metric("Win Rate",     f"{p['win_rate']:.1f}%")
                    pc3.metric("Avg Return",   f"{p['avg_return']:+.2f}%")
                    pc4.metric("Sample Size",  f"{p['sample_size']} days")

                    st.markdown(
                        f"<div style='color:#7a8299;font-size:13px;margin-top:4px'>"
                        f"📋 {p['description']} · "
                        f"Std dev: {p['std']:.2f}%</div>",
                        unsafe_allow_html=True
                    )

                    # Mini bar showing win rate
                    wr = p["win_rate"]
                    st.markdown(f"""
                    <div style='margin-top:10px'>
                        <div style='font-size:11px;color:#7a8299;margin-bottom:4px'>
                            Win rate: {wr:.0f}% bullish outcomes
                        </div>
                        <div style='background:#1e2538;border-radius:4px;height:8px'>
                            <div style='background:{sig_color};width:{wr}%;
                                        height:8px;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — AI PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 🤖 AI Prediction: Tomorrow's NIFTY Direction")

        if fii_models is None:
            st.error("Model could not be trained — insufficient data.")
        else:
            st.caption(
                f"Model trained on **{fii_models['n_train']} trading days** · "
                f"Cross-val accuracy: **{fii_models['cv_accuracy']:.1f}%**"
            )

            prediction = predict_next_day(fii_models, fii_df)

            if prediction:
                direction = prediction["direction"]
                conf      = prediction["confidence"]
                mag       = prediction["magnitude"]
                prob_up   = prediction["prob_up"]
                prob_down = prediction["prob_down"]

                pred_color = "#00d4aa" if direction == "Bullish" else "#ff6b6b"
                pred_icon  = "📈" if direction == "Bullish" else "📉"

                # Big prediction card
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,{pred_color}22,{pred_color}11);
                            border:1px solid {pred_color}55;border-radius:16px;
                            padding:28px;text-align:center;margin:16px 0'>
                    <div style='font-size:3rem'>{pred_icon}</div>
                    <div style='font-size:2rem;font-weight:800;color:{pred_color};margin:8px 0'>
                        {direction}
                    </div>
                    <div style='color:#7a8299;font-size:0.9rem'>
                        Expected move: <strong style='color:#e4e8f0'>{mag:+.2f}%</strong> ·
                        Confidence: <strong style='color:#e4e8f0'>{conf:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                pa, pb = st.columns(2)
                with pa:
                    st.markdown(f"""
                    <div style='background:#111520;border:1px solid #1e2538;
                                border-radius:12px;padding:16px;text-align:center'>
                        <div style='font-size:12px;color:#7a8299;margin-bottom:8px'>
                            📈 Bullish Probability
                        </div>
                        <div style='font-size:1.6rem;font-weight:700;color:#00d4aa'>
                            {prob_up:.1f}%
                        </div>
                        <div style='background:#1e2538;border-radius:4px;
                                    height:8px;margin-top:8px'>
                            <div style='background:#00d4aa;width:{prob_up}%;
                                        height:8px;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with pb:
                    st.markdown(f"""
                    <div style='background:#111520;border:1px solid #1e2538;
                                border-radius:12px;padding:16px;text-align:center'>
                        <div style='font-size:12px;color:#7a8299;margin-bottom:8px'>
                            📉 Bearish Probability
                        </div>
                        <div style='font-size:1.6rem;font-weight:700;color:#ff6b6b'>
                            {prob_down:.1f}%
                        </div>
                        <div style='background:#1e2538;border-radius:4px;
                                    height:8px;margin-top:8px'>
                            <div style='background:#ff6b6b;width:{prob_down}%;
                                        height:8px;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Current FII/DII context
                st.markdown("<br>**Current FII/DII Context (last 5 days)**",
                            unsafe_allow_html=True)
                recent5 = fii_df.tail(5)[
                    ["date", "fii_net", "dii_net",
                     "fii_dii_net", "nifty_return"]
                ].copy()
                recent5.columns = [
                    "Date", "FII Net (₹Cr)", "DII Net (₹Cr)",
                    "Combined (₹Cr)", "NIFTY Return %"
                ]
                recent5 = recent5.sort_values("Date", ascending=False)
                st.dataframe(
                    recent5.style.format({
                        "FII Net (₹Cr)":    "{:,.0f}",
                        "DII Net (₹Cr)":    "{:,.0f}",
                        "Combined (₹Cr)":   "{:,.0f}",
                        "NIFTY Return %":   "{:.2f}%",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

            # Disclaimer
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "⚠️ **Disclaimer:** FII/DII flow predictions are based on "
                "historical patterns and do not guarantee future market movement. "
                "Markets are influenced by many factors beyond institutional flows. "
                "This is not financial advice.",
                icon=None,
            )


# ══════════════════════════════════════════════════════════════
# 📸 CHART ANALYZER
# ══════════════════════════════════════════════════════════════
elif page == "📸 Chart Analyzer":

    st.markdown("""
    <div style='margin-bottom:24px'>
        <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                   background:linear-gradient(135deg,#00d4aa,#6c63ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            📸 AI Chart Analyzer
        </h2>
        <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
            Upload any stock chart screenshot — get instant AI technical analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── API key check ─────────────────────────────────────────
    if not get_api_key():
        st.info(
            "💡 **Running in rule-based mode** — no Claude API key detected. "
            "Analysis will use image processing instead of AI vision. "
            "Add `ANTHROPIC_API_KEY` to Streamlit secrets for full AI analysis.",
            icon=None,
        )

    # ── Upload section ────────────────────────────────────────
    st.markdown("#### 📤 Upload Chart Screenshot")
    st.caption("Supports screenshots from Zerodha, Groww, TradingView, Google Finance, NSE, or any charting platform")

    col_up, col_ctx = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Drop chart screenshot here",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a screenshot of any stock chart",
            label_visibility="collapsed",
        )

    with col_ctx:
        extra_context = st.text_area(
            "Optional context",
            placeholder="e.g. This is Reliance Industries daily chart, I am a swing trader looking for entry points...",
            height=120,
            help="Give the AI extra context for better analysis",
        )

    if uploaded:
        # Show uploaded image
        st.image(uploaded, caption=f"Uploaded: {uploaded.name}",
                 use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍 Analyse Chart", type="primary")

        if analyse_btn:
            with st.spinner("🧠 Claude is analysing your chart… (10-20 seconds)"):
                image_bytes = uploaded.read()
                result      = analyse_chart(image_bytes, uploaded.name, extra_context)

            # ── Error handling ────────────────────────────────
            if "error" in result:
                st.error(f"❌ {result['error']}")
                st.stop()

            # ── Fallback banner ───────────────────────────────
            if result.get("_fallback"):
                reason = result.get("_fallback_reason", "")
                st.warning(
                    f"⚡ **Rule-based analysis mode** — {reason}\n\n"
                    "Add Claude API credits at console.anthropic.com for "
                    "full AI-powered analysis with precise price levels, "
                    "exact pattern names, and RSI/MACD readings.",
                    icon=None,
                )

            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # ── Header: stock name + timeframe ────────────────
            stock_name = result.get("stock_name", "Unknown")
            timeframe  = result.get("timeframe",  "Unknown")
            cur_price  = result.get("current_price")

            h1, h2, h3 = st.columns(3)
            h1.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:14px;text-align:center'>
                <div style='font-size:11px;color:#7a8299'>STOCK</div>
                <div style='font-size:1.1rem;font-weight:700;color:#e4e8f0'>
                    {stock_name}</div></div>""", unsafe_allow_html=True)
            h2.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:14px;text-align:center'>
                <div style='font-size:11px;color:#7a8299'>TIMEFRAME</div>
                <div style='font-size:1.1rem;font-weight:700;color:#e4e8f0'>
                    {timeframe}</div></div>""", unsafe_allow_html=True)
            h3.markdown(f"""<div style='background:#111520;border:1px solid #1e2538;
                border-radius:12px;padding:14px;text-align:center'>
                <div style='font-size:11px;color:#7a8299'>PRICE</div>
                <div style='font-size:1.1rem;font-weight:700;color:#e4e8f0'>
                    {f"₹{cur_price}" if cur_price else "—"}</div></div>""",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Recommendation (big card) ─────────────────────
            rec    = result.get("recommendation", {})
            action = rec.get("action", "Hold")
            conf   = rec.get("confidence", 50)
            color, icon = rec_style(action)

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}22,{color}11);
                        border:2px solid {color}55;border-radius:16px;
                        padding:24px 32px;text-align:center;margin-bottom:24px'>
                <div style='font-size:2.5rem'>{icon}</div>
                <div style='font-size:2rem;font-weight:800;color:{color};margin:6px 0'>
                    {action}
                </div>
                <div style='color:#7a8299;font-size:0.85rem'>
                    AI Confidence: <strong style='color:#e4e8f0'>{conf}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.progress(conf / 100)

            # Entry / SL / Target
            r1, r2, r3 = st.columns(3)
            r1.metric("📍 Entry Zone",  rec.get("entry_zone",  "—"))
            r2.metric("🛡️ Stop Loss",   rec.get("stop_loss",   "—"))
            r3.metric("🎯 Target",       rec.get("target",      "—"))

            st.markdown(
                f"<div style='background:#111520;border-left:3px solid {color};"
                f"padding:12px 16px;border-radius:0 8px 8px 0;margin:16px 0;"
                f"color:#e4e8f0;font-size:0.9rem'>{rec.get('rationale','')}</div>",
                unsafe_allow_html=True
            )

            st.markdown("---")

            # ── Trend + Momentum ──────────────────────────────
            col_t, col_m = st.columns(2)

            with col_t:
                trend = result.get("trend", {})
                t_dir = trend.get("direction", "Unknown")
                t_str = trend.get("strength",  "Unknown")
                t_col = ("#00d4aa" if t_dir == "Bullish"
                         else "#ff6b6b" if t_dir == "Bearish" else "#ffb347")
                st.markdown(f"""
                <div style='background:#111520;border:1px solid #1e2538;
                            border-radius:12px;padding:18px;height:160px'>
                    <div style='font-size:12px;color:#7a8299;margin-bottom:8px'>
                        📈 TREND
                    </div>
                    <div style='font-size:1.4rem;font-weight:700;color:{t_col}'>
                        {trend_icon(t_dir)} {t_dir}
                    </div>
                    <div style='font-size:12px;color:#ffb347;margin:4px 0'>
                        Strength: {t_str}
                    </div>
                    <div style='font-size:12px;color:#7a8299;margin-top:6px'>
                        {trend.get("description","")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_m:
                mom    = result.get("momentum", {})
                m_sig  = mom.get("signal", "Neutral")
                m_col  = ("#00d4aa" if m_sig == "Bullish"
                          else "#ff6b6b" if m_sig == "Bearish" else "#ffb347")
                rsi    = mom.get("rsi_value")
                macd   = mom.get("macd_signal", "")
                st.markdown(f"""
                <div style='background:#111520;border:1px solid #1e2538;
                            border-radius:12px;padding:18px;height:160px'>
                    <div style='font-size:12px;color:#7a8299;margin-bottom:8px'>
                        ⚡ MOMENTUM
                    </div>
                    <div style='font-size:1.4rem;font-weight:700;color:{m_col}'>
                        {m_sig}
                    </div>
                    <div style='font-size:12px;color:#7a8299;margin:4px 0'>
                        {f"RSI: {rsi}" if rsi else "RSI: not visible"} ·
                        {f"MACD: {macd}" if macd else ""}
                    </div>
                    <div style='font-size:12px;color:#7a8299;margin-top:6px'>
                        {mom.get("description","")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Chart Patterns ────────────────────────────────
            patterns = result.get("patterns", [])
            if patterns:
                st.markdown("#### 🔷 Chart Patterns Detected")
                pcols = st.columns(min(len(patterns), 3))
                for i, p in enumerate(patterns[:3]):
                    ptype = p.get("type", "")
                    pcol  = pattern_type_color(ptype)
                    rel   = p.get("reliability", "Medium")
                    rel_c = ("#00d4aa" if rel == "High"
                             else "#ffb347" if rel == "Medium" else "#ff6b6b")
                    pcols[i].markdown(f"""
                    <div style='background:#111520;border:1px solid {pcol}44;
                                border-radius:12px;padding:16px;text-align:center'>
                        <div style='font-weight:700;color:#e4e8f0;margin-bottom:6px'>
                            {p.get("name","")}
                        </div>
                        <div style='font-size:11px;color:{pcol};margin-bottom:4px'>
                            {ptype}
                        </div>
                        <div style='font-size:11px;color:{rel_c}'>
                            Reliability: {rel}
                        </div>
                        <div style='font-size:11px;color:#7a8299;margin-top:6px'>
                            {p.get("description","")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Candlestick Patterns ──────────────────────────
            candles = result.get("candlestick_patterns", [])
            if candles:
                st.markdown("#### 🕯️ Candlestick Patterns")
                for c in candles:
                    c_sig = c.get("signal", "Neutral")
                    c_col = ("#00d4aa" if c_sig == "Bullish"
                             else "#ff6b6b" if c_sig == "Bearish" else "#ffb347")
                    st.markdown(
                        f"<div style='background:#111520;border-left:3px solid {c_col};"
                        f"padding:10px 14px;border-radius:0 8px 8px 0;margin:6px 0;"
                        f"font-size:13px;color:#e4e8f0'>"
                        f"<strong>{c.get('name','')}</strong> — "
                        f"<span style='color:{c_col}'>{c_sig}</span> · "
                        f"{c.get('location','')}</div>",
                        unsafe_allow_html=True
                    )

            # ── Support & Resistance ──────────────────────────
            sr = result.get("support_resistance", {})
            if sr:
                st.markdown("#### 🎯 Support & Resistance Levels")
                sr1, sr2 = st.columns(2)
                supports   = sr.get("key_support", [])
                resistances = sr.get("key_resistance", [])
                with sr1:
                    st.markdown("**🟢 Key Support Levels**")
                    for s in supports:
                        st.markdown(
                            f"<div style='background:#00d4aa22;border:1px solid #00d4aa44;"
                            f"border-radius:8px;padding:8px 12px;margin:4px 0;"
                            f"font-weight:600;color:#00d4aa'>₹ {s}</div>",
                            unsafe_allow_html=True
                        )
                with sr2:
                    st.markdown("**🔴 Key Resistance Levels**")
                    for r in resistances:
                        st.markdown(
                            f"<div style='background:#ff6b6b22;border:1px solid #ff6b6b44;"
                            f"border-radius:8px;padding:8px 12px;margin:4px 0;"
                            f"font-weight:600;color:#ff6b6b'>₹ {r}</div>",
                            unsafe_allow_html=True
                        )
                if sr.get("description"):
                    st.caption(sr["description"])

            # ── Risk Factors ──────────────────────────────────
            risks = result.get("risk_factors", [])
            if risks:
                st.markdown("#### ⚠️ Risk Factors")
                for risk in risks:
                    st.markdown(
                        f"<div style='background:#ff6b6b11;border-left:3px solid #ff6b6b;"
                        f"padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;"
                        f"font-size:13px;color:#e4e8f0'>⚠️ {risk}</div>",
                        unsafe_allow_html=True
                    )

            # ── Summary ───────────────────────────────────────
            summary = result.get("summary", "")
            if summary:
                st.markdown("---")
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#6c63ff22,#00d4aa11);"
                    f"border:1px solid #6c63ff44;border-radius:12px;padding:16px 20px;"
                    f"color:#e4e8f0;font-size:0.95rem;line-height:1.6'>"
                    f"💡 <strong>Summary:</strong> {summary}</div>",
                    unsafe_allow_html=True
                )

            # ── Disclaimer ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "⚠️ **Disclaimer:** This analysis is generated by AI and is for "
                "educational purposes only. It is not financial advice. "
                "Always do your own research before making investment decisions.",
                icon=None,
            )

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#7a8299'>
            <div style='font-size:4rem;margin-bottom:16px'>📸</div>
            <div style='font-size:1.1rem;font-weight:600;color:#e4e8f0;margin-bottom:8px'>
                Upload a stock chart screenshot to get started
            </div>
            <div style='font-size:0.9rem;max-width:400px;margin:0 auto'>
                Works with screenshots from Zerodha Kite, Groww, TradingView,
                Google Finance, NSE India, Moneycontrol, or any charting platform
            </div>
            <br>
            <div style='font-size:0.85rem;color:#4a5568'>
                📈 Patterns &nbsp;·&nbsp; 🎯 Support/Resistance &nbsp;·&nbsp;
                ⚡ Momentum &nbsp;·&nbsp; 🚀 Recommendation
            </div>
        </div>
        """, unsafe_allow_html=True)