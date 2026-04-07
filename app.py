# =============================================================
# app.py — NiftyMind · AI Stock Advisor
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure the root directory is in the Python path for robust imports on Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load API keys from .env file
load_dotenv()

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
from src.sentiment    import analyse_sentiment, has_news_api_key
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
from src.insight      import generate_ai_insight, generate_chart_insights
from src.ui.components  import (show_brand, show_status_heartbeat, show_insight,
                                 get_base64_bin_file)


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
    --surface:     #111520;
    --surface2:    #181d2e;
    --border:      #1e2538;
    --accent:      #00d4aa;
    --accent2:     #6c63ff;
    --danger:      #ff5c7c;
    --warning:     #ffb347;
    --text:        #e4e8f0;
    --muted:       #7a8299;
    --card-radius: 14px;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--card-radius);
    padding: 14px 18px;
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,212,170,.12);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px; }
[data-testid="stMetricValue"] { color: var(--text)  !important; font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 600; }

/* ── Section headers ── */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* ── Tabs ── */
[data-testid="stTab"] { font-family: 'DM Sans', sans-serif; font-weight: 500; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--card-radius) !important;
}

/* ── Selectbox / multiselect ── */
[data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #0a0d14;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .85; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: var(--accent) !important; }

/* ── Alert banners ── */
.alert-danger  { background: rgba(255,92,124,.12); border-left: 3px solid var(--danger);  border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-warning { background: rgba(255,179,71,.12);  border-left: 3px solid var(--warning); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-info    { background: rgba(108,99,255,.12);  border-left: 3px solid var(--accent2); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.alert-ok      { background: rgba(0,212,170,.12);   border-left: 3px solid var(--accent);  border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }

/* ── Rec badge ── */
.rec-badge {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
    padding: 10px 28px;
    border-radius: 50px;
    letter-spacing: .5px;
}
.badge-strong-buy { background: rgba(0,212,170,.15); color: #00d4aa; border: 1.5px solid #00d4aa; }
.badge-buy        { background: rgba(0,212,170,.08); color: #00d4aa; border: 1.5px solid rgba(0,212,170,.4); }
.badge-hold       { background: rgba(255,179,71,.10); color: #ffb347; border: 1.5px solid rgba(255,179,71,.4); }
.badge-sell       { background: rgba(255,92,124,.10); color: #ff5c7c; border: 1.5px solid rgba(255,92,124,.4); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 24px 0; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    # 1. Branding
    show_brand()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🧠 AI Prediction", "📰 Market Intelligence",
         "💼 Portfolio Optimizer", "🔁 Backtesting", "🔬 Model Evaluation",
         "🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Stock Search ──────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;color:#7a8299;margin-bottom:6px'>🔍 SEARCH ANY NSE STOCK</div>",
        unsafe_allow_html=True
    )

    search_query = st.text_input(
        "Search",
        placeholder="e.g. Reliance, TATASTEEL, HDFC Bank…",
        label_visibility="collapsed",
    )

    # Initialise session state — None shows welcome page on first load
    if "selected_stock" not in st.session_state:
        st.session_state.selected_stock      = None
        st.session_state.selected_stock_name = None

    # Live suggestions while typing
    if search_query and len(search_query) >= 2:
        suggestions = get_suggestions(search_query, max_results=5)

        if suggestions:
            st.markdown(
                "<div style='font-size:11px;color:#7a8299;margin-bottom:4px'>Suggestions:</div>",
                unsafe_allow_html=True
            )
            # Use index + query hash so keys are always unique across rerenders
            q_hash = abs(hash(search_query)) % 100000
            for idx, s in enumerate(suggestions):
                btn_label = f"{s['name']}  `{s['ticker']}`"
                btn_key   = f"sug_{q_hash}_{idx}_{s['ticker'].replace('.','_')}"
                if st.button(btn_label, key=btn_key, width='stretch'):
                    st.session_state.selected_stock      = s["ticker"]
                    st.session_state.selected_stock_name = s["name"]
                    st.rerun()

        # Search button — key includes query so it never clashes with presets
        search_btn_key = f"search_btn_{abs(hash(search_query)) % 100000}"
        if st.button("🔎 Search", key=search_btn_key, width='stretch'):
            with st.spinner("Searching…"):
                ticker, name = resolve_ticker(search_query)
            if ticker:
                st.session_state.selected_stock      = ticker
                st.session_state.selected_stock_name = name or ticker
                st.success(f"Found: {name} ({ticker})")
                st.rerun()
            else:
                st.error("Stock not found. Try a different name or NSE ticker.")

    # ── Popular Stocks (quick-select) ─────────────────────────
    st.markdown(
        "<div style='font-size:12px;color:#7a8299;margin:12px 0 6px'>⚡ POPULAR STOCKS</div>",
        unsafe_allow_html=True
    )

    # Show presets in a 2-column grid
    # Keys use "preset_" prefix + index so they never clash with suggestion keys
    preset_cols = st.columns(2)
    for i, ticker in enumerate(STOCK_LIST):
        label = STOCK_LABELS.get(ticker, ticker).split()[0]
        btn_key = f"preset_{i}_{ticker.replace('.','_')}"
        if preset_cols[i % 2].button(label, key=btn_key, width='stretch'):
            st.session_state.selected_stock      = ticker
            st.session_state.selected_stock_name = STOCK_LABELS.get(ticker, ticker)
            st.rerun()

    st.markdown("---")

    # ── Compare Stock ──────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;color:#7a8299;margin-bottom:6px'>📊 COMPARE WITH ANOTHER STOCK</div>",
        unsafe_allow_html=True
    )
    
    if "compare_ticker" not in st.session_state:
        st.session_state.compare_ticker = None
        st.session_state.compare_name   = None

    compare_query = st.text_input(
        "Compare",
        placeholder="e.g. TCS, INFY, ICICIBANK...",
        label_visibility="collapsed",
        key="compare_input"
    )

    if compare_query and len(compare_query) >= 2:
        comp_suggestions = get_suggestions(compare_query, max_results=3)
        if comp_suggestions:
            for idx, s in enumerate(comp_suggestions):
                if st.button(f"Compare: {s['name']}", key=f"comp_sug_{idx}"):
                    st.session_state.compare_ticker = s["ticker"]
                    st.session_state.compare_name   = s["name"]
                    st.rerun()

    if st.session_state.compare_ticker:
        st.info(f"Comparing with: **{st.session_state.compare_name}**")
        if st.button("❌ Remove Comparison"):
            st.session_state.compare_ticker = None
            st.session_state.compare_name   = None
            st.rerun()

    st.markdown("---")

    # ── Timeframe Selectors ──────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;color:#7a8299;margin:12px 0 6px'>🗓️ TIMEFRAME</div>",
        unsafe_allow_html=True
    )
    
    if 'period' not in st.session_state:
        st.session_state.period = '1y'

    # Horizontal button grid for periods
    p_row1 = st.columns(4)
    p_row2 = st.columns(3)
    
    periods_r1 = [("6M", "6mo"), ("1Y", "1y"), ("2Y", "2y"), ("5Y", "5y")]
    periods_r2 = [("10Y", "10y"), ("15Y", "15y"), ("30Y", "30y")]
    
    for i, (label, val) in enumerate(periods_r1):
        if p_row1[i].button(label, key=f"p_{val}",
                             type="primary" if st.session_state.period == val else "secondary"):
            st.session_state.period = val
            st.rerun()
            
    for i, (label, val) in enumerate(periods_r2):
        if p_row2[i].button(label, key=f"p_{val}",
                             type="primary" if st.session_state.period == val else "secondary"):
            st.session_state.period = val
            st.rerun()
    
    period = st.session_state.period

    st.markdown("---")

    # 4. Data Governance
    with st.sidebar.expander("🛡️ DATA GOVERNANCE"):
        st.markdown(f"""
        <div style="font-size: 11px; color: #7a8299; line-height: 1.6;">
        <b>PRICE DATA</b>: Yahoo Finance & Stooq (EOD / 15m Delayed)<br>
        <b>SENTIMENT</b>: NewsAPI.org & yfinance Headlines<br>
        <b>ANALYTICS</b>: AI Inference on {st.session_state.get('period', '1y')} window<br>
        <b>FII/DII</b>: NSE India Official Disclosure
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Determine system status
    _sys_status = "live"
    if st.session_state.get("data_source") == "Alpha Vantage": _sys_status = "delayed"

    # 5. Heartbeat Status
    show_status_heartbeat(status=_sys_status)
    
    # Correct for IST (UTC + 5.5) as Streamlit Cloud servers run on UTC
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.sidebar.caption(f"Last updated: {ist_now.strftime('%H:%M:%S IST')}")


# ══════════════════════════════════════════════════════════════
# WELCOME PAGE — shown until user picks a stock
# ══════════════════════════════════════════════════════════════

_STOCK_FREE_PAGES = {"🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer", "💼 Portfolio Optimizer"}

if st.session_state.get("selected_stock") is None and page not in _STOCK_FREE_PAGES:
    # Get base64 logo for the hero section
    logo_path = os.path.join("assets", "logo.svg")
    logo_html = "🧠" # Fallback
    if os.path.exists(logo_path):
        logo_base64 = get_base64_bin_file(logo_path)
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_base64}" width="80" height="80" style="margin-bottom:16px">'

    st.markdown(f"""
    <div style='text-align:center;padding:60px 20px 30px'>
        <div style='display:flex;justify-content:center'>{logo_html}</div>
        <h1 style='font-size:2.4rem;font-weight:700;
                   background:linear-gradient(135deg,#00d4aa,#6c63ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:8px'>Welcome to NiftyMind</h1>
        <p style='color:#7a8299;font-size:1.1rem;max-width:520px;margin:0 auto 32px'>
            AI-powered stock intelligence for Indian markets.<br>
            Search any NSE stock or pick one below to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div style='background:#111520;border:1px solid #1e2538;
            border-radius:12px;padding:20px;text-align:center'>
            <div style='font-size:28px'>📈</div>
            <div style='font-weight:600;color:#e4e8f0;margin:8px 0 4px'>AI Prediction</div>
            <div style='font-size:12px;color:#7a8299'>GradientBoosting model trained on 1 year of price history</div>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style='background:#111520;border:1px solid #1e2538;
            border-radius:12px;padding:20px;text-align:center'>
            <div style='font-size:28px'>📰</div>
            <div style='font-weight:600;color:#e4e8f0;margin:8px 0 4px'>Sentiment Analysis</div>
            <div style='font-size:12px;color:#7a8299'>Live news sentiment powered by VADER NLP</div>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style='background:#111520;border:1px solid #1e2538;
            border-radius:12px;padding:20px;text-align:center'>
            <div style='font-size:28px'>💼</div>
            <div style='font-weight:600;color:#e4e8f0;margin:8px 0 4px'>Portfolio Optimizer</div>
            <div style='font-size:12px;color:#7a8299'>Sharpe-ratio optimised allocation across NSE stocks</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick-pick popular stocks
    st.markdown(
        "<div style='text-align:center;font-size:13px;color:#7a8299;"
        "margin-bottom:16px'>⚡ Popular stocks — click to explore</div>",
        unsafe_allow_html=True
    )
    qcols = st.columns(len(STOCK_LIST))
    for i, ticker in enumerate(STOCK_LIST):
        label = STOCK_LABELS.get(ticker, ticker).split()[0]
        if qcols[i].button(label, key=f"welcome_{i}_{ticker.replace('.','_')}"):
            st.session_state.selected_stock      = ticker
            st.session_state.selected_stock_name = STOCK_LABELS.get(ticker, ticker)
            st.rerun()

    st.stop()   # don't render anything else until a stock is selected


# ══════════════════════════════════════════════════════════════
# DATA LOADING  (skipped for stock-free pages)
# ══════════════════════════════════════════════════════════════

data         = None
close_prices = None

if page not in _STOCK_FREE_PAGES:
    stock = st.session_state.selected_stock
    
    load_status = st.empty()
    with load_status.container():
        with st.status("📊 Fetching market data...", expanded=True) as status:
            import logging
            logging.basicConfig(level=logging.INFO)
            data = load_stock_data(stock, period=period)
            
            # Load comparison data if selected
            compare_data = None
            if st.session_state.get("compare_ticker"):
                st.write(f"Fetching comparison data for {st.session_state.compare_ticker}...")
                compare_data = load_stock_data(st.session_state.compare_ticker, period=period)
            
            status.update(label="✅ Market Data Loaded", state="complete", expanded=False)
    
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
    with load_status.container():
        with st.status("🧠 Initializing AI Engine...", expanded=True) as status:
            cp_hash = hash(close_prices.values.tobytes())
            model, scaler, last_sequence, train_mae, test_mae = train_lstm_model(st.session_state.selected_stock, cp_hash, close_prices)
            status.update(label="✅ AI Engine Ready", state="complete", expanded=False)

        predicted_price = predict_next_price(model, scaler, last_sequence)
        current_price   = float(close_prices.iloc[-1])
        forecast_5d     = forecast_prices(model, scaler, last_sequence, days=5)

        with st.status("🔍 Deep Intelligence Analysis...", expanded=True) as status:
            st.write("Fetching news & calculating sentiment...")
            _sent = analyse_sentiment(st.session_state.selected_stock, company_name=st.session_state.get("selected_stock_name"))
            sentiment_score  = _sent["score"]
            sentiment_label  = _sent["label"]
            articles         = _sent["articles"]
            headlines        = [a["title"] for a in articles]
            sentiment_source = _sent["model"]

            st.write("Calculating multi-factor risk metrics...")
            # Risk metrics expect returns, not absolute prices
            risk_metrics = calculate_risk_metrics(
                close_prices.pct_change().dropna(),
                stock_ticker=st.session_state.selected_stock,
                period=period
            )
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)
    
    # Hide all loading indicators now that everything is ready
    load_status.empty()

    volatility   = risk_metrics.get("volatility", 0.0)
    sharpe_ratio = risk_metrics.get("sharpe", 0.0)
    max_drawdown = risk_metrics.get("max_drawdown", 0.0)
    
    # Enhanced advisor call with test_mae for better confidence
    advisor = run_ai_advisor(
        current_price, predicted_price, volatility, sentiment_score,
        test_mae=test_mae
    )


# ══════════════════════════════════════════════════════════════
# GLOBAL KPI STRIP  (only for stock pages)
# ══════════════════════════════════════════════════════════════

if page not in _STOCK_FREE_PAGES:
    st.markdown(f"## {STOCK_LABELS.get(st.session_state.selected_stock, st.session_state.selected_stock)}")
    st.caption(f"`{st.session_state.selected_stock}` · {period} view · Prices in ₹")

    k1, k2, k3, k4, k5 = st.columns(5)
    price_delta = f"{advisor['price_change']:+.2f}%"
    k1.metric("Current Price",   f"₹{current_price:,.2f}", help="The latest closing price fetched from various data sources.")
    k2.metric("AI Predicted",    f"₹{predicted_price:,.2f}", price_delta, help="Next-day price predicted by our LSTM model. The delta shows projected movement.")
    k3.metric("Sentiment",       sentiment_label, help="Combined news sentiment score from VADER and news headlines.")
    k4.metric("Risk",            advisor["risk"], help="Multi-factor risk assessment based on volatility and drawdowns.")
    k5.metric("Recommendation",  advisor["recommendation"], help="Final AI signal integrating price, technicals, and sentiment.")

    st.markdown("---")

    # ── Data source badge ────────────────────────────────────
    _source = st.session_state.get("data_source", "Yahoo Finance")
    _source_color = {"Yahoo Finance": "#00d4aa", "Stooq": "#6c63ff", "Alpha Vantage": "#ffb347"}.get(_source, "#7a8299")
    st.caption(f'Data sourced from: :{_source_color}[**{_source}**]')


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 📊 DASHBOARD
# ─────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.subheader("Price Chart & Technical Indicators")

    extended = create_extended_features(data)

    chart_tab, bb_tab, macd_tab = st.tabs(["Price & MAs", "Bollinger Bands", "MACD"])

    with chart_tab:
        # Google-style logic: color based on net performance in period
        is_up      = close_prices.iloc[-1] >= close_prices.iloc[0]
        chart_color = "#00d4aa" if is_up else "#ff5c7c"
        fill_color  = "rgba(0, 212, 170, 0.08)" if is_up else "rgba(255, 92, 124, 0.08)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close_prices.index, y=close_prices,
            name="Close",
            line=dict(color=chart_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hoverinfo="x+y",
        ))

        # ── News Overlay ─────────────────────────────────────────
        if articles:
            news_dates = []
            news_prices = []
            news_texts = []
            
            for art in articles[:15]: # Max 15 markers
                try:
                    dt = pd.to_datetime(art['publishedAt']).tz_localize(None).normalize()
                    if dt in close_prices.index:
                        news_dates.append(dt)
                        news_prices.append(close_prices.loc[dt])
                        news_texts.append(f"<b>{art['source']}</b>: {art['title'][:60]}...")
                except: continue
            
            if news_dates:
                fig.add_trace(go.Scatter(
                    x=news_dates, y=news_prices,
                    mode='markers', name='News Markers',
                    marker=dict(symbol='diamond', size=10, color='#6C63FF', line=dict(width=1, color='white')),
                    text=news_texts, hoverinfo='text'
                ))

        # Thinner SMAs with coordinated colors
        if "SMA20" in extended:
            fig.add_trace(go.Scatter(
                x=extended.index, y=extended["SMA20"],
                name="SMA 20", line=dict(color="rgba(108, 99, 255, 0.5)", width=1.2, dash="dash")
            ))
        if "SMA50" in extended:
            fig.add_trace(go.Scatter(
                x=extended.index, y=extended["SMA50"],
                name="SMA 50", line=dict(color="rgba(255, 179, 71, 0.5)", width=1.2, dash="dash")
            ))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                tickfont=dict(color="#7a8299", size=10),
            ),
            yaxis=dict(
                gridcolor="rgba(122, 130, 153, 0.1)",
                zeroline=False,
                showline=False,
                tickfont=dict(color="#7a8299", size=10),
                side="right",
            ),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        # ── AI Insights Snackbars ────────────────────────────────
        chart_insights = generate_chart_insights(data, extended)
        for insight_text in chart_insights:
            show_insight(insight_text)

        # Volume chart
        if "Volume" in data.columns:
            st.subheader("Trading Volume", help="Daily trading volume. Massive spikes often indicate institutional entry or exit.")
            vol_fig = go.Figure(go.Bar(
                x=data.index, y=data["Volume"],
                marker_color="rgba(122, 130, 153, 0.3)",
                name="Volume"
            ))
            vol_fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(vol_fig, width="stretch")

        # ── Comparison Chart (if active) ──────────────────────────
        if st.session_state.get("compare_ticker"):
            st.markdown("---")
            st.subheader(f"📊 Relative Performance: {stock} vs {st.session_state.compare_ticker}")
            
            with st.status("Fetching comparison data...", expanded=False):
                comp_data = load_stock_data(st.session_state.compare_ticker, period=period)
            
            if not comp_data.empty:
                c_close = get_close_prices(comp_data)
                
                # Re-index to match primary data dates
                c_close = c_close.reindex(close_prices.index, method='ffill')
                
                # Normalize to 0% at start
                norm_main = ((close_prices / close_prices.iloc[0]) - 1) * 100
                norm_comp = ((c_close / c_close.iloc[0]) - 1) * 100
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=norm_main.index, y=norm_main, name=stock,
                    line=dict(color="#00d4aa", width=2)
                ))
                fig_comp.add_trace(go.Scatter(
                    x=norm_comp.index, y=norm_comp, name=st.session_state.compare_ticker,
                    line=dict(color="#6c63ff", width=2)
                ))
                
                fig_comp.update_layout(
                    template="plotly_dark", height=400,
                    yaxis_title="Percent Change (%)",
                    hovermode="x unified",
                    margin=dict(l=0,r=0,t=20,b=0),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_comp, width="stretch")
                st.caption("Performance is normalized to 0% at the beginning of the period.")

    with bb_tab:
        st.subheader("Bollinger Bands", help="Measures market volatility. Price near the upper band suggests overbought conditions; near the lower band suggests oversold.")
        if "BB_Upper" in extended:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Upper"],
                                      name="Upper Band", line=dict(color="#ff5c7c", dash="dot", width=1)))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Mid"],
                                      name="Middle (SMA20)", line=dict(color="#7a8299", width=1)))
            fig2.add_trace(go.Scatter(x=extended.index, y=extended["BB_Lower"],
                                      name="Lower Band", line=dict(color="#00d4aa", dash="dot", width=1),
                                      fill="tonexty", fillcolor="rgba(0,212,170,0.04)"))
            fig2.add_trace(go.Scatter(x=close_prices.index, y=close_prices,
                                      name="Close", line=dict(color="white", width=1.5)))
            fig2.update_layout(template="plotly_dark", height=400,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Not enough data for Bollinger Bands.")

    with macd_tab:
        st.subheader("MACD (Moving Average Convergence Divergence)", help="A trend-following momentum indicator. Crossovers between the MACD and Signal line indicate potential buy/sell signals.")
        if "MACD" in extended:
            fig3 = go.Figure()
            macd_hist = extended["MACD"] - extended["MACD_Signal"]
            fig3.add_trace(go.Bar(x=extended.index, y=macd_hist, name="Histogram",
                                  marker_color=["#00d4aa" if v >= 0 else "#ff5c7c" for v in macd_hist]))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD"],
                                      name="MACD", line=dict(color="#6c63ff")))
            fig3.add_trace(go.Scatter(x=extended.index, y=extended["MACD_Signal"],
                                      name="Signal", line=dict(color="#ffb347", dash="dot")))
            fig3.update_layout(template="plotly_dark", height=380,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for MACD.")

    # 5-day forecast strip
    st.subheader("5-Day Price Forecast", help="AI-generated projections for the next 5 trading days based on the LSTM model's inference.")
    today  = datetime.today()
    fdates = [(today + timedelta(days=i+1)).strftime("%a %d %b") for i in range(5)]
    fcols  = st.columns(5)
    for i, (d, p) in enumerate(zip(fdates, forecast_5d)):
        delta = ((p - current_price) / current_price) * 100
        fcols[i].metric(d, f"₹{p:,.2f}", f"{delta:+.2f}%")



# ─────────────────────────────────────────────────────────────
# 🧠 AI PREDICTION
# ─────────────────────────────────────────────────────────────
elif page == "🧠 AI Prediction":
    st.markdown("""
        <div style='margin-bottom:24px'>
            <h2 style='margin:0;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4aa,#6c63ff);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🧠 AI Investment Intelligence
            </h2>
            <p style='color:#7a8299;margin:4px 0 0;font-size:0.9rem'>
                Multi-factor signal decomposition · Institutional-grade consensus · Risk-adjusted verdict
            </p>
        </div>
    """, unsafe_allow_html=True)

    top_l, top_r = st.columns([1, 1.2])

    with top_l:
        # Decision Badge
        badge_class = {
            "Strong Buy": "badge-strong-buy", "Buy": "badge-buy",
            "Hold": "badge-hold", "Sell": "badge-sell",
        }.get(advisor["recommendation"], "badge-hold")
        
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;height:100%">
          <div style="color:var(--muted);font-size:12px;margin-bottom:12px;letter-spacing:1px;font-weight:600">PREDICTED VERDICT</div>
          <span class="rec-badge {badge_class}" style="font-size:32px;padding:12px 36px">{advisor['recommendation']}</span>
          <div style="margin-top:20px;display:flex;justify-content:center;gap:32px">
            <div>
                <div style="color:var(--muted);font-size:10px">CONFIDENCE</div>
                <div style="font-size:18px;font-weight:700;color:var(--accent)">{advisor['confidence']}%</div>
            </div>
            <div style="border-left:1px solid var(--border);height:30px"></div>
            <div>
                <div style="color:var(--muted);font-size:10px">RISK LEVEL</div>
                <div style="font-size:18px;font-weight:700;color:#ffb347">{advisor['risk']}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with top_r:
        # Decision Factor Radar
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=list(advisor["factor_scores"].values()),
            theta=list(advisor["factor_scores"].keys()),
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.25)',
            line=dict(color='#00d4aa', width=3),
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(122,130,153,0.1)"),
                angularaxis=dict(gridcolor="rgba(122,130,153,0.1)", linecolor="rgba(122,130,153,0.1)", tickfont=dict(size=10, color="#7a8299")),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False, height=220,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_fig, width='stretch')

    st.markdown("---")

    col_l, col_r = st.columns([1.1, 0.9])

    with col_l:
        st.markdown("### 🔍 Model Explainability")
        imp = compute_feature_importance(model, scaler, close_prices)
        st.plotly_chart(shap_chart(imp), width='stretch')
        
        # Technical Signal Capsules (Symmetric Grid)
        signals = get_technical_signals(close_prices)
        if signals:
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            s_rows = st.columns(3)
            # We map signals to these slots
            for i, (name, (val, label, desc)) in enumerate(signals.items()):
                target_col = s_rows[i % 3]
                color = "#00d4aa" if "🟢" in label else "#ff5c7c" if "🔴" in label else "#ffb347"
                icon  = "●"
                target_col.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px;text-align:center;margin-bottom:12px">
                  <div style="font-size:9px;color:var(--muted);margin-bottom:6px;font-weight:600">{name.upper()}</div>
                  <div style="font-size:14px;font-weight:700;color:{color};line-height:1.2">{label}</div>
                  <div style="font-size:18px;color:{color if icon == '●' else '#fff'}">{icon}</div>
                  <div style="font-size:10px;color:var(--muted);margin-top:2px">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Detailed Analysis & Reasoning")
        
        reasoning_icons = ["📈", "📰", "🛡️", "📊", "🧭"]
        for i, reason in enumerate(advisor["reasons"]):
            icon = reasoning_icons[i % len(reasoning_icons)]
            style = "alert-info"
            if "📈" in reason or "🟢" in reason or "Bullish" in reason: style = "alert-ok"
            elif "📉" in reason or "🔴" in reason or "Bearish" in reason: style = "alert-danger"
            elif "⚠️" in reason or "🟡" in reason: style = "alert-warning"

            st.markdown(f"""
            <div class="{style}" style="display:flex;align-items:flex-start;gap:12px;padding:16px;border-radius:12px;margin-bottom:12px">
                <span style="font-size:20px;margin-top:-2px">{icon}</span>
                <div style="font-size:14px;line-height:1.5">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Risk Profile metrics")
        r1, r2 = st.columns(2)
        r1.metric("Sortino Ratio", f"{risk_metrics.get('sortino', 0):.2f}")
        r2.metric("Beta vs NIFTY", f"{risk_metrics.get('beta', 1.0):.2f}")



elif page == "📰 Market Intelligence":
    # ── Sentiment breakdown ────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Sentiment", sentiment_label, f"{sentiment_score:+.3f}")
    c2.metric("Intelligence Layer",  sentiment_source)
    c3.metric("Coverage (Articles)", str(len(articles)))

    # Alerts
    st.markdown("#### 💡 AI Market Alerts")
    alerts = detect_market_alerts(close_prices, sentiment_score)

    if not alerts:
        st.markdown('<div class="alert-ok">✅ No immediate high-risk signals detected in current sentiment/price flow.</div>', unsafe_allow_html=True)
    else:
        for a in alerts:
            css_class = {"danger": "alert-danger", "warning": "alert-warning"}.get(a["level"], "alert-info")
            icon      = {"danger": "🚨", "warning": "⚠️"}.get(a["level"], "ℹ️")
            st.markdown(f'<div class="{css_class}">{icon} {a["message"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Symmetric News & RSI Layout
    news_l, chart_r = st.columns([1.2, 0.8])

    with news_l:
        header_col1, header_col2 = st.columns([1, 1])
        header_col1.markdown("#### 📰 Latest Intelligence")
        if header_col2.button("🔄 Refresh News", key="refresh_news_btn", width='stretch'):
            st.cache_data.clear()
            st.rerun()
        if not articles:
            st.warning("No recent headlines found for this ticker.")
        else:
            for a in articles[:10]:
                source = a.get('source', 'Market News')
                url = a.get('url', '#')
                score = a.get('_score', 0.0)
                
                # Sentiment dot color
                dot_color = "#00d4aa" if score > 0.05 else "#ff5c7c" if score < -0.05 else "#7a8299"
                
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px;position:relative;overflow:hidden">
                    <div style="position:absolute;top:0;left:0;width:4px;height:100%;background:{dot_color}"></div>
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                        <span style="font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px">
                            {source}
                        </span>
                        <span style="font-size:10px;font-weight:700;color:{dot_color}">
                            SCORE: {score:+.2f}
                        </span>
                    </div>
                    <div style="font-size:15px;font-weight:600;margin-bottom:6px;color:var(--text);line-height:1.4">
                        {a['title']}
                    </div>
                    <div style="font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.5">
                        {a.get('description', 'No description available...')[:160]}...
                    </div>
                    <div style="display:flex;justify-content:flex-end">
                        <a href="{url}" target="_blank" style="text-decoration:none;font-size:12px;color:var(--accent);font-weight:600;padding:4px 8px;border:1px solid var(--accent);border-radius:6px;transition:0.2s">
                            Read Full Story ↗
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with chart_r:
        st.markdown("#### 🌊 Momentum Flow (RSI)")
        ext = create_extended_features(data)
        if "RSI" in ext.columns:
            rsi_fig = go.Figure()
            # Dynamic coloring for RSI
            current_rsi = ext["RSI"].iloc[-1]
            rsi_color   = "#ff5c7c" if current_rsi > 70 else "#00d4aa" if current_rsi < 30 else "#6c63ff"
            
            rsi_fig.add_hline(y=70, line=dict(color="rgba(255,92,124,0.3)", dash="dot"))
            rsi_fig.add_hline(y=30, line=dict(color="rgba(0,212,170,0.3)", dash="dot"))
            rsi_fig.add_trace(go.Scatter(
                x=ext.index, y=ext["RSI"], name="RSI (14)",
                line=dict(color=rsi_color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(rsi_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.05])}"
            ))
            rsi_fig.update_layout(
                template="plotly_dark", height=320,
                yaxis=dict(range=[0,100], gridcolor="rgba(122,130,153,0.1)", side="right"),
                xaxis=dict(showgrid=False),
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(rsi_fig, width='stretch')
            
            st.markdown(f"""
            <div style="background:var(--surface);border-radius:10px;padding:16px;border:1px solid var(--border)">
                <div style="color:var(--muted);font-size:11px">CURRENT RSI</div>
                <div style="font-size:24px;font-weight:700;color:{rsi_color}">{current_rsi:.1f}</div>
                <div style="font-size:12px;color:var(--muted)">{'Overbought conditions' if current_rsi > 70 else 'Oversold opportunity' if current_rsi < 30 else 'Neutral momentum'}</div>
            </div>
            """, unsafe_allow_html=True)


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
        add_clicked = st.button("➕ Add Stock", key="pf_add_btn", width='stretch')

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
                if sugg_cols[idx].button(label, key=f"pf_sugg_{q_h}_{idx}", width='stretch', disabled=already):
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
            if col.button(sector_label, key=f"sec_{sector_label}", width='stretch'):
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
                if st.button("✕ Remove", key=f"rm_{idx}_{t.replace('.','_')}", width='stretch'):
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
        st.plotly_chart(pie_fig, width='stretch')
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
            width='stretch',
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
    st.plotly_chart(ef_fig, width='stretch')

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
    st.plotly_chart(heat, width='stretch')


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
    st.plotly_chart(bt_fig, width='stretch')

    # Drawdown chart
    from src.backtest import build_drawdown_fig, build_monthly_heatmap
    st.plotly_chart(build_drawdown_fig(comparison), width='stretch')

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
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width='stretch')

    # Monthly heatmap for AI Strategy
    if "AI Strategy" in comparison.columns:
        st.markdown("#### AI Strategy Monthly Returns")
        st.plotly_chart(build_monthly_heatmap(comparison["AI Strategy"], "AI Strategy"), width='stretch')


# ─────────────────────────────────────────────────────────────
# 🔬 MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Model Evaluation":
    st.subheader("LSTM Model Evaluation")

    with st.spinner("Evaluating model…"):
        evaluation = evaluate_model(close_prices, model, scaler)

    st.plotly_chart(evaluation["chart"], width='stretch')

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("MAE (Test)",   f"₹{evaluation['test_mae']:.2f}")
    e2.metric("RMSE (Test)",  f"₹{evaluation['test_rmse']:.2f}")
    e3.metric("MAPE",         f"{evaluation['mape']:.2f}%")
    e4.metric("Confidence",   f"{evaluation['confidence']:.1f}%")

    st.markdown("#### Model Confidence")
    st.progress(int(evaluation["confidence"]) / 100)

    # Residuals
    st.markdown("#### Prediction Residuals")
    st.plotly_chart(evaluation["residual_chart"], width='stretch')
    
    actual      = evaluation["actual"]
    predictions = evaluation["predictions"]
    residuals   = actual - predictions

    # Residual distribution
    hist_fig = px.histogram(
        x=residuals, nbins=40,
        labels={"x": "Residual (₹)"},
        color_discrete_sequence=["#00d4aa"],
        height=260,
    )
    hist_fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(hist_fig, width='stretch')


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
                                 width='stretch')
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
            }), hide_index=True, width='stretch')

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
        st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig2, width='stretch')

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
                    width='stretch',
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
                 width='stretch')

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