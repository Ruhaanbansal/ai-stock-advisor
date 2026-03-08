# =============================================================
# app.py — NiftyMind · AI Stock Advisor
# =============================================================

import streamlit as st
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
from src.sentiment    import get_news_sentiment, get_stock_news
from src.risk         import calculate_risk_metrics
from src.explainability import get_technical_signals, compute_feature_importance, shap_chart
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
    st.markdown("""
    <div style="padding:12px 0 20px">
      <span style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#00d4aa">
        NiftyMind
      </span>
      <span style="font-size:11px;color:#7a8299;display:block;margin-top:2px">
        AI-Powered Indian Stock Intelligence
      </span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🧠 AI Prediction", "📰 Market Intelligence",
         "💼 Portfolio Optimizer", "🔁 Backtesting", "🔬 Model Evaluation"],
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

    # Initialise session state for selected stock
    if "selected_stock" not in st.session_state:
        st.session_state.selected_stock = "RELIANCE.NS"
    if "selected_stock_name" not in st.session_state:
        st.session_state.selected_stock_name = "Reliance Industries"

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

    # ── Currently selected ────────────────────────────────────
    stock = st.session_state.selected_stock

    st.markdown(
        f"<div style='background:#111520;border:1px solid #1e2538;border-radius:8px;"
        f"padding:8px 12px;margin-top:10px'>"
        f"<div style='font-size:10px;color:#7a8299'>ANALYSING</div>"
        f"<div style='font-size:14px;font-weight:600;color:#00d4aa'>"
        f"{st.session_state.selected_stock_name}</div>"
        f"<div style='font-size:11px;color:#7a8299'>{stock}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    period = st.select_slider(
        "Data Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")


# ══════════════════════════════════════════════════════════════
# DATA LOADING  (cached — runs once per stock/period combo)
# ══════════════════════════════════════════════════════════════

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

# Guard: need at least 65 data points for LSTM sequences
if len(close_prices) < 65:
    st.warning(
        f"Not enough historical data for **{stock}** (only {len(close_prices)} days found). "
        "Try selecting a longer period or a different stock."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
# MODEL TRAINING  (cached per stock — no retraining on re-render)
# ══════════════════════════════════════════════════════════════

with st.spinner("Loading AI model…"):
    cp_hash = hash(close_prices.values.tobytes())
    model, scaler, last_sequence = train_lstm_model(stock, cp_hash, close_prices)

predicted_price = predict_next_price(model, scaler, last_sequence)
current_price   = float(close_prices.iloc[-1])
forecast_5d     = forecast_prices(model, scaler, last_sequence, days=5)


# ══════════════════════════════════════════════════════════════
# SENTIMENT, RISK, ADVISOR  (all cached via sub-functions)
# ══════════════════════════════════════════════════════════════

with st.spinner("Analysing sentiment & risk…"):
    sentiment_score, sentiment_label, headlines, sentiment_source = get_news_sentiment(stock)
    risk_metrics = calculate_risk_metrics(close_prices)

volatility   = risk_metrics["volatility"]
sharpe_ratio = risk_metrics["sharpe_ratio"]
max_drawdown = risk_metrics["max_drawdown"]

advisor = run_ai_advisor(current_price, predicted_price, volatility, sentiment_score)


# ══════════════════════════════════════════════════════════════
# GLOBAL KPI STRIP  (always visible)
# ══════════════════════════════════════════════════════════════

st.markdown(f"## {STOCK_LABELS.get(stock, stock)}")
st.caption(f"`{stock}` · {period} view · Prices in ₹")

k1, k2, k3, k4, k5 = st.columns(5)
price_delta = f"{advisor['price_change']:+.2f}%"
k1.metric("Current Price",   f"₹{current_price:,.2f}")
k2.metric("AI Predicted",    f"₹{predicted_price:,.2f}", price_delta)
k3.metric("Sentiment",       sentiment_label)
k4.metric("Risk",            advisor["risk"])
k5.metric("Recommendation",  advisor["recommendation"])

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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices,
                                 name="Close", line=dict(color="#00d4aa", width=1.8)))
        if "SMA20" in extended:
            fig.add_trace(go.Scatter(x=extended.index, y=extended["SMA20"],
                                     name="SMA 20", line=dict(color="#6c63ff", width=1, dash="dot")))
        if "SMA50" in extended:
            fig.add_trace(go.Scatter(x=extended.index, y=extended["SMA50"],
                                     name="SMA 50", line=dict(color="#ffb347", width=1, dash="dot")))
        fig.update_layout(template="plotly_dark", height=420,
                          xaxis_title="Date", yaxis_title="Price (₹)",
                          legend=dict(orientation="h", y=1.02),
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width='stretch')

    with bb_tab:
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
        st.plotly_chart(vol_fig, width='stretch')


# ─────────────────────────────────────────────────────────────
# 🧠 AI PREDICTION
# ─────────────────────────────────────────────────────────────
elif page == "🧠 AI Prediction":
    st.subheader("AI Investment Recommendation")

    left, right = st.columns([1, 2])

    with left:
        # Recommendation badge
        badge_class = {
            "Strong Buy": "badge-strong-buy",
            "Buy":        "badge-buy",
            "Hold":       "badge-hold",
            "Sell":       "badge-sell",
        }.get(advisor["recommendation"], "badge-hold")

        st.markdown(f"""
        <div style="text-align:center;padding:20px 0">
          <div style="color:#7a8299;font-size:12px;margin-bottom:8px">AI DECISION</div>
          <span class="rec-badge {badge_class}">{advisor['recommendation']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        conf = advisor["confidence"]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            number={"suffix": "%", "font": {"size": 28, "color": "#e4e8f0"}},
            title={"text": "Confidence", "font": {"size": 13, "color": "#7a8299"}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#7a8299"},
                "bar":   {"color": "#00d4aa"},
                "bgcolor": "#111520",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,92,124,.15)"},
                    {"range": [40, 70], "color": "rgba(255,179,71,.15)"},
                    {"range": [70,100], "color": "rgba(0,212,170,.15)"},
                ],
                "threshold": {"line": {"color": "#00d4aa", "width": 3}, "value": conf}
            }
        ))
        gauge.update_layout(
            height=220, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(gauge, width='stretch')

        # Risk metrics
        st.markdown("**Risk Metrics**")
        st.metric("Volatility",    f"{volatility:.2%}")
        st.metric("Sharpe Ratio",  f"{sharpe_ratio:.2f}")
        st.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
        st.metric("Max Drawdown",  f"{max_drawdown:.2%}")
        st.metric("VaR (95%)",     f"{risk_metrics['var_95']:.2%}")
        st.metric("CVaR (95%)",    f"{risk_metrics['cvar_95']:.2%}")

    with right:
        st.markdown("**AI Reasoning**")
        for r in advisor["reasons"]:
            st.markdown(f"› {r}")

        st.markdown("---")

        st.markdown("**Narrative Insight**")
        insight = generate_ai_insight(
            advisor["recommendation"], advisor["price_change"],
            sentiment_label, volatility
        )
        for line in insight:
            st.markdown(line)

        st.markdown("---")

        st.markdown("**Model Explainability**")
        exp_tab1, exp_tab2 = st.tabs(["Technical Signals", "Feature Importance"])

        with exp_tab1:
            signals = get_technical_signals(close_prices)
            if signals:
                for indicator, (value, signal) in signals.items():
                    color = {"Overbought":"🔴","Oversold":"🟢","Bullish":"🟢",
                             "Bearish":"🔴","Golden Cross ↑":"🟢","Death Cross ↓":"🔴"
                             }.get(signal, "🟡")
                    st.markdown(f"{color} **{indicator}**: `{value}` — {signal}")
            else:
                st.info("Not enough data to compute signals.")

        with exp_tab2:
            with st.spinner("Computing feature importance…"):
                imp   = compute_feature_importance(model, scaler, close_prices)
                s_fig = shap_chart(imp)
            if imp:
                st.plotly_chart(s_fig, width='stretch')
            else:
                st.info("Not enough data to compute feature importance.")


# ─────────────────────────────────────────────────────────────
# 📰 MARKET INTELLIGENCE
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
            st.markdown(f"• {h}")

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
        st.plotly_chart(rsi_fig, width='stretch')


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

    # Metrics table
    st.markdown("#### Strategy Performance")
    rows = []
    for strat, m in metrics.items():
        rows.append({
            "Strategy":     strat,
            "Total Return": f"{m['total_return']:+.2f}%",
            "Max Drawdown": f"{m['max_drawdown']:.2f}%",
            "Sharpe Ratio": f"{m['sharpe_ratio']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), width='stretch')


# ─────────────────────────────────────────────────────────────
# 🔬 MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Model Evaluation":
    st.subheader("LSTM Model Evaluation")

    with st.spinner("Evaluating model…"):
        evaluation = evaluate_model(close_prices, model, scaler)

    st.plotly_chart(evaluation["chart"], width='stretch')

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("MAE",        f"₹{evaluation['mae']:.2f}")
    e2.metric("RMSE",       f"₹{evaluation['rmse']:.2f}")
    e3.metric("MAPE",       f"{evaluation['mape']:.2f}%")
    e4.metric("Confidence", f"{evaluation['confidence']:.1f}%")

    st.markdown("#### Model Confidence")
    st.progress(int(evaluation["confidence"]) / 100)

    # Residuals
    st.markdown("#### Prediction Residuals")
    actual      = evaluation["actual"]
    predictions = evaluation["predictions"]
    residuals   = actual - predictions

    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(
        y=residuals, mode="lines",
        line=dict(color="#6c63ff", width=1),
        name="Residual",
    ))
    res_fig.add_hline(y=0, line=dict(color="#7a8299", dash="dot"))
    res_fig.update_layout(
        template="plotly_dark", height=280,
        yaxis_title="Residual (₹)",
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(res_fig, width='stretch')

    # Residual distribution
    hist_fig = px.histogram(
        x=residuals, nbins=40,
        labels={"x": "Residual (₹)"},
        color_discrete_sequence=["#00d4aa"],
        height=260,
    )
    hist_fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(hist_fig, width='stretch')