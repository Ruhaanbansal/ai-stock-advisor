import streamlit as st
from datetime import datetime
from src.config import STOCK_LIST, STOCK_LABELS
from src.stock_search import resolve_ticker, get_suggestions
from src.dataloader import load_stock_data
from src.ui.components import show_brand, show_status_heartbeat

def show_sidebar(status="live"):
    """Displays the professional sidebar navigation and filters."""
    with st.sidebar:
        # 1. Branding
        show_brand()

        # 2. Page Navigation
        page = st.radio(
            "Navigate",
            ["📊 Dashboard", "🧠 AI Prediction", "📰 Market Intelligence",
             "💼 Portfolio Optimizer", "🔁 Backtesting", "🔬 Model Evaluation",
             "🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 3. Stock Search & Quick Picks
        if page not in {"🚀 IPO Predictor", "📡 FII/DII Tracker", "📸 Chart Analyzer", "💼 Portfolio Optimizer"}:
            st.markdown(
                "<div style='font-size:12px;color:#7a8299;margin-bottom:6px'>🔍 SEARCH ANY NSE STOCK</div>",
                unsafe_allow_html=True
            )

            search_query = st.text_input(
                "Search",
                placeholder="e.g. Reliance, TATASTEEL, HDFC Bank…",
                label_visibility="collapsed",
                key="sidebar_search"
            )

            if search_query and len(search_query) >= 2:
                suggestions = get_suggestions(search_query, max_results=5)
                if suggestions:
                    st.markdown("<div style='font-size:11px;color:#7a8299;margin-bottom:4px'>Suggestions:</div>", unsafe_allow_html=True)
                    for idx, s in enumerate(suggestions):
                        if st.button(f"{s['name']}  `{s['ticker']}`", key=f"sug_{idx}", width='stretch'):
                            st.session_state.selected_stock      = s["ticker"]
                            st.session_state.selected_stock_name = s["name"]
                            st.rerun()

            # Popular Stocks
            st.markdown("<div style='font-size:12px;color:#7a8299;margin:12px 0 6px'>⚡ POPULAR STOCKS</div>", unsafe_allow_html=True)
            preset_cols = st.columns(2)
            for i, ticker in enumerate(STOCK_LIST):
                label = STOCK_LABELS.get(ticker, ticker).split()[0]
                if preset_cols[i % 2].button(label, key=f"preset_{i}", width='stretch'):
                    st.session_state.selected_stock      = ticker
                    st.session_state.selected_stock_name = STOCK_LABELS.get(ticker, ticker)
                    st.rerun()

            st.markdown("---")
            
            # Comparison Ticker
            st.markdown("<div style='font-size:12px;color:#7a8299;margin-bottom:6px'>📊 COMPARE WITH STOCK</div>", unsafe_allow_html=True)
            compare_query = st.text_input("Compare", placeholder="e.g. TCS, INFY...", label_visibility="collapsed", key="sidebar_compare")
            if compare_query and len(compare_query) >= 2:
                comp_sugs = get_suggestions(compare_query, max_results=3)
                for idx, s in enumerate(comp_sugs):
                    if st.button(f"Compare: {s['name']}", key=f"comp_{idx}"):
                        st.session_state.compare_ticker = s["ticker"]
                        st.session_state.compare_name   = s["name"]
                        st.rerun()
            
            if st.session_state.get("compare_ticker"):
                st.info(f"Comparing: **{st.session_state.compare_name}**")
                if st.button("❌ Clear Comparison"):
                    st.session_state.compare_ticker = None
                    st.rerun()

            st.markdown("---")

            # Timeframe
            st.markdown("<div style='font-size:12px;color:#7a8299;margin:12px 0 6px'>🗓️ TIMEFRAME</div>", unsafe_allow_html=True)
            if 'period' not in st.session_state: st.session_state.period = '1y'
            p_row1 = st.columns(4)
            p_row2 = st.columns(3)
            periods_r1 = [("6M", "6mo"), ("1Y", "1y"), ("2Y", "2y"), ("5Y", "5y")]
            periods_r2 = [("10Y", "10y"), ("15Y", "15y"), ("30Y", "30y")]
            for i, (label, val) in enumerate(periods_r1):
                if p_row1[i].button(label, key=f"p_{val}", type="primary" if st.session_state.period == val else "secondary"):
                    st.session_state.period = val; st.rerun()
            for i, (label, val) in enumerate(periods_r2):
                if p_row2[i].button(label, key=f"p_{val}", type="primary" if st.session_state.period == val else "secondary"):
                    st.session_state.period = val; st.rerun()

        # 4. Data Governance
        with st.sidebar.expander("🛡️ DATA GOVERNANCE"):
            st.markdown("""
            <div style="font-size: 11px; color: #7a8299; line-height: 1.6;">
            <b>PRICE DATA</b>: Yahoo Finance (EOD / 15m Delayed)<br>
            <b>SENTIMENT</b>: NewsAPI.org & yfinance Headlines<br>
            <b>ANALYTICS</b>: AI Inference on {st.session_state.period} window<br>
            <b>FII/DII</b>: NSE India Official Disclosure
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 5. Heartbeat Status
        show_status_heartbeat(status=status)
        st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S IST')}")
        
        return page
