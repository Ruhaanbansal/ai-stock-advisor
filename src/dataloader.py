# =============================================================
# dataloader.py — Data Loading (delegates to data_sources.py)
# =============================================================

import pandas as pd
import streamlit as st

from src.config       import DEFAULT_PERIOD, PORTFOLIO_PERIOD
from src.data_sources import fetch_stock_data, fetch_portfolio_data


# ─────────────────────────────────────────────────────────────
# Load Single Stock
# ─────────────────────────────────────────────────────────────

def load_stock_data(
    stock:  str,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame | None:
    """
    Load OHLCV data using the waterfall strategy:
    Yahoo Finance → Stooq → Alpha Vantage.
    Stores the source name in st.session_state for display.
    """
    data, source = fetch_stock_data(stock, period)
    st.session_state["data_source"] = source
    return data


# ─────────────────────────────────────────────────────────────
# Extract Clean Close Series
# ─────────────────────────────────────────────────────────────

def get_close_prices(data: pd.DataFrame) -> pd.Series:
    """Extract a clean numeric Close price Series."""
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return pd.to_numeric(close, errors="coerce").dropna()


# ─────────────────────────────────────────────────────────────
# Load Multiple Stocks for Portfolio
# ─────────────────────────────────────────────────────────────

def load_portfolio_data(
    stocks: tuple,
    period: str = PORTFOLIO_PERIOD,
) -> pd.DataFrame | None:
    """
    Load close prices for a portfolio of stocks.
    Falls back per-ticker through all data sources.
    """
    df, source_map = fetch_portfolio_data(stocks, period)
    st.session_state["portfolio_sources"] = source_map
    return df


# ─────────────────────────────────────────────────────────────
# Daily Returns
# ─────────────────────────────────────────────────────────────

def calculate_returns(price_data: pd.DataFrame | pd.Series):
    """Calculate daily percentage returns."""
    return price_data.pct_change().dropna()


# ─────────────────────────────────────────────────────────────
# Quick Stock Info
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(stock: str) -> dict:
    """Fetch basic fundamental info. Returns empty dict on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(stock).info
        return {
            "name":       info.get("longName", stock),
            "sector":     info.get("sector", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio":   info.get("trailingPE"),
            "52w_high":   info.get("fiftyTwoWeekHigh"),
            "52w_low":    info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {}