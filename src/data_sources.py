# =============================================================
# data_sources.py — Multi-Source Stock Data Fetcher
# =============================================================
#
# Waterfall strategy:
#   1. Yahoo Finance with browser-spoofed session (primary)
#   2. Alpha Vantage (API key required)
#   3. Stooq via pandas_datareader (free fallback)
#
# Streamlit Cloud blocks plain yfinance calls — we fix this by
# injecting browser-like headers into the requests session that
# yfinance uses internally, which bypasses the rate-limiting.
# =============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import requests

from src.config import DEFAULT_PERIOD, PORTFOLIO_PERIOD

_AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "E7RBUJ17S6H24GJO")

_PERIOD_DAYS = {
    "1mo": 30,  "3mo": 90,  "6mo": 180,
    "1y": 365,  "2y": 730,  "5y": 1825,
}

# Browser headers to bypass Yahoo Finance bot detection
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}


# ─────────────────────────────────────────────────────────────
# Source 1 — Yahoo Finance (with session spoofing)
# ─────────────────────────────────────────────────────────────

def _make_yf_session() -> requests.Session:
    """Create a requests session with browser headers for yfinance."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def _flatten_yf(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten any MultiIndex columns from yfinance."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data
    known  = {"Close", "Open", "High", "Low", "Volume"}
    level0 = set(data.columns.get_level_values(0))
    level1 = set(data.columns.get_level_values(1))
    if level0 & known:
        data.columns = data.columns.get_level_values(0)
    elif level1 & known:
        data.columns = data.columns.get_level_values(1)
    return data.loc[:, ~data.columns.duplicated()]


def _fetch_yahoo(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Fetch from Yahoo Finance using a browser-spoofed session.
    This bypasses Streamlit Cloud's outbound request filtering.
    """
    try:
        import yfinance as yf
        session = _make_yf_session()
        tk      = yf.Ticker(ticker, session=session)
        data    = tk.history(period=period, auto_adjust=True)

        if data is None or data.empty:
            # Fallback: try yf.download with session
            data = yf.download(
                ticker, period=period,
                auto_adjust=True, progress=False,
                session=session,
            )

        if data is None or data.empty:
            return None

        data = _flatten_yf(data)
        if "Close" not in data.columns:
            return None

        return data.dropna(subset=["Close"])

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Source 2 — Alpha Vantage
# ─────────────────────────────────────────────────────────────

def _to_av_symbol(ticker: str) -> str:
    base = ticker.replace(".NS", "").replace(".BO", "")
    return f"{base}.NSE"


def _fetch_alpha_vantage(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch from Alpha Vantage TIME_SERIES_DAILY_ADJUSTED."""
    try:
        symbol     = _to_av_symbol(ticker)
        outputsize = "compact" if _PERIOD_DAYS.get(period, 180) <= 100 else "full"

        for sym in [symbol, ticker.replace(".NS", "").replace(".BO", "")]:
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={sym}&outputsize={outputsize}&apikey={_AV_KEY}"
            )
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue

            data = resp.json()
            ts   = data.get("Time Series (Daily)", {})
            if not ts:
                continue

            records = []
            for date_str, vals in ts.items():
                records.append({
                    "Date":   pd.to_datetime(date_str),
                    "Open":   float(vals.get("1. open",  0)),
                    "High":   float(vals.get("2. high",  0)),
                    "Low":    float(vals.get("3. low",   0)),
                    "Close":  float(vals.get("5. adjusted close",
                                             vals.get("4. close", 0))),
                    "Volume": float(vals.get("6. volume", 0)),
                })

            df     = pd.DataFrame(records).set_index("Date").sort_index()
            days   = _PERIOD_DAYS.get(period, 180)
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
            df     = df[df.index >= cutoff]

            if not df.empty:
                return df

    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# Source 3 — Stooq via pandas_datareader
# ─────────────────────────────────────────────────────────────

def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch from Stooq using pandas_datareader."""
    try:
        import pandas_datareader.data as web
        from datetime import datetime, timedelta

        days  = _PERIOD_DAYS.get(period, 180)
        end   = datetime.today()
        start = end - timedelta(days=days)

        # Stooq uses lowercase tickers
        symbol = ticker.lower()
        df     = web.DataReader(symbol, "stooq", start, end)

        if df is None or df.empty:
            return None

        df = df.sort_index()
        df.columns = [c.strip().title() for c in df.columns]

        if "Close" not in df.columns:
            return None

        return df.dropna(subset=["Close"])

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Main Waterfall Entry Point
# ─────────────────────────────────────────────────────────────

_SOURCE_NAMES = {
    "yahoo":         "Yahoo Finance",
    "alpha_vantage": "Alpha Vantage",
    "stooq":         "Stooq",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> tuple:
    """
    Try each data source in order. Returns (DataFrame, source_name).
    Returns (None, 'none') only if all sources fail.
    """
    sources = [
        ("yahoo",         lambda: _fetch_yahoo(ticker, period)),
        ("alpha_vantage", lambda: _fetch_alpha_vantage(ticker, period)),
        ("stooq",         lambda: _fetch_stooq(ticker, period)),
    ]

    for key, fn in sources:
        try:
            data = fn()
            if data is not None and not data.empty and "Close" in data.columns:
                return data, _SOURCE_NAMES[key]
        except Exception:
            continue

    return None, "none"


# ─────────────────────────────────────────────────────────────
# Portfolio Multi-Ticker Fetch
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_data(tickers: tuple, period: str = PORTFOLIO_PERIOD) -> tuple:
    """
    Fetch close prices for multiple tickers.
    Returns (prices_df, source_map {ticker: source_name}).
    """
    tickers    = list(tickers)
    prices     = {}
    source_map = {}

    # Try Yahoo batch first
    try:
        import yfinance as yf
        session = _make_yf_session()
        batch   = yf.download(
            tickers, period=period,
            auto_adjust=True, progress=False,
            session=session,
        )
        if batch is not None and not batch.empty:
            closes = _extract_batch_closes(batch, tickers)
            for t, s in closes.items():
                if s is not None and len(s) > 10:
                    prices[t]     = s
                    source_map[t] = "Yahoo Finance"
    except Exception:
        pass

    # Individual fallback for missing tickers
    missing = [t for t in tickers if t not in prices]
    for ticker in missing:
        data, source = fetch_stock_data(ticker, period)
        if data is not None and "Close" in data.columns:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            prices[ticker]     = pd.to_numeric(close, errors="coerce")
            source_map[ticker] = source

    if not prices:
        return None, {}

    df = pd.DataFrame(prices)
    df = df.ffill(limit=3).dropna(how="all").dropna(axis=1, how="all")
    return (df if not df.empty else None), source_map


def _extract_batch_closes(data: pd.DataFrame, tickers: list) -> dict:
    result = {}
    if not isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns and len(tickers) == 1:
            result[tickers[0]] = pd.to_numeric(data["Close"], errors="coerce")
        return result

    level0 = set(data.columns.get_level_values(0))
    level1 = set(data.columns.get_level_values(1))

    for t in tickers:
        try:
            if "Close" in level0:
                close_df = data["Close"]
                if isinstance(close_df, pd.DataFrame) and t in close_df.columns:
                    result[t] = pd.to_numeric(close_df[t], errors="coerce")
                elif isinstance(close_df, pd.Series):
                    result[tickers[0]] = pd.to_numeric(close_df, errors="coerce")
            elif t in level0 and "Close" in level1:
                result[t] = pd.to_numeric(data[t]["Close"], errors="coerce")
        except Exception:
            continue

    return result