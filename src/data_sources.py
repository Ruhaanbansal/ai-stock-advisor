# =============================================================
# data_sources.py — Multi-Source Stock Data Fetcher
# =============================================================
#
# Waterfall strategy:
#   1. Yahoo Finance  (yfinance)      — primary, no key needed
#   2. Stooq                          — free, no key needed
#   3. Alpha Vantage                  — requires API key
#
# Each source is tried in order. If one fails or returns empty
# data, the next is attempted automatically.
# =============================================================

import os
import time
import pandas as pd
import numpy as np
import streamlit as st

from src.config import DEFAULT_PERIOD, PORTFOLIO_PERIOD

# ── Alpha Vantage key — env var takes priority over hardcoded ──
_AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "E7RBUJ17S6H24GJO")

# Period string → approximate days mapping (for non-yfinance sources)
_PERIOD_DAYS = {
    "1mo": 30,  "3mo": 90,  "6mo": 180,
    "1y":  365, "2y":  730, "5y": 1825,
}


# ─────────────────────────────────────────────────────────────
# Helper — normalise NSE ticker for each source
# ─────────────────────────────────────────────────────────────

def _to_stooq_symbol(ticker: str) -> str:
    """
    Convert NSE ticker to stooq format.
    RELIANCE.NS → RELIANCE.NS (stooq uses same format for NSE)
    """
    return ticker.replace(".NS", ".NS").replace(".BO", ".BO")


def _to_av_symbol(ticker: str) -> str:
    """
    Alpha Vantage uses BSE format for Indian stocks: e.g. RELIANCE.BSE
    NSE tickers need .NS → .NSE or just the base symbol
    """
    base = ticker.replace(".NS", "").replace(".BO", "")
    return f"{base}.NSE"


# ─────────────────────────────────────────────────────────────
# Source 1 — Yahoo Finance
# ─────────────────────────────────────────────────────────────

def _fetch_yahoo(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data is None or data.empty:
            return None

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(data.columns.get_level_values(0))
            level1 = set(data.columns.get_level_values(1))
            known  = {"Close", "Open", "High", "Low", "Volume"}
            if level0 & known:
                data.columns = data.columns.get_level_values(0)
            elif level1 & known:
                data.columns = data.columns.get_level_values(1)
        data = data.loc[:, ~data.columns.duplicated()]

        if "Close" not in data.columns:
            return None

        return data.dropna(subset=["Close"])

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Source 2 — Stooq (free, no API key)
# ─────────────────────────────────────────────────────────────

def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Stooq provides free historical data for NSE stocks.
    URL format: https://stooq.com/q/d/l/?s=RELIANCE.NS&i=d
    """
    try:
        import requests
        from io import StringIO

        # Stooq uses lowercase tickers
        symbol = ticker.lower()
        url    = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or "No data" in resp.text:
            return None

        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df.columns = [c.strip().title() for c in df.columns]

        # Filter to period
        days = _PERIOD_DAYS.get(period, 180)
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]

        return df if not df.empty else None

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Source 3 — Alpha Vantage
# ─────────────────────────────────────────────────────────────

def _fetch_alpha_vantage(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED endpoint.
    Returns up to 5 years of daily data (outputsize=full).
    """
    try:
        import requests

        symbol     = _to_av_symbol(ticker)
        outputsize = "compact" if _PERIOD_DAYS.get(period, 180) <= 100 else "full"
        url        = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}"
            f"&outputsize={outputsize}"
            f"&apikey={_AV_KEY}"
        )

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()

        # Alpha Vantage returns error messages in JSON
        if "Error Message" in data or "Note" in data or "Information" in data:
            # Try plain symbol without exchange suffix as fallback
            base_symbol = ticker.replace(".NS", "").replace(".BO", "")
            url2 = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={base_symbol}"
                f"&outputsize={outputsize}"
                f"&apikey={_AV_KEY}"
            )
            resp2 = requests.get(url2, timeout=15)
            if resp2.status_code != 200:
                return None
            data = resp2.json()
            if "Error Message" in data or "Note" in data or "Information" in data:
                return None

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None

        records = []
        for date_str, vals in ts.items():
            records.append({
                "Date":   pd.to_datetime(date_str),
                "Open":   float(vals.get("1. open",  0)),
                "High":   float(vals.get("2. high",  0)),
                "Low":    float(vals.get("3. low",   0)),
                "Close":  float(vals.get("5. adjusted close", vals.get("4. close", 0))),
                "Volume": float(vals.get("6. volume", 0)),
            })

        df = pd.DataFrame(records).set_index("Date").sort_index()

        # Filter to requested period
        days   = _PERIOD_DAYS.get(period, 180)
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        df     = df[df.index >= cutoff]

        return df if not df.empty else None

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Main Entry — Waterfall Fetch
# ─────────────────────────────────────────────────────────────

_SOURCE_NAMES = {
    "yahoo":         "Yahoo Finance",
    "stooq":         "Stooq",
    "alpha_vantage": "Alpha Vantage",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> tuple[pd.DataFrame | None, str]:
    """
    Try each data source in order and return the first successful result.
    Returns (DataFrame, source_name) or (None, "none") if all fail.
    """
    sources = [
        ("yahoo",         lambda: _fetch_yahoo(ticker, period)),
        ("stooq",         lambda: _fetch_stooq(ticker, period)),
        ("alpha_vantage", lambda: _fetch_alpha_vantage(ticker, period)),
    ]

    for source_key, fetch_fn in sources:
        try:
            data = fetch_fn()
            if data is not None and not data.empty and "Close" in data.columns:
                return data, _SOURCE_NAMES[source_key]
        except Exception:
            continue

    return None, "none"


# ─────────────────────────────────────────────────────────────
# Portfolio Multi-Ticker Fetch
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_data(
    tickers: tuple,
    period:  str = PORTFOLIO_PERIOD,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Fetch close prices for multiple tickers using the waterfall strategy.
    Returns (prices_df, source_map) where source_map = {ticker: source_name}.
    """
    tickers    = list(tickers)
    prices     = {}
    source_map = {}

    # ── Try Yahoo batch first (fastest) ───────────────────────
    try:
        import yfinance as yf
        batch = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if batch is not None and not batch.empty:
            closes = _extract_batch_closes(batch, tickers)
            for t, s in closes.items():
                if s is not None and len(s) > 10:
                    prices[t]     = s
                    source_map[t] = "Yahoo Finance"
    except Exception:
        pass

    # ── Fallback: fetch missing tickers individually ──────────
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
    """Extract per-ticker Close series from a yfinance batch download."""
    result = {}

    if not isinstance(data.columns, pd.MultiIndex):
        # Flat — only one ticker
        if "Close" in data.columns and len(tickers) == 1:
            result[tickers[0]] = pd.to_numeric(data["Close"], errors="coerce")
        return result

    level0 = set(data.columns.get_level_values(0))
    level1 = set(data.columns.get_level_values(1))

    for t in tickers:
        try:
            if "Close" in level0:
                # (metric, ticker) format
                close_df = data["Close"]
                if isinstance(close_df, pd.DataFrame) and t in close_df.columns:
                    result[t] = pd.to_numeric(close_df[t], errors="coerce")
                elif isinstance(close_df, pd.Series):
                    result[tickers[0]] = pd.to_numeric(close_df, errors="coerce")
            elif t in level0 and "Close" in level1:
                # (ticker, metric) format
                result[t] = pd.to_numeric(data[t]["Close"], errors="coerce")
        except Exception:
            continue

    return result