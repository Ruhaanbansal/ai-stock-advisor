# =============================================================
# data_sources.py — Multi-Source Stock Data Fetcher
# =============================================================
#
# Waterfall strategy:
#   1. Alpha Vantage  (most reliable on Streamlit Cloud)
#   2. Yahoo Finance  (with browser session spoofing)
#   3. Stooq          (via direct CSV download)
# =============================================================

import os
import pandas as pd
import streamlit as st
import requests
import logging

from src.config import DEFAULT_PERIOD, PORTFOLIO_PERIOD

logger  = logging.getLogger(__name__)
_AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "E7RBUJ17S6H24GJO")

_PERIOD_DAYS = {
    "1mo": 30,  "3mo": 90,  "6mo": 180,
    "1y": 365,  "2y": 730,  "5y": 1825,
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


# ─────────────────────────────────────────────────────────────
# Source 1 — Alpha Vantage (PRIMARY on cloud)
# ─────────────────────────────────────────────────────────────

def _to_av_symbols(ticker: str) -> list[str]:
    """Return all symbol formats to try for Alpha Vantage."""
    base = ticker.replace(".NS", "").replace(".BO", "")
    return [
        f"{base}.BSE",   # BSE format (most supported)
        f"{base}.NSE",   # NSE format
        base,            # bare symbol
    ]


def _fetch_alpha_vantage(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch from Alpha Vantage — works reliably on Streamlit Cloud."""
    outputsize = "compact" if _PERIOD_DAYS.get(period, 180) <= 100 else "full"
    days       = _PERIOD_DAYS.get(period, 365)
    cutoff     = pd.Timestamp.today() - pd.Timedelta(days=days)

    for sym in _to_av_symbols(ticker):
        try:
            url  = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={sym}&outputsize={outputsize}&apikey={_AV_KEY}"
            )
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            if resp.status_code != 200:
                continue

            raw = resp.json()

            # Detect API errors / rate limits
            if any(k in raw for k in ("Error Message", "Note", "Information")):
                msg = raw.get("Note") or raw.get("Information") or raw.get("Error Message", "")
                logger.warning(f"Alpha Vantage [{sym}]: {msg[:120]}")
                continue

            ts = raw.get("Time Series (Daily)", {})
            if not ts:
                continue

            rows = []
            for date_str, v in ts.items():
                rows.append({
                    "Date":   pd.to_datetime(date_str),
                    "Open":   float(v.get("1. open",  0)),
                    "High":   float(v.get("2. high",  0)),
                    "Low":    float(v.get("3. low",   0)),
                    "Close":  float(v.get("5. adjusted close",
                                          v.get("4. close", 0))),
                    "Volume": float(v.get("6. volume", 0)),
                })

            df = (pd.DataFrame(rows)
                    .set_index("Date")
                    .sort_index()
                    .pipe(lambda d: d[d.index >= cutoff]))

            if not df.empty and len(df) > 5:
                logger.info(f"Alpha Vantage OK: {sym} → {len(df)} rows")
                return df

        except Exception as e:
            logger.warning(f"Alpha Vantage [{sym}] error: {e}")
            continue

    return None


# ─────────────────────────────────────────────────────────────
# Source 2 — Yahoo Finance (session spoofing)
# ─────────────────────────────────────────────────────────────

def _fetch_yahoo(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch from Yahoo Finance with a browser-spoofed session."""
    try:
        import yfinance as yf

        session = requests.Session()
        session.headers.update(_HEADERS)

        # Prefer .history() over .download() — more reliable on cloud
        tk   = yf.Ticker(ticker, session=session)
        data = tk.history(period=period, auto_adjust=True, raise_errors=False)

        if data is None or data.empty:
            data = yf.download(
                ticker, period=period,
                auto_adjust=True, progress=False,
                session=session,
            )

        if data is None or data.empty:
            return None

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            known  = {"Close", "Open", "High", "Low", "Volume"}
            level0 = set(data.columns.get_level_values(0))
            level1 = set(data.columns.get_level_values(1))
            if level0 & known:
                data.columns = data.columns.get_level_values(0)
            elif level1 & known:
                data.columns = data.columns.get_level_values(1)
        data = data.loc[:, ~data.columns.duplicated()]

        if "Close" not in data.columns:
            return None

        result = data.dropna(subset=["Close"])
        if len(result) > 5:
            logger.info(f"Yahoo Finance OK: {ticker} → {len(result)} rows")
            return result

    except Exception as e:
        logger.warning(f"Yahoo Finance [{ticker}] error: {e}")

    return None


# ─────────────────────────────────────────────────────────────
# Source 3 — Stooq (direct CSV)
# ─────────────────────────────────────────────────────────────

def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch from Stooq via direct CSV download."""
    from io import StringIO
    from datetime import datetime, timedelta

    days  = _PERIOD_DAYS.get(period, 365)
    start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
    end   = datetime.today().strftime("%Y%m%d")

    # Stooq symbol formats for Indian stocks
    symbol_variants = [
        ticker.lower(),                            # reliance.ns
        ticker.replace(".NS", ".ns").lower(),      # reliance.ns
        ticker.replace(".NS", "").lower() + ".ns", # reliance.ns
    ]

    for sym in symbol_variants:
        try:
            url  = f"https://stooq.com/q/d/l/?s={sym}&d1={start}&d2={end}&i=d"
            resp = requests.get(url, headers=_HEADERS, timeout=15)

            if resp.status_code != 200 or len(resp.text) < 50:
                continue
            if "No data" in resp.text or "Przekroczon" in resp.text:
                continue

            df = pd.read_csv(StringIO(resp.text))
            if df.empty or "Close" not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            if len(df) > 5:
                logger.info(f"Stooq OK: {sym} → {len(df)} rows")
                return df

        except Exception as e:
            logger.warning(f"Stooq [{sym}] error: {e}")
            continue

    return None


# ─────────────────────────────────────────────────────────────
# Main Waterfall — Alpha Vantage first on cloud
# ─────────────────────────────────────────────────────────────

_SOURCE_NAMES = {
    "alpha_vantage": "Alpha Vantage",
    "yahoo":         "Yahoo Finance",
    "stooq":         "Stooq",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> tuple:
    """
    Try each data source in order. Returns (DataFrame, source_name).
    Alpha Vantage is tried first since it works reliably on Streamlit Cloud.
    """
    sources = [
        ("alpha_vantage", lambda: _fetch_alpha_vantage(ticker, period)),
        ("yahoo",         lambda: _fetch_yahoo(ticker, period)),
        ("stooq",         lambda: _fetch_stooq(ticker, period)),
    ]

    errors = []
    for key, fn in sources:
        try:
            data = fn()
            if data is not None and not data.empty and "Close" in data.columns:
                return data, _SOURCE_NAMES[key]
            errors.append(f"{_SOURCE_NAMES[key]}: empty response")
        except Exception as e:
            errors.append(f"{_SOURCE_NAMES[key]}: {e}")

    logger.error(f"All sources failed for {ticker}: {errors}")
    return None, "none"


# ─────────────────────────────────────────────────────────────
# Portfolio Multi-Ticker Fetch
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_data(tickers: tuple, period: str = PORTFOLIO_PERIOD) -> tuple:
    """Fetch close prices for multiple tickers using the waterfall strategy."""
    tickers    = list(tickers)
    prices     = {}
    source_map = {}

    # Try Yahoo batch first (fastest when it works)
    try:
        import yfinance as yf
        session = requests.Session()
        session.headers.update(_HEADERS)

        batch = yf.download(
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

    # Individual fallback for any missing tickers
    for ticker in [t for t in tickers if t not in prices]:
        data, source = fetch_stock_data(ticker, period)
        if data is not None and "Close" in data.columns:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            prices[ticker]     = pd.to_numeric(close, errors="coerce")
            source_map[ticker] = source

    if not prices:
        return None, {}

    df = pd.DataFrame(prices).ffill(limit=3).dropna(how="all").dropna(axis=1, how="all")
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