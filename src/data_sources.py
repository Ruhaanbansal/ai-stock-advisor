# =============================================================
# data_sources.py — Resilient Multi-Source Stock Data Fetcher
# =============================================================
#
# Strategy for Streamlit Cloud compatibility:
#
#   1. yfinance with curl_cffi (bypasses TLS fingerprinting blocks)
#   2. yfinance with requests session + browser headers
#   3. Alpha Vantage REST API
#   4. Stooq direct CSV
#   5. Bundled CSV fallback (repo data/raw/ folder)
#
# curl_cffi mimics a real Chrome TLS fingerprint — this is the
# most reliable way to bypass cloud IP blocks on Yahoo Finance.
# =============================================================

import os
import pandas as pd
import streamlit as st
import logging
from datetime import datetime, timedelta

logger  = logging.getLogger(__name__)
_AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "E7RBUJ17S6H24GJO")

_PERIOD_DAYS = {
    "1mo": 30,  "3mo": 90,  "6mo": 180,
    "1y": 365,  "2y": 730,  "5y": 1825,
}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _flatten(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data
    known  = {"Close","Open","High","Low","Volume"}
    l0, l1 = set(data.columns.get_level_values(0)), set(data.columns.get_level_values(1))
    if l0 & known:
        data.columns = data.columns.get_level_values(0)
    elif l1 & known:
        data.columns = data.columns.get_level_values(1)
    return data.loc[:, ~data.columns.duplicated()]


def _valid(df) -> bool:
    return df is not None and not df.empty and "Close" in df.columns and len(df) > 5


def _cutoff(period: str) -> pd.Timestamp:
    days = _PERIOD_DAYS.get(period, 365)
    return pd.Timestamp.today() - pd.Timedelta(days=days)


# ─────────────────────────────────────────────────────────────
# Source 1 — yfinance with curl_cffi (Chrome TLS fingerprint)
# ─────────────────────────────────────────────────────────────

def _fetch_yf_curl(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Uses curl_cffi to impersonate Chrome's TLS fingerprint.
    This is the most reliable bypass for Yahoo Finance on cloud IPs.
    curl_cffi is already in yfinance's dependencies.
    """
    try:
        from curl_cffi import requests as curl_requests
        import yfinance as yf

        session = curl_requests.Session(impersonate="chrome120")
        tk      = yf.Ticker(ticker, session=session)
        data    = tk.history(period=period, auto_adjust=True)

        if _valid(data):
            data = _flatten(data)
            logger.info(f"curl_cffi OK: {ticker} → {len(data)} rows")
            return data.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"curl_cffi [{ticker}]: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Source 2 — yfinance with requests session
# ─────────────────────────────────────────────────────────────

def _fetch_yf_requests(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import requests, yfinance as yf
        session = requests.Session()
        session.headers.update(_BROWSER_HEADERS)

        tk   = yf.Ticker(ticker, session=session)
        data = tk.history(period=period, auto_adjust=True)
        if not _valid(data):
            data = yf.download(ticker, period=period,
                               auto_adjust=True, progress=False, session=session)
        if _valid(data):
            data = _flatten(data)
            return data.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"yf_requests [{ticker}]: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Source 3 — Alpha Vantage
# ─────────────────────────────────────────────────────────────

def _fetch_alpha_vantage(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import requests
        base   = ticker.replace(".NS","").replace(".BO","")
        syms   = [f"{base}.BSE", f"{base}.NSE", base]
        out    = "compact" if _PERIOD_DAYS.get(period,180) <= 100 else "full"
        cutoff = _cutoff(period)

        for sym in syms:
            url  = (f"https://www.alphavantage.co/query"
                    f"?function=TIME_SERIES_DAILY_ADJUSTED"
                    f"&symbol={sym}&outputsize={out}&apikey={_AV_KEY}")
            r    = requests.get(url, headers=_BROWSER_HEADERS, timeout=20)
            raw  = r.json()
            ts   = raw.get("Time Series (Daily)", {})
            if not ts:
                continue

            rows = [{"Date": pd.to_datetime(d),
                     "Open":   float(v["1. open"]),
                     "High":   float(v["2. high"]),
                     "Low":    float(v["3. low"]),
                     "Close":  float(v.get("5. adjusted close", v["4. close"])),
                     "Volume": float(v["6. volume"])}
                    for d,v in ts.items()]
            df = (pd.DataFrame(rows).set_index("Date").sort_index()
                    .pipe(lambda d: d[d.index >= cutoff]))
            if _valid(df):
                logger.info(f"AlphaVantage OK: {sym} → {len(df)} rows")
                return df
    except Exception as e:
        logger.warning(f"AlphaVantage [{ticker}]: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Source 4 — Stooq direct CSV
# ─────────────────────────────────────────────────────────────

def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        import requests
        from io import StringIO
        days  = _PERIOD_DAYS.get(period, 365)
        end   = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
        sym   = ticker.lower()

        url  = f"https://stooq.com/q/d/l/?s={sym}&d1={start}&d2={end}&i=d"
        r    = requests.get(url, headers=_BROWSER_HEADERS, timeout=15)
        if r.status_code != 200 or len(r.text) < 50 or "No data" in r.text:
            return None

        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if _valid(df):
            logger.info(f"Stooq OK: {sym} → {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"Stooq [{ticker}]: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Source 5 — Bundled CSV fallback
# ─────────────────────────────────────────────────────────────

# Map NSE tickers to bundled CSV files in data/raw/
_CSV_MAP = {
    "RELIANCE.NS": "data/raw/RELIANCE.NS.csv",
    "TCS.NS":      "data/raw/TCS.NS.csv",
    "HDFCBANK.NS": "data/raw/HDFCBANK.NS.csv",
}

def _fetch_csv_fallback(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Load from a pre-downloaded CSV in the repo.
    This ALWAYS works — no network needed.
    Shows a warning that data may not be current.
    """
    path = _CSV_MAP.get(ticker)
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Filter to requested period
        df = df[df.index >= _cutoff(period)]

        # Normalise column names
        df.columns = [c.strip().title() for c in df.columns]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})

        if _valid(df):
            logger.info(f"CSV fallback OK: {ticker} → {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"CSV fallback [{ticker}]: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Main Waterfall
# ─────────────────────────────────────────────────────────────

_SOURCE_NAMES = {
    "curl":          "Yahoo Finance",
    "yf_requests":   "Yahoo Finance",
    "alpha_vantage": "Alpha Vantage",
    "stooq":         "Stooq",
    "csv":           "Cached Data",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> tuple:
    sources = [
        ("curl",          lambda: _fetch_yf_curl(ticker, period)),
        ("yf_requests",   lambda: _fetch_yf_requests(ticker, period)),
        ("alpha_vantage", lambda: _fetch_alpha_vantage(ticker, period)),
        ("stooq",         lambda: _fetch_stooq(ticker, period)),
        ("csv",           lambda: _fetch_csv_fallback(ticker, period)),
    ]
    for key, fn in sources:
        try:
            data = fn()
            if _valid(data):
                return data.dropna(subset=["Close"]), _SOURCE_NAMES[key]
        except Exception as e:
            logger.warning(f"Source {key} failed for {ticker}: {e}")
    return None, "none"


# ─────────────────────────────────────────────────────────────
# Portfolio Multi-Ticker Fetch
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_data(tickers: tuple, period: str = "1y") -> tuple:
    tickers    = list(tickers)
    prices     = {}
    source_map = {}

    # Try batch Yahoo first
    try:
        from curl_cffi import requests as curl_requests
        import yfinance as yf
        session = curl_requests.Session(impersonate="chrome120")
        batch   = yf.download(tickers, period=period,
                               auto_adjust=True, progress=False, session=session)
        if batch is not None and not batch.empty:
            closes = _extract_batch_closes(batch, tickers)
            for t, s in closes.items():
                if s is not None and len(s) > 10:
                    prices[t], source_map[t] = s, "Yahoo Finance"
    except Exception:
        pass

    # Individual fallback
    for t in [t for t in tickers if t not in prices]:
        data, src = fetch_stock_data(t, period)
        if data is not None and "Close" in data.columns:
            c = data["Close"]
            if isinstance(c, pd.DataFrame): c = c.squeeze()
            prices[t], source_map[t] = pd.to_numeric(c, errors="coerce"), src

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
    l0 = set(data.columns.get_level_values(0))
    l1 = set(data.columns.get_level_values(1))
    for t in tickers:
        try:
            if "Close" in l0:
                cd = data["Close"]
                if isinstance(cd, pd.DataFrame) and t in cd.columns:
                    result[t] = pd.to_numeric(cd[t], errors="coerce")
                elif isinstance(cd, pd.Series):
                    result[tickers[0]] = pd.to_numeric(cd, errors="coerce")
            elif t in l0 and "Close" in l1:
                result[t] = pd.to_numeric(data[t]["Close"], errors="coerce")
        except Exception:
            continue
    return result