# =============================================================
# fii_dii_analyzer.py — FII/DII Pattern Detection & Prediction
# =============================================================
#
# Two engines:
#   1. Pattern Detector — finds historical patterns like
#      "when FII buys 3 consecutive days, NIFTY rises X% next day"
#   2. ML Predictor — GradientBoosting model that predicts
#      next-day NIFTY direction based on FII/DII flows
#
# Data: data/raw/fii_dii_historical.csv (5 years, ~1280 days)
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble       import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import accuracy_score

DATA_PATH  = "data/raw/fii_dii_historical.csv"
MODEL_PATH = "models/fii_dii_model.pkl"


# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_fii_dii_data() -> pd.DataFrame | None:
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from raw FII/DII data."""
    f = pd.DataFrame()

    # Raw flows
    f["fii_net"]          = df["fii_net"]
    f["dii_net"]          = df["dii_net"]
    f["fii_dii_net"]      = df["fii_net"] + df["dii_net"]

    # Rolling sums
    f["fii_3d"]           = df["fii_net"].rolling(3).sum()
    f["fii_5d"]           = df["fii_net"].rolling(5).sum()
    f["fii_10d"]          = df["fii_net"].rolling(10).sum()
    f["dii_3d"]           = df["dii_net"].rolling(3).sum()
    f["dii_5d"]           = df["dii_net"].rolling(5).sum()

    # Momentum
    f["fii_momentum"]     = df["fii_net"] - df["fii_net"].shift(5)
    f["dii_momentum"]     = df["dii_net"] - df["dii_net"].shift(5)

    # Consecutive buy/sell streaks
    f["fii_streak"]       = (
        df["fii_net"].gt(0)
        .groupby((df["fii_net"].gt(0) != df["fii_net"].gt(0).shift()).cumsum())
        .cumsum() * np.where(df["fii_net"].gt(0), 1, -1)
    )
    f["dii_streak"]       = (
        df["dii_net"].gt(0)
        .groupby((df["dii_net"].gt(0) != df["dii_net"].gt(0).shift()).cumsum())
        .cumsum() * np.where(df["dii_net"].gt(0), 1, -1)
    )

    # FII/DII ratio (who is dominating)
    total = df["fii_net"].abs() + df["dii_net"].abs() + 1
    f["fii_dominance"]    = df["fii_net"] / total
    f["dii_dominance"]    = df["dii_net"] / total

    # NIFTY context
    if "nifty_return" in df.columns:
        f["nifty_return_1d"]  = df["nifty_return"]
        f["nifty_return_5d"]  = df["nifty_return"].rolling(5).sum()
        f["nifty_return_10d"] = df["nifty_return"].rolling(10).sum()

    # FII vs NIFTY divergence
    if "nifty_return" in df.columns:
        f["fii_nifty_div"]    = np.sign(df["fii_net"]) - np.sign(df["nifty_return"])

    return f


# ─────────────────────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────────────────────

def _train(df: pd.DataFrame) -> dict:
    """Train FII/DII → NIFTY direction prediction model."""
    df = df.dropna(subset=["nifty_next_return"]).copy()

    X = _build_features(df).dropna()
    valid_idx = X.index
    df = df.loc[valid_idx]

    # Target 1: direction (up/down)
    y_dir = (df["nifty_next_return"] > 0).astype(int)

    # Target 2: magnitude
    y_mag = df["nifty_next_return"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Direction classifier
    m_dir = GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    m_dir.fit(X_scaled, y_dir)

    # Magnitude regressor
    m_mag = GradientBoostingRegressor(
        n_estimators=150, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    m_mag.fit(X_scaled, y_mag)

    # Cross-val accuracy
    cv_acc = cross_val_score(
        m_dir, X_scaled, y_dir,
        cv=5, scoring="accuracy"
    ).mean()

    return {
        "model_dir":    m_dir,
        "model_mag":    m_mag,
        "scaler":       scaler,
        "feature_names": list(X.columns),
        "cv_accuracy":  round(float(cv_acc) * 100, 1),
        "n_train":      len(df),
    }


# ─────────────────────────────────────────────────────────────
# Load / Train (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_fii_dii_model() -> dict | None:
    df = load_fii_dii_data()
    if df is None or len(df) < 50:
        return None

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_train") == len(df):
                return saved
        except Exception:
            pass

    models = _train(df)
    os.makedirs("models", exist_ok=True)
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)
    except Exception:
        pass

    return models


# ─────────────────────────────────────────────────────────────
# Predict Next Day
# ─────────────────────────────────────────────────────────────

def predict_next_day(models: dict, df: pd.DataFrame) -> dict:
    """
    Predict tomorrow's NIFTY direction based on today's FII/DII flows.
    Uses the last row of the dataset as input.
    """
    features = _build_features(df).dropna()
    if features.empty:
        return {}

    X_last  = features.iloc[[-1]]
    X_scaled = models["scaler"].transform(X_last)

    direction_proba = models["model_dir"].predict_proba(X_scaled)[0]
    direction       = int(np.argmax(direction_proba))
    magnitude       = float(models["model_mag"].predict(X_scaled)[0])
    confidence      = float(direction_proba[direction]) * 100

    return {
        "direction":    "Bullish" if direction == 1 else "Bearish",
        "magnitude":    round(magnitude, 2),
        "confidence":   round(confidence, 1),
        "prob_up":      round(float(direction_proba[1]) * 100, 1),
        "prob_down":    round(float(direction_proba[0]) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# Pattern Detection Engine
# ─────────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Find statistically significant FII/DII patterns.
    Returns list of patterns with win rate, avg return, sample size.
    """
    df = df.dropna(subset=["fii_net", "dii_net", "nifty_next_return"]).copy()
    patterns = []

    def pattern_stats(mask, name: str, description: str) -> dict | None:
        subset = df[mask]
        if len(subset) < 10:
            return None
        avg_return  = float(subset["nifty_next_return"].mean())
        win_rate    = float((subset["nifty_next_return"] > 0).mean() * 100)
        sample_size = len(subset)
        std         = float(subset["nifty_next_return"].std())
        return {
            "name":        name,
            "description": description,
            "avg_return":  round(avg_return, 2),
            "win_rate":    round(win_rate, 1),
            "sample_size": sample_size,
            "std":         round(std, 2),
            "signal":      "Bullish" if avg_return > 0.1 else (
                           "Bearish" if avg_return < -0.1 else "Neutral"),
        }

    # ── Pattern 1: FII heavy buying ──────────────────────────
    fii_75th = df["fii_net"].quantile(0.75)
    p = pattern_stats(
        df["fii_net"] > fii_75th,
        "FII Heavy Buying",
        f"FII net buy > ₹{fii_75th:.0f} Cr"
    )
    if p: patterns.append(p)

    # ── Pattern 2: FII heavy selling ─────────────────────────
    fii_25th = df["fii_net"].quantile(0.25)
    p = pattern_stats(
        df["fii_net"] < fii_25th,
        "FII Heavy Selling",
        f"FII net sell < ₹{fii_25th:.0f} Cr"
    )
    if p: patterns.append(p)

    # ── Pattern 3: FII buys 3 consecutive days ───────────────
    fii_3d_buy = (
        df["fii_net"].gt(0) &
        df["fii_net"].shift(1).gt(0) &
        df["fii_net"].shift(2).gt(0)
    )
    p = pattern_stats(fii_3d_buy, "FII Buys 3 Days in a Row",
                      "FII net positive for 3 consecutive days")
    if p: patterns.append(p)

    # ── Pattern 4: FII sells 3 consecutive days ──────────────
    fii_3d_sell = (
        df["fii_net"].lt(0) &
        df["fii_net"].shift(1).lt(0) &
        df["fii_net"].shift(2).lt(0)
    )
    p = pattern_stats(fii_3d_sell, "FII Sells 3 Days in a Row",
                      "FII net negative for 3 consecutive days")
    if p: patterns.append(p)

    # ── Pattern 5: FII buys 5 consecutive days ───────────────
    fii_5d_buy = all([
        df["fii_net"].shift(i).gt(0) for i in range(5)
    ]) if False else df["fii_net"].rolling(5).min().gt(0)
    p = pattern_stats(fii_5d_buy, "FII Buys 5 Days in a Row",
                      "Strong FII accumulation signal")
    if p: patterns.append(p)

    # ── Pattern 6: DII buys while FII sells ──────────────────
    p = pattern_stats(
        df["dii_net"].gt(0) & df["fii_net"].lt(0),
        "DII Buying, FII Selling",
        "Domestic institutions defending while foreigners exit"
    )
    if p: patterns.append(p)

    # ── Pattern 7: Both FII and DII buying ───────────────────
    p = pattern_stats(
        df["fii_net"].gt(0) & df["dii_net"].gt(0),
        "Both FII & DII Buying",
        "Strong consensus buying — most bullish signal"
    )
    if p: patterns.append(p)

    # ── Pattern 8: Both FII and DII selling ──────────────────
    p = pattern_stats(
        df["fii_net"].lt(0) & df["dii_net"].lt(0),
        "Both FII & DII Selling",
        "Strong consensus selling — most bearish signal"
    )
    if p: patterns.append(p)

    # ── Pattern 9: FII buys after 5-day sell streak ──────────
    fii_5d_neg = df["fii_net"].rolling(5).max().shift(1).lt(0)
    p = pattern_stats(
        fii_5d_neg & df["fii_net"].gt(0),
        "FII Reversal (Buy after 5-day sell)",
        "FII turns buyer after extended selling — contrarian signal"
    )
    if p: patterns.append(p)

    # ── Pattern 10: Mega FII sell (>₹5000 Cr) ───────────────
    p = pattern_stats(
        df["fii_net"] < -5000,
        "Mega FII Sell (>₹5000 Cr)",
        "Panic selling or large block exits by FIIs"
    )
    if p: patterns.append(p)

    return sorted(patterns, key=lambda x: abs(x["avg_return"]), reverse=True)


# ─────────────────────────────────────────────────────────────
# Live FII/DII Fetch (today's data from NSE)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_fii_dii() -> dict | None:
    """Fetch today's / latest FII/DII data from NSE India."""
    try:
        import requests
        from datetime import datetime, timedelta

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Referer":    "https://www.nseindia.com/",
            "Accept":     "application/json",
        })
        session.get("https://www.nseindia.com/", timeout=10)

        end   = datetime.today()
        start = end - timedelta(days=7)
        url   = (
            "https://www.nseindia.com/api/fiidiiTradeReact"
            f"?startDate={start.strftime('%d-%b-%Y')}"
            f"&endDate={end.strftime('%d-%b-%Y')}"
        )
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        if not data:
            return None

        def _safe_float(val) -> float:
            try:
                return float(str(val).replace(",","").replace("−","-").strip() or "0")
            except Exception:
                return 0.0

        # Find most recent entry with non-zero data
        latest = None
        for entry in data:
            fii_n = _safe_float(entry.get("fiiNetValue") or
                                 entry.get("fii_net") or 0)
            dii_n = _safe_float(entry.get("diiNetValue") or
                                 entry.get("dii_net") or 0)
            if abs(fii_n) > 0 or abs(dii_n) > 0:
                latest = entry
                break

        if latest is None:
            return None   # all zeros — trigger fallback

        # Handle both camelCase and snake_case keys
        def _get(d, *keys):
            for k in keys:
                if k in d:
                    return _safe_float(d[k])
            return 0.0

        return {
            "date":     latest.get("date",""),
            "fii_buy":  _get(latest, "fiiBuyValue",  "fii_buy"),
            "fii_sell": _get(latest, "fiiSellValue", "fii_sell"),
            "fii_net":  _get(latest, "fiiNetValue",  "fii_net"),
            "dii_buy":  _get(latest, "diiBuyValue",  "dii_buy"),
            "dii_sell": _get(latest, "diiSellValue", "dii_sell"),
            "dii_net":  _get(latest, "diiNetValue",  "dii_net"),
            "all_days": data[:10],
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Monthly Summary
# ─────────────────────────────────────────────────────────────

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FII/DII data by month for trend chart."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        fii_net=("fii_net", "sum"),
        dii_net=("dii_net", "sum"),
        nifty_return=("nifty_return", "sum"),
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    return monthly.tail(36)  # last 3 years


# ─────────────────────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> dict:
    """Compute correlations between FII/DII flows and NIFTY returns."""
    df = df.dropna(subset=["fii_net", "dii_net",
                            "nifty_return", "nifty_next_return"])
    return {
        "fii_same_day":   round(float(df["fii_net"].corr(df["nifty_return"])), 3),
        "fii_next_day":   round(float(df["fii_net"].corr(df["nifty_next_return"])), 3),
        "dii_same_day":   round(float(df["dii_net"].corr(df["nifty_return"])), 3),
        "dii_next_day":   round(float(df["dii_net"].corr(df["nifty_next_return"])), 3),
        "fii_dii_corr":   round(float(df["fii_net"].corr(df["dii_net"])), 3),
    }