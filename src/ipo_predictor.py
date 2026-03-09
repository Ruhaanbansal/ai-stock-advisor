# =============================================================
# ipo_predictor.py — IPO Success Prediction Model
# =============================================================
#
# Model: GradientBoostingRegressor (listing gain) +
#        GradientBoostingRegressor (30-day return) +
#        GradientBoostingClassifier (recommendation label)
#
# Features:
#   - Subscription rates (total, QIB, HNI, retail)
#   - GMP % (grey market premium)
#   - Sector (one-hot encoded)
#   - NIFTY trend 30 days before listing
#   - Issue price (log-scaled)
#
# Training data: data/raw/ipo_historical.csv (58 IPOs, 2015-2025)
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble          import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import cross_val_score
from sklearn.metrics           import mean_absolute_error

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DATA_PATH  = "data/raw/ipo_historical.csv"
MODEL_PATH = "models/ipo_model.pkl"

SECTORS = [
    "Technology", "Fintech", "Consumer Tech", "Banking", "NBFC",
    "Pharma", "Healthcare", "Chemicals", "Manufacturing", "Infrastructure",
    "Renewable Energy", "Consumer", "FMCG", "Insurance", "Asset Management",
    "Logistics", "Retail", "Real Estate", "Defence", "Energy",
    "Auto", "EV", "Travel Tech", "HR Tech", "Media", "Telecom", "Education",
]

FEATURES = [
    "sub_total", "sub_qib", "sub_hni", "sub_retail",
    "gmp_percent", "nifty_trend_30d", "issue_price_log",
] + [f"sector_{s.replace(' ','_')}" for s in SECTORS]


# ─────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────

def _build_feature_row(
    sub_total:      float,
    sub_qib:        float,
    sub_hni:        float,
    sub_retail:     float,
    gmp_percent:    float,
    nifty_trend:    float,
    issue_price:    float,
    sector:         str,
) -> pd.DataFrame:
    """Convert raw IPO inputs into the model feature vector."""
    row = {
        "sub_total":       sub_total,
        "sub_qib":         sub_qib,
        "sub_hni":         sub_hni,
        "sub_retail":      sub_retail,
        "gmp_percent":     gmp_percent,
        "nifty_trend_30d": nifty_trend,
        "issue_price_log": np.log1p(issue_price),
    }
    # One-hot encode sector
    for s in SECTORS:
        row[f"sector_{s.replace(' ','_')}"] = 1 if s == sector else 0

    return pd.DataFrame([row])[FEATURES]


def _prepare_training_data(df: pd.DataFrame):
    """Build X, y_gain, y_return, y_label from historical CSV."""
    rows = []
    for _, r in df.iterrows():
        row = {
            "sub_total":       r["sub_total"],
            "sub_qib":         r["sub_qib"],
            "sub_hni":         r["sub_hni"],
            "sub_retail":      r["sub_retail"],
            "gmp_percent":     r["gmp_percent"],
            "nifty_trend_30d": r.get("nifty_trend_30d", 0),
            "issue_price_log": np.log1p(r["issue_price"]),
        }
        for s in SECTORS:
            row[f"sector_{s.replace(' ','_')}"] = (
                1 if r.get("sector", "") == s else 0
            )
        rows.append(row)

    X        = pd.DataFrame(rows)[FEATURES]
    y_gain   = df["listing_gain"].values
    y_return = df["return_30d"].fillna(0).values
    y_label  = df["recommendation"].values
    return X, y_gain, y_return, y_label


# ─────────────────────────────────────────────────────────────
# Train Models
# ─────────────────────────────────────────────────────────────

def _train(df: pd.DataFrame) -> dict:
    """Train all three models and return them as a dict."""
    X, y_gain, y_return, y_label = _prepare_training_data(df)

    # ── Model 1: Listing gain regressor ──────────────────────
    m_gain = GradientBoostingRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    m_gain.fit(X, y_gain)

    # ── Model 2: 30-day return regressor ─────────────────────
    m_return = GradientBoostingRegressor(
        n_estimators=150, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    m_return.fit(X, y_return)

    # ── Model 3: Recommendation classifier ───────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y_label)
    m_label = GradientBoostingClassifier(
        n_estimators=150, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    m_label.fit(X, y_enc)

    # ── Cross-val MAE for listing gain ────────────────────────
    cv_scores = cross_val_score(
        m_gain, X, y_gain,
        cv=min(5, len(df)//5),
        scoring="neg_mean_absolute_error",
    )
    mae_cv = float(-cv_scores.mean())

    return {
        "model_gain":   m_gain,
        "model_return": m_return,
        "model_label":  m_label,
        "label_encoder": le,
        "mae_cv":       round(mae_cv, 2),
        "n_train":      len(df),
        "feature_names": FEATURES,
    }


# ─────────────────────────────────────────────────────────────
# Load / Train (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_ipo_model() -> dict | None:
    """
    Load cached model if available, otherwise train from CSV.
    Returns None if training data not found.
    """
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    if df.empty or len(df) < 10:
        return None

    # Check if saved model is up to date
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_train") == len(df):
                return saved
        except Exception:
            pass

    # Train fresh
    models = _train(df)

    # Save to disk
    os.makedirs("models", exist_ok=True)
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(models, f)
    except Exception:
        pass

    return models


# ─────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────

def predict_ipo(
    models:      dict,
    sub_total:   float,
    sub_qib:     float,
    sub_hni:     float,
    sub_retail:  float,
    gmp_percent: float,
    nifty_trend: float,
    issue_price: float,
    sector:      str,
) -> dict:
    """
    Predict listing gain, 30-day return, and recommendation
    for a new IPO given its subscription and GMP data.

    Returns dict with:
        listing_gain    — predicted listing day gain %
        return_30d      — predicted 30-day post-listing return %
        recommendation  — "Strong Subscribe" / "Subscribe" / "Neutral" / "Avoid"
        confidence      — model confidence % (0-100)
        probabilities   — per-class probabilities dict
        feature_importance — top features driving this prediction
    """
    X = _build_feature_row(
        sub_total, sub_qib, sub_hni, sub_retail,
        gmp_percent, nifty_trend, issue_price, sector,
    )

    listing_gain = float(models["model_gain"].predict(X)[0])
    return_30d   = float(models["model_return"].predict(X)[0])

    # Classification
    le          = models["label_encoder"]
    proba       = models["model_label"].predict_proba(X)[0]
    pred_class  = int(np.argmax(proba))
    recommendation = le.inverse_transform([pred_class])[0]
    confidence  = float(proba[pred_class]) * 100

    # Per-class probabilities
    proba_dict = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }

    # Top 5 features driving this prediction
    importances = models["model_gain"].feature_importances_
    feat_imp    = sorted(
        zip(FEATURES, importances),
        key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "listing_gain":       round(listing_gain, 1),
        "return_30d":         round(return_30d, 1),
        "recommendation":     recommendation,
        "confidence":         round(confidence, 1),
        "probabilities":      proba_dict,
        "feature_importance": feat_imp,
        "mae_cv":             models.get("mae_cv", 0),
        "n_train":            models.get("n_train", 0),
    }


# ─────────────────────────────────────────────────────────────
# Fetch current NIFTY trend (for auto-fill)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_current_nifty_trend() -> float:
    """Return NIFTY 30-day trend % for auto-filling market conditions."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end   = datetime.today()
        start = end - timedelta(days=35)
        df    = yf.download("^NSEI",
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is not None and len(df) >= 5:
            return round(
                ((float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0]))
                 / float(df["Close"].iloc[0])) * 100, 2
            )
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────
# Similar past IPOs (for context)
# ─────────────────────────────────────────────────────────────

def find_similar_ipos(
    sub_total:  float,
    gmp_percent: float,
    sector:     str,
    n:          int = 3,
) -> pd.DataFrame:
    """Find the most similar past IPOs from training data."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Simple similarity: same sector + closest subscription
    same_sector = df[df["sector"] == sector].copy()
    if same_sector.empty:
        same_sector = df.copy()

    same_sector["similarity"] = (
        abs(same_sector["sub_total"] - sub_total) / (sub_total + 1) +
        abs(same_sector["gmp_percent"] - gmp_percent) / (abs(gmp_percent) + 1)
    )
    top = same_sector.nsmallest(n, "similarity")[
        ["name", "listing_date", "issue_price",
         "sub_total", "gmp_percent", "listing_gain", "recommendation"]
    ]
    return top.reset_index(drop=True)