# =============================================================
# explainability.py — Feature Importance (no SHAP/TensorFlow)
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config    import SEQUENCE_LENGTH
from src.features  import calculate_rsi


# ─────────────────────────────────────────────────────────────
# Feature Importance via Permutation
# ─────────────────────────────────────────────────────────────

def compute_feature_importance(
    model, scaler, close_prices: pd.Series
) -> dict[str, float]:
    """
    Estimate feature importance using permutation importance.
    Shuffles each feature and measures prediction error increase.
    Works with any scikit-learn compatible model.
    """
    from src.model import _build_features

    X, y, _, _ = _build_features(close_prices)
    if len(X) == 0:
        return {}

    baseline_preds = model.predict(X)
    baseline_err   = np.mean((baseline_preds - y) ** 2)

    # Feature names: last seq_len lags + 5 rolling features
    feature_names = (
        [f"Lag {i+1}"  for i in range(SEQUENCE_LENGTH)] +
        ["Roll Mean 5d", "Roll Mean 10d", "Roll Mean 20d", "Roll Std 5d", "1d Return"]
    )

    importance = {}
    for i, name in enumerate(feature_names):
        X_perm       = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_err     = np.mean((model.predict(X_perm) - y) ** 2)
        importance[name] = max(0.0, perm_err - baseline_err)

    # Normalise to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


# ─────────────────────────────────────────────────────────────
# Technical Indicator Signals (independent of model)
# ─────────────────────────────────────────────────────────────

def get_technical_signals(close_prices: pd.Series) -> dict:
    """
    Derive human-readable signal labels from technical indicators.
    Returns a dict of {indicator: (value, signal_label)}.
    """
    signals = {}

    # RSI
    try:
        rsi_series = calculate_rsi(close_prices)
        rsi_val    = float(rsi_series.iloc[-1])
        if rsi_val > 70:
            rsi_sig = "Overbought"
        elif rsi_val < 30:
            rsi_sig = "Oversold"
        else:
            rsi_sig = "Neutral"
        signals["RSI (14d)"] = (round(rsi_val, 1), rsi_sig)
    except Exception:
        pass

    # 50-day vs 200-day moving average
    try:
        ma50  = float(close_prices.rolling(50).mean().iloc[-1])
        ma200 = float(close_prices.rolling(200).mean().iloc[-1])
        cross = "Golden Cross ↑" if ma50 > ma200 else "Death Cross ↓"
        signals["MA50 vs MA200"] = (round(ma50 / ma200, 3), cross)
    except Exception:
        pass

    # Price momentum (20-day return)
    try:
        mom = (float(close_prices.iloc[-1]) / float(close_prices.iloc[-20]) - 1) * 100
        signals["20d Momentum"] = (round(mom, 2), "Bullish" if mom > 0 else "Bearish")
    except Exception:
        pass

    # Volatility (20-day rolling std as %)
    try:
        vol = float(close_prices.pct_change().rolling(20).std().iloc[-1]) * 100
        signals["Volatility (20d)"] = (round(vol, 2), "High" if vol > 2 else "Normal")
    except Exception:
        pass

    return signals


# ─────────────────────────────────────────────────────────────
# SHAP-style Importance Chart
# ─────────────────────────────────────────────────────────────

def shap_chart(importance: dict) -> go.Figure:
    """
    Horizontal bar chart showing the top feature importances.
    Styled to look like a SHAP summary plot.
    """
    if not importance:
        return go.Figure()

    # Show top 10 most important features
    top    = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [t[0] for t in top]
    values = [t[1] for t in top]

    colors = [
        f"rgba(0, 212, 170, {0.4 + 0.6 * v / max(values)})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=max(300, len(labels) * 32),
        margin=dict(l=0, r=60, t=20, b=0),
        xaxis=dict(title="Importance", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig