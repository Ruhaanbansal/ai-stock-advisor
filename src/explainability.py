# =============================================================
# explainability.py — Model & Signal Explainability
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import SEQUENCE_LENGTH
from src.features import calculate_rsi, calculate_macd, calculate_bollinger_bands


# ─────────────────────────────────────────────────────────────
# Feature Importance via Permutation (grouped)
# ─────────────────────────────────────────────────────────────

def compute_feature_importance(
    model, scaler, close_prices: pd.Series
) -> dict[str, float]:
    """
    Permutation importance — grouped for readability.
    Groups: Short-term history, Medium-term history, Long-term history,
            Rolling stats, Return signals.
    """
    from src.model import _build_features

    X, y, _, _ = _build_features(close_prices)
    if len(X) == 0:
        return {}

    baseline_preds = model.predict(X)
    baseline_err   = np.mean((baseline_preds - y) ** 2)

    # Total feature count: SEQUENCE_LENGTH lags + 7 engineered features
    n_lags     = SEQUENCE_LENGTH
    n_features = X.shape[1]

    # Feature groups (indices)
    groups = {
        "Short-term history (1-10d)":  list(range(0, 10)),
        "Medium-term history (11-30d)": list(range(10, 30)),
        "Long-term history (31-60d)":   list(range(30, n_lags)),
        "Rolling Mean (5/10/20d)":      [n_lags, n_lags+1, n_lags+2],
        "Rolling Std (5d)":             [n_lags+3],
        "Return Signals (1/5/10d)":     [n_lags+4, n_lags+5, n_lags+6],
    }

    importance = {}
    for group_name, indices in groups.items():
        valid_idx = [i for i in indices if i < n_features]
        if not valid_idx:
            continue
        X_perm = X.copy()
        for i in valid_idx:
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_err = np.mean((model.predict(X_perm) - y) ** 2)
        importance[group_name] = max(0.0, perm_err - baseline_err)

    # Normalise to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


# ─────────────────────────────────────────────────────────────
# Technical Indicator Signals
# ─────────────────────────────────────────────────────────────

def get_technical_signals(close_prices: pd.Series, data: pd.DataFrame | None = None) -> dict:
    """
    Returns a dict of {indicator: (value, signal_label, description)}.
    Safely handles insufficient data for each indicator.
    """
    signals = {}

    # ── RSI ───────────────────────────────────────────────────
    try:
        rsi_series = calculate_rsi(close_prices)
        rsi_val    = float(rsi_series.dropna().iloc[-1])
        if rsi_val > 70:
            sig, desc = "Overbought 🔴", "RSI above 70 — potential reversal"
        elif rsi_val < 30:
            sig, desc = "Oversold 🟢", "RSI below 30 — potential rebound"
        else:
            sig, desc = "Neutral ⚪", "RSI in neutral band (30–70)"
        signals["RSI (14d)"] = (round(rsi_val, 1), sig, desc)
    except Exception:
        pass

    # ── MACD ─────────────────────────────────────────────────
    try:
        macd, macd_sig = calculate_macd(close_prices)
        m_val = float(macd.dropna().iloc[-1])
        s_val = float(macd_sig.dropna().iloc[-1])
        hist  = m_val - s_val
        if hist > 0:
            sig, desc = "Bullish Crossover 🟢", "MACD above signal line — upward momentum"
        else:
            sig, desc = "Bearish Crossover 🔴", "MACD below signal line — downward momentum"
        signals["MACD"] = (round(hist, 4), sig, desc)
    except Exception:
        pass

    # ── MA50 vs MA200 ─────────────────────────────────────────
    try:
        if len(close_prices) >= 200:
            ma50  = float(close_prices.rolling(50).mean().iloc[-1])
            ma200 = float(close_prices.rolling(200).mean().iloc[-1])
            cross = "Golden Cross 🟢" if ma50 > ma200 else "Death Cross 🔴"
            desc  = ("MA50 above MA200 — long-term bull market" if ma50 > ma200
                     else "MA50 below MA200 — long-term bear market")
            signals["MA50 vs MA200"] = (round(ma50 / ma200, 3), cross, desc)
        elif len(close_prices) >= 50:
            ma20 = float(close_prices.rolling(20).mean().iloc[-1])
            ma50 = float(close_prices.rolling(50).mean().iloc[-1])
            cross = "Bullish Bias 🟢" if ma20 > ma50 else "Bearish Bias 🔴"
            desc  = "MA20 vs MA50 (not enough data for MA200)"
            signals["MA20 vs MA50"] = (round(ma20 / ma50, 3), cross, desc)
    except Exception:
        pass

    # ── Price momentum (20-day) ───────────────────────────────
    try:
        if len(close_prices) >= 20:
            mom  = (float(close_prices.iloc[-1]) / float(close_prices.iloc[-20]) - 1) * 100
            sig  = "Bullish 🟢" if mom > 0 else "Bearish 🔴"
            desc = f"Price is {abs(mom):.1f}% {'above' if mom > 0 else 'below'} level 20 days ago"
            signals["20d Momentum"] = (round(mom, 2), sig, desc)
    except Exception:
        pass

    # ── Bollinger Band squeeze ────────────────────────────────
    try:
        upper, mid, lower = calculate_bollinger_bands(close_prices)
        width     = float(((upper - lower) / mid).dropna().iloc[-1])
        cur_price = float(close_prices.iloc[-1])
        bb_pos    = float((cur_price - float(lower.iloc[-1])) /
                          (float(upper.iloc[-1]) - float(lower.iloc[-1]) + 1e-8))
        if bb_pos > 0.8:
            sig, desc = "Near Upper Band 🔴", "Price near resistance — possible pullback"
        elif bb_pos < 0.2:
            sig, desc = "Near Lower Band 🟢", "Price near support — possible bounce"
        else:
            sig, desc = "Inside Bands ⚪", "Price in normal range"
        signals["Bollinger Bands"] = (round(bb_pos, 2), sig, desc)
    except Exception:
        pass

    # ── Volume trend ──────────────────────────────────────────
    try:
        if data is not None and "Volume" in data.columns:
            vol      = data["Volume"]
            vol_ma5  = float(vol.rolling(5).mean().iloc[-1])
            vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
            ratio    = vol_ma5 / (vol_ma20 + 1e-8)
            if ratio > 1.3:
                sig, desc = "High Volume 🟢", "Recent volume 30%+ above average — strong signal"
            elif ratio < 0.7:
                sig, desc = "Low Volume ⚪", "Volume below average — weak trend"
            else:
                sig, desc = "Normal Volume ⚪", "Volume in normal range"
            signals["Volume Trend"] = (round(ratio, 2), sig, desc)
    except Exception:
        pass

    # ── Volatility ────────────────────────────────────────────
    try:
        vol  = float(close_prices.pct_change().rolling(20).std().dropna().iloc[-1]) * 100
        if vol > 3:
            sig, desc = "High 🔴", "Daily volatility above 3% — elevated risk"
        elif vol > 1.5:
            sig, desc = "Medium ⚪", "Moderate daily volatility"
        else:
            sig, desc = "Low 🟢", "Daily volatility below 1.5% — stable"
        signals["Volatility (20d)"] = (round(vol, 2), sig, desc)
    except Exception:
        pass

    return signals


# ─────────────────────────────────────────────────────────────
# Feature Importance Chart
# ─────────────────────────────────────────────────────────────

def shap_chart(importance: dict) -> go.Figure | None:
    """Horizontal bar chart showing grouped feature importances."""
    if not importance:
        return None

    top    = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in top]
    values = [t[1] for t in top]
    max_v  = max(values) if values else 1

    colors = [
        f"rgba(0, 212, 170, {0.4 + 0.6 * v / max_v})"
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
        height=max(280, len(labels) * 44),
        margin=dict(l=0, r=80, t=20, b=0),
        xaxis=dict(title="Relative Importance", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig