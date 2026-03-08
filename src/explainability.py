# =============================================================
# explainability.py — Model Explainability
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import FEATURE_NAMES


# ─────────────────────────────────────────────────────────────
# Rule-based Explanation
# ─────────────────────────────────────────────────────────────

def generate_explanation(close_prices: pd.Series, predicted_price: float) -> list[str]:
    """
    Produce human-readable bullet points explaining the prediction.
    """
    explanations = []
    current = float(close_prices.iloc[-1])

    sma20 = close_prices.rolling(20).mean().iloc[-1]
    sma50 = close_prices.rolling(50).mean().iloc[-1]

    delta    = close_prices.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(com=13).mean().iloc[-1]
    rsi      = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

    # Direction
    if predicted_price > current:
        explanations.append(f"Model forecasts price increase to ₹{predicted_price:.2f}.")
    else:
        explanations.append(f"Model forecasts price decline to ₹{predicted_price:.2f}.")

    # RSI
    if rsi < 30:
        explanations.append(f"RSI ({rsi:.1f}) signals oversold — potential rebound ahead.")
    elif rsi > 70:
        explanations.append(f"RSI ({rsi:.1f}) signals overbought — correction possible.")
    else:
        explanations.append(f"RSI ({rsi:.1f}) is in neutral territory.")

    # Moving average crossover
    if sma20 > sma50:
        explanations.append("Golden cross: SMA20 > SMA50 (bullish momentum signal).")
    else:
        explanations.append("Death cross: SMA20 < SMA50 (bearish momentum signal).")

    # Price vs SMA20
    if current > sma20:
        explanations.append("Price above 20-day SMA — short-term uptrend intact.")
    else:
        explanations.append("Price below 20-day SMA — short-term momentum is weak.")

    return explanations


# ─────────────────────────────────────────────────────────────
# SHAP-style Feature Importance
# ─────────────────────────────────────────────────────────────

def shap_explanation(
    model,
    X_input:       np.ndarray,
    feature_names: list[str] = FEATURE_NAMES,
) -> dict[str, float]:
    """
    Approximate feature importance from the magnitude of the last timestep.
    """
    try:
        last_step  = X_input[0][-1]
        importance = {name: float(abs(last_step[i])) for i, name in enumerate(feature_names)}
        return importance
    except Exception:
        return {name: 0.0 for name in feature_names}


def shap_chart(importance: dict) -> go.Figure:
    """Return a horizontal bar chart of SHAP-style importance values."""
    names  = list(importance.keys())
    values = list(importance.values())

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(
            color=values,
            colorscale="Teal",
            showscale=False,
        )
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Feature Importance (SHAP-style)",
        xaxis_title="Magnitude",
        height=280,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LIME-style Local Explanation
# ─────────────────────────────────────────────────────────────

def lime_explanation(
    model,
    X_input:       np.ndarray,
    feature_names: list[str] = FEATURE_NAMES,
) -> list[tuple[str, float]]:
    """
    Simple local explanation: feature weights from the last timestep,
    sorted by absolute magnitude.
    """
    try:
        last_step    = X_input[0][-1]
        explanations = [(name, float(last_step[i])) for i, name in enumerate(feature_names)]
        return sorted(explanations, key=lambda x: abs(x[1]), reverse=True)
    except Exception:
        return []