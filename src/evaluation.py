# =============================================================
# evaluation.py — Model Evaluation (GradientBoosting compatible)
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import SEQUENCE_LENGTH


def evaluate_model(
    close_prices:    pd.Series,
    model,
    scaler,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Evaluate the GradientBoosting model on historical data.
    Rebuilds features using the same pipeline as model.py so
    predictions are directly comparable to actual prices.
    """
    from src.model import _build_features

    X, y, _, scaled = _build_features(close_prices, seq_len=sequence_length)

    if len(X) == 0:
        raise ValueError("Insufficient data for evaluation.")

    # ── Predictions in scaled space ───────────────────────────
    pred_scaled = model.predict(X)

    # ── Inverse transform to price space ─────────────────────
    predictions = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    actual = scaler.inverse_transform(
        y.reshape(-1, 1)
    ).flatten()

    # ── Metrics ───────────────────────────────────────────────
    mae  = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mape = float(np.mean(np.abs((actual - predictions) /
                                np.where(actual == 0, 1, actual))) * 100)

    latest_price = float(close_prices.iloc[-1])
    confidence   = max(0.0, min(100.0, 100.0 - (rmse / latest_price) * 100))

    # ── Date index for chart ───────────────────────────────────
    # X starts at index `sequence_length` of close_prices
    dates = close_prices.index[sequence_length : sequence_length + len(actual)]

    # ── Plotly Chart ──────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines", name="Actual",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predictions,
        mode="lines", name="Predicted",
        line=dict(color="#ff6b6b", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Actual vs Predicted Close Price",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ── Residuals ─────────────────────────────────────────────
    residuals = actual - predictions
    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(
        x=dates, y=residuals,
        mode="lines", name="Residual",
        line=dict(color="#6c63ff", width=1),
    ))
    res_fig.add_hline(y=0, line=dict(color="#7a8299", dash="dot"))
    res_fig.update_layout(
        template="plotly_dark",
        title="Prediction Residuals",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    return {
        "mae":         round(mae, 2),
        "rmse":        round(rmse, 2),
        "mape":        round(mape, 2),
        "confidence":  round(confidence, 2),
        "chart":       fig,
        "residual_chart": res_fig,
        "actual":      actual,
        "predictions": predictions,
    }