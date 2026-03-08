# =============================================================
# evaluation.py — Model Evaluation & Diagnostics
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import create_features, create_lstm_sequences
from src.config import SEQUENCE_LENGTH


# ─────────────────────────────────────────────────────────────
# Evaluate Model
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    close_prices:    pd.Series,
    model,
    scaler,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Run the trained model over historical data and compute error metrics.
    Returns a dict with mae, rmse, mape, confidence, and a Plotly figure.
    """
    feature_df = create_features(close_prices)
    X, _       = create_lstm_sequences(scaler.transform(feature_df.values), sequence_length)

    if X.shape[0] == 0:
        raise ValueError("Insufficient data for evaluation.")

    # Predictions
    pred_scaled = model.predict(X, verbose=0)
    n_features  = X.shape[2]

    padded = np.concatenate(
        [pred_scaled, np.zeros((len(pred_scaled), n_features - 1))],
        axis=1
    )
    predictions  = scaler.inverse_transform(padded)[:, 0]
    actual       = feature_df["Close"].iloc[sequence_length:].values

    # ── Metrics ───────────────────────────────────────────────
    mae   = float(mean_absolute_error(actual, predictions))
    rmse  = float(np.sqrt(mean_squared_error(actual, predictions)))
    mape  = float(np.mean(np.abs((actual - predictions) / actual)) * 100)

    latest_price = actual[-1]
    confidence   = max(0.0, min(100.0, 100.0 - (rmse / latest_price) * 100))

    # ── Plotly Chart ──────────────────────────────────────────
    dates = feature_df.index[sequence_length:]

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

    return {
        "mae":        round(mae,   2),
        "rmse":       round(rmse,  2),
        "mape":       round(mape,  2),
        "confidence": round(confidence, 2),
        "chart":      fig,
        "actual":     actual,
        "predictions": predictions,
    }