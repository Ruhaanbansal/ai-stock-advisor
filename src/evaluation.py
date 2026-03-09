# =============================================================
# evaluation.py — Model Evaluation (GradientBoosting)
# =============================================================
#
# Key design: we do NOT use the scaler from train_lstm_model
# because it was fit inside _build_features on a fresh call.
# Instead we refit a fresh scaler here on the same data,
# which is guaranteed to match what the model was trained on.
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics        import mean_absolute_error, mean_squared_error
from sklearn.preprocessing  import MinMaxScaler

from src.config import SEQUENCE_LENGTH


def evaluate_model(
    close_prices:    pd.Series,
    model,
    scaler,                     # kept for API compatibility, not used directly
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Evaluate the GradientBoosting model on historical close prices.
    Rebuilds features from scratch using the same pipeline as model.py.
    """
    prices = close_prices.dropna()

    if len(prices) < sequence_length + 10:
        raise ValueError(
            f"Need at least {sequence_length + 10} data points, "
            f"got {len(prices)}."
        )

    # ── Refit scaler on close prices (matches model training) ─
    eval_scaler = MinMaxScaler()
    scaled      = eval_scaler.fit_transform(
        prices.values.reshape(-1, 1)
    ).flatten()

    # ── Rebuild feature matrix (same logic as model._build_features) ─
    X, y = [], []
    for i in range(sequence_length, len(scaled) - 1):
        window = scaled[i - sequence_length : i]
        r5     = np.mean(window[-5:])
        r10    = np.mean(window[-10:]) if len(window) >= 10 else r5
        r20    = np.mean(window[-20:]) if len(window) >= 20 else r10
        s5     = np.std(window[-5:])
        ret    = window[-1] - window[-2]
        X.append(np.concatenate([window, [r5, r10, r20, s5, ret]]))
        y.append(scaled[i])

    if len(X) == 0:
        raise ValueError("Not enough data to build evaluation features.")

    X = np.array(X)
    y = np.array(y)

    # ── Run predictions ───────────────────────────────────────
    pred_scaled = model.predict(X)

    # ── Inverse transform to ₹ price space ───────────────────
    actual      = eval_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    predictions = eval_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    # ── Metrics ───────────────────────────────────────────────
    mae  = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    eps  = np.where(np.abs(actual) < 1e-8, 1e-8, actual)
    mape = float(np.mean(np.abs((actual - predictions) / eps)) * 100)

    latest_price = float(prices.iloc[-1])
    confidence   = max(0.0, min(100.0, 100.0 - (rmse / latest_price) * 100))

    # ── Date axis ─────────────────────────────────────────────
    dates = prices.index[sequence_length : sequence_length + len(actual)]

    # ── Actual vs Predicted chart ─────────────────────────────
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

    # ── Residuals chart ───────────────────────────────────────
    residuals = actual - predictions
    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(
        x=dates, y=residuals,
        mode="lines", name="Residual",
        line=dict(color="#6c63ff", width=1),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    res_fig.add_hline(y=0, line=dict(color="#7a8299", dash="dot"))
    res_fig.update_layout(
        template="plotly_dark",
        title="Prediction Residuals (Actual − Predicted)",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    return {
        "mae":            round(mae, 2),
        "rmse":           round(rmse, 2),
        "mape":           round(mape, 2),
        "confidence":     round(confidence, 2),
        "chart":          fig,
        "residual_chart": res_fig,
        "actual":         actual,
        "predictions":    predictions,
    }