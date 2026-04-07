# =============================================================
# evaluation.py — GradientBoosting Model Evaluation
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics        import mean_absolute_error, mean_squared_error
from sklearn.preprocessing  import MinMaxScaler

from src.config import SEQUENCE_LENGTH


def evaluate_model(
    close_prices:    pd.Series,
    model,
    scaler,           # kept for API compatibility
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Evaluate the GradientBoosting model on historical close prices.
    Uses an 80/20 split to produce honest out-of-sample test metrics.
    Returns train vs test MAE, MAPE, directional accuracy, and charts.
    """
    prices = close_prices.dropna()

    if len(prices) < sequence_length + 10:
        raise ValueError(
            f"Need at least {sequence_length + 10} data points, got {len(prices)}."
        )

    from src.model import _build_features

    X, y, eval_scaler, scaled = _build_features(prices)

    if len(X) == 0:
        raise ValueError("Not enough data to build evaluation features.")

    # 80/20 split (same as training)
    split_idx   = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Predictions
    train_pred_s = model.predict(X_train)
    test_pred_s  = model.predict(X_test)

    # Inverse transform
    def inv(arr):
        return eval_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    train_actual = inv(y_train)
    test_actual  = inv(y_test)
    train_pred   = inv(train_pred_s)
    test_pred    = inv(test_pred_s)

    def _mae(a, p):  return float(mean_absolute_error(a, p))
    def _rmse(a, p): return float(np.sqrt(mean_squared_error(a, p)))
    def _mape(a, p):
        eps = np.where(np.abs(a) < 1e-8, 1e-8, a)
        return float(np.mean(np.abs((a - p) / eps)) * 100)
    def _dir_acc(a, p):
        a_dir = np.diff(a) > 0
        p_dir = np.diff(p) > 0
        return float(np.mean(a_dir == p_dir) * 100)

    train_mae  = _mae(train_actual, train_pred)
    test_mae   = _mae(test_actual,  test_pred)
    train_rmse = _rmse(train_actual, train_pred)
    test_rmse  = _rmse(test_actual,  test_pred)
    test_mape  = _mape(test_actual,  test_pred)
    test_dir   = _dir_acc(test_actual, test_pred)

    latest_price = float(prices.iloc[-1])
    confidence   = max(0.0, min(100.0, 100.0 - (test_rmse / latest_price) * 100))

    # ── Date axes ─────────────────────────────────────────────
    offset_start = sequence_length
    offset_split = offset_start + split_idx

    train_dates = prices.index[offset_start : offset_start + len(train_actual)]
    test_dates  = prices.index[offset_split : offset_split + len(test_actual)]

    # ── Actual vs Predicted chart ─────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_dates, y=train_actual,
        mode="lines", name="Train Actual",
        line=dict(color="#7a8299", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=train_dates, y=train_pred,
        mode="lines", name="Train Predicted",
        line=dict(color="#6c63ff", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_actual,
        mode="lines", name="Test Actual",
        line=dict(color="#00d4aa", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_pred,
        mode="lines", name="Test Predicted",
        line=dict(color="#ff6b6b", width=1.8, dash="dash"),
    ))
    # Vertical divider at train/test boundary
    if len(test_dates) > 0:
        divider_x = test_dates[0]
        fig.add_vline(x=divider_x, line=dict(color="#ffb347", dash="dot", width=1))
        fig.add_annotation(
            x=divider_x, y=1, yref="paper",
            text="Test start", showarrow=False,
            xanchor="left", xshift=5, yanchor="top",
            font=dict(color="#ffb347", size=10),
        )
    fig.update_layout(
        template="plotly_dark",
        title="Actual vs Predicted Close Price (Train / Test Split)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=440,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # ── Residuals chart (test only) ───────────────────────────
    residuals = test_actual - test_pred
    res_fig = go.Figure()
    res_fig.add_trace(go.Scatter(
        x=test_dates, y=residuals,
        mode="lines", name="Residual (₹)",
        line=dict(color="#6c63ff", width=1),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    res_fig.add_hline(y=0, line=dict(color="#7a8299", dash="dot"))
    res_fig.update_layout(
        template="plotly_dark",
        title="Test Set Residuals (Actual − Predicted)",
        height=240,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    return {
        "train_mae":   round(train_mae, 2),
        "test_mae":    round(test_mae, 2),
        "train_rmse":  round(train_rmse, 2),
        "test_rmse":   round(test_rmse, 2),
        "mape":        round(test_mape, 2),
        "dir_acc":     round(test_dir, 1),
        "confidence":  round(confidence, 2),
        "chart":       fig,
        "residual_chart": res_fig,
        "actual":      test_actual,
        "predictions": test_pred,
    }