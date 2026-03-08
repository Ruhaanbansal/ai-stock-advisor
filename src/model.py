# =============================================================
# model.py — Stock Price Prediction (scikit-learn)
# =============================================================
#
# Replaced Keras LSTM with scikit-learn GradientBoostingRegressor.
# Reasons:
#   - No TensorFlow/Keras dependency (incompatible with Python 3.14)
#   - Works on all Python versions with zero C++ build deps
#   - Comparable prediction accuracy for financial time series
#   - Much faster to train (seconds vs minutes)
#
# Architecture:
#   Features: lagged returns + rolling stats (60-day window)
#   Model: GradientBoostingRegressor (100 estimators)
#   Output: next-day price prediction + 5-day forecast
# =============================================================

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

from src.config import SEQUENCE_LENGTH


# ─────────────────────────────────────────────────────────────
# Feature Engineering for Time-Series
# ─────────────────────────────────────────────────────────────

def _build_features(prices: pd.Series, seq_len: int = SEQUENCE_LENGTH) -> tuple:
    """
    Build supervised learning features from a price series.

    For each day t, features include:
      - Last `seq_len` normalised prices (lagged window)
      - Rolling mean and std over 5, 10, 20 days
      - Day-over-day return
    Target: price at day t+1
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(seq_len, len(scaled) - 1):
        window = scaled[i - seq_len : i]

        # Rolling stats features
        r5  = np.mean(window[-5:])
        r10 = np.mean(window[-10:]) if len(window) >= 10 else r5
        r20 = np.mean(window[-20:]) if len(window) >= 20 else r10
        s5  = np.std(window[-5:])
        ret = window[-1] - window[-2]   # 1-day return

        features = np.concatenate([window, [r5, r10, r20, s5, ret]])
        X.append(features)
        y.append(scaled[i])             # predict next close

    return np.array(X), np.array(y), scaler, scaled


# ─────────────────────────────────────────────────────────────
# Train / Load Model
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_lstm_model(stock: str, data_hash: int, close_prices: pd.Series):
    """
    Train a GradientBoostingRegressor on historical close prices.
    Cached by (stock, data_hash) so it only retrains when data changes.

    Returns (model, scaler, last_sequence) matching the old LSTM interface
    so the rest of app.py needs no changes.
    """
    model_path = f"models/{stock.replace('.', '_')}.pkl"

    X, y, scaler, scaled = _build_features(close_prices)

    # ── Try loading saved model ───────────────────────────────
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("data_hash") == data_hash:
                model         = saved["model"]
                last_sequence = scaled[-SEQUENCE_LENGTH:]
                return model, scaler, last_sequence
        except Exception:
            pass

    # ── Train new model ───────────────────────────────────────
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    try:
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "data_hash": data_hash}, f)
    except Exception:
        pass

    last_sequence = scaled[-SEQUENCE_LENGTH:]
    return model, scaler, last_sequence


# ─────────────────────────────────────────────────────────────
# Predict Next Price
# ─────────────────────────────────────────────────────────────

def predict_next_price(model, scaler, last_sequence: np.ndarray) -> float:
    """Predict the next day's closing price."""
    window = last_sequence[-SEQUENCE_LENGTH:]
    r5     = np.mean(window[-5:])
    r10    = np.mean(window[-10:]) if len(window) >= 10 else r5
    r20    = np.mean(window[-20:]) if len(window) >= 20 else r10
    s5     = np.std(window[-5:])
    ret    = window[-1] - window[-2]

    features = np.concatenate([window, [r5, r10, r20, s5, ret]]).reshape(1, -1)
    pred_scaled = model.predict(features)[0]
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])


# ─────────────────────────────────────────────────────────────
# 5-Day Forecast
# ─────────────────────────────────────────────────────────────

def forecast_prices(model, scaler, last_sequence: np.ndarray, days: int = 5) -> list[float]:
    """
    Iteratively predict `days` future prices, feeding each
    prediction back as input for the next step.
    """
    sequence = last_sequence.copy()
    forecasts = []

    for _ in range(days):
        window  = sequence[-SEQUENCE_LENGTH:]
        r5      = np.mean(window[-5:])
        r10     = np.mean(window[-10:]) if len(window) >= 10 else r5
        r20     = np.mean(window[-20:]) if len(window) >= 20 else r10
        s5      = np.std(window[-5:])
        ret     = window[-1] - window[-2]

        features    = np.concatenate([window, [r5, r10, r20, s5, ret]]).reshape(1, -1)
        pred_scaled = model.predict(features)[0]
        price       = float(scaler.inverse_transform([[pred_scaled]])[0][0])

        forecasts.append(price)
        sequence = np.append(sequence, pred_scaled)

    return forecasts