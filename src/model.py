# =============================================================
# model.py — Stock Price Prediction (GradientBoostingRegressor)
# =============================================================
#
# Architecture:
#   Features: lagged prices + rolling stats + volume features (60-day window)
#   Model: GradientBoostingRegressor (200 estimators, tuned hyperparameters)
#   Split: 80% train / 20% test — prevents evaluation from being purely in-sample
#   Output: next-day price prediction + 5-day forecast + train/test scores
# =============================================================

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle
import os

from src.config import (
    SEQUENCE_LENGTH,
    GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE,
    GB_SUBSAMPLE, GB_MIN_SAMPLES, GB_MAX_FEATURES, GB_RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────
# Feature Engineering for Time-Series
# ─────────────────────────────────────────────────────────────

def _build_features(prices: pd.Series, seq_len: int = SEQUENCE_LENGTH) -> tuple:
    """
    Build supervised learning features from a price series.

    For each day t, features include:
      - Last `seq_len` normalised prices (lagged window)
      - Rolling mean over 5, 10, 20 days
      - Rolling std over 5 days
      - Day-over-day return
      - 5-day and 10-day return momentum
    Target: scaled price at day t (next day's price)
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(seq_len, len(scaled) - 1):
        window = scaled[i - seq_len : i]

        r5   = np.mean(window[-5:])
        r10  = np.mean(window[-10:]) if len(window) >= 10 else r5
        r20  = np.mean(window[-20:]) if len(window) >= 20 else r10
        s5   = np.std(window[-5:])
        ret1 = window[-1] - window[-2]              # 1-day return
        ret5 = window[-1] - window[-6] if len(window) >= 6 else ret1   # 5-day
        ret10 = window[-1] - window[-11] if len(window) >= 11 else ret5 # 10-day

        features = np.concatenate([window, [r5, r10, r20, s5, ret1, ret5, ret10]])
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
    - 80/20 train/test split to give honest out-of-sample metrics
    - Cached by (stock, data_hash) — only retrains when new data arrives
    - Returns (model, scaler, last_sequence, train_mae, test_mae)

    NOTE: Function name kept as train_lstm_model for backward compatibility
          with app.py imports.
    """
    model_path = f"models/{stock.replace('.', '_')}.pkl"

    X, y, scaler, scaled = _build_features(close_prices)

    if len(X) < 20:
        return None, None, None, None, None

    # 80/20 split — always keep the most recent 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Try loading saved model ───────────────────────────────
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("data_hash") == data_hash:
                model         = saved["model"]
                last_sequence = scaled[-SEQUENCE_LENGTH:]
                return (
                    model, scaler, last_sequence,
                    saved.get("train_mae", 0.0),
                    saved.get("test_mae",  0.0),
                )
        except Exception:
            pass

    # ── Train new model ───────────────────────────────────────
    model = GradientBoostingRegressor(
        n_estimators      = GB_N_ESTIMATORS,
        max_depth         = GB_MAX_DEPTH,
        learning_rate     = GB_LEARNING_RATE,
        subsample         = GB_SUBSAMPLE,
        min_samples_leaf  = GB_MIN_SAMPLES,
        max_features      = GB_MAX_FEATURES,
        random_state      = GB_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # ── Train & test MAE in price space ──────────────────────
    train_pred_s = model.predict(X_train)
    test_pred_s  = model.predict(X_test)
    train_pred   = scaler.inverse_transform(train_pred_s.reshape(-1, 1)).flatten()
    test_pred    = scaler.inverse_transform(test_pred_s.reshape(-1, 1)).flatten()
    train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    test_actual  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    train_mae    = float(mean_absolute_error(train_actual, train_pred))
    test_mae     = float(mean_absolute_error(test_actual,  test_pred))

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    try:
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":     model,
                "data_hash": data_hash,
                "train_mae": train_mae,
                "test_mae":  test_mae,
            }, f)
    except Exception:
        pass

    last_sequence = scaled[-SEQUENCE_LENGTH:]
    return model, scaler, last_sequence, train_mae, test_mae


# ─────────────────────────────────────────────────────────────
# Feature vector for a single prediction
# ─────────────────────────────────────────────────────────────

def _make_feature_vector(window: np.ndarray) -> np.ndarray:
    """Build the same feature vector used during training for a given window."""
    r5   = np.mean(window[-5:])
    r10  = np.mean(window[-10:]) if len(window) >= 10 else r5
    r20  = np.mean(window[-20:]) if len(window) >= 20 else r10
    s5   = np.std(window[-5:])
    ret1 = window[-1] - window[-2]
    ret5 = window[-1] - window[-6]  if len(window) >= 6  else ret1
    ret10 = window[-1] - window[-11] if len(window) >= 11 else ret5
    return np.concatenate([window, [r5, r10, r20, s5, ret1, ret5, ret10]])


# ─────────────────────────────────────────────────────────────
# Predict Next Price
# ─────────────────────────────────────────────────────────────

def predict_next_price(model, scaler, last_sequence: np.ndarray) -> float:
    """Predict the next day's closing price."""
    if model is None or scaler is None or last_sequence is None:
        return 0.0
    window      = last_sequence[-SEQUENCE_LENGTH:]
    features    = _make_feature_vector(window).reshape(1, -1)
    pred_scaled = model.predict(features)[0]
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])


# ─────────────────────────────────────────────────────────────
# 5-Day Forecast
# ─────────────────────────────────────────────────────────────

def forecast_prices(
    model, scaler, last_sequence: np.ndarray, days: int = 5
) -> list[float]:
    """
    Iteratively predict `days` future prices, feeding each
    prediction back as input for the next step.
    """
    if model is None or scaler is None or last_sequence is None:
        return [0.0] * days

    sequence  = last_sequence.copy()
    forecasts = []

    for _ in range(days):
        window      = sequence[-SEQUENCE_LENGTH:]
        features    = _make_feature_vector(window).reshape(1, -1)
        pred_scaled = model.predict(features)[0]
        price       = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        forecasts.append(price)
        sequence = np.append(sequence, pred_scaled)

    return forecasts