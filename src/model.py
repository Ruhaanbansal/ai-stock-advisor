# =============================================================
# model.py — LSTM Model: Train, Save, Load & Predict
# =============================================================

import os
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.features import create_features, create_lstm_sequences
from src.config import (
    SEQUENCE_LENGTH, LSTM_UNITS_1, LSTM_UNITS_2,
    DROPOUT_RATE, EPOCHS, BATCH_SIZE
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Build Architecture
# ─────────────────────────────────────────────────────────────

def _build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ─────────────────────────────────────────────────────────────
# Train (or load cached) LSTM Model
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_lstm_model(stock: str, close_prices_hash: int, _close_prices):
    """
    Train LSTM and cache the result for the session.
    Pass `close_prices_hash` (hash of the series) so Streamlit
    knows to retrain only when the underlying data changes.

    Returns: (model, scaler, last_sequence)
    """
    close_prices = _close_prices

    model_path = os.path.join(MODEL_DIR, f"{stock.replace('.', '_')}.keras")

    feature_df = create_features(close_prices)
    scaler     = MinMaxScaler()
    scaled     = scaler.fit_transform(feature_df.values)
    X, y       = create_lstm_sequences(scaled, SEQUENCE_LENGTH)

    if X.shape[0] == 0:
        raise ValueError("Not enough data to create training sequences.")

    # ── Load pre-trained weights if they exist ────────────────
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = _build_model(input_shape=(SEQUENCE_LENGTH, X.shape[2]))
        early_stop = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
        model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[early_stop]
        )
        model.save(model_path)

    last_sequence = X[-1]
    return model, scaler, last_sequence


# ─────────────────────────────────────────────────────────────
# Predict Next Price
# ─────────────────────────────────────────────────────────────

def predict_next_price(model, scaler, last_sequence: np.ndarray) -> float:
    """
    Predict the next trading day's closing price.
    """
    n_features = last_sequence.shape[1]
    X_input    = last_sequence.reshape(1, SEQUENCE_LENGTH, n_features)

    predicted_scaled = model.predict(X_input, verbose=0)

    # Inverse-transform: pad zeros for the non-Close features
    padded    = np.concatenate(
        [predicted_scaled, np.zeros((1, n_features - 1))],
        axis=1
    )
    predicted = scaler.inverse_transform(padded)[:, 0]
    return float(predicted[0])


# ─────────────────────────────────────────────────────────────
# Multi-step Forecast
# ─────────────────────────────────────────────────────────────

def forecast_prices(
    model,
    scaler,
    last_sequence: np.ndarray,
    days: int = 5
) -> list[float]:
    """
    Iteratively forecast `days` trading days into the future.
    """
    n_features = last_sequence.shape[1]
    seq        = last_sequence.copy()
    forecasts  = []

    for _ in range(days):
        X_in   = seq.reshape(1, SEQUENCE_LENGTH, n_features)
        pred_s = model.predict(X_in, verbose=0)

        # Build next timestep (keep non-Close features from last step)
        next_step    = seq[-1].copy()
        next_step[0] = pred_s[0, 0]         # update only the Close feature
        seq          = np.vstack([seq[1:], next_step])

        padded = np.concatenate(
            [pred_s, np.zeros((1, n_features - 1))],
            axis=1
        )
        forecasts.append(float(scaler.inverse_transform(padded)[0, 0]))

    return forecasts