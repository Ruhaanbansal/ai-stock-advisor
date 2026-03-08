# =============================================================
# features.py — Technical Indicator Engineering
# =============================================================

import pandas as pd
import numpy as np

from src.config import FEATURE_NAMES


# ─────────────────────────────────────────────────────────────
# Individual Indicators
# ─────────────────────────────────────────────────────────────

def calculate_sma(close_prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return close_prices.rolling(window).mean()


def calculate_ema(close_prices: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return close_prices.ewm(span=window, adjust=False).mean()


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    Uses Wilder's smoothing (ewm) for a more accurate RSI.
    """
    delta = close_prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close_prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    MACD Line and Signal Line.
    Standard 12/26/9 configuration.
    """
    ema12  = calculate_ema(close_prices, 12)
    ema26  = calculate_ema(close_prices, 26)
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(
    close_prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: upper, middle (SMA), lower.
    """
    sma   = close_prices.rolling(window).mean()
    std   = close_prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range — measures volatility.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ─────────────────────────────────────────────────────────────
# Feature Engineering Pipeline (used by LSTM)
# ─────────────────────────────────────────────────────────────

def create_features(close_prices: pd.Series) -> pd.DataFrame:
    """
    Generate the 4-feature DataFrame used by the LSTM model.
    Column order must match FEATURE_NAMES in config.
    """
    df = pd.DataFrame()
    df["Close"] = close_prices
    df["SMA20"]  = calculate_sma(close_prices, 20)
    df["SMA50"]  = calculate_sma(close_prices, 50)
    df["RSI"]    = calculate_rsi(close_prices)
    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────
# Extended Features (used by charts / explainability)
# ─────────────────────────────────────────────────────────────

def create_extended_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature set for display on the dashboard.
    Expects a DataFrame with at least a 'Close' column.
    """
    close = data["Close"]
    df    = data.copy()

    df["SMA20"]         = calculate_sma(close, 20)
    df["SMA50"]         = calculate_sma(close, 50)
    df["EMA12"]         = calculate_ema(close, 12)
    df["RSI"]           = calculate_rsi(close)
    df["MACD"], df["MACD_Signal"] = calculate_macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calculate_bollinger_bands(close)

    if all(c in data.columns for c in ["High", "Low", "Close"]):
        df["ATR"] = calculate_atr(data["High"], data["Low"], close)

    return df


# ─────────────────────────────────────────────────────────────
# LSTM Sequence Preparation
# ─────────────────────────────────────────────────────────────

def create_lstm_sequences(
    scaled_data: np.ndarray,
    sequence_length: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over scaled_data to produce (X, y) pairs for LSTM training.
    y is the Close price of the next timestep (index 0).
    """
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length : i])
        y.append(scaled_data[i, 0])       # predict Close only

    return np.array(X), np.array(y)