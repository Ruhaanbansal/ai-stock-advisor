import pandas as pd
import numpy as np


# -----------------------------------
# Moving Averages
# -----------------------------------

def calculate_sma(close_prices, window):
    """
    Calculate Simple Moving Average
    """
    return close_prices.rolling(window).mean()


# -----------------------------------
# RSI Indicator
# -----------------------------------

def calculate_rsi(close_prices, window=14):
    """
    Calculate Relative Strength Index
    """

    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# -----------------------------------
# Feature Engineering Pipeline
# -----------------------------------

def create_features(close_prices):
    """
    Generate all model features
    """

    feature_df = pd.DataFrame()

    feature_df["Close"] = close_prices

    # Moving averages
    feature_df["SMA20"] = calculate_sma(close_prices, 20)
    feature_df["SMA50"] = calculate_sma(close_prices, 50)

    # RSI
    feature_df["RSI"] = calculate_rsi(close_prices)

    # Remove NaN rows caused by rolling indicators
    feature_df = feature_df.dropna()

    return feature_df


# -----------------------------------
# Prepare Sequences for LSTM
# -----------------------------------

def create_lstm_sequences(scaled_data, sequence_length=60):
    """
    Convert time series data into LSTM sequences
    """

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):

        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    return X, y