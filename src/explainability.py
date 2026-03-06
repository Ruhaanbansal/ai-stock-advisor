import numpy as np
import pandas as pd


# -------------------------------------------------
# 1. Rule-based explanation (Human readable reasons)
# -------------------------------------------------

def generate_explanation(close_prices, predicted_price):

    explanations = []

    current_price = float(close_prices.iloc[-1])

    sma20 = close_prices.rolling(20).mean().iloc[-1]
    sma50 = close_prices.rolling(50).mean().iloc[-1]

    # RSI calculation
    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]

    rs = avg_gain / avg_loss if avg_loss != 0 else 0

    rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50

    # Prediction direction
    if predicted_price > current_price:
        explanations.append("Model predicts price increase based on recent momentum.")
    else:
        explanations.append("Model predicts price decrease based on recent trend.")

    # RSI signals
    if rsi < 30:
        explanations.append("RSI indicates oversold conditions (potential rebound).")

    elif rsi > 70:
        explanations.append("RSI indicates overbought conditions (possible correction).")

    # Moving average crossover
    if sma20 > sma50:
        explanations.append("Short-term trend is stronger than long-term trend (bullish signal).")

    else:
        explanations.append("Long-term trend dominates short-term trend (bearish signal).")

    # Price vs SMA
    if current_price > sma20:
        explanations.append("Current price is above 20-day moving average (uptrend support).")

    else:
        explanations.append("Current price is below 20-day moving average (weak momentum).")

    return explanations


# -------------------------------------------------
# 2. SHAP-style Feature Importance
# -------------------------------------------------

def shap_explanation(model, X_input, feature_names):

    """
    Returns approximate feature importance values
    based on feature magnitude.
    """

    try:

        # Use last timestep of sequence
        last_features = X_input[0][-1]

        importance = {}

        for i, name in enumerate(feature_names):

            importance[name] = float(abs(last_features[i]))

        return importance

    except Exception:

        return {name: 0 for name in feature_names}


# -------------------------------------------------
# 3. LIME-style Local Explanation
# -------------------------------------------------

def lime_explanation(model, X_input, feature_names):

    """
    Generates simple local explanation weights
    for each feature.
    """

    try:

        last_features = X_input[0][-1]

        explanations = []

        for i, feature in enumerate(feature_names):

            weight = float(last_features[i])

            explanations.append((feature, weight))

        # Sort by absolute importance
        explanations = sorted(
            explanations,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return explanations

    except Exception:

        return []