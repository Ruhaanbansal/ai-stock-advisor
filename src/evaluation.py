import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------------------------
# Create Technical Indicators
# -------------------------------------------------

def create_features(close_prices):

    data = pd.DataFrame()

    data["Close"] = close_prices
    data["SMA20"] = close_prices.rolling(20).mean()
    data["SMA50"] = close_prices.rolling(50).mean()

    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    data = data.dropna()

    return data


# -------------------------------------------------
# Prepare LSTM Input
# -------------------------------------------------

def prepare_sequences(data, scaler, sequence_length):

    scaled = scaler.transform(data.values)

    X = []
    y = []

    for i in range(sequence_length, len(scaled)):

        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X, y


# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------

def evaluate_model(close_prices, model, scaler, sequence_length=60):

    data = create_features(close_prices)

    X, y = prepare_sequences(data, scaler, sequence_length)

    predictions_scaled = model.predict(X, verbose=0)

    predictions = scaler.inverse_transform(
        np.concatenate(
            [predictions_scaled, np.zeros((len(predictions_scaled), 3))],
            axis=1
        )
    )[:, 0]

    actual_prices = data["Close"].iloc[sequence_length:].values

    mae = mean_absolute_error(actual_prices, predictions)

    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))

    latest_price = actual_prices[-1]

    confidence = max(
        0,
        100 - (rmse / latest_price) * 100
    )

    # Chart
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(actual_prices, label="Actual Price")
    ax.plot(predictions, label="Predicted Price")

    ax.set_title("Actual vs Predicted Stock Price")
    ax.legend()

    return {
        "mae": mae,
        "rmse": rmse,
        "confidence": confidence,
        "chart": fig
    }