import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.features import create_features, create_lstm_sequences


# -------------------------------------------------
# Model Configuration
# -------------------------------------------------

SEQUENCE_LENGTH = 60


# -------------------------------------------------
# Train LSTM Model
# -------------------------------------------------

def train_lstm_model(close_prices):
    """
    Train LSTM model on stock price features
    """

    # Generate feature dataframe
    feature_df = create_features(close_prices)

    # Scale features
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(feature_df)

    # Create sequences
    X, y = create_lstm_sequences(
        scaled_data,
        sequence_length=SEQUENCE_LENGTH
    )

    # Build LSTM model
    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, X.shape[2])
        )
    )

    model.add(Dropout(0.2))

    model.add(LSTM(32))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # Train model
    model.fit(
        X,
        y,
        epochs=3,
        batch_size=16,
        verbose=0
    )

    # Return trained components
    last_sequence = X[-1]

    return model, scaler, last_sequence


# -------------------------------------------------
# Predict Next Price
# -------------------------------------------------

def predict_next_price(model, scaler, last_sequence):
    """
    Predict next-day stock price
    """

    X_input = last_sequence.reshape(1, SEQUENCE_LENGTH, last_sequence.shape[1])

    predicted_scaled = model.predict(X_input, verbose=0)

    # Inverse transform to original price scale
    predicted_price = scaler.inverse_transform(
        np.concatenate(
            [predicted_scaled, np.zeros((1, last_sequence.shape[1] - 1))],
            axis=1
        )
    )[:, 0]

    return float(predicted_price[0])