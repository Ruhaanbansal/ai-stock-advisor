import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Stock Advisor", layout="wide")

st.title("📈 AI Stock Advisory System")

st.write("Welcome to the AI-powered stock prediction dashboard.")

# ---- Stock Selector ----
stock = st.selectbox(
    "Select NSE Stock",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
)

st.write("Selected Stock:", stock)

# ---- Download Live Data ----
data = yf.download(stock, period="6mo", auto_adjust=True)

st.write("Data rows downloaded:", len(data))

if data.empty:
    st.error("No data received from Yahoo Finance.")
else:
    # Always flatten columns (important fix)
    data.columns = data.columns.get_level_values(0)

    # Extract close price safely
    close_prices = data["Close"]

    # Ensure numeric
    close_prices = pd.to_numeric(close_prices, errors="coerce")

    st.subheader("Last 6 Months Price Chart")
    st.line_chart(close_prices)

# -------------------------------
# Risk Analysis Section
# -------------------------------

st.subheader("📊 Risk Analysis")

# Calculate daily returns
returns = close_prices.pct_change().dropna()

# Annualized Volatility
annual_volatility = returns.std() * (252 ** 0.5)

st.write("Annualized Volatility:", round(annual_volatility, 4))

# Risk Category
if annual_volatility < 0.25:
    risk_category = "Low Risk"
elif annual_volatility < 0.40:
    risk_category = "Medium Risk"
else:
    risk_category = "High Risk"

st.write("Risk Category:", risk_category)


# -------------------------------
# LSTM Price Forecast Section (Optimized)
# -------------------------------

st.subheader("🤖 LSTM Next-Day Price Forecast")



@st.cache_resource
def train_lstm_model(close_prices):
    
    price_data = close_prices.values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(price_data)
    
    sequence_length = 20
    
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = Sequential()
    model.add(LSTM(32, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    
    return model, scaler, X[-1]

model, scaler, last_sequence = train_lstm_model(close_prices)

# Predict
X_input = last_sequence.reshape(1, 20, 1)
predicted_scaled = model.predict(X_input, verbose=0)
predicted_price = scaler.inverse_transform(predicted_scaled)

st.write("Predicted Next-Day Price: ₹", round(float(predicted_price[0][0]), 2))



# -------------------------------
# Recommendation Engine
# -------------------------------

st.subheader("📌 Investment Recommendation")

current_price = float(close_prices.iloc[-1])
predicted_price_value = float(predicted_price[0][0])

price_change_percent = ((predicted_price_value - current_price) / current_price) * 100



# Decision Logic
if price_change_percent > 1 and risk_category == "Low Risk":
    recommendation = "Strong Buy"
elif price_change_percent > 0:
    recommendation = "Buy"
elif price_change_percent < -1:
    recommendation = "Sell"
else:
    recommendation = "Hold"


st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Current Price",
        value=f"₹ {round(current_price, 2)}"
    )

with col2:
    st.metric(
        label="Predicted Next-Day Price",
        value=f"₹ {round(predicted_price_value, 2)}",
        delta=f"{round(price_change_percent, 2)}%"
    )

with col3:
    st.metric(
        label="Risk Category",
        value=risk_category
    )

st.markdown("---")

# Highlight Recommendation

if recommendation in ["Strong Buy", "Buy"]:
    st.success(f"📢 Final Recommendation: {recommendation}")
elif recommendation == "Sell":
    st.error(f"📢 Final Recommendation: {recommendation}")
else:
    st.warning(f"📢 Final Recommendation: {recommendation}")

