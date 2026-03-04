import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

sequence_length = 60

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------------
# Dashboard Header
# -----------------------------------

st.set_page_config(
    page_title="AI Stock Advisor",
    layout="wide"
)

st.title("📈 AI-Powered Stock Advisor Dashboard")

st.markdown(
"""
This dashboard uses **Machine Learning and Deep Learning (LSTM)** to analyze stock market data,
predict future prices, evaluate risk levels, and provide investment recommendations.

Features included:
- LSTM-based stock price forecasting
- Portfolio optimization (Modern Portfolio Theory)
- Multi-stock performance comparison
- Strategy backtesting
- Model evaluation metrics
"""
)

st.set_page_config(page_title="AI Stock Advisor", layout="wide")



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

# -----------------------------------
# Download Processed Data
# -----------------------------------

st.markdown("---")
st.subheader("Download Data")

csv = close_prices.to_csv().encode("utf-8")

st.download_button(
    label="Download Stock Data as CSV",
    data=csv,
    file_name="stock_data.csv",
    mime="text/csv",
)

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
    
    # Create feature dataframe
    feature_df = pd.DataFrame()

    feature_df["Close"] = close_prices
    feature_df["SMA20"] = close_prices.rolling(20).mean()
    feature_df["SMA50"] = close_prices.rolling(50).mean()

    # RSI calculation
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    feature_df["RSI"] = 100 - (100 / (1 + rs))

    # Drop missing values
    feature_df = feature_df.dropna()

    price_data = feature_df.values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(price_data)
    
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 4)))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )
    
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    
    return model, scaler, X[-1]

model, scaler, last_sequence = train_lstm_model(close_prices)

# Predict
X_input = last_sequence.reshape(1, sequence_length, 4)

predicted_scaled = model.predict(X_input, verbose=0)

predicted_price = scaler.inverse_transform(
    np.concatenate([predicted_scaled, np.zeros((1,3))], axis=1)
)[:,0]

predicted_price_value = float(predicted_price[0])



# -------------------------------
# Recommendation Engine
# -------------------------------

st.subheader("📌 Investment Recommendation")

current_price = float(close_prices.iloc[-1])
predicted_price_value = float(predicted_price[0])

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

# -----------------------------------
# Portfolio Allocation Section
# -----------------------------------

st.markdown("---")
st.header("📊 Portfolio Allocation Optimizer")

portfolio_stocks = st.multiselect(
    "Select Stocks for Portfolio",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    default=["RELIANCE.NS", "TCS.NS"]
)

st.write("Selected Portfolio Stocks:", portfolio_stocks)

# Download portfolio stock data
if len(portfolio_stocks) > 0:
    
    portfolio_data = yf.download(portfolio_stocks, period="1y", auto_adjust=True)

    # Extract closing prices
    if isinstance(portfolio_data.columns, pd.MultiIndex):
        portfolio_prices = portfolio_data["Close"]
    else:
        portfolio_prices = portfolio_data[["Close"]]

    st.subheader("Portfolio Stock Prices")
    st.line_chart(portfolio_prices)

    # Calculate daily returns
portfolio_returns = portfolio_prices.pct_change().dropna()

st.subheader("Daily Returns (Portfolio Stocks)")
st.write(portfolio_returns.tail())

# -------------------------------
# Portfolio Optimization
# -------------------------------

import numpy as np
from scipy.optimize import minimize

# Expected annual returns
mean_returns = portfolio_returns.mean() * 252

# Covariance matrix
cov_matrix = portfolio_returns.cov() * 252

num_assets = len(mean_returns)

# Portfolio performance function
def portfolio_performance(weights):
    
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return returns, volatility


# Sharpe Ratio (to maximize)
def negative_sharpe(weights):
    
    returns, volatility = portfolio_performance(weights)
    
    return -returns / volatility


# Constraints (weights must sum to 1)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for each weight
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess
init_guess = num_assets * [1. / num_assets]

# Optimize
opt_result = minimize(
    negative_sharpe,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = opt_result.x


# -------------------------------
# Display Portfolio Allocation
# -------------------------------

allocation = pd.DataFrame({
    "Stock": mean_returns.index,
    "Optimal Weight": optimal_weights
})

st.subheader("📊 Optimal Portfolio Allocation")

st.write(allocation)

# Pie chart visualization


import matplotlib.pyplot as plt

# Clean weights (remove near-zero allocations)
allocation = allocation[allocation["Optimal Weight"] > 0.01]

st.subheader("Portfolio Allocation Chart")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6))

ax.pie(
    allocation["Optimal Weight"],
    labels=allocation["Stock"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize":10}
)

ax.axis("equal")

st.pyplot(fig)


# -----------------------------------
# Multi-Stock Comparison Section
# -----------------------------------

st.markdown("---")
st.header("📈 Multi-Stock Performance Comparison")

comparison_stocks = st.multiselect(
    "Select Stocks to Compare",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    default=["RELIANCE.NS", "TCS.NS"]
)

st.write("Selected Comparison Stocks:", comparison_stocks)


# Download comparison stock data
if len(comparison_stocks) > 0:

    comparison_data = yf.download(comparison_stocks, period="1y", auto_adjust=True)

    # Extract closing prices safely
    if isinstance(comparison_data.columns, pd.MultiIndex):
        comparison_prices = comparison_data["Close"]
    else:
        comparison_prices = comparison_data[["Close"]]

    # Normalize prices to start at 100
    normalized_prices = comparison_prices / comparison_prices.iloc[0] * 100

    st.subheader("📊 Normalized Stock Performance (Base = 100)")
    st.line_chart(normalized_prices)


# -----------------------------------
# Backtesting Strategy Performance
# -----------------------------------

st.markdown("---")
st.header("📉 Strategy Backtesting")

st.write(
    "This section simulates how the AI strategy would have performed historically "
    "compared to a simple buy-and-hold strategy."
)

# Download historical data for backtesting
backtest_data = yf.download(stock, period="1y")

# Extract Close prices safely
if "Close" in backtest_data.columns:
    backtest_prices = backtest_data["Close"]
else:
    st.error("Close price not found in downloaded data")
    st.stop()

# Calculate daily returns
backtest_returns = backtest_prices.pct_change()

# Drop NaN values
backtest_returns = backtest_returns.dropna()

# Buy & Hold cumulative returns
buy_hold = (1 + backtest_returns).cumprod()

# Convert to DataFrame for Streamlit
buy_hold_df = pd.DataFrame(buy_hold)
buy_hold_df.columns = ["Buy & Hold"]

st.subheader("Buy & Hold Strategy Performance")
st.line_chart(buy_hold_df)


# -----------------------------------
# Simple AI Strategy Simulation
# -----------------------------------

# Create signal using previous day's return
signal = backtest_returns.shift(1)

# Convert to binary signal
signal = (signal > 0).astype(int)

# Strategy returns
strategy_returns = backtest_returns * signal

# Cumulative strategy returns
strategy_cumulative = (1 + strategy_returns).cumprod()

strategy_df = pd.DataFrame(strategy_cumulative)
strategy_df.columns = ["AI Strategy"]


# -----------------------------------
# Combine both strategies
# -----------------------------------

comparison_df = pd.concat([buy_hold_df, strategy_df], axis=1)

st.subheader("📊 Strategy Comparison: AI vs Buy & Hold")

st.line_chart(comparison_df)

# -----------------------------------
# Prepare Results for Download
# -----------------------------------

results_df = comparison_df.copy()

results_df = results_df.reset_index(drop=True)

csv_results = results_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Backtest Results",
    data=csv_results,
    file_name="backtest_results.csv",
    mime="text/csv",
)

# -----------------------------------
# Strategy Performance Summary
# -----------------------------------

buy_hold_final = comparison_df["Buy & Hold"].iloc[-1]
ai_strategy_final = comparison_df["AI Strategy"].iloc[-1]

buy_hold_return = (buy_hold_final - 1) * 100
ai_strategy_return = (ai_strategy_final - 1) * 100

st.subheader("Strategy Performance Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Buy & Hold Return",
        value=f"{buy_hold_return:.2f}%"
    )

with col2:
    st.metric(
        label="AI Strategy Return",
        value=f"{ai_strategy_return:.2f}%"
    )

# -----------------------------------
# Model Evaluation
# -----------------------------------

st.markdown("---")
st.header("📊 Model Evaluation")

st.write(
    "This section evaluates how well the LSTM model predicts stock prices."
)


# -----------------------------------
# Prepare data for evaluation
# -----------------------------------

# Use same feature dataframe
# Recreate feature dataframe for evaluation
evaluation_data = pd.DataFrame()

evaluation_data["Close"] = close_prices
evaluation_data["SMA20"] = close_prices.rolling(20).mean()
evaluation_data["SMA50"] = close_prices.rolling(50).mean()

# RSI calculation
delta = close_prices.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
evaluation_data["RSI"] = 100 - (100 / (1 + rs))

evaluation_data = evaluation_data.dropna()

# Scale features
scaled_eval = scaler.transform(evaluation_data.values)

X_eval = []
y_eval = []

for i in range(sequence_length, len(scaled_eval)):
    X_eval.append(scaled_eval[i-sequence_length:i])
    y_eval.append(scaled_eval[i, 0])  # Close price

X_eval = np.array(X_eval)
y_eval = np.array(y_eval)

# Predict using trained model
predictions_scaled = model.predict(X_eval, verbose=0)

# Convert predictions back to original price scale
predictions = scaler.inverse_transform(
    np.concatenate([predictions_scaled, np.zeros((len(predictions_scaled),3))], axis=1)
)[:,0]

actual_prices = evaluation_data["Close"].iloc[sequence_length:].values

# -----------------------------------
# Plot Actual vs Predicted
# -----------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(actual_prices, label="Actual Price")
ax.plot(predictions, label="Predicted Price")

ax.set_title("Actual vs Predicted Stock Price")
ax.legend()

st.pyplot(fig)


# -----------------------------------
# Model Error Metrics
# -----------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(actual_prices, predictions)
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))

# -----------------------------------
# Prediction Confidence Score
# -----------------------------------

latest_price = actual_prices[-1]

confidence_score = max(
    0,
    100 - (rmse / latest_price) * 100
)

st.subheader("Model Error Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Mean Absolute Error (MAE)",
        value=f"{mae:.2f}"
    )

with col2:
    st.metric(
        label="Root Mean Squared Error (RMSE)",
        value=f"{rmse:.2f}"
    )

# -----------------------------------
# Prediction Confidence Indicator
# -----------------------------------

st.subheader("Prediction Confidence")

st.progress(min(int(confidence_score), 100))

st.write(
    f"Model Confidence Score: **{confidence_score:.2f}%**"
)



# -----------------------------------
# Model Information
# -----------------------------------

st.markdown("---")
st.header("🤖 Model Information")

st.markdown("""
This dashboard uses a **Long Short-Term Memory (LSTM) neural network** for stock price forecasting.

**Model Characteristics**

- Sequence length: 60 trading days
- Input features: Close Price, SMA20, SMA50, RSI
- Architecture:
  - LSTM (64 units)
  - Dropout (0.2)
  - LSTM (32 units)
  - Dropout (0.2)
  - Dense output layer
- Optimizer: Adam
- Loss Function: Mean Squared Error

The model learns temporal patterns in stock prices and technical indicators to estimate future price movements.
""")

