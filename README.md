---

# 📈 AI-Powered Stock Advisor

## 🌐 Live Demo

🔗 **Try the live dashboard:**
[https://moneymeow.streamlit.app/](https://moneymeow.streamlit.app/) *(replace if your deployed link is different)*

---

# 📊 Project Overview

The **AI-Powered Stock Advisor** is a machine learning and deep learning–based financial analytics system designed to analyze stock market data, forecast future prices, evaluate risk, and generate investment recommendations.

The system integrates **time-series forecasting using LSTM neural networks**, **portfolio optimization techniques**, and **strategy backtesting** to create an interactive investment decision-support dashboard.

The dashboard allows users to:

* Analyze stock performance
* Predict future price movements
* Optimize investment portfolios
* Compare stock returns
* Evaluate trading strategies
* Visualize model performance

The application is deployed using **Streamlit Cloud**, making the analytics platform accessible through a web interface.

---

# 🚀 Key Features

### 🤖 LSTM Price Forecasting

* Deep learning model predicts future stock prices.
* Uses **60-day historical sequence windows**.
* Incorporates technical indicators like:

  * SMA20
  * SMA50
  * RSI

### 📊 Portfolio Optimization

Implements **Modern Portfolio Theory (Mean-Variance Optimization)** to calculate optimal asset allocation.

Features:

* Expected return estimation
* Covariance matrix calculation
* Sharpe ratio maximization
* Portfolio allocation visualization

---

### 📈 Multi-Stock Performance Comparison

Users can compare the historical performance of multiple stocks using normalized price charts.

Capabilities:

* Select multiple stocks
* Normalize price movements
* Identify best performing assets

---

### 📉 Strategy Backtesting

The dashboard simulates trading strategies and compares them with traditional buy-and-hold investing.

Strategies compared:

| Strategy    | Description                                      |
| ----------- | ------------------------------------------------ |
| Buy & Hold  | Invest and hold asset for entire period          |
| AI Strategy | Invest only when positive momentum signal occurs |

This enables evaluation of **strategy profitability over time**.

---

### 📊 Model Evaluation

The project includes model performance diagnostics such as:

* **Actual vs Predicted Price Visualization**
* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

These metrics help quantify prediction accuracy.

---

### 📉 Risk Assessment

Stocks are classified based on volatility into:

* Low Risk
* Medium Risk
* High Risk

Risk is calculated using **rolling volatility and return statistics**.

---

### 📥 Data Export

Users can download:

* Processed stock datasets
* Backtest results
* Strategy comparison data

This allows deeper offline analysis.

---

# 🧠 Machine Learning Model

The forecasting model uses a **Long Short-Term Memory (LSTM) neural network**, which is well suited for time-series prediction.

### Model Architecture

```
Input Features:
Close Price
SMA20
SMA50
RSI

Architecture:
LSTM (64 units)
Dropout (0.2)
LSTM (32 units)
Dropout (0.2)
Dense Output Layer
```

### Training Setup

| Parameter       | Value              |
| --------------- | ------------------ |
| Sequence Length | 60 trading days    |
| Loss Function   | Mean Squared Error |
| Optimizer       | Adam               |
| Scaling         | MinMaxScaler       |

The model learns temporal patterns from historical price movements and technical indicators to forecast future prices.

---

# 📊 Dashboard Components

The interactive dashboard includes multiple analytical modules:

### Stock Analysis

* Price history visualization
* Risk classification
* Investment recommendation

### Portfolio Optimization

* Multi-asset portfolio allocation
* Sharpe ratio maximization
* Portfolio allocation chart

### Multi-Stock Comparison

* Performance comparison across stocks
* Normalized growth visualization

### Strategy Backtesting

* AI strategy vs Buy-and-Hold comparison
* Cumulative return visualization

### Model Evaluation

* Actual vs Predicted price chart
* Error metrics (MAE, RMSE)
* Model confidence indicator

---

# ⚙️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

| Category         | Tools              |
| ---------------- | ------------------ |
| Data Processing  | Pandas, NumPy      |
| Machine Learning | Scikit-learn       |
| Deep Learning    | TensorFlow / Keras |
| Financial Data   | yfinance           |
| Visualization    | Matplotlib         |
| Web Dashboard    | Streamlit          |

---

# 📂 Project Structure

```
ai-stock-advisor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_portfolio_optimization.ipynb
│   └── 06_lstm_model.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
└── assets/
```

---

# 🔬 Methodology

The project follows a structured machine learning pipeline:

1. Data Collection
   Historical stock data is downloaded using **Yahoo Finance API**.

2. Data Preprocessing

   * Missing value handling
   * Feature scaling
   * Time-series sequencing

3. Feature Engineering

   * Moving averages
   * Momentum indicators
   * Technical signals

4. Model Training
   LSTM neural network trained on historical sequences.

5. Strategy Simulation
   Trading strategy evaluated using backtesting.

6. Deployment
   Interactive dashboard deployed using **Streamlit Cloud**.

---

# 📈 Example Use Cases

This system can be used for:

* Stock market analysis
* Investment strategy testing
* Financial data visualization
* ML-based forecasting research
* Portfolio allocation experimentation

---

# 🔮 Future Improvements

Possible enhancements include:

* Transformer-based time-series models
* Reinforcement learning trading agents
* Real-time market data streaming
* News sentiment analysis integration
* Automated portfolio rebalancing
* Advanced risk metrics (VaR, CVaR)

---

# 👨‍💻 Author

**Ruhaan Bansal**

Computer Science Engineering
Machine Learning & Data Analytics Enthusiast

---

# ⭐ If you found this project useful, consider giving it a star!

