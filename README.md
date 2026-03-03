# AI-Powered Stock Investment Decision System

An end-to-end machine learning pipeline that integrates **time-series feature engineering, multi-class trend prediction, explainable AI (SHAP), and quantitative risk modeling** to generate intelligent **Buy / Hold / Sell recommendations** for NSE-listed stocks.

---

## 🚀 Project Overview

This project builds a data-driven stock advisory system that:

* Collects 5 years of historical NSE stock data
* Engineers technical and statistical features
* Predicts 5-day future market direction
* Explains model decisions using SHAP
* Computes risk metrics (Volatility & Sharpe Ratio)
* Generates actionable investment recommendations

The system simulates a simplified **AI-powered robo-advisor**.

---

## 🏗️ System Architecture

```
Yahoo Finance Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Target Creation (5-Day Future Trend)
        ↓
Chronological Train/Test Split
        ↓
Model Training (Random Forest & XGBoost)
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Risk Scoring
        ↓
Buy / Hold / Sell Recommendation Engine
```

---

## 📊 Features Engineered

### 📈 Momentum Indicators

* RSI (Relative Strength Index)

### 📉 Trend Indicators

* MACD
* SMA (20, 50, 200)

### 📊 Statistical Features

* Daily Returns
* Rolling Volatility (20-day)
* Volume Ratio

### 🎯 Target Variable

* Multi-class classification:

  * `1` → Uptrend (> +2% in 5 days)
  * `0` → Stable
  * `-1` → Downtrend (< -2% in 5 days)

---

## 🤖 Models Implemented

### 1️⃣ Random Forest (Baseline Model)

* Handles non-linear feature interactions
* Provides feature importance

### 2️⃣ XGBoost (Advanced Model)

* Gradient boosting
* Improved classification performance
* Used for final predictions

---

## 🔍 Explainable AI (SHAP)

* Global feature importance visualization
* Local prediction explanation (Waterfall plots)
* Identifies which indicators influence each decision

This ensures model transparency and interpretability.

---

## 📉 Risk Modeling Module

The system calculates:

* **Annualized Volatility**
* **Sharpe Ratio**
* **Composite Risk Score (0–100 scale)**

Stocks are categorized as:

* Low Risk
* Medium Risk
* High Risk

---

## 🧠 Final Recommendation Engine

The system combines:

* Predicted Trend
* Model Confidence
* Risk Category

To generate:

* Strong Buy
* Buy
* Hold
* Sell

---

## 📁 Project Structure

```
ai-stock-advisor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib / Seaborn
* yfinance API

---

## 📌 Key Learning Outcomes

* Time-series aware ML modeling
* Avoiding data leakage in financial prediction
* Multi-class classification
* Feature importance & explainability
* Financial risk metric computation
* Decision-engine design

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice.

---

## 📎 Future Improvements

* Live stock data integration
* Hyperparameter tuning
* Portfolio optimization
* Streamlit web application deployment
* LSTM-based deep learning model

---

## 👨‍💻 Author

Ruhaan Bansal
CSE (Business Systems)
Machine Learning & Quantitative Finance Enthusiast
