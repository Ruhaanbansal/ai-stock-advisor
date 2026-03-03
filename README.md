# 📈 AI-Powered Stock Advisory System

An end-to-end quantitative machine learning system that integrates **time-series feature engineering, multi-class trend prediction (XGBoost), SHAP explainability, and quantitative risk modeling** to generate intelligent **Buy / Hold / Sell recommendations** for NSE-listed stocks.

---
## 🌐 Live Demo

🔗 https://ai-stock-advisor-umgtb43rb9fhdxdxeqy6me.streamlit.app/

## 🔎 Problem Statement

Financial markets are noisy, non-linear, and risk-sensitive.
The goal of this project is to design a structured ML pipeline that:

* Predicts 5-day future stock direction
* Quantifies investment risk
* Provides interpretable model explanations
* Outputs actionable investment decisions

---

## 🧠 Core Contributions

✔ Built time-series aware ML pipeline (no data leakage)
✔ Engineered 10+ financial indicators (RSI, MACD, SMA, Volatility, Volume Ratio)
✔ Implemented multi-class classification (-1, 0, +1 trend prediction)
✔ Compared Random Forest vs XGBoost performance
✔ Integrated SHAP for global & local model explainability
✔ Designed composite risk scoring using Volatility & Sharpe Ratio
✔ Developed rule-based recommendation engine

---

## 📊 Feature Engineering

### Momentum

* RSI (14-day)

### Trend

* MACD & Signal Line
* SMA (20, 50, 200)

### Statistical

* Daily Returns
* 20-day Rolling Volatility
* Volume Ratio

---

## 🤖 Model Performance

| Model         | Purpose                        |
| ------------- | ------------------------------ |
| Random Forest | Baseline non-linear classifier |
| XGBoost       | Final gradient boosting model  |

Chronological 80/20 split used to simulate real-world forecasting.

Evaluation Metrics:

* Accuracy
* Weighted F1-Score
* Confusion Matrix

---

## 🔍 Explainable AI

Implemented SHAP to:

* Identify most influential financial indicators
* Explain individual Buy/Sell predictions
* Improve model transparency

---

## 📉 Risk Modeling

Calculated:

* Annualized Volatility
* Sharpe Ratio
* Composite Risk Score (0–100 scale)

Risk categories:

* Low
* Medium
* High

---

## 🏗️ System Pipeline

```
Data Collection → Feature Engineering → Target Creation
→ Chronological Split → Model Training
→ Evaluation → SHAP Explainability
→ Risk Scoring → Recommendation Engine
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib / Seaborn
* yfinance

---

## 📁 Project Structure

```
ai-stock-advisor/
│
├── data/
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

## 🚀 Future Improvements

* Hyperparameter tuning
* LSTM-based deep learning model
* Portfolio optimization module
* Streamlit web deployment

---

## 👨‍💻 Author

Ruhaan Bansal
CSE (Business Systems)
Focused on ML, Quantitative Finance & Decision Intelligence
