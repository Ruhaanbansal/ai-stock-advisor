# 🧠 NiftyMind — AI-Powered Indian Stock Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://niftymind.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

NiftyMind is a full-stack AI stock analysis platform for Indian markets (NSE/BSE). It combines machine learning price prediction, real-time sentiment analysis, portfolio optimisation, and backtesting — all in a sleek dark-themed web app.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📈 **AI Price Prediction** | GradientBoosting model trained on 1 year of historical data with 5-day forecast |
| 📰 **Market Intelligence** | Live news sentiment via VADER NLP + NewsAPI integration |
| 💼 **Portfolio Optimizer** | Sharpe-ratio, inverse-volatility & momentum weighted allocation with efficient frontier |
| 🔁 **Backtesting** | Compare Buy & Hold vs AI vs Momentum vs Mean Reversion strategies |
| 📊 **Model Evaluation** | MAE, RMSE, MAPE metrics with actual vs predicted chart |
| 🔍 **NSE Stock Search** | 150+ NSE stocks with live autocomplete suggestions |
| 🌐 **Multi-Source Data** | Yahoo Finance → Alpha Vantage → Stooq waterfall with auto-fallback |

---

## 🚀 Live Demo

👉 **[niftymind.streamlit.app](https://niftymind.streamlit.app)**

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly, custom CSS
- **ML Model:** scikit-learn GradientBoostingRegressor
- **Data Sources:** yfinance, Alpha Vantage, Stooq, NSE India API
- **NLP:** VADER Sentiment, NewsAPI
- **Portfolio Math:** scipy, numpy (Markowitz optimisation)

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/Ruhaanbansal/ai-stock-advisor.git
cd ai-stock-advisor
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the root directory:
```env
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alphavantage_key_here
```

Get free API keys from:
- **NewsAPI:** https://newsapi.org/register (free tier: 100 requests/day)
- **Alpha Vantage:** https://www.alphavantage.co/support/#api-key (free tier: 25 requests/day)

> **Note:** Both keys are optional. The app works without them using yfinance + VADER as fallbacks.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-stock-advisor/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to git)
├── models/                 # Saved ML models (auto-created, gitignored)
├── data/
│   ├── raw/                # Historical CSV files
│   └── processed/          # Feature-engineered data
├── notebooks/              # Jupyter notebooks for EDA & model training
└── src/
    ├── config.py           # Constants and configuration
    ├── data_sources.py     # Multi-source data fetching waterfall
    ├── dataloader.py       # Data loading interface
    ├── stock_search.py     # NSE stock search & autocomplete
    ├── model.py            # GradientBoosting price prediction
    ├── features.py         # Technical indicators (RSI, MACD, BB, ATR)
    ├── sentiment.py        # News sentiment analysis
    ├── risk.py             # Risk metrics (VaR, CVaR, Sharpe, Sortino)
    ├── advisor.py          # AI recommendation engine
    ├── alerts.py           # Market alert detection
    ├── portfolio.py        # Portfolio optimisation
    ├── backtest.py         # Strategy backtesting
    ├── evaluation.py       # Model evaluation metrics
    ├── explainability.py   # Feature importance & technical signals
    └── insight.py          # Natural language market insights
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**
3. Select your repo, branch `main`, and main file `app.py`
4. Go to **Settings → Secrets** and add:
```toml
NEWS_API_KEY = "your_key_here"
ALPHA_VANTAGE_KEY = "your_key_here"
```
5. Click **Deploy**

---

## 📊 Data Sources

NiftyMind uses a **waterfall fallback** strategy to ensure data is always available:

1. **Yahoo Finance** (primary) — free, no key, real-time
2. **Alpha Vantage** (fallback) — requires free API key
3. **Stooq** (fallback) — free, no key
4. **Bundled CSVs** (last resort) — repo data for major stocks

---

## ⚠️ Disclaimer

This application is built for **educational purposes only**. Nothing on NiftyMind constitutes financial advice. Always do your own research before making investment decisions.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ for Indian markets
</div>