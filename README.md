# 🧠 NiftyMind — AI-Powered Indian Stock Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Live](https://img.shields.io/badge/Live-niftymind.streamlit.app-brightgreen)

**A full-stack AI stock intelligence platform for Indian markets (NSE/BSE)**
*Built with Streamlit · scikit-learn · Claude Vision API · VADER NLP*

[🚀 Live App](https://niftymind.streamlit.app) · [📂 GitHub](https://github.com/Ruhaanbansal/ai-stock-advisor)

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [Features](#-features)
4. [Architecture](#-architecture)
5. [Machine Learning — Deep Dive](#-machine-learning--deep-dive)
6. [Data Sources](#-data-sources)
7. [File Structure](#-file-structure)
8. [Setup & Installation](#-setup--installation)
9. [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
10. [API Keys](#-api-keys)
11. [Configuration](#-configuration)
12. [Known Limitations](#-known-limitations)
13. [Tech Stack](#-tech-stack)

---

## Project Overview

NiftyMind is a free, open-source AI stock intelligence platform built specifically for **Indian retail investors**. It combines machine learning, real-time data, and AI vision into a single Streamlit web app accessible at [niftymind.streamlit.app](https://niftymind.streamlit.app).

### What makes it different from existing tools?

| Feature | Tickertape | Screener | Groww | **NiftyMind** |
|---------|-----------|---------|-------|--------------|
| AI price prediction | ❌ | ❌ | ❌ | ✅ |
| IPO success prediction | ❌ | ❌ | ❌ | ✅ |
| FII/DII pattern detection | ❌ | ❌ | ❌ | ✅ |
| Chart screenshot analysis | ❌ | ❌ | ❌ | ✅ |
| Portfolio optimization | ❌ | ❌ | ❌ | ✅ |
| Strategy backtesting | ❌ | ❌ | ❌ | ✅ |
| Free & open source | ❌ | ✅ | ❌ | ✅ |

### Design Philosophy

- **Zero mandatory API keys** — every feature has a free fallback
- **Python 3.14 compatible** — no TensorFlow, no PyTorch, no heavy C++ dependencies
- **Indian market focused** — NSE/BSE tickers, INR prices, NIFTY benchmarks, Indian risk-free rate
- **Production-grade fallbacks** — 6-layer data source waterfall ensures the app never goes blank

---

## Live Demo

> **URL:** [https://niftymind.streamlit.app](https://niftymind.streamlit.app)

The app is deployed on Streamlit Community Cloud and is **completely free** to use. No login, no account, no credit card.

---

## ✨ Features

### 1. Dashboard — Price Charts & Technical Indicators

The entry point for any stock analysis. Displays:

- **Candlestick / line chart** with 20-day and 50-day Simple Moving Averages (SMA)
- **Bollinger Bands** — 20-day mean ± 2 standard deviations, visualising volatility envelopes
- **MACD** (Moving Average Convergence Divergence) — 12/26-day EMA difference + 9-day signal line + histogram
- **Volume histogram** with colour-coded buying/selling pressure
- **RSI** (Relative Strength Index) with overbought (>70) and oversold (<30) threshold lines
- **KPI strip** — Current price, AI-predicted price, sentiment label, risk category, and recommendation

All charts are fully interactive (Plotly) — zoom, hover tooltips, pan, and download as PNG.

---

### 2. AI Prediction — Next Day & 5-Day Forecast

The core ML feature of the app. Given a stock's historical close prices, the model:

1. Builds a supervised learning feature matrix from a 60-day rolling window
2. Trains a **Gradient Boosting Regressor** on 1 year of daily close prices
3. Predicts **tomorrow's closing price** with percentage change delta
4. Iteratively auto-regresses to produce a **5-day price forecast**

**Output:**
- Predicted next-day price with % change from current price
- 5-day forecast chart overlaid on historical prices
- Confidence interval derived from cross-validation MAE
- Feature importance chart — which lagged prices and rolling statistics drove the prediction

> See the [Machine Learning section](#-machine-learning--deep-dive) for full technical explanation.

---

### 3. Market Intelligence — News Sentiment Analysis

Fetches and analyses the latest news headlines for the selected stock using NLP.

**Two-tier pipeline:**

| Tier | Condition | News Source | Sentiment Model |
|------|-----------|------------|----------------|
| 1 | `NEWS_API_KEY` is configured | NewsAPI.org | VADER NLP |
| 2 | No API key | yfinance built-in news feed | VADER NLP |

**What is VADER?**
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based lexical sentiment model designed for social media and financial text. It uses a hand-crafted dictionary of words with pre-assigned sentiment scores and applies grammatical rules (negation, intensifiers, punctuation). It returns a compound score in [-1, +1] with no training required.

**Output:**
- Overall sentiment score + Bullish / Neutral / Bearish classification
- Latest 10 headlines with individual sentiment scores colour-coded
- Sentiment trend chart (rolling average over recent headlines)
- News source attribution badge

---

### 4. Portfolio Optimizer — Mean-Variance Optimisation

Allows users to build and optimise a multi-stock portfolio using Modern Portfolio Theory.

**Input:**
- Up to 10 NSE/BSE stocks (typed or searched)
- Investment amount in ₹
- Risk appetite: Low / Medium / High
- Time period for historical data

**Three allocation strategies:**

| Risk Level | Strategy | Mathematical Basis |
|-----------|---------|-------------------|
| Low | Inverse-volatility weighting | `weight ∝ 1/σ` — safer stocks get more capital |
| Medium | Sharpe-ratio maximisation | SLSQP constrained optimisation |
| High | Momentum weighting | `weight ∝ max(annualised_return, 0)` |

**Output:**
- Allocation table: stock, weight %, amount in ₹, expected return, volatility, Sharpe ratio
- **Efficient Frontier** — 2,000 Monte Carlo simulated portfolios plotted as risk vs. return scatter
- Optimal portfolio highlighted on the efficient frontier
- Correlation heatmap across all selected stocks
- Per-stock risk/return metrics

---

### 5. Backtesting — Strategy Comparison Engine

Tests 4 trading strategies on the selected stock's full historical data and compares their performance.

**Strategies:**

| Strategy | Logic | Type |
|---------|-------|------|
| Buy & Hold | Always fully invested | Baseline benchmark |
| AI Strategy | Enter when previous day's return was positive | Lag-1 momentum |
| Momentum | Enter when 5-day rolling average return is positive | Medium-term trend following |
| Mean Reversion | Enter when previous day's return was negative | Counter-trend |

**Performance Metrics (per strategy):**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Total Return % | `(final_value / initial_value - 1) × 100` | Overall compounded growth |
| Maximum Drawdown % | `min((cum_ret - peak) / peak) × 100` | Worst peak-to-trough loss |
| Sharpe Ratio | `(μ_excess / σ) × √252` | Risk-adjusted return |

**Output:**
- Cumulative return chart with all 4 strategies overlaid
- Side-by-side metrics comparison table
- Best-performing strategy highlighted

---

### 6. Model Evaluation — Out-of-Sample Accuracy

Evaluates the AI prediction model's real-world performance on held-out data for the selected stock.

**Metrics computed:**

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| MAE | `mean(|actual - predicted|)` in ₹ | Average absolute prediction error |
| RMSE | `√mean((actual - predicted)²)` in ₹ | Penalises large errors more than MAE |
| MAPE | `mean(|actual - predicted| / actual) × 100` % | Scale-independent percentage error |
| Confidence | `max(0, 100 - MAPE × 2)` % | Human-readable accuracy score |

**Output:**
- Actual vs. predicted price overlay chart (last 60 trading days)
- Residual chart — prediction error over time (a good model has random residuals, no trend)
- Residual distribution histogram (should be centred near zero, roughly normal)
- All 4 metrics in a KPI strip

---

### 7. 🚀 IPO Success Predictor

Predicts whether an upcoming NSE IPO is worth subscribing to, based on 10 years of historical IPO data.

**Training dataset:** 58 NSE IPOs spanning 2015–2025, covering:
- Multiple market cycles (bull run 2017, crash 2020, boom 2021, correction 2022, recovery 2023–24)
- Diverse sectors: Technology, Pharma, NBFC, Defence, EV, Renewable Energy, and more
- Features: subscription rates (Total/QIB/HNI/Retail), GMP %, sector, NIFTY trend, issue price
- Labels: actual listing gains and 30-day post-listing returns fetched from yfinance

**Three models trained simultaneously:**

| Model | Target Variable | Algorithm |
|-------|---------------|-----------|
| `model_gain` | Listing day gain % | GradientBoosting Regressor |
| `model_return` | 30-day post-listing return % | GradientBoosting Regressor |
| `model_label` | Strong Subscribe / Subscribe / Neutral / Avoid | GradientBoosting Classifier |

**Input fields:**
- IPO name (optional, display only)
- Issue price (₹)
- Sector (27 options)
- Total, QIB, HNI, Retail subscription multiples
- GMP % (Grey Market Premium)
- NIFTY 30-day trend (auto-filled with live NIFTY data)

**Output:**
- 🟢 Strong Subscribe / 🔵 Subscribe / 🟡 Neutral / 🔴 Avoid badge with confidence %
- Predicted listing day gain % and expected listing price
- Predicted 30-day return % and total return from issue price
- Probability breakdown across all 4 outcomes with visual bars
- Top 5 feature importance drivers for this specific prediction
- 4 similar past IPOs from history for reference

---

### 8. 📡 FII/DII Tracker — Institutional Flow Analysis

Tracks Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII) activity across the entire Indian equity market (NSE as a whole, not stock-specific).

**Why FII/DII matters:**
FIIs and DIIs collectively move billions of rupees daily. When FIIs sell heavily, NIFTY typically falls; when DIIs absorb FII selling, markets find a floor. Understanding these flows gives retail investors an institutional-grade market perspective.

**Data:** 5 years of daily FII/DII flows (2020–2025), ~1,280 trading days

**4 Tabs:**

**Tab 1 — Live Data**
- Today's FII buy, sell, net and DII buy, sell, net (from NSE India API, refreshes every 30 min)
- Graceful fallback to historical data on weekends/holidays with explanation
- Last 10 trading days summary table with combined signal (🟢 Bullish / 🔴 Bearish)
- 5 correlation cards: FII → same-day NIFTY, FII → next-day NIFTY, DII → same-day, etc.

**Tab 2 — Trends**
- Monthly grouped bar chart: FII net + DII net + NIFTY monthly return overlay (3-year view)
- Rolling 30-day FII net flow chart with area fill — instantly reveals accumulation vs. distribution phases

**Tab 3 — Patterns**
Ten statistically validated patterns mined from 5 years of data. Each pattern card shows:
- Win rate (% of times NIFTY rose next day after this pattern)
- Average next-day NIFTY return
- Sample size (number of occurrences — patterns with <10 occurrences are filtered out)
- Standard deviation (measures consistency)

Example patterns:
- **Both FII & DII Buying** → strongest bullish signal (highest win rate)
- **FII Buys 3 Consecutive Days** → sustained foreign accumulation
- **Mega FII Sell (>₹5,000 Cr)** → panic exits, sharp next-day drops
- **DII Buying while FII Sells** → domestic institutional support signal
- **FII Reversal after 5-day Sell** → contrarian entry opportunity

**Tab 4 — AI Prediction**
- GradientBoosting classifier trained on 15 FII/DII flow features
- Predicts tomorrow's NIFTY direction: 📈 Bullish or 📉 Bearish
- Expected percentage magnitude of move
- Model confidence score and probability bars
- Last 5 days FII/DII context table for transparency

---

### 9. Chart Analyzer — AI Vision Analysis

Upload any stock chart screenshot and receive instant, structured technical analysis. Works with screenshots from any charting platform.

**Supported input formats:** PNG, JPG, JPEG, WEBP
**Supported platforms:** Zerodha Kite, Groww, TradingView, Google Finance, NSE India, Moneycontrol, Angel One, Upstox

**Two modes:**

| Mode | Condition | Engine | Accuracy |
|------|-----------|--------|---------|
| Full AI Analysis | `ANTHROPIC_API_KEY` set + credits | Claude claude-sonnet-4-20250514 Vision | High |
| Rule-based Fallback | No key or no credits | PIL + NumPy image analysis | Medium |

**Full AI mode output:**
- **Recommendation card** — Strong Buy / Buy / Hold / Sell / Strong Sell with confidence %
- Entry zone, stop loss, and price target (in ₹ where visible)
- Trend card: Bullish/Bearish/Sideways with strength (Strong/Moderate/Weak) and description
- Momentum card: RSI value (if visible on chart), MACD signal
- Chart pattern cards: Head & Shoulders, Double Top/Bottom, Bull Flag, Cup & Handle, etc. — with type (Continuation/Reversal) and reliability rating
- Candlestick patterns: Doji, Hammer, Shooting Star, Engulfing, Harami, etc.
- Support & Resistance levels in ₹ (colour-coded green/red)
- Risk factors list
- Plain-English summary written for retail investors

**Cost:** ~₹0.80–1.60 per analysis (Claude claude-sonnet-4-20250514 pricing, ~$0.01–0.02 USD)

---

## Architecture

```
User Browser (Chrome / Safari / Firefox)
        │
        ▼
┌──────────────────────────────────────────┐
│           Streamlit Frontend             │
│   app.py (~1,900 lines)                  │
│   Session state, navigation, 9 pages     │
└──────────┬───────────────────────────────┘
           │
  ┌────────┴────────────────────┐
  │        src/ modules         │
  └────────┬────────────────────┘
           │
  ┌────────┼──────────────────────────────┐
  ▼        ▼                              ▼
Data     ML Pipeline                External APIs
Layer    (scikit-learn)              ─────────────
──────   ────────────                NSE India API
data_    model.py                    Yahoo Finance
sources  ipo_predictor.py            Alpha Vantage
.py      fii_dii_analyzer.py         Stooq CSV
         portfolio.py                NewsAPI.org
         evaluation.py               Anthropic Claude
         explainability.py
         sentiment.py (VADER)
         risk.py
         backtest.py
         advisor.py
```

### Data Flow Per Page Load

```
Step 1 │ User selects stock from sidebar search or quick-pick buttons
       │
Step 2 │ data_sources.py — 6-layer waterfall fetch
       │ NSE India → Alpha Vantage → Stooq → yfinance .NS → yfinance .BO → CSV
       │
Step 3 │ features.py — compute all technical indicators
       │ SMA, EMA, RSI, MACD, Bollinger Bands, Volume trends
       │
Step 4 │ model.py — GradientBoosting train/predict
       │ Cached by (stock_ticker, data_hash) — retrains only when data changes
       │
Step 5 │ sentiment.py — fetch news + VADER analysis
       │ Cached for 1 hour to avoid hitting API limits
       │
Step 6 │ risk.py — compute full risk dashboard
       │ Volatility, Sharpe, Sortino, VaR, CVaR, Max Drawdown
       │
Step 7 │ advisor.py — rule-based recommendation
       │ Combines price signal + sentiment + risk into Buy/Hold/Sell
       │
Step 8 │ Streamlit renders the selected page
```

### Session State Management

Streamlit reruns the entire script on every interaction. Session state persists across reruns:

```python
st.session_state.selected_stock      # current ticker (e.g. "RELIANCE.NS")
st.session_state.selected_stock_name # display name (e.g. "Reliance Industries")
st.session_state.data_source         # which source returned data ("Yahoo Finance" etc.)
```

### Stock-Free Pages

IPO Predictor, FII/DII Tracker, Portfolio Optimizer, and Chart Analyzer do not require a stock to be selected. They are gated by:

```python
_STOCK_FREE_PAGES = {
    "🚀 IPO Predictor",
    "📡 FII/DII Tracker",
    "💼 Portfolio Optimizer",
    "📸 Chart Analyzer",
}
# Data loading and model training are skipped for these pages
```

---

## Machine Learning — Deep Dive

### 1. Stock Price Prediction — Gradient Boosting Regressor

**File:** `src/model.py`

#### Why Gradient Boosting Instead of LSTM?

The original model used a Keras LSTM (Long Short-Term Memory) neural network. It was replaced for these reasons:

1. **Python 3.14 incompatibility** — TensorFlow has no wheels for Python 3.14, making deployment on Streamlit Cloud impossible
2. **Comparable accuracy** — For financial time series with engineered features, tree ensembles match LSTM performance
3. **Speed** — GBM trains in 1–2 seconds; LSTM required minutes
4. **Zero dependencies** — No GPU, no CUDA, no C++ build tools needed

#### Feature Engineering

The model converts raw price history into a structured feature matrix. For each trading day `t`:

```
Window:   [p_{t-60}, p_{t-59}, ..., p_{t-1}]   ← 60 normalised prices (MinMaxScaler)

Rolling statistics computed from the window:
  r5  = mean of last 5 prices     (short-term trend proxy)
  r10 = mean of last 10 prices    (medium-term trend proxy)
  r20 = mean of last 20 prices    (long-term trend proxy)
  s5  = std dev of last 5 prices  (short-term volatility)
  ret = p_{t-1} - p_{t-2}         (1-day momentum)

Feature vector = [window(60), r5, r10, r20, s5, ret]
Total features = 65 per sample
```

**Normalisation:** `MinMaxScaler` maps all prices to the [0, 1] range before training. This prevents the model from being dominated by the absolute price level. Predictions are inverse-transformed back to ₹.

**Target:** `y = scaled_price[t]` (predict next day's normalised close price)

#### Model Hyperparameters

```python
GradientBoostingRegressor(
    n_estimators  = 100,   # 100 trees in the ensemble
    max_depth     = 4,     # shallow trees prevent overfitting
    learning_rate = 0.05,  # shrinkage factor — slow learning = better generalisation
    subsample     = 0.8,   # stochastic: each tree sees 80% of rows (reduces variance)
    random_state  = 42,    # reproducibility
)
```

#### How Gradient Boosting Works

Gradient Boosting sequentially builds an ensemble of weak learners (shallow decision trees), where each tree corrects the errors of all previous trees:

```
Initialise: F₀(x) = mean(y)   ← first prediction: just the average price

For m = 1 to M:
    1. Compute residuals: rᵢ = yᵢ - F_{m-1}(xᵢ)   ← what's wrong with current model
    2. Fit a shallow tree hₘ to predict residuals
    3. F_m(x) = F_{m-1}(x) + η × hₘ(x)             ← small improvement

Final prediction: F_M(x) = F₀ + η×h₁ + η×h₂ + ... + η×h_M
```

Where `η` = learning rate (0.05). The algorithm minimises Mean Squared Error loss by computing negative gradients (residuals) at each step — hence the name **gradient** boosting.

#### 5-Day Forecast — Recursive Multi-Step Prediction

```python
sequence = last_60_days.copy()

for day in range(5):
    features = build_feature_vector(sequence[-60:])
    pred_scaled = model.predict(features)
    price = scaler.inverse_transform(pred_scaled)

    forecasts.append(price)
    sequence = append(sequence, pred_scaled)  # feed prediction back as input
```

This is **recursive multi-step forecasting** (also called iterated one-step forecasting). Each prediction is treated as if it were a true observation and fed back into the model. Error compounds over time — day-5 forecasts are less reliable than day-1.

#### Caching Architecture

```python
@st.cache_resource(show_spinner=False)
def train_lstm_model(stock: str, data_hash: int, close_prices: pd.Series):
    ...
```

The function name `train_lstm_model` is preserved from the original LSTM version to avoid breaking `app.py` imports. The cache key is `(stock, data_hash)` where:

```python
data_hash = hash(close_prices.values.tobytes())
```

This means:
- **First load:** Trains in ~2 seconds, saves model to `models/{stock}.pkl`
- **Same data:** Returns from Streamlit's in-memory cache instantly
- **New trading day:** `data_hash` changes → model retrains automatically

---

### 2. IPO Success Prediction — Multi-Output GBM

**File:** `src/ipo_predictor.py`

#### Training Dataset Construction

The 58-IPO dataset was built by:
1. Curating IPO subscription data from NSE/BSE filings and Chittorgarh.com
2. Fetching actual listing prices and 30-day returns from yfinance
3. Fetching NIFTY 50 returns for the 30 days before each listing (market condition feature)
4. Using GMP % as a fallback listing gain estimate where yfinance data was unavailable

#### Feature Vector (33 features per IPO)

```python
features = [
    sub_total,            # total subscription multiple (e.g. 38.25 = 38× oversubscribed)
    sub_qib,              # QIB (Qualified Institutional Buyers) subscription
    sub_hni,              # HNI (High Net Worth Individuals) subscription
    sub_retail,           # retail investor subscription
    gmp_percent,          # Grey Market Premium as % of issue price
    nifty_trend_30d,      # NIFTY 30-day return before listing date
    log1p(issue_price),   # log of issue price (reduces skew from large-cap IPOs)
    sector_Technology,    # one-hot encoded: 1 if sector == Technology
    sector_Pharma,        # one-hot encoded: 1 if sector == Pharma
    ...                   # 27 total sector binary features
]
```

**Why log-transform the issue price?**
IPO issue prices range from ₹20 (Utkarsh Small Finance Bank) to ₹2,150 (Paytm). Without log-scaling, large-price IPOs would dominate the model. `log(price + 1)` compresses this range.

**Why does QIB subscription matter most?**
QIBs are institutional investors (mutual funds, insurance companies, FIIs) who conduct detailed fundamental analysis before bidding. High QIB subscription is a proxy for informed institutional conviction. Historically, QIB subscription × GMP are the two strongest predictors of listing gain.

**Why does GMP matter?**
Grey Market Premium is the unofficial pre-listing price. It reflects aggregated retail and HNI sentiment before official trading. A GMP of +30% means the market expects a 30% listing gain. GMP is a strong leading indicator because it incorporates information that subscription data alone cannot capture (e.g. demand intensity, grey market supply/demand).

#### Three Parallel Models

```python
# Model 1: How much % gain on listing day?
model_gain = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8
)
model_gain.fit(X, y_listing_gain)

# Model 2: What's the 30-day return after listing?
model_return = GradientBoostingRegressor(
    n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8
)
model_return.fit(X, y_30d_return)

# Model 3: What's the subscription recommendation?
model_label = GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8
)
model_label.fit(X, y_encoded_label)
```

Label thresholds:
```
listing_gain >= 30%  →  "Strong Subscribe"
listing_gain 10–30%  →  "Subscribe"
listing_gain  0–10%  →  "Neutral"
listing_gain  < 0%   →  "Avoid"
```

#### Cross-Validation

```python
cv_scores = cross_val_score(
    model_gain, X, y_gain,
    cv=5,                              # 5-fold CV
    scoring='neg_mean_absolute_error'
)
mae_cv = -cv_scores.mean()            # displayed as model accuracy in UI
```

5-fold CV means the model is trained on 80% of IPOs and tested on the remaining 20%, five times, each with a different held-out fold. The MAE shown in the UI is the average prediction error across these 5 held-out sets — a realistic estimate of real-world accuracy.

#### Similar IPO Lookup

```python
def find_similar_ipos(sub_total, gmp_percent, sector, n=4):
    similarity = (
        abs(df.sub_total - sub_total) / (sub_total + 1) +
        abs(df.gmp_percent - gmp_percent) / (|gmp_percent| + 1)
    )
    return df.nsmallest(n, 'similarity')
```

This finds the most similar historical IPOs by subscription pattern and GMP within the same sector, giving users concrete reference points.

---

### 3. FII/DII → NIFTY Direction Prediction

**File:** `src/fii_dii_analyzer.py`

#### Feature Engineering (15 features)

```python
features = {
    # Raw flows (₹ Crores)
    "fii_net":          today's FII net buy/sell,
    "dii_net":          today's DII net buy/sell,
    "fii_dii_net":      combined institutional flow,

    # Rolling sums (accumulation/distribution signals)
    "fii_3d":           3-day rolling FII net,
    "fii_5d":           5-day rolling FII net,
    "fii_10d":          10-day rolling FII net,
    "dii_3d":           3-day rolling DII net,
    "dii_5d":           5-day rolling DII net,

    # Momentum (acceleration of flows)
    "fii_momentum":     fii_net - fii_net.shift(5),
    "dii_momentum":     dii_net - dii_net.shift(5),

    # Streak (consecutive direction signals)
    "fii_streak":       +3 = FII bought 3 days in a row, -2 = sold 2 days,
    "dii_streak":       same for DII,

    # Dominance (who is in control today)
    "fii_dominance":    fii_net / (|fii_net| + |dii_net|),
    "dii_dominance":    dii_net / (|fii_net| + |dii_net|),

    # Divergence (are flows contradicting price action?)
    "fii_nifty_div":    sign(fii_net) - sign(nifty_return),
}
```

#### Feature Scaling

Unlike price data (all in ₹), FII/DII features span very different scales:
- Raw flows: ₹-100,000 Cr to ₹+100,000 Cr
- Streak counts: -10 to +10
- Dominance ratio: -1 to +1

`StandardScaler` normalises each feature to mean=0, std=1:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This prevents large-magnitude features (raw flows) from drowning out informative small-scale features (streaks, dominance).

#### Two-Model Architecture

```python
# Direction: will NIFTY go up or down tomorrow?
model_dir = GradientBoostingClassifier(n_estimators=200, ...)
y_dir = (df["nifty_next_return"] > 0).astype(int)   # 1 = up, 0 = down
model_dir.fit(X_scaled, y_dir)

# Magnitude: by how much % will NIFTY move?
model_mag = GradientBoostingRegressor(n_estimators=150, ...)
model_mag.fit(X_scaled, df["nifty_next_return"])
```

#### Statistical Pattern Detection (Non-Parametric)

Beyond ML, the app mines patterns directly from data:

```python
# Example: "Both FII & DII Buying" pattern
mask = (df["fii_net"] > 0) & (df["dii_net"] > 0)
subset = df[mask]

avg_next_return = subset["nifty_next_return"].mean()
win_rate = (subset["nifty_next_return"] > 0).mean() * 100
sample_size = len(subset)
```

This approach:
- Makes no distributional assumptions (non-parametric)
- Is fully transparent — you can inspect every data point
- Requires minimum 10 occurrences to ensure statistical validity
- Sorts patterns by `|avg_return|` to surface the most impactful signals first

This is a form of **rule-based data mining** — letting the historical data surface its own patterns rather than imposing a model.

---

### 4. Portfolio Optimisation — Modern Portfolio Theory

**File:** `src/portfolio.py`

#### Mathematical Foundation (Markowitz, 1952)

For a portfolio of `n` stocks with weight vector `w`:

```
Expected annual return:   μ_p = wᵀ μ        (dot product of weights × mean returns)
Annual variance:          σ²_p = wᵀ Σ w      (quadratic form with covariance matrix)
Annual volatility:        σ_p = √(σ²_p)
Sharpe Ratio:             S = (μ_p - r_f) / σ_p
```

Where:
- `μ` = vector of annualised mean returns (daily_return × 252)
- `Σ` = annualised covariance matrix (daily_cov × 252)
- `r_f` = risk-free rate (Indian 10Y bond = 6.5%)

#### Maximum Sharpe Optimisation

```python
result = scipy.optimize.minimize(
    fun         = lambda w: -sharpe_ratio(w),   # maximise Sharpe = minimise -Sharpe
    x0          = np.repeat(1/n, n),             # equal weights as starting guess
    method      = "SLSQP",                       # Sequential Least Squares Programming
    bounds      = [(0.0, 1.0)] × n,             # no short-selling (weights ≥ 0)
    constraints = [{"type": "eq",
                    "fun": lambda w: np.sum(w) - 1}]  # fully invested (weights sum to 1)
)
```

**SLSQP** is a gradient-based constrained optimisation algorithm. It iteratively moves `w` in the direction that improves the objective while satisfying constraints. It finds the portfolio at the **tangency point** of the Capital Market Line and the efficient frontier.

#### Efficient Frontier (Monte Carlo Simulation)

```python
for _ in range(2000):
    w = np.random.dirichlet(np.ones(n))          # random weights summing to 1
    ret, vol = portfolio_performance(w, μ, Σ)
    sharpe = (ret - r_f) / vol
    plot(vol, ret, color=sharpe)                 # colour by Sharpe
```

`np.random.dirichlet(ones(n))` generates uniformly distributed random weight vectors that sum to 1. 2,000 simulations fill out the feasible portfolio space. The curve traced by the top-left edge of the scatter is the **efficient frontier** — the set of portfolios that offer maximum return for each level of risk.

#### Three Risk Strategies in Detail

**Low Risk — Inverse Volatility Weighting:**
```python
inv_vol = 1.0 / σ_i    for each stock i
weights = inv_vol / sum(inv_vol)
```
Intuition: A stock with half the volatility gets twice the allocation. This naturally diversifies risk.

**Medium Risk — Sharpe Maximisation:**
Uses the SLSQP optimisation above. Produces the theoretically optimal risk-adjusted portfolio.

**High Risk — Momentum Weighting:**
```python
pos_ret = max(annualised_return_i, 0)   for each stock i
weights = pos_ret / sum(pos_ret)
```
Concentrates capital in recent winners. Equivalent to a momentum factor tilt.

---

### 5. Risk Analytics — Quantitative Risk Measures

**File:** `src/risk.py`

#### Historical Volatility
```
σ_annual = std(daily_returns) × √252
```
The `√252` factor annualises daily volatility (252 trading days per year). This is the standard industry convention.

#### Sharpe Ratio
```
Sharpe = mean(daily_excess_returns) / std(daily_returns) × √252
daily_excess_return = daily_return - (r_f / 252)
```
Measures return per unit of total risk. The Indian 10-year government bond yield (6.5%) is used as the risk-free rate.

#### Sortino Ratio
```
Sortino = mean(daily_excess_returns) / std(negative_daily_returns) × √252
```
Like Sharpe, but only penalises **downside** volatility. A stock that goes up a lot and rarely drops scores much better on Sortino than Sharpe.

#### Value at Risk (Historical VaR)
```
VaR₉₅ = -percentile(daily_returns, 5)
```
The 5th percentile of the daily return distribution. Interpretation: "On 95% of trading days, the loss will not exceed VaR₉₅." No normality assumption — uses the actual empirical distribution.

#### Conditional VaR (Expected Shortfall)
```
CVaR₉₅ = -mean(daily_returns | daily_returns ≤ -VaR₉₅)
```
The average loss in the worst 5% of days. Always ≥ VaR. More useful for tail risk management because it captures the severity (not just the probability) of extreme losses.

#### Maximum Drawdown
```
cum_returns = (1 + daily_returns).cumprod()
peak = cum_returns.cummax()
drawdown = (cum_returns - peak) / peak
MDD = drawdown.min()
```
The largest percentage decline from a peak to a subsequent trough over the full period. Critical for understanding the worst-case scenario a buy-and-hold investor would have faced.

---

### 6. Sentiment Analysis — VADER NLP

**File:** `src/sentiment.py`

#### VADER Scoring

For each headline, VADER computes 4 scores:
- `pos` — proportion of text expressing positive sentiment
- `neg` — proportion of text expressing negative sentiment
- `neu` — proportion of text expressing neutral sentiment
- `compound` — normalised weighted composite score in [-1, +1]

The compound score aggregates all three with a sigmoid normalisation:
```
compound = (sum_of_weighted_valence_scores) / √((sum²) + α)
```
Where `α = 15` (normalisation constant).

**Classification thresholds (from `src/config.py`):**
```
compound > 0.05   →  Bullish
compound < -0.05  →  Bearish
otherwise         →  Neutral
```

**Aggregation:**
```python
overall_score = mean([vader.polarity_scores(headline)["compound"]
                      for headline in headlines])
```

#### Why VADER instead of FinBERT?

FinBERT (a financial-domain fine-tuned BERT model) was the original plan. It was replaced because:
- Requires `transformers` + `torch` (~2GB install)
- PyTorch has no Python 3.14 compatible wheels (as of 2025)
- VADER achieves strong performance on short financial headlines
- Zero installation complexity, works offline

---

### 7. Chart Analysis — Claude Vision + Rule-Based Fallback

**File:** `src/chart_analyzer.py`, `src/chart_analyzer_fallback.py`

#### Prompt Engineering for Claude claude-sonnet-4-20250514

The prompt instructs Claude to:
1. Return **only valid JSON** (no markdown, no preamble) for easy parsing
2. Use `null` for anything not visible — avoids hallucination of specific price levels
3. Consider Indian market context (NSE/BSE, prices in ₹)
4. Be specific about price levels only when they are clearly readable in the chart

The structured JSON schema ensures the UI can reliably render every section without error handling for missing keys.

#### Rule-Based Fallback (PIL + NumPy)

When the Claude API is unavailable:

```python
# Step 1: Resize image to 400×300
img = Image.open(image_bytes).convert("RGB").resize((400, 300))
arr = np.array(img)

# Step 2: Extract price curve
gray  = np.mean(arr, axis=2)                    # convert to grayscale
curve = np.argmax(gray, axis=0)                  # brightest row per column ≈ price line
curve = img_height - curve                       # invert (higher price = higher y)
curve = np.convolve(curve, np.ones(10)/10, 'same')  # smooth noise

# Step 3: Trend detection
slope = np.polyfit(np.arange(len(curve)), curve, 1)[0]
direction = "Bullish" if slope > 0.05 else "Bearish" if slope < -0.05 else "Sideways"

# Step 4: Pattern detection
left  = mean(curve[:n//3])
mid   = mean(curve[n//3:2*n//3])
right = mean(curve[2*n//3:])
# Double Top: left ≈ right >> mid
# Double Bottom: left ≈ right << mid
# V-shape: minimum in the middle third
```

The fallback provides directional guidance but cannot detect exact patterns or read price levels from the axis. It is honest about this — showing relative % levels rather than absolute ₹ values.

---

## Data Sources

### 6-Layer Data Waterfall (`src/data_sources.py`)

The app never shows a blank chart. If one source fails, it silently moves to the next:

```
Layer 1 │ NSE India public API
        │ Cookie-based session, free, best for freshly listed stocks
        │ Endpoint: https://www.nseindia.com/api/chart-databyindex
        ↓ (if unavailable, rate-limited, or returns empty)

Layer 2 │ Alpha Vantage
        │ Requires ALPHA_VANTAGE_KEY in secrets
        │ Supports NSE (e.g. RELIANCE.BSE) and BSE symbols
        ↓ (if unavailable or key not set)

Layer 3 │ Stooq direct CSV download
        │ Free, no API key, Polish financial data aggregator
        │ Tries: symbol.NS, symbol.IN, symbol
        ↓ (if unavailable)

Layer 4 │ yfinance (primary: .NS suffix)
        │ Auto-retries .BO (BSE) if .NS returns 404
        │ Known bad tickers pre-registered in _YAHOO_USE_BO set
        ↓ (if unavailable)

Layer 5 │ yfinance download() (alternative internal method)
        ↓ (if all API sources fail)

Layer 6 │ Bundled CSV fallback
        │ Pre-downloaded files in data/raw/
        │ RELIANCE.NS.csv, TCS.NS.csv, HDFCBANK.NS.csv
        │ May be stale but always available
```

### Auto-Ticker Registration

Yahoo Finance silently drops some NSE tickers when they update their feed. The `_YAHOO_USE_BO` set handles this:

```python
_YAHOO_USE_BO = {
    "TATAMOTORS.NS",   # dropped from NSE feed, use BSE (.BO)
    "ZOMATO.NS",
    "NYKAA.NS",
    "PAYTM.NS",
    "POLICYBZR.NS",
}
```

When a ticker in this set is requested, the code skips directly to `.BO` without attempting `.NS` first. New bad tickers are auto-registered at runtime via `_mark_use_bo()`.

### FII/DII Data Construction

```
Live:       NSE India /api/fiidiiTradeReact (post-market, trading days only)
Historical: Built from curated monthly NSE archive data (2020–2025)
            Monthly totals distributed across trading days using
            np.random.normal() with controlled noise to simulate
            realistic daily variation while preserving monthly sums
NIFTY:      yfinance "^NSEI" for daily returns and monthly returns
```

---

## File Structure

```
ai-stock-advisor/
│
├── app.py                          # Main Streamlit application (~1,900 lines)
│                                   # 9 pages, sidebar navigation, session state
│
├── requirements.txt                # Python package dependencies
├── .env                            # Local API keys (never committed — in .gitignore)
├── .gitignore                      # Excludes: models/, data/processed/, .env, __pycache__
├── README.md                       # This documentation file
│
├── data/
│   ├── fetch_ipo_data.py           # One-time dataset builder: 58 NSE IPOs (2015–2025)
│   ├── fetch_fii_dii_data.py       # One-time dataset builder: 5yr FII/DII history
│   ├── raw/
│   │   ├── ipo_historical.csv      # IPO training dataset (58 rows, 15 columns)
│   │   ├── fii_dii_historical.csv  # FII/DII training dataset (~1,280 rows)
│   │   ├── RELIANCE.NS.csv         # Bundled fallback: Reliance price history
│   │   ├── TCS.NS.csv              # Bundled fallback: TCS price history
│   │   └── HDFCBANK.NS.csv         # Bundled fallback: HDFC Bank price history
│   └── processed/                  # Auto-generated feature cache (gitignored)
│
├── models/                         # Trained model pickle files (gitignored)
│   ├── RELIANCE_NS.pkl             # GBM model for Reliance
│   ├── TCS_NS.pkl                  # GBM model for TCS
│   ├── ipo_model.pkl               # IPO prediction model bundle
│   └── fii_dii_model.pkl           # FII/DII prediction model bundle
│
└── src/
    ├── __init__.py                 # Package initialisation
    ├── config.py                   # All constants and thresholds
    ├── data_sources.py             # 6-layer data waterfall + auto-ticker registration
    ├── dataloader.py               # load_stock_data(), get_close_prices() wrappers
    ├── stock_search.py             # NSE ticker database + fuzzy search
    ├── features.py                 # Technical indicator computation (SMA, RSI, MACD, etc.)
    ├── model.py                    # GradientBoosting price prediction + 5-day forecast
    ├── evaluation.py               # Out-of-sample MAE/RMSE/MAPE evaluation
    ├── explainability.py           # Permutation importance + technical signal detection
    ├── sentiment.py                # VADER NLP sentiment pipeline (2-tier)
    ├── risk.py                     # VaR, CVaR, Sharpe, Sortino, drawdown
    ├── advisor.py                  # Rule-based Buy/Hold/Sell recommendation engine
    ├── alerts.py                   # Price alert detection logic
    ├── backtest.py                 # 4-strategy backtesting engine
    ├── portfolio.py                # Mean-variance portfolio optimisation (SLSQP)
    ├── insight.py                  # AI-generated stock insight text
    ├── ipo_predictor.py            # IPO success prediction (3 parallel GBM models)
    ├── fii_dii_analyzer.py         # FII/DII pattern detection + ML NIFTY prediction
    ├── chart_analyzer.py           # Claude claude-sonnet-4-20250514 vision chart analysis
    └── chart_analyzer_fallback.py  # Rule-based fallback (PIL + NumPy)
```

---


---

## API Keys

All API keys are **completely optional**. Every feature has a free fallback that works without any key.

| Key | Source | Feature Enabled | Free Tier |
|-----|--------|----------------|-----------|
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org/register) | Better news coverage (100+ sources) | 100 req/day |
| `ALPHA_VANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Reliable stock data fallback | 25 req/day |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Full AI chart analysis | Pay-as-you-go |

**Without any keys:**
- News comes from yfinance's built-in feed (works for all NSE stocks)
- Stock data comes from NSE India → Stooq → yfinance (waterfall)
- Chart analysis uses PIL-based rule-based analysis

### Adding to Streamlit Cloud Secrets

In your Streamlit Cloud dashboard → Manage App → Settings → Secrets:

```toml
NEWS_API_KEY       = "abc123yourkeyhere"
ALPHA_VANTAGE_KEY  = "XYZ789yourkeyhere"
ANTHROPIC_API_KEY  = "sk-ant-api03-yourkeyhere"
```

### Adding to Local `.env`

```env
NEWS_API_KEY=abc123yourkeyhere
ALPHA_VANTAGE_KEY=XYZ789yourkeyhere
ANTHROPIC_API_KEY=sk-ant-api03-yourkeyhere
```

---

## Configuration

All tunable constants are in `src/config.py`:

| Constant | Default | Description |
|---------|---------|-------------|
| `SEQUENCE_LENGTH` | `60` | Days of price history used as model input |
| `RISK_FREE_RATE` | `0.065` | Indian 10Y government bond yield (6.5% p.a.) |
| `LOW_RISK_THRESHOLD` | `0.15` | Annualised volatility below which = "Low Risk" |
| `HIGH_RISK_THRESHOLD` | `0.30` | Annualised volatility above which = "High Risk" |
| `BULLISH_THRESHOLD` | `0.05` | VADER compound score above which = "Bullish" |
| `BEARISH_THRESHOLD` | `-0.05` | VADER compound score below which = "Bearish" |
| `BACKTEST_PERIOD` | `"2y"` | Default yfinance period for backtesting data |
| `NEWS_PAGE_SIZE` | `10` | Headlines fetched per sentiment analysis call |

---

## Known Limitations

**1. Price data delay:** Free APIs provide 15-minute delayed data. `auto_adjust=True` further adjusts for dividends and stock splits. Small differences from broker terminals are expected and normal.

**2. Yahoo Finance rate limiting:** Streamlit Cloud's shared IP addresses are occasionally rate-limited by Yahoo Finance. The 6-layer waterfall handles this transparently, but brief loading delays can occur during heavy traffic periods.

**3. IPO model dataset size:** The training set contains 58 IPOs. While sufficient to learn general patterns, it is small compared to production-grade models. Accuracy improves as more IPOs are added to `ipo_historical.csv`. The model is most reliable for IPOs with extreme subscription (very high or very low) and clear GMP signals.

**4. FII/DII weekend data:** NSE publishes FII/DII data only after market close on trading days (by ~6 PM IST). The Live Data tab shows historical data on weekends, public holidays, and before 6 PM on trading days.

**5. Chart Analyzer without API credits:** The PIL-based fallback analyses image brightness patterns and cannot read actual price levels from chart axes. Support/Resistance levels are shown as "X% of visible range" rather than absolute ₹ values. Adding ₹5 of Anthropic credits enables full AI analysis.

**6. Portfolio optimization limitations:** Uses historical returns to estimate future expected returns — a standard but imperfect assumption. The SLSQP optimizer finds a local optimum which may not be globally optimal for highly non-convex problems with many assets. The efficient frontier simulation uses 2,000 random portfolios which provides good coverage for up to 10 stocks.

**7. VADER vs FinBERT:** VADER is a general-purpose sentiment model, not finance-specific. FinBERT would provide better accuracy on financial text but requires PyTorch (incompatible with Python 3.14). VADER performs acceptably on short headlines but may misinterpret domain-specific financial terminology.

---

## Tech Stack

| Category | Technology | Version | Purpose |
|---------|-----------|---------|---------|
| **Web Framework** | Streamlit | ≥1.35.0 | UI, routing, state management, deployment |
| **Data Manipulation** | pandas | ≥2.0.0 | DataFrames, time series, all data ops |
| **Numerical Computing** | NumPy | ≥1.24.0 | Arrays, statistics, linear algebra |
| **Financial Data** | yfinance | ≥0.2.50 | Historical prices, news feed |
| **HTTP / TLS** | curl_cffi | ≥0.6.0 | Chrome fingerprint spoofing for Yahoo Finance |
| **Machine Learning** | scikit-learn | ≥1.3.0 | GradientBoosting, scalers, CV, metrics |
| **Optimisation** | SciPy | ≥1.10.0 | SLSQP portfolio optimisation |
| **NLP / Sentiment** | vaderSentiment | ≥3.3.2 | News headline sentiment scoring |
| **News API** | newsapi-python | ≥0.2.7 | Financial headline fetching |
| **Visualisation** | Plotly | ≥5.15.0 | Interactive charts and graphs |
| **Image Processing** | Pillow | ≥10.0.0 | Chart analyzer fallback |
| **AI Vision** | Anthropic SDK | ≥0.25.0 | Claude claude-sonnet-4-20250514 chart analysis |
| **Deployment** | Streamlit Cloud | — | Free hosting, CI/CD from GitHub |
| **Language** | Python | 3.14 | Application runtime |

---

## License

MIT License

Copyright (c) 2025 Ruhaan Bansal

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

## Acknowledgements

- [NSE India](https://www.nseindia.com) — public market data and FII/DII APIs
- [Yahoo Finance](https://finance.yahoo.com) via yfinance — historical price data
- [Anthropic](https://anthropic.com) — Claude claude-sonnet-4-20250514 multimodal AI
- [Streamlit](https://streamlit.io) — deployment platform and web framework
- [scikit-learn](https://scikit-learn.org) — machine learning toolkit
- [Alpha Vantage](https://alphavantage.co) — reliable financial data API
- [NewsAPI](https://newsapi.org) — financial news aggregation

---

<div align="center">

Built with ❤️ for Indian retail investors

[niftymind.streamlit.app](https://niftymind.streamlit.app) · [GitHub](https://github.com/Ruhaanbansal/ai-stock-advisor)

*"Making institutional-grade AI tools accessible to every retail investor"*

</div>
