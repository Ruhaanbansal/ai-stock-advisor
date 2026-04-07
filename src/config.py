# =============================================================
# config.py — Centralized Configuration
# =============================================================

# ── Stock Universe ────────────────────────────────────────────
STOCK_LIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "LTIM.NS", "AXISBANK.NS"
]

STOCK_LABELS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "INFY.NS":      "Infosys",
    "HDFCBANK.NS":  "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "WIPRO.NS":     "Wipro",
    "LTIM.NS":      "LTIMindtree",
    "AXISBANK.NS":  "Axis Bank",
}

# ── Data ─────────────────────────────────────────────────────
DEFAULT_PERIOD   = "6mo"
BACKTEST_PERIOD  = "1y"
PORTFOLIO_PERIOD = "1y"

# ── GradientBoosting Model ────────────────────────────────────
SEQUENCE_LENGTH   = 60        # lookback window (days)
GB_N_ESTIMATORS   = 200
GB_MAX_DEPTH      = 3
GB_LEARNING_RATE  = 0.05
GB_SUBSAMPLE      = 0.8
GB_MIN_SAMPLES    = 5
GB_MAX_FEATURES   = 0.8
GB_RANDOM_STATE   = 42

# ── Risk Thresholds ───────────────────────────────────────────
LOW_RISK_THRESHOLD  = 0.25
HIGH_RISK_THRESHOLD = 0.40

# ── Advisor Signal Weights (must sum to 1.0) ──────────────────
ADVISOR_PRICE_WEIGHT      = 0.30   # AI predicted price change
ADVISOR_SENTIMENT_WEIGHT  = 0.20   # news sentiment
ADVISOR_RSI_WEIGHT        = 0.20   # RSI overbought/oversold
ADVISOR_MACD_WEIGHT       = 0.15   # MACD crossover
ADVISOR_VOLATILITY_WEIGHT = 0.15   # volatility-adjusted signal

# Score → Recommendation bands
STRONG_BUY_SCORE = 70   # score >= 70 → Strong Buy
BUY_SCORE        = 50   # score >= 50 → Buy
HOLD_SCORE       = 35   # score >= 35 → Hold  (else Sell)

# ── Sentiment ─────────────────────────────────────────────────
BULLISH_THRESHOLD  = 0.15
BEARISH_THRESHOLD  = -0.15
NEWS_PAGE_SIZE     = 15
FINBERT_MODEL      = "ProsusAI/finbert"   # HuggingFace model id

# ── Alerts ────────────────────────────────────────────────────
PRICE_MOVE_THRESHOLD     = 0.03   # 3% daily move triggers alert
VOLATILITY_ALERT_LEVEL   = 0.40

# ── Portfolio ─────────────────────────────────────────────────
RISK_FREE_RATE           = 0.07   # 7% — approximate Indian 10Y bond yield (2024)

# ── UI ────────────────────────────────────────────────────────
APP_TITLE      = "NiftyMind · AI Stock Advisor"
APP_ICON       = "🧠"
CURRENCY       = "₹"