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

# ── LSTM Model ────────────────────────────────────────────────
SEQUENCE_LENGTH  = 60
LSTM_UNITS_1     = 64
LSTM_UNITS_2     = 32
DROPOUT_RATE     = 0.2
EPOCHS           = 3
BATCH_SIZE       = 16
FEATURE_NAMES    = ["Close", "SMA20", "SMA50", "RSI"]

# ── Risk Thresholds ───────────────────────────────────────────
LOW_RISK_THRESHOLD  = 0.25
HIGH_RISK_THRESHOLD = 0.40

# ── Advisor Signal Thresholds ─────────────────────────────────
STRONG_BUY_CHANGE     = 2.0      # % predicted change
STRONG_BUY_SENTIMENT  = 0.2
BUY_CHANGE            = 0.0
SELL_CHANGE           = -1.0
SELL_SENTIMENT        = -0.3

# ── Sentiment ─────────────────────────────────────────────────
BULLISH_THRESHOLD = 0.2
BEARISH_THRESHOLD = -0.2
NEWS_PAGE_SIZE    = 10

# ── Alerts ────────────────────────────────────────────────────
PRICE_MOVE_THRESHOLD     = 0.03   # 3% daily move triggers alert
VOLATILITY_ALERT_LEVEL   = 0.40

# ── Portfolio ─────────────────────────────────────────────────
RISK_FREE_RATE           = 0.06   # 6% — approximate Indian 10Y bond yield

# ── UI ────────────────────────────────────────────────────────
APP_TITLE      = "NiftyMind · AI Stock Advisor"
APP_ICON       = "🧠"
CURRENCY       = "₹"