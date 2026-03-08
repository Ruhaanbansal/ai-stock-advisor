# =============================================================
# alerts.py — Market Alert Detection
# =============================================================

import numpy as np
import pandas as pd

from src.config import PRICE_MOVE_THRESHOLD, VOLATILITY_ALERT_LEVEL, BEARISH_THRESHOLD


def detect_market_alerts(close_prices: pd.Series, sentiment_score: float) -> list[dict]:
    """
    Return a list of alert dicts with 'level' and 'message' keys.
    Levels: 'warning' | 'danger' | 'info'
    """
    alerts  = []
    returns = close_prices.pct_change().dropna()

    if returns.empty:
        return alerts

    # ── Unusual single-day move ───────────────────────────────
    latest_move = returns.iloc[-1]
    if abs(latest_move) > PRICE_MOVE_THRESHOLD:
        direction = "surge" if latest_move > 0 else "drop"
        alerts.append({
            "level":   "warning",
            "message": f"Unusual price {direction} detected today ({latest_move:.2%})",
        })

    # ── High volatility ───────────────────────────────────────
    vol = returns.std() * np.sqrt(252)
    if vol > VOLATILITY_ALERT_LEVEL:
        alerts.append({
            "level":   "danger",
            "message": f"Elevated annualised volatility: {vol:.2%}",
        })

    # ── Negative sentiment ────────────────────────────────────
    if sentiment_score < BEARISH_THRESHOLD:
        alerts.append({
            "level":   "warning",
            "message": "Strong negative news sentiment detected",
        })

    # ── RSI Overbought / Oversold (simple) ────────────────────
    if len(close_prices) >= 14:
        delta    = close_prices.diff()
        gain     = delta.clip(lower=0).ewm(com=13).mean()
        loss     = (-delta.clip(upper=0)).ewm(com=13).mean()
        rsi      = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        last_rsi = rsi.iloc[-1]

        if last_rsi > 70:
            alerts.append({
                "level":   "info",
                "message": f"RSI at {last_rsi:.1f} — stock may be overbought",
            })
        elif last_rsi < 30:
            alerts.append({
                "level":   "info",
                "message": f"RSI at {last_rsi:.1f} — stock may be oversold (potential rebound)",
            })

    # ── Price vs 50-day SMA ───────────────────────────────────
    if len(close_prices) >= 50:
        sma50       = close_prices.rolling(50).mean().iloc[-1]
        current     = close_prices.iloc[-1]
        pct_vs_sma  = (current - sma50) / sma50

        if pct_vs_sma < -0.05:
            alerts.append({
                "level":   "warning",
                "message": f"Price is {abs(pct_vs_sma):.1%} below 50-day SMA",
            })

    return alerts