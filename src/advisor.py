# =============================================================
# advisor.py — AI Investment Advisory Engine
# =============================================================
#
# Multi-factor weighted scoring system (0-100 score):
#   Price change signal   : 30%
#   Sentiment signal      : 20%
#   RSI signal            : 20%
#   MACD crossover signal : 15%
#   Volatility signal     : 15%
#
# Score bands:
#   >= 70  → Strong Buy
#   >= 50  → Buy
#   >= 35  → Hold
#   <  35  → Sell
# =============================================================

import numpy as np

from src.config import (
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
    ADVISOR_PRICE_WEIGHT, ADVISOR_SENTIMENT_WEIGHT,
    ADVISOR_RSI_WEIGHT, ADVISOR_MACD_WEIGHT, ADVISOR_VOLATILITY_WEIGHT,
    STRONG_BUY_SCORE, BUY_SCORE, HOLD_SCORE,
)


# ─────────────────────────────────────────────────────────────
# Individual Signal Scorers (each returns 0–100)
# ─────────────────────────────────────────────────────────────

def _price_signal(current: float, predicted: float) -> tuple[float, float]:
    """
    Score based on predicted % price change.
    Returns (sub_score 0-100, price_change_pct).
    """
    if current <= 0:
        return 50.0, 0.0
    pct = ((predicted - current) / current) * 100
    # Map: -5% → 0, 0% → 50, +5% → 100 (clipped)
    score = np.clip(50 + pct * 10, 0, 100)
    return float(score), float(pct)


def _sentiment_signal(sentiment_score: float) -> float:
    """
    Score based on sentiment compound score (-1 to +1).
    Returns sub_score 0-100.
    """
    # Map: -1 → 0, 0 → 50, +1 → 100
    return float(np.clip(50 + sentiment_score * 50, 0, 100))


def _rsi_signal(rsi: float | None) -> float:
    """
    Score based on RSI (0-100 indicator).
    RSI < 30 = oversold = bullish opportunity → high score
    RSI > 70 = overbought = danger → low score
    30-70 = neutral band → proportional
    """
    if rsi is None:
        return 50.0
    if rsi < 30:
        # Oversold — strong buy signal, score 70-100
        return float(np.clip(100 - rsi, 70, 100))
    elif rsi > 70:
        # Overbought — sell signal, score 0-30
        return float(np.clip(100 - rsi, 0, 30))
    else:
        # Neutral band — linear mapping 30-70 → 45-55
        return float(45 + (rsi - 30) / 40 * 10)


def _macd_signal(macd: float | None, macd_signal: float | None) -> float:
    """
    Score based on MACD crossover.
    MACD > Signal (bullish crossover) → high score
    MACD < Signal (bearish crossover) → low score
    """
    if macd is None or macd_signal is None:
        return 50.0
    diff = macd - macd_signal
    # Map: strong crossover (+ve) → 100, strong bearish → 0
    score = np.clip(50 + diff / (abs(diff) + 1e-8) * 30, 20, 80)
    return float(score)


def _volatility_signal(volatility: float) -> float:
    """
    Score based on volatility (annualised).
    Low volatility = more predictable = higher score.
    High volatility = risky = lower score.
    """
    if volatility < LOW_RISK_THRESHOLD:
        return 70.0
    elif volatility < HIGH_RISK_THRESHOLD:
        return 50.0
    else:
        return 30.0


def _risk_label(volatility: float) -> str:
    if volatility < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif volatility < HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


# ─────────────────────────────────────────────────────────────
# Composite Score → Recommendation
# ─────────────────────────────────────────────────────────────

def _score_to_recommendation(score: float) -> str:
    if score >= STRONG_BUY_SCORE:
        return "Strong Buy"
    elif score >= BUY_SCORE:
        return "Buy"
    elif score >= HOLD_SCORE:
        return "Hold"
    return "Sell"


# ─────────────────────────────────────────────────────────────
# Confidence Score
# ─────────────────────────────────────────────────────────────

def calculate_confidence(
    current_price: float,
    predicted_price: float,
    test_mae: float | None = None,
) -> float:
    """
    Confidence based on model's actual test MAE relative to current price.
    Falls back to score-spread method if test_mae not available.
    - High confidence = low relative MAE = model is precise
    """
    if test_mae is not None and current_price > 0 and test_mae > 0:
        relative_error = (test_mae / current_price) * 100
        # 0% error → 100% confidence; 10% error → 0% confidence
        return float(np.clip(100 - relative_error * 10, 0, 100))
    else:
        # Fallback: smaller price deviation = higher confidence
        if current_price > 0:
            deviation = abs((predicted_price - current_price) / current_price) * 100
            return float(np.clip(100 - deviation * 5, 40, 95))
        return 60.0


# ─────────────────────────────────────────────────────────────
# Human-readable Reasoning
# ─────────────────────────────────────────────────────────────

def generate_ai_reasoning(
    price_change:    float,
    sentiment_score: float,
    volatility:      float,
    rsi:             float | None = None,
    macd_diff:       float | None = None,
    composite_score: float = 50.0,
) -> list[str]:
    reasons = []

    # Price signal
    if price_change > 2:
        reasons.append(f"📈 AI model projects a strong upside of <b>{price_change:+.2f}%</b> — bullish momentum signal.")
    elif price_change > 0:
        reasons.append(f"📈 AI model forecasts a modest gain of <b>{price_change:+.2f}%</b>.")
    elif price_change > -2:
        reasons.append(f"📉 AI model expects a slight decline of <b>{price_change:.2f}%</b> — watch for reversal.")
    else:
        reasons.append(f"📉 AI model projects a significant drop of <b>{price_change:.2f}%</b> — exercise caution.")

    # Sentiment signal
    if sentiment_score > 0.2:
        reasons.append("📰 News sentiment is <b>Bullish</b> — positive news flow supports the outlook.")
    elif sentiment_score < -0.2:
        reasons.append("📰 News sentiment is <b>Bearish</b> — negative headlines add downside risk.")
    else:
        reasons.append("📰 News sentiment is <b>Neutral</b> — no strong macro tailwinds or headwinds.")

    # RSI signal
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"⚡ RSI at <b>{rsi:.1f}</b> — stock is in oversold territory, potential rebound signal.")
        elif rsi > 70:
            reasons.append(f"⚡ RSI at <b>{rsi:.1f}</b> — stock is overbought, elevated reversal risk.")
        else:
            reasons.append(f"⚡ RSI at <b>{rsi:.1f}</b> — neutral momentum, no extremes detected.")

    # MACD signal
    if macd_diff is not None:
        if macd_diff > 0:
            reasons.append("📊 MACD is <b>above</b> signal line — bullish crossover confirms upward momentum.")
        else:
            reasons.append("📊 MACD is <b>below</b> signal line — bearish crossover, momentum is weakening.")

    # Volatility / Risk
    if volatility < LOW_RISK_THRESHOLD:
        reasons.append(f"🛡️ Annualised volatility {volatility:.1%} is <b>low</b> — price is behaving predictably.")
    elif volatility < HIGH_RISK_THRESHOLD:
        reasons.append(f"⚠️ Moderate volatility {volatility:.1%} — suitable for medium-risk investors.")
    else:
        reasons.append(f"🚨 High volatility {volatility:.1%} — short-term positions carry elevated risk.")

    return reasons


# ─────────────────────────────────────────────────────────────
# Full Advisor Pipeline
# ─────────────────────────────────────────────────────────────

def run_ai_advisor(
    current_price:   float,
    predicted_price: float,
    volatility:      float,
    sentiment_score: float,
    rsi:             float | None = None,
    macd:            float | None = None,
    macd_signal_val: float | None = None,
    test_mae:        float | None = None,
) -> dict:
    """
    Multi-factor weighted scoring advisor.
    Returns a dict suitable for direct use by the Streamlit app.
    """
    # ── Individual sub-scores ────────────────────────────────
    price_sub, price_change = _price_signal(current_price, predicted_price)
    sentiment_sub           = _sentiment_signal(sentiment_score)
    rsi_sub                 = _rsi_signal(rsi)
    macd_sub                = _macd_signal(macd, macd_signal_val)
    vol_sub                 = _volatility_signal(volatility)

    # ── Weighted composite score ──────────────────────────────
    composite = (
        price_sub     * ADVISOR_PRICE_WEIGHT     +
        sentiment_sub * ADVISOR_SENTIMENT_WEIGHT +
        rsi_sub       * ADVISOR_RSI_WEIGHT       +
        macd_sub      * ADVISOR_MACD_WEIGHT      +
        vol_sub       * ADVISOR_VOLATILITY_WEIGHT
    )
    composite = float(np.clip(composite, 0, 100))

    # ── Recommendation & risk ─────────────────────────────────
    recommendation = _score_to_recommendation(composite)
    risk           = _risk_label(volatility)
    confidence     = calculate_confidence(current_price, predicted_price, test_mae)

    # ── Reasoning ─────────────────────────────────────────────
    macd_diff = (macd - macd_signal_val) if (macd is not None and macd_signal_val is not None) else None
    reasons   = generate_ai_reasoning(
        price_change, sentiment_score, volatility,
        rsi, macd_diff, composite
    )

    # ── Sub-score breakdown for display ──────────────────────
    factor_scores = {
        "Price Signal":    round(price_sub, 1),
        "Sentiment":       round(sentiment_sub, 1),
        "RSI":             round(rsi_sub, 1),
        "MACD":            round(macd_sub, 1),
        "Volatility/Risk": round(vol_sub, 1),
    }

    return {
        "recommendation":  recommendation,
        "composite_score": round(composite, 1),
        "confidence":      round(confidence, 2),
        "risk":            risk,
        "price_change":    round(price_change, 4),
        "reasons":         reasons,
        "factor_scores":   factor_scores,
    }