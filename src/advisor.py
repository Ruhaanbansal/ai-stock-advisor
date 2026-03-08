# =============================================================
# advisor.py — AI Investment Advisory Engine
# =============================================================

from src.config import (
    STRONG_BUY_CHANGE, STRONG_BUY_SENTIMENT,
    BUY_CHANGE, SELL_CHANGE, SELL_SENTIMENT,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _price_change_pct(current: float, predicted: float) -> float:
    """Expected price change as a percentage."""
    return ((predicted - current) / current) * 100


def _risk_label(volatility: float) -> str:
    if volatility < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif volatility < HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


# ─────────────────────────────────────────────────────────────
# Recommendation Logic
# ─────────────────────────────────────────────────────────────

def generate_recommendation(
    current_price:   float,
    predicted_price: float,
    volatility:      float,
    sentiment_score: float,
) -> tuple[str, float, str]:
    """
    Returns (recommendation, price_change_pct, risk_label).
    """
    price_change = _price_change_pct(current_price, predicted_price)
    risk         = _risk_label(volatility)

    if (
        price_change > STRONG_BUY_CHANGE
        and sentiment_score > STRONG_BUY_SENTIMENT
        and risk == "Low Risk"
    ):
        recommendation = "Strong Buy"

    elif price_change > STRONG_BUY_CHANGE and sentiment_score > STRONG_BUY_SENTIMENT:
        recommendation = "Buy"

    elif price_change > BUY_CHANGE and sentiment_score >= 0:
        recommendation = "Buy"

    elif sentiment_score < SELL_SENTIMENT:
        recommendation = "Sell"

    elif price_change < SELL_CHANGE:
        recommendation = "Sell"

    else:
        recommendation = "Hold"

    return recommendation, price_change, risk


# ─────────────────────────────────────────────────────────────
# Confidence Score
# ─────────────────────────────────────────────────────────────

def calculate_confidence(current_price: float, predicted_price: float) -> float:
    """
    Confidence = 100 minus the predicted deviation from current price.
    Clamped to [0, 100].
    """
    deviation = abs(_price_change_pct(current_price, predicted_price))
    return max(0.0, min(100.0, 100.0 - deviation))


# ─────────────────────────────────────────────────────────────
# Human-readable Reasoning
# ─────────────────────────────────────────────────────────────

def generate_ai_reasoning(
    price_change:    float,
    sentiment_score: float,
    volatility:      float,
) -> list[str]:
    reasons = []

    if price_change > 0:
        reasons.append(f"AI model projects a +{price_change:.2f}% price move.")
    else:
        reasons.append(f"AI model projects a {price_change:.2f}% price decline.")

    if sentiment_score > 0.2:
        reasons.append("Financial news sentiment is positive (Bullish).")
    elif sentiment_score < -0.2:
        reasons.append("Financial news sentiment is negative (Bearish).")
    else:
        reasons.append("Market news sentiment is neutral.")

    if volatility < LOW_RISK_THRESHOLD:
        reasons.append("Low annualised volatility — stable price behaviour expected.")
    elif volatility < HIGH_RISK_THRESHOLD:
        reasons.append("Moderate volatility — watch for short-term swings.")
    else:
        reasons.append("High volatility — elevated short-term risk.")

    return reasons


# ─────────────────────────────────────────────────────────────
# Full Advisor Pipeline
# ─────────────────────────────────────────────────────────────

def run_ai_advisor(
    current_price:   float,
    predicted_price: float,
    volatility:      float,
    sentiment_score: float,
) -> dict:
    """
    Single entry point for the AI advisor.
    Returns a dict suitable for direct use by the Streamlit app.
    """
    recommendation, price_change, risk = generate_recommendation(
        current_price, predicted_price, volatility, sentiment_score
    )
    confidence = calculate_confidence(current_price, predicted_price)
    reasons    = generate_ai_reasoning(price_change, sentiment_score, volatility)

    return {
        "recommendation": recommendation,
        "confidence":     round(confidence, 2),
        "risk":           risk,
        "price_change":   round(price_change, 4),
        "reasons":        reasons,
    }