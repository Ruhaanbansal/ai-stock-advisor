# =============================================================
# insight.py — Narrative Investment Insight Generator
# =============================================================

from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD


def generate_ai_insight(
    recommendation:  str,
    price_change:    float,
    sentiment_label: str,
    volatility:      float,
) -> list[str]:
    """
    Return a list of plain-English insight sentences combining
    price trend, sentiment and volatility context.
    """
    insight = []

    # Price trend
    if price_change > 2:
        insight.append(
            f"The model projects a notable upside of {price_change:.2f}%, "
            "suggesting positive short-term momentum."
        )
    elif price_change > 0:
        insight.append(
            f"The model forecasts a modest gain of {price_change:.2f}%."
        )
    elif price_change > -1:
        insight.append(
            "The model expects relatively flat price movement in the near term."
        )
    else:
        insight.append(
            f"The model anticipates a decline of {abs(price_change):.2f}%. "
            "Exercise caution."
        )

    # Sentiment
    sentiment_map = {
        "Bullish":  "Positive news flow supports the bullish outlook.",
        "Bearish":  "Negative news sentiment adds downside risk.",
        "Neutral":  "News sentiment is neutral — no strong macro tailwinds or headwinds.",
    }
    insight.append(sentiment_map.get(sentiment_label, "Sentiment data unavailable."))

    # Volatility
    if volatility < LOW_RISK_THRESHOLD:
        insight.append(
            f"Annualised volatility of {volatility:.2%} is low — "
            "the stock is behaving predictably."
        )
    elif volatility < HIGH_RISK_THRESHOLD:
        insight.append(
            f"Moderate volatility ({volatility:.2%}) — "
            "suitable for medium-risk investors."
        )
    else:
        insight.append(
            f"High volatility ({volatility:.2%}) — "
            "short-term positions carry elevated risk."
        )

    # Final call
    action_map = {
        "Strong Buy": "The combined signals are strongly positive. The AI recommends **Strong Buy**.",
        "Buy":        "Overall conditions are favourable. The AI recommends **Buy**.",
        "Hold":       "Mixed signals suggest maintaining current positions. The AI recommends **Hold**.",
        "Sell":       "Risk factors outweigh upside potential. The AI recommends **Sell**.",
    }
    insight.append(action_map.get(recommendation, f"AI recommendation: **{recommendation}**."))

    return insight