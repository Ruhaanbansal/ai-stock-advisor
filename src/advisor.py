import numpy as np


# -------------------------------------------------
# Calculate Price Change %
# -------------------------------------------------

def calculate_price_change(current_price, predicted_price):
    """
    Calculate expected price change percentage
    """

    change_percent = ((predicted_price - current_price) / current_price) * 100

    return change_percent


# -------------------------------------------------
# Generate Recommendation
# -------------------------------------------------

def generate_recommendation(
    current_price,
    predicted_price,
    volatility,
    sentiment_score
):

    price_change = ((predicted_price - current_price) / current_price) * 100

    # Risk classification
    if volatility < 0.25:
        risk = "Low Risk"
    elif volatility < 0.40:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # ---------------------------------
    # Sentiment-adjusted decision logic
    # ---------------------------------

    if price_change > 2 and sentiment_score > 0.2 and risk == "Low Risk":
        recommendation = "Strong Buy"

    elif price_change > 0 and sentiment_score >= 0:
        recommendation = "Buy"

    elif sentiment_score < -0.3:
        recommendation = "Sell"

    elif price_change < -1:
        recommendation = "Sell"

    else:
        recommendation = "Hold"

    return recommendation, price_change, risk

# -------------------------------------------------
# Confidence Score
# -------------------------------------------------

def calculate_confidence(current_price, predicted_price):
    """
    Estimate confidence level of prediction
    """

    confidence = max(
        0,
        100 - (abs(predicted_price - current_price) / current_price) * 100
    )

    return min(confidence, 100)


# -------------------------------------------------
# Generate AI Explanation
# -------------------------------------------------

def generate_ai_reasoning(price_change, sentiment_score, volatility):

    reasons = []

    # price movement
    if price_change > 0:
        reasons.append("AI predicts upward price movement")
    else:
        reasons.append("AI predicts potential price decline")

    # sentiment impact
    if sentiment_score > 0.2:
        reasons.append("Positive financial news sentiment detected")

    elif sentiment_score < -0.2:
        reasons.append("Negative market news detected")

    else:
        reasons.append("Neutral market sentiment")

    # volatility
    if volatility < 0.25:
        reasons.append("Low volatility indicates stable stock")

    elif volatility < 0.40:
        reasons.append("Moderate volatility")

    else:
        reasons.append("High volatility risk")

    return reasons

# -------------------------------------------------
# Full AI Advisor Pipeline
# -------------------------------------------------

def run_ai_advisor(
    current_price,
    predicted_price,
    volatility,
    sentiment_score
):
    """
    Complete AI investment advisory output
    """

    recommendation, price_change, risk = generate_recommendation(
        current_price,
        predicted_price,
        volatility,
        sentiment_score
    )

    confidence = calculate_confidence(
        current_price,
        predicted_price
    )

    reasons = generate_ai_reasoning(
        price_change,
        sentiment_score,
        volatility
    )

    result = {
        "recommendation": recommendation,
        "confidence": confidence,
        "risk": risk,
        "price_change": price_change,
        "reasons": reasons
    }

    return result