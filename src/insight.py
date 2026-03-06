def generate_ai_insight(
    recommendation,
    price_change,
    sentiment_label,
    volatility
):

    insight = []

    # --------------------------------
    # Price Trend Insight
    # --------------------------------

    if price_change > 1:
        insight.append(
            "The model predicts a positive short-term price movement."
        )

    elif price_change < -1:
        insight.append(
            "The model indicates a potential short-term price decline."
        )

    else:
        insight.append(
            "The model predicts relatively stable short-term price movement."
        )

    # --------------------------------
    # Sentiment Insight
    # --------------------------------

    if sentiment_label == "Bullish":
        insight.append(
            "Recent financial news shows positive sentiment around the company."
        )

    elif sentiment_label == "Bearish":
        insight.append(
            "Recent news sentiment around the company is negative."
        )

    else:
        insight.append(
            "Market news sentiment remains neutral."
        )

    # --------------------------------
    # Volatility Insight
    # --------------------------------

    if volatility < 0.25:
        insight.append(
            "The stock currently exhibits relatively low volatility, suggesting stable price behavior."
        )

    elif volatility < 0.40:
        insight.append(
            "The stock shows moderate volatility levels."
        )

    else:
        insight.append(
            "High volatility indicates elevated risk levels for short-term trading."
        )

    # --------------------------------
    # Final Recommendation Insight
    # --------------------------------

    insight.append(
        f"Based on the combined analysis, the AI system suggests a **{recommendation}** strategy."
    )

    return insight