# =============================================================
# insight.py — Narrative Investment Insight Generator
# =============================================================

import pandas as pd
import numpy as np
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
        "Strong Buy": "The combined signals are strongly positive. The AI recommends <b>Strong Buy</b>.",
        "Buy":        "Overall conditions are favourable. The AI recommends <b>Buy</b>.",
        "Hold":       "Mixed signals suggest maintaining current positions. The AI recommends <b>Hold</b>.",
        "Sell":       "Risk factors outweigh upside potential. The AI recommends <b>Sell</b>.",
    }
    insight.append(action_map.get(recommendation, f"AI recommendation: <b>{recommendation}</b>."))

    return insight


def generate_chart_insights(data, extended) -> list[str]:
    """
    Analyzes technical indicators to generate short, actionable chart observations.
    Used for the 'AI Insight Snackbars' under charts.
    """
    if data is None or data.empty or extended is None:
        return ["Insufficient data for chart analysis."]

    insights = []
    curr_price = float(data['Close'].iloc[-1])
    
    # 1. Trend Analysis (Price vs SMAs)
    if 'SMA20' in extended and 'SMA50' in extended:
        sma20 = float(extended['SMA20'].iloc[-1])
        sma50 = float(extended['SMA50'].iloc[-1])
        if curr_price > sma20 > sma50:
            insights.append("🚀 <b>Strong Uptrend</b>: Price is holding above both 20 & 50-day MAs.")
        elif curr_price < sma20 < sma50:
            insights.append("📉 <b>Bearish Trend</b>: Price is currently below short & medium term MAs.")

    # 2. Momentum (RSI)
    if 'RSI' in extended:
        rsi = float(extended['RSI'].iloc[-1])
        if rsi > 70:
            insights.append(f"⚠️ <b>Overbought</b>: RSI ({rsi:.1f}) suggests potential profit booking soon.")
        elif rsi < 30:
            insights.append(f"⚡ <b>Oversold</b>: RSI ({rsi:.1f}) indicates high recovery potential.")

    # 3. Volume Analysis
    if 'Volume' in data.columns:
        avg_vol = data['Volume'].tail(20).mean()
        curr_vol = data['Volume'].iloc[-1]
        if curr_vol > avg_vol * 1.5:
            insights.append(f"📊 <b>Volume Spike</b>: Significant institutional activity detected.")

    # 4. Bollinger Band Context
    if 'BB_Upper' in extended and 'BB_Lower' in extended:
        upper = float(extended['BB_Upper'].iloc[-1])
        if curr_price > upper:
            insights.append("🔥 <b>Price Breakout</b>: Trading above upper Bollinger Band — high momentum.")

    return insights[:3] # Return top 3 most relevant insights