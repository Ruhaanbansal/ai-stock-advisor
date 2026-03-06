import numpy as np


def detect_market_alerts(close_prices, sentiment_score):

    alerts = []

    # --------------------------------
    # Price Movement Alert
    # --------------------------------

    returns = close_prices.pct_change().dropna()

    if len(returns) > 0:

        latest_move = returns.iloc[-1]

        if abs(latest_move) > 0.03:
            alerts.append("⚠️ Unusual price movement detected")

    # --------------------------------
    # Volatility Alert
    # --------------------------------

    volatility = returns.std() * np.sqrt(252)

    if volatility > 0.40:
        alerts.append("⚠️ High market volatility")

    # --------------------------------
    # Negative News Alert
    # --------------------------------

    if sentiment_score < -0.3:
        alerts.append("⚠️ Strong negative news sentiment")

    return alerts