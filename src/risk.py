import numpy as np


# -------------------------------------------------
# Calculate Daily Returns
# -------------------------------------------------

def calculate_returns(close_prices):
    """
    Calculate daily percentage returns
    """

    returns = close_prices.pct_change().dropna()

    return returns


# -------------------------------------------------
# Calculate Volatility
# -------------------------------------------------

def calculate_volatility(returns):
    """
    Annualized volatility
    """

    volatility = returns.std() * np.sqrt(252)

    return volatility


# -------------------------------------------------
# Calculate Sharpe Ratio
# -------------------------------------------------

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Sharpe Ratio calculation
    """

    excess_returns = returns - (risk_free_rate / 252)

    sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)

    return sharpe_ratio


# -------------------------------------------------
# Calculate Maximum Drawdown
# -------------------------------------------------

def calculate_max_drawdown(close_prices):
    """
    Maximum drawdown calculation
    """

    cumulative_returns = (1 + close_prices.pct_change().dropna()).cumprod()

    peak = cumulative_returns.cummax()

    drawdown = (cumulative_returns - peak) / peak

    max_drawdown = drawdown.min()

    return max_drawdown


# -------------------------------------------------
# Risk Category Classification
# -------------------------------------------------

def get_risk_category(volatility):
    """
    Categorize asset risk level
    """

    if volatility < 0.25:
        return "Low Risk"

    elif volatility < 0.40:
        return "Medium Risk"

    else:
        return "High Risk"


# -------------------------------------------------
# Main Risk Dashboard Function
# -------------------------------------------------

def calculate_risk_metrics(close_prices):
    """
    Calculate all risk metrics for dashboard
    """

    returns = calculate_returns(close_prices)

    volatility = calculate_volatility(returns)

    sharpe_ratio = calculate_sharpe_ratio(returns)

    max_drawdown = calculate_max_drawdown(close_prices)

    return volatility, sharpe_ratio, max_drawdown