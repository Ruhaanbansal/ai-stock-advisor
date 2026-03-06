import pandas as pd
import numpy as np
import yfinance as yf


# -------------------------------------------------
# Load Historical Data
# -------------------------------------------------

def load_backtest_data(stock, period="1y"):
    """
    Download historical stock data for backtesting
    """

    data = yf.download(stock, period=period, auto_adjust=True)

    if "Close" not in data.columns:
        raise ValueError("Close price not found in data")

    return data["Close"]


# -------------------------------------------------
# Calculate Returns
# -------------------------------------------------

def calculate_returns(close_prices):
    """
    Calculate daily percentage returns
    """

    returns = close_prices.pct_change().dropna()

    return returns


# -------------------------------------------------
# Buy & Hold Strategy
# -------------------------------------------------

def buy_and_hold_strategy(returns):
    """
    Simulate buy and hold strategy
    """

    cumulative_returns = (1 + returns).cumprod()

    result = pd.DataFrame(cumulative_returns)

    result.columns = ["Buy & Hold"]

    return result


# -------------------------------------------------
# Simple AI Strategy
# -------------------------------------------------

def ai_strategy_simulation(returns):
    """
    Simple AI-based trading signal
    Uses previous day return as signal
    """

    signal = returns.shift(1)

    signal = (signal > 0).astype(int)

    strategy_returns = returns * signal

    cumulative_returns = (1 + strategy_returns).cumprod()

    result = pd.DataFrame(cumulative_returns)

    result.columns = ["AI Strategy"]

    return result


# -------------------------------------------------
# Combine Strategy Results
# -------------------------------------------------

def compare_strategies(close_prices):
    """
    Compare AI strategy vs Buy & Hold
    """

    returns = calculate_returns(close_prices)

    buy_hold = buy_and_hold_strategy(returns)

    ai_strategy = ai_strategy_simulation(returns)

    comparison = pd.concat([buy_hold, ai_strategy], axis=1)

    return comparison


# -------------------------------------------------
# Strategy Performance Metrics
# -------------------------------------------------

def calculate_strategy_performance(comparison_df):
    """
    Calculate final returns for both strategies
    """

    buy_hold_final = comparison_df["Buy & Hold"].iloc[-1]

    ai_final = comparison_df["AI Strategy"].iloc[-1]

    buy_hold_return = (buy_hold_final - 1) * 100

    ai_return = (ai_final - 1) * 100

    results = {
        "buy_hold_return": buy_hold_return,
        "ai_return": ai_return
    }

    return results


# -------------------------------------------------
# Full Backtest Pipeline
# -------------------------------------------------

def run_backtest(stock):
    """
    Run full backtesting workflow
    """

    close_prices = load_backtest_data(stock)

    comparison = compare_strategies(close_prices)

    performance = calculate_strategy_performance(comparison)

    return comparison, performance