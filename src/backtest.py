# =============================================================
# backtest.py — Strategy Backtesting Engine
# =============================================================

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

from src.config import BACKTEST_PERIOD, RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_data(stock: str, period: str = BACKTEST_PERIOD) -> pd.Series:
    data = yf.download(stock, period=period, auto_adjust=True, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No Close price data found for {stock}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return close.dropna()


# ─────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────

def buy_and_hold(returns: pd.Series) -> pd.Series:
    """Always invested — baseline."""
    return (1 + returns).cumprod().rename("Buy & Hold")


def momentum_strategy(returns: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Enter when the previous `lookback`-day return is positive.
    A simple but common momentum signal.
    """
    signal  = returns.rolling(lookback).mean().shift(1) > 0
    strat_r = returns * signal.astype(int)
    return (1 + strat_r).cumprod().rename("Momentum")


def mean_reversion_strategy(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Fade recent moves: enter when yesterday's return is negative
    (i.e. bet on reversal).
    """
    signal  = returns.shift(1) < 0
    strat_r = returns * signal.astype(int)
    return (1 + strat_r).cumprod().rename("Mean Reversion")


def ai_strategy(returns: pd.Series) -> pd.Series:
    """
    Original 1-day lag positive-momentum strategy kept for comparison.
    """
    signal  = (returns.shift(1) > 0).astype(int)
    strat_r = returns * signal
    return (1 + strat_r).cumprod().rename("AI Strategy")


# ─────────────────────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────────────────────

def _strategy_metrics(cum_series: pd.Series, returns: pd.Series) -> dict:
    final_return = (cum_series.iloc[-1] - 1) * 100
    peak         = cum_series.cummax()
    dd           = (cum_series - peak) / peak
    max_dd       = dd.min() * 100

    daily_rf     = RISK_FREE_RATE / 252
    excess       = returns - daily_rf
    sharpe       = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0

    return {
        "total_return": round(final_return, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  round(sharpe, 3),
    }


# ─────────────────────────────────────────────────────────────
# Full Comparison
# ─────────────────────────────────────────────────────────────

def compare_strategies(close_prices: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Run all strategies and return (cumulative_df, metrics_dict).
    """
    returns = close_prices.pct_change().dropna()

    bh  = buy_and_hold(returns)
    ai  = ai_strategy(returns)
    mom = momentum_strategy(returns)
    mr  = mean_reversion_strategy(returns)

    comparison = pd.concat([bh, ai, mom, mr], axis=1).dropna()

    # Per-strategy returns for metric calculation
    strat_returns = {
        "Buy & Hold":     returns,
        "AI Strategy":    returns * (returns.shift(1) > 0).astype(int),
        "Momentum":       returns * (returns.rolling(5).mean().shift(1) > 0).astype(int),
        "Mean Reversion": returns * (returns.shift(1) < 0).astype(int),
    }

    metrics = {
        name: _strategy_metrics(comparison[name], strat_returns[name])
        for name in comparison.columns
    }

    return comparison, metrics


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def run_backtest(stock: str) -> tuple[pd.DataFrame, dict]:
    close_prices = load_backtest_data(stock)
    return compare_strategies(close_prices)