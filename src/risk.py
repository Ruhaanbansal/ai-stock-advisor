# =============================================================
# risk.py — Risk Metrics Suite
# =============================================================
#
# Metrics implemented:
#   Volatility, Sharpe, Sortino (fixed MAR-based), Max Drawdown,
#   VaR (95%/99%), CVaR, Beta vs NIFTY, Treynor Ratio, Calmar Ratio
# =============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from src.config import RISK_FREE_RATE


_TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────
# Fetch NIFTY returns for Beta calculation
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nifty_returns(period: str = "1y") -> pd.Series | None:
    """Fetch NIFTY 50 returns for Beta calculation."""
    try:
        df = yf.download("^NSEI", period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # Ensure timezone-naive to avoid concatenation errors
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        close = df["Close"].squeeze()
        return close.pct_change().dropna()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────

def calculate_volatility(returns: pd.Series) -> float:
    """Annualised volatility."""
    return float(returns.std() * np.sqrt(_TRADING_DAYS))


def calculate_sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """
    Annualised Sharpe Ratio.
    rf is annual risk-free rate (default: 7% Indian 10Y bond yield).
    """
    daily_rf   = rf / _TRADING_DAYS
    excess     = returns - daily_rf
    if returns.std() < 1e-10:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(_TRADING_DAYS))


def calculate_sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """
    Annualised Sortino Ratio using proper downside deviation.
    MAR = daily risk-free rate.
    Downside deviation = std of returns below MAR (not just negative returns).
    """
    daily_rf       = rf / _TRADING_DAYS
    excess         = returns - daily_rf
    downside_rets  = excess[excess < 0]
    if len(downside_rets) == 0:
        return float("inf")
    downside_dev = float(np.sqrt((downside_rets ** 2).mean()) * np.sqrt(_TRADING_DAYS))
    if downside_dev < 1e-10:
        return 0.0
    ann_excess_return = float(excess.mean() * _TRADING_DAYS)
    return ann_excess_return / downside_dev


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a positive percentage."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.
    Returns the loss at the given confidence level as a positive %.
    """
    return float(np.percentile(returns, (1 - confidence) * 100) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).
    Average of returns below VaR threshold.
    """
    var  = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var]
    if tail.empty:
        return float(var * 100)
    return float(tail.mean() * 100)


def calculate_beta(stock_returns: pd.Series, nifty_returns: pd.Series) -> float:
    """
    Beta vs NIFTY 50.
    Beta > 1: more volatile than market.
    Beta < 1: less volatile.
    """
    # Ensure both are timezone-naive to avoid 'Cannot join tz-naive with tz-aware' error
    if stock_returns.index.tz is not None:
        stock_returns.index = stock_returns.index.tz_localize(None)
    if nifty_returns.index.tz is not None:
        nifty_returns.index = nifty_returns.index.tz_localize(None)

    # Align on common dates
    aligned = pd.concat([stock_returns, nifty_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0  # default to market beta if insufficient data
    s_ret = aligned.iloc[:, 0]
    m_ret = aligned.iloc[:, 1]
    cov   = s_ret.cov(m_ret)
    var   = m_ret.var()
    if var < 1e-10:
        return 1.0
    return float(cov / var)


def calculate_treynor_ratio(returns: pd.Series, beta: float, rf: float = RISK_FREE_RATE) -> float:
    """
    Treynor Ratio = (Portfolio Return - Rf) / Beta.
    Measures excess return per unit of market risk.
    """
    if abs(beta) < 1e-6:
        return 0.0
    ann_return   = float((1 + returns.mean()) ** _TRADING_DAYS - 1)
    excess_return = ann_return - rf
    return excess_return / beta


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar Ratio = Annualised Return / |Max Drawdown|.
    Higher is better. < 0 = consistently losing.
    """
    ann_return   = float((1 + returns.mean()) ** _TRADING_DAYS - 1)
    max_dd       = abs(calculate_max_drawdown(returns) / 100)
    if max_dd < 1e-6:
        return 0.0
    return ann_return / max_dd


# ─────────────────────────────────────────────────────────────
# Full Risk Report
# ─────────────────────────────────────────────────────────────

def calculate_risk_metrics(
    returns:      pd.Series,
    stock_ticker: str = "",
    period:       str = "1y",
) -> dict:
    """
    Compute the full suite of risk metrics.
    Returns a flat dict ready for display.
    """
    returns = returns.dropna()
    if len(returns) < 10:
        return {}

    volatility   = calculate_volatility(returns)
    sharpe       = calculate_sharpe_ratio(returns)
    sortino      = calculate_sortino_ratio(returns)
    max_dd       = calculate_max_drawdown(returns)
    var_95       = calculate_var(returns, 0.95)
    var_99       = calculate_var(returns, 0.99)
    cvar_95      = calculate_cvar(returns, 0.95)
    calmar       = calculate_calmar_ratio(returns)

    # Beta — fetch NIFTY and align
    beta    = 1.0
    treynor = 0.0
    nifty_rets = _fetch_nifty_returns(period)
    if nifty_rets is not None:
        beta    = calculate_beta(returns, nifty_rets)
        treynor = calculate_treynor_ratio(returns, beta)

    return {
        "volatility":  round(volatility, 4),
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3),
        "max_drawdown": round(max_dd, 2),
        "var_95":      round(var_95, 2),
        "var_99":      round(var_99, 2),
        "cvar_95":     round(cvar_95, 2),
        "beta":        round(beta, 3),
        "treynor":     round(treynor, 4),
        "calmar":      round(calmar, 3),
    }