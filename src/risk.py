# =============================================================
# risk.py — Comprehensive Risk Metrics
# =============================================================

import numpy as np
import pandas as pd

from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────

def calculate_returns(close_prices: pd.Series) -> pd.Series:
    return close_prices.pct_change().dropna()


# ─────────────────────────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────────────────────────

def calculate_volatility(returns: pd.Series) -> float:
    """Annualised historical volatility."""
    return float(returns.std() * np.sqrt(252))


# ─────────────────────────────────────────────────────────────
# Sharpe Ratio
# ─────────────────────────────────────────────────────────────

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> float:
    """
    Annualised Sharpe Ratio.
    Uses Indian 10Y government bond yield as the default risk-free rate.
    """
    daily_rf = risk_free_rate / 252
    excess   = returns - daily_rf
    std      = returns.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


# ─────────────────────────────────────────────────────────────
# Maximum Drawdown
# ─────────────────────────────────────────────────────────────

def calculate_max_drawdown(close_prices: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown expressed as a negative fraction.
    """
    cum_ret = (1 + close_prices.pct_change().dropna()).cumprod()
    peak    = cum_ret.cummax()
    dd      = (cum_ret - peak) / peak
    return float(dd.min())


# ─────────────────────────────────────────────────────────────
# Value at Risk & Conditional VaR (Expected Shortfall)
# ─────────────────────────────────────────────────────────────

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.
    Returns a positive number representing the potential loss.
    """
    return float(-np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — average loss beyond VaR.
    """
    var      = calculate_var(returns, confidence)
    tail     = returns[returns <= -var]
    if tail.empty:
        return var
    return float(-tail.mean())


# ─────────────────────────────────────────────────────────────
# Sortino Ratio
# ─────────────────────────────────────────────────────────────

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> float:
    """
    Sortino Ratio — like Sharpe but penalises only downside volatility.
    """
    daily_rf      = risk_free_rate / 252
    excess        = returns - daily_rf
    downside_std  = returns[returns < 0].std()
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(252))


# ─────────────────────────────────────────────────────────────
# Risk Category
# ─────────────────────────────────────────────────────────────

def get_risk_category(volatility: float) -> str:
    if volatility < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif volatility < HIGH_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


# ─────────────────────────────────────────────────────────────
# All Risk Metrics
# ─────────────────────────────────────────────────────────────

def calculate_risk_metrics(close_prices: pd.Series) -> dict:
    """
    Compute the full risk dashboard in one call.
    Returns a dict of named metrics.
    """
    returns = calculate_returns(close_prices)

    volatility    = calculate_volatility(returns)
    sharpe        = calculate_sharpe_ratio(returns)
    sortino       = calculate_sortino_ratio(returns)
    max_dd        = calculate_max_drawdown(close_prices)
    var_95        = calculate_var(returns, 0.95)
    cvar_95       = calculate_cvar(returns, 0.95)
    risk_category = get_risk_category(volatility)

    return {
        "volatility":    volatility,
        "sharpe_ratio":  sharpe,
        "sortino_ratio": sortino,
        "max_drawdown":  max_dd,
        "var_95":        var_95,
        "cvar_95":       cvar_95,
        "risk_category": risk_category,
    }