# =============================================================
# portfolio.py — Portfolio Optimization (Mean-Variance)
# =============================================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.config import RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────

def calculate_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    return price_data.pct_change().dropna()


# ─────────────────────────────────────────────────────────────
# Portfolio Statistics
# ─────────────────────────────────────────────────────────────

def portfolio_performance(
    weights:      np.ndarray,
    mean_returns: pd.Series,
    cov_matrix:   pd.DataFrame,
) -> tuple[float, float]:
    """Annualised return and volatility."""
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return ret, vol


def portfolio_sharpe(
    weights:      np.ndarray,
    mean_returns: pd.Series,
    cov_matrix:   pd.DataFrame,
    rf:           float = RISK_FREE_RATE,
) -> float:
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return (ret - rf) / vol if vol > 0 else 0.0


def _negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_sharpe(weights, mean_returns, cov_matrix)


# ─────────────────────────────────────────────────────────────
# Optimize — Maximum Sharpe
# ─────────────────────────────────────────────────────────────

def optimize_portfolio(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Find the portfolio with the highest Sharpe Ratio.
    """
    mean_ret   = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n          = len(mean_ret)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    init_guess  = np.repeat(1 / n, n)

    result = minimize(
        _negative_sharpe,
        init_guess,
        args=(mean_ret, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    ret, vol = portfolio_performance(weights, mean_ret, cov_matrix)

    return pd.DataFrame({
        "Stock":           mean_ret.index,
        "Optimal Weight":  np.round(weights, 4),
        "Exp. Return (%)": np.round(mean_ret.values * 100, 2),
        "Volatility (%)":  np.round(
            [np.sqrt(cov_matrix.iloc[i, i]) * 100 for i in range(n)], 2
        ),
    }), round(ret, 4), round(vol, 4)


# ─────────────────────────────────────────────────────────────
# Risk-based Allocation
# ─────────────────────────────────────────────────────────────

def recommend_portfolio(
    returns:           pd.DataFrame,
    investment_amount: float,
    risk_level:        str,
) -> pd.DataFrame:
    """
    Allocate capital based on selected risk appetite.

    - Low    → inverse-volatility weighted (lower vol = higher weight)
    - Medium → Sharpe-ratio optimized (max risk-adjusted return)
    - High   → momentum weighted (higher recent return = higher weight)

    Each strategy produces genuinely different weights, not equal splits.
    """
    n        = len(returns.columns)
    mean_ret = returns.mean() * 252
    vols     = returns.std() * np.sqrt(252)

    if risk_level == "Low":
        # Inverse-volatility weighting — safer stocks get more capital
        inv_vol = 1.0 / vols.replace(0, np.nan)
        inv_vol = inv_vol.fillna(0)
        total   = inv_vol.sum()
        weights = (inv_vol / total).values if total > 0 else np.repeat(1/n, n)

    elif risk_level == "Medium":
        # Sharpe-optimized allocation
        try:
            alloc_df, _, _ = optimize_portfolio(returns)
            weights = alloc_df["Optimal Weight"].values
        except Exception:
            weights = np.repeat(1 / n, n)

    else:  # High
        # Momentum weighting — stocks with higher annualised return get more weight
        # Clip negative returns to zero so we don't short anything
        pos_ret = mean_ret.clip(lower=0)
        total   = pos_ret.sum()
        if total > 0:
            weights = (pos_ret / total).values
        else:
            # Fallback: concentrate on best performer
            weights          = np.zeros(n)
            weights[mean_ret.argmax()] = 1.0

    # ── Per-stock metrics for the allocation table ─────────────
    sharpe_per_stock = []
    for col in returns.columns:
        r = returns[col]
        excess = r - (RISK_FREE_RATE / 252)
        s = (excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        sharpe_per_stock.append(round(s, 2))

    allocation = pd.DataFrame({
        "Stock":           returns.columns,
        "Weight (%)":      np.round(weights * 100, 2),
        "Amount (₹)":      np.round(weights * investment_amount, 2),
        "Exp. Return (%)": np.round(mean_ret.values * 100, 2),
        "Volatility (%)":  np.round(vols.values * 100, 2),
        "Sharpe":          sharpe_per_stock,
    })

    return allocation


# ─────────────────────────────────────────────────────────────
# Efficient Frontier (Monte Carlo)
# ─────────────────────────────────────────────────────────────

def generate_efficient_frontier(
    returns:       pd.DataFrame,
    num_portfolios: int = 2000,
) -> pd.DataFrame:
    """
    Simulate random portfolios to draw the efficient frontier.
    """
    mean_ret   = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n          = len(mean_ret)
    records    = []

    for _ in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v = portfolio_performance(w, mean_ret, cov_matrix)
        records.append({
            "Return (%)":    round(r * 100, 3),
            "Volatility (%)": round(v * 100, 3),
            "Sharpe":         round((r - RISK_FREE_RATE) / v if v > 0 else 0, 3),
        })

    return pd.DataFrame(records)