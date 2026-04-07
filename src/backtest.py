# =============================================================
# backtest.py — Strategy Backtesting Engine
# =============================================================
#
# Strategies:
#   1. Buy & Hold          — passive benchmark
#   2. AI Strategy         — uses trained GradientBoosting model (REAL ML)
#   3. Momentum            — buy when 20d return > 0
#   4. Mean Reversion      — buy when price > 5% below 20d MA
#
# All strategies include:
#   - 0.1% per-trade transaction cost (realistic for Indian markets)
#   - Monthly return heatmap data
#   - Full drawdown series
# =============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_sources import fetch_stock_data
from src.config import RISK_FREE_RATE, SEQUENCE_LENGTH

_COST = 0.001          # 0.1% round-trip transaction cost
_TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────
# Data Loading (uses resilient waterfall)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_backtest_data(stock: str) -> pd.Series | None:
    """Load close prices using the resilient waterfall data_sources."""
    data, _ = fetch_stock_data(stock, "2y")
    if data is None or data.empty:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return pd.to_numeric(close, errors="coerce").dropna()


# ─────────────────────────────────────────────────────────────
# Shared performance metrics
# ─────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series) -> float:
    daily_rf = RISK_FREE_RATE / _TRADING_DAYS
    excess   = returns - daily_rf
    if excess.std() < 1e-10:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(_TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return float(dd.min() * 100)


def _win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns) * 100)


def _compute_metrics(equity: pd.Series, name: str) -> dict:
    daily_rets   = equity.pct_change().dropna()
    total_return = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    return {
        "strategy":     name,
        "total_return": round(total_return, 2),
        "max_drawdown": round(_max_drawdown(equity), 2),
        "sharpe_ratio": round(_sharpe(daily_rets), 3),
        "win_rate":     round(_win_rate(daily_rets), 1),
    }


# ─────────────────────────────────────────────────────────────
# AI Strategy — uses the actual trained GBM
# ─────────────────────────────────────────────────────────────

def _run_ai_strategy(close: pd.Series) -> pd.Series:
    """
    Run the AI strategy using a rolling walk-forward approach.
    For each day in the out-of-sample period:
      - Train model on all prior data
      - Predict next day's price
      - Go long if predicted > current + 0.3%, else cash
    Uses a single model trained on the first 70% of data then re-used
    rolling for the remaining 30% (trade-off: speed vs. accuracy).
    """
    from src.model import _build_features, _make_feature_vector, GB_N_ESTIMATORS
    from sklearn.ensemble import GradientBoostingRegressor
    from src.config import (
        GB_MAX_DEPTH, GB_LEARNING_RATE, GB_SUBSAMPLE,
        GB_MIN_SAMPLES, GB_MAX_FEATURES, GB_RANDOM_STATE,
    )

    prices = close.values
    n      = len(prices)

    if n < SEQUENCE_LENGTH + 30:
        return pd.Series(np.ones(n), index=close.index)

    # Train on first 70% of data
    train_end = int(n * 0.70)
    train_prices = pd.Series(prices[:train_end], index=close.index[:train_end])

    X_tr, y_tr, scaler_tr, scaled_tr = _build_features(train_prices)
    if len(X_tr) < 10:
        return pd.Series(np.ones(n), index=close.index)

    model = GradientBoostingRegressor(
        n_estimators     = GB_N_ESTIMATORS,
        max_depth        = GB_MAX_DEPTH,
        learning_rate    = GB_LEARNING_RATE,
        subsample        = GB_SUBSAMPLE,
        min_samples_leaf = GB_MIN_SAMPLES,
        max_features     = GB_MAX_FEATURES,
        random_state     = GB_RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)

    # Walk-forward on remaining 30%
    equity = [1.0]
    in_position = False
    entry_price = None

    for i in range(train_end, n - 1):
        # Rebuild scaler on expanding window for proper normalisation
        hist = pd.Series(prices[max(0, i - SEQUENCE_LENGTH * 5):i + 1])
        from sklearn.preprocessing import MinMaxScaler
        sc    = MinMaxScaler()
        sc.fit(hist.values.reshape(-1, 1))
        scaled_window = sc.transform(hist.tail(SEQUENCE_LENGTH).values.reshape(-1, 1)).flatten()

        if len(scaled_window) < SEQUENCE_LENGTH:
            equity.append(equity[-1])
            continue

        features        = _make_feature_vector(scaled_window).reshape(1, -1)
        pred_scaled     = model.predict(features)[0]
        predicted_price = float(sc.inverse_transform([[pred_scaled]])[0][0])
        current_price   = prices[i]
        next_price      = prices[i + 1]

        threshold = current_price * 1.003   # 0.3% min expected gain

        if predicted_price > threshold and not in_position:
            # Buy — deduct transaction cost
            in_position = True
            entry_price = current_price
            daily_ret   = (next_price / current_price - 1) - _COST
        elif predicted_price <= threshold and in_position:
            # Sell — deduct transaction cost
            in_position = False
            daily_ret   = (next_price / current_price - 1) - _COST
        elif in_position:
            daily_ret = next_price / current_price - 1
        else:
            daily_ret = 0.0  # in cash

        equity.append(equity[-1] * (1 + daily_ret))

    # Pad beginning (buy & hold for training period)
    bh_prefix = prices[:train_end] / prices[0]
    full_series = np.concatenate([bh_prefix, equity])
    # Align lengths
    full_series = full_series[:n]
    return pd.Series(full_series, index=close.index)


# ─────────────────────────────────────────────────────────────
# Strategy Implementations
# ─────────────────────────────────────────────────────────────

def _buy_and_hold(close: pd.Series) -> pd.Series:
    return close / close.iloc[0]


def _momentum_strategy(close: pd.Series, window: int = 20) -> pd.Series:
    """Go long when 20d momentum is positive, cash otherwise."""
    momentum = close.pct_change(window)
    equity   = [1.0]
    pos      = False
    for i in range(1, len(close)):
        m = momentum.iloc[i]
        c = close.iloc[i]
        p = close.iloc[i - 1]
        if pd.isna(m):
            equity.append(equity[-1])
            continue
        going_long = m > 0
        if going_long and not pos:
            pos = True
            ret = (c / p - 1) - _COST
        elif not going_long and pos:
            pos = False
            ret = (c / p - 1) - _COST
        elif pos:
            ret = c / p - 1
        else:
            ret = 0.0
        equity.append(equity[-1] * (1 + ret))
    return pd.Series(equity, index=close.index)


def _mean_reversion_strategy(close: pd.Series, window: int = 20, band: float = 0.05) -> pd.Series:
    """Mean reversion: go long if price dropped >5% below 20d MA."""
    ma     = close.rolling(window).mean()
    equity = [1.0]
    pos    = False
    for i in range(1, len(close)):
        if pd.isna(ma.iloc[i]):
            equity.append(equity[-1])
            continue
        c   = close.iloc[i]
        p   = close.iloc[i - 1]
        dev = (c - ma.iloc[i]) / ma.iloc[i]
        # Buy: price is >5% below MA; Sell: price has reverted to MA
        buy_signal  = dev < -band
        sell_signal = dev >= 0
        if buy_signal and not pos:
            pos = True
            ret = (c / p - 1) - _COST
        elif sell_signal and pos:
            pos = False
            ret = (c / p - 1) - _COST
        elif pos:
            ret = c / p - 1
        else:
            ret = 0.0
        equity.append(equity[-1] * (1 + ret))
    return pd.Series(equity, index=close.index)


# ─────────────────────────────────────────────────────────────
# Drawdown Series
# ─────────────────────────────────────────────────────────────

def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return ((equity - peak) / peak) * 100


# ─────────────────────────────────────────────────────────────
# Monthly Return Heatmap Data
# ─────────────────────────────────────────────────────────────

def _monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Build a year × month matrix of monthly returns."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly.index = monthly.index.to_period("M")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years  = sorted(set(monthly.index.year))
    matrix = pd.DataFrame(index=years, columns=months, dtype=float)
    for period, val in monthly.items():
        yr = period.year
        mo = months[period.month - 1]
        matrix.loc[yr, mo] = round(val * 100, 2)
    return matrix


# ─────────────────────────────────────────────────────────────
# Drawdown Figure
# ─────────────────────────────────────────────────────────────

def build_drawdown_fig(comparison: pd.DataFrame) -> go.Figure:
    """Build a drawdown chart for all strategies."""
    colors = {
        "Buy & Hold":      "#7a8299",
        "AI Strategy":     "#00d4aa",
        "Momentum":        "#6c63ff",
        "Mean Reversion":  "#ffb347",
    }
    fig = go.Figure()
    for col in comparison.columns:
        dd = _drawdown_series(comparison[col])
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=col,
            line=dict(color=colors.get(col, "#fff"), width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(colors.get(col,'#7a8299').lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},0.05)",
        ))
    fig.update_layout(
        template="plotly_dark", height=280,
        title="Drawdown (% from peak)",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Monthly Heatmap Figure
# ─────────────────────────────────────────────────────────────

def build_monthly_heatmap(equity: pd.Series, strategy_name: str) -> go.Figure:
    """Build a monthly returns heatmap (year × month)."""
    matrix = _monthly_returns(equity)
    fig = go.Figure(go.Heatmap(
        z=matrix.values.astype(float),
        x=list(matrix.columns),
        y=[str(y) for y in matrix.index],
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in matrix.values.astype(float)],
        texttemplate="%{text}",
        textfont_size=10,
        colorbar=dict(title="Return %"),
    ))
    fig.update_layout(
        template="plotly_dark", height=max(200, len(matrix) * 40 + 80),
        title=f"Monthly Returns — {strategy_name}",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def run_backtest(stock: str) -> tuple[pd.DataFrame, dict]:
    """
    Run all 4 strategies and return (comparison_df, metrics_dict).
    comparison_df: each column is a normalised equity curve (starts at 1).
    """
    close = _load_backtest_data(stock)
    if close is None or len(close) < 100:
        empty_df  = pd.DataFrame()
        empty_met = {}
        return empty_df, empty_met

    # ── Run strategies ────────────────────────────────────────
    bh  = _buy_and_hold(close)
    ai  = _run_ai_strategy(close)
    mom = _momentum_strategy(close)
    mr  = _mean_reversion_strategy(close)

    comparison = pd.DataFrame({
        "Buy & Hold":      bh.values,
        "AI Strategy":     ai.values,
        "Momentum":        mom.values,
        "Mean Reversion":  mr.values,
    }, index=close.index)

    # ── Metrics ───────────────────────────────────────────────
    metrics = {
        "Buy & Hold":      _compute_metrics(bh,  "Buy & Hold"),
        "AI Strategy":     _compute_metrics(ai,  "AI Strategy"),
        "Momentum":        _compute_metrics(mom, "Momentum"),
        "Mean Reversion":  _compute_metrics(mr,  "Mean Reversion"),
    }

    return comparison, metrics