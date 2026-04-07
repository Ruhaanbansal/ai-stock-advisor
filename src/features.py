# =============================================================
# features.py — Technical Indicator Engineering
# =============================================================

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Price-based Indicators
# ─────────────────────────────────────────────────────────────

def calculate_sma(close_prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return close_prices.rolling(window).mean()


def calculate_ema(close_prices: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return close_prices.ewm(span=window, adjust=False).mean()


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (ewm).
    Returns values 0-100.
    """
    delta    = close_prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close_prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MACD Line and Signal Line (12/26/9 standard config)."""
    ema12  = calculate_ema(close_prices, 12)
    ema26  = calculate_ema(close_prices, 26)
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_bollinger_bands(
    close_prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, middle (SMA), lower."""
    sma   = close_prices.rolling(window).mean()
    std   = close_prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range — measures volatility."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window).mean()


def calculate_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator %K and %D.
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = 3-period SMA of %K
    """
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom        = highest_high - lowest_low
    stoch_k      = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    stoch_d      = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Commodity Channel Index.
    CCI > +100 = overbought, CCI < -100 = oversold.
    """
    typical_price = (high + low + close) / 3
    sma           = typical_price.rolling(window).mean()
    mean_dev      = typical_price.rolling(window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical_price - sma) / (0.015 * mean_dev.replace(0, np.nan))


# ─────────────────────────────────────────────────────────────
# Volume-based Indicators
# ─────────────────────────────────────────────────────────────

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume — cumulative volume flow.
    Rising OBV + rising price = healthy uptrend.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Rolling VWAP over a window (daily approximation).
    Price above VWAP = bullish; below = bearish.
    """
    typical_price  = (high + low + close) / 3
    tp_vol         = typical_price * volume
    rolling_tp_vol = tp_vol.rolling(window).sum()
    rolling_vol    = volume.rolling(window).sum()
    return rolling_tp_vol / rolling_vol.replace(0, np.nan)


def calculate_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume ratio: today's volume vs. rolling average — spike detection."""
    return volume / volume.rolling(window).mean().replace(0, np.nan)


# ─────────────────────────────────────────────────────────────
# Feature Engineering Pipeline (used by model.py)
# ─────────────────────────────────────────────────────────────

def create_extended_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full technical feature set for dashboard charts and explainability.
    Expects a DataFrame with at least a 'Close' column.
    Safely handles missing High/Low/Volume columns.
    """
    close = data["Close"]
    df    = data.copy()

    # ── Price indicators (always available) ──────────────────
    df["SMA20"] = calculate_sma(close, 20)
    df["SMA50"] = calculate_sma(close, 50)
    df["EMA12"] = calculate_ema(close, 12)
    df["RSI"]   = calculate_rsi(close)

    df["MACD"], df["MACD_Signal"] = calculate_macd(close)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calculate_bollinger_bands(close)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # ── OHLC-dependent indicators ─────────────────────────────
    if all(c in data.columns for c in ["High", "Low", "Close"]):
        df["ATR"]              = calculate_atr(data["High"], data["Low"], close)
        df["Stoch_K"], df["Stoch_D"] = calculate_stochastic(
            data["High"], data["Low"], close
        )
        df["CCI"] = calculate_cci(data["High"], data["Low"], close)

    # ── Volume indicators ─────────────────────────────────────
    if "Volume" in data.columns and data["Volume"].sum() > 0:
        df["OBV"]           = calculate_obv(close, data["Volume"])
        df["Vol_Ratio"]     = calculate_volume_ratio(data["Volume"])
        if all(c in data.columns for c in ["High", "Low"]):
            df["VWAP"] = calculate_vwap(
                data["High"], data["Low"], close, data["Volume"]
            )

    return df