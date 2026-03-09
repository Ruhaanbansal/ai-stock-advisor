# =============================================================
# chart_analyzer_fallback.py — Rule-based chart analysis
# =============================================================
# Used when Claude API key is missing or has no credits.
# Uses PIL + numpy to analyse the chart image directly.
# =============================================================

import numpy as np
from PIL import Image
from io import BytesIO


def _load_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((400, 300))
    return np.array(img)


def _extract_price_curve(img: np.ndarray) -> np.ndarray:
    """
    Extract a rough price curve by finding the darkest/brightest
    horizontal band per column (simulates candlestick highs).
    """
    gray = np.mean(img, axis=2)
    # Find row with max brightness per column (price line approximation)
    curve = np.argmax(gray, axis=0).astype(float)
    # Invert so higher price = higher value
    curve = gray.shape[0] - curve
    # Smooth
    kernel = np.ones(10) / 10
    curve  = np.convolve(curve, kernel, mode="same")
    return curve


def _trend_analysis(curve: np.ndarray) -> dict:
    """Detect trend from price curve slope."""
    n      = len(curve)
    first  = np.mean(curve[:n//4])
    last   = np.mean(curve[3*n//4:])
    mid    = np.mean(curve[n//4:3*n//4])
    change = (last - first) / (first + 1e-8) * 100

    # Fit linear regression
    x     = np.arange(n)
    slope = np.polyfit(x, curve, 1)[0]

    if slope > 0.05:
        direction, strength = "Bullish", ("Strong" if slope > 0.2 else "Moderate")
    elif slope < -0.05:
        direction, strength = "Bearish", ("Strong" if slope < -0.2 else "Moderate")
    else:
        direction, strength = "Sideways", "Weak"

    return {
        "direction":   direction,
        "strength":    strength,
        "description": (
            f"Price moved approximately {change:+.1f}% over the visible period. "
            f"The {'upward' if slope > 0 else 'downward' if slope < 0 else 'flat'} "
            f"slope indicates a {strength.lower()} {direction.lower()} trend."
        ),
    }


def _support_resistance(curve: np.ndarray, img_height: int) -> dict:
    """Find support and resistance from price curve extremes."""
    n      = len(curve)
    # Rolling min/max to find S/R zones
    window = n // 5

    lows  = []
    highs = []
    for i in range(0, n - window, window // 2):
        segment = curve[i:i+window]
        lows.append(float(np.min(segment)))
        highs.append(float(np.max(segment)))

    # Convert pixel positions to relative price levels
    price_min = float(np.min(curve))
    price_max = float(np.max(curve))
    price_range = price_max - price_min + 1e-8

    def pix_to_pct(v):
        return round((v - price_min) / price_range * 100, 1)

    unique_lows  = sorted(set([pix_to_pct(l) for l in lows]))[:2]
    unique_highs = sorted(set([pix_to_pct(h) for h in highs]), reverse=True)[:2]

    return {
        "key_support":    [f"{v}% of range" for v in unique_lows],
        "key_resistance": [f"{v}% of range" for v in unique_highs],
        "description":    (
            "Support and resistance estimated from visible price extremes. "
            "For precise levels, use the actual price scale on your chart."
        ),
    }


def _detect_patterns(curve: np.ndarray) -> list:
    """Detect basic chart patterns from curve shape."""
    n      = len(curve)
    third  = n // 3
    left   = np.mean(curve[:third])
    mid    = np.mean(curve[third:2*third])
    right  = np.mean(curve[2*third:])
    recent = curve[3*n//4:]

    patterns = []

    # Double Top
    if (left > mid * 1.02 and right > mid * 1.02
            and abs(left - right) / (left + 1e-8) < 0.05):
        patterns.append({
            "name":        "Possible Double Top",
            "type":        "Reversal",
            "reliability": "Medium",
            "description": "Two similar highs with a trough in between — bearish reversal signal.",
        })

    # Double Bottom
    elif (left < mid * 0.98 and right < mid * 0.98
              and abs(left - right) / (left + 1e-8) < 0.05):
        patterns.append({
            "name":        "Possible Double Bottom",
            "type":        "Reversal",
            "reliability": "Medium",
            "description": "Two similar lows with a peak in between — bullish reversal signal.",
        })

    # V-shaped recovery
    min_idx = int(np.argmin(curve))
    if (0.2 * n < min_idx < 0.8 * n
            and curve[min_idx] < np.mean(curve) * 0.9):
        patterns.append({
            "name":        "V-Shape Recovery",
            "type":        "Reversal",
            "reliability": "Low",
            "description": "Sharp decline followed by equally sharp recovery.",
        })

    # Uptrend channel
    slope = np.polyfit(np.arange(n), curve, 1)[0]
    if slope > 0.1:
        patterns.append({
            "name":        "Rising Channel",
            "type":        "Continuation",
            "reliability": "Medium",
            "description": "Consistent higher highs and higher lows — bullish continuation.",
        })
    elif slope < -0.1:
        patterns.append({
            "name":        "Falling Channel",
            "type":        "Continuation",
            "reliability": "Medium",
            "description": "Consistent lower highs and lower lows — bearish continuation.",
        })

    # Consolidation / flag
    recent_std = np.std(recent)
    full_std   = np.std(curve)
    if recent_std < full_std * 0.5:
        patterns.append({
            "name":        "Consolidation / Flag",
            "type":        "Continuation",
            "reliability": "Low",
            "description": "Price compressing after a move — potential breakout pending.",
        })

    return patterns[:3] if patterns else [{
        "name":        "No Clear Pattern",
        "type":        "Neutral",
        "reliability": "Low",
        "description": "No strong chart pattern detected. Market may be in transition.",
    }]


def _momentum_analysis(curve: np.ndarray) -> dict:
    """Estimate momentum from recent price action."""
    n        = len(curve)
    recent   = curve[int(n * 0.8):]
    earlier  = curve[int(n * 0.6):int(n * 0.8)]
    momentum = np.mean(recent) - np.mean(earlier)

    if momentum > 1:
        signal = "Bullish"
        desc   = "Recent price action shows accelerating upward momentum."
    elif momentum < -1:
        signal = "Bearish"
        desc   = "Recent price action shows accelerating downward momentum."
    else:
        signal = "Neutral"
        desc   = "Momentum is neutral — no strong directional bias detected."

    return {
        "signal":       signal,
        "rsi_visible":  False,
        "rsi_value":    None,
        "macd_visible": False,
        "macd_signal":  None,
        "description":  desc,
    }


def _recommendation(trend: dict, momentum: dict, patterns: list) -> dict:
    """Generate recommendation from trend + momentum + patterns."""
    bull_score = 0

    if trend["direction"] == "Bullish":
        bull_score += 2 if trend["strength"] == "Strong" else 1
    elif trend["direction"] == "Bearish":
        bull_score -= 2 if trend["strength"] == "Strong" else 1

    if momentum["signal"] == "Bullish":
        bull_score += 1
    elif momentum["signal"] == "Bearish":
        bull_score -= 1

    for p in patterns:
        if p["type"] == "Continuation" and trend["direction"] == "Bullish":
            bull_score += 0.5
        elif p["type"] == "Reversal":
            bull_score -= 0.5

    if bull_score >= 2.5:
        action, conf = "Strong Buy", 78
    elif bull_score >= 1:
        action, conf = "Buy",        65
    elif bull_score <= -2.5:
        action, conf = "Strong Sell", 75
    elif bull_score <= -1:
        action, conf = "Sell",        62
    else:
        action, conf = "Hold",        55

    return {
        "action":     action,
        "confidence": conf,
        "entry_zone": "See support levels",
        "stop_loss":  "Below nearest support",
        "target":     "Nearest resistance level",
        "rationale":  (
            f"Rule-based analysis detected a {trend['strength'].lower()} "
            f"{trend['direction'].lower()} trend with {momentum['signal'].lower()} "
            f"momentum. Pattern analysis identified: "
            f"{patterns[0]['name'] if patterns else 'no clear pattern'}. "
            f"This is a simplified analysis — upload to Claude AI for precise levels."
        ),
    }


def analyse_chart_fallback(image_bytes: bytes, filename: str) -> dict:
    """
    Rule-based chart analysis using image processing.
    No API key required. Less accurate than Claude Vision.
    """
    try:
        img   = _load_image(image_bytes)
        curve = _extract_price_curve(img)

        trend    = _trend_analysis(curve)
        sr       = _support_resistance(curve, img.shape[0])
        patterns = _detect_patterns(curve)
        momentum = _momentum_analysis(curve)
        rec      = _recommendation(trend, momentum, patterns)

        return {
            "stock_name":    "Unknown (add API key for auto-detection)",
            "timeframe":     "Unknown",
            "current_price": None,
            "trend":         trend,
            "momentum":      momentum,
            "patterns":      patterns,
            "support_resistance": sr,
            "candlestick_patterns": [],
            "recommendation": rec,
            "risk_factors": [
                "Rule-based analysis — less accurate than AI vision analysis",
                "Price levels are relative, not absolute",
                "Add ANTHROPIC_API_KEY for precise pattern detection",
            ],
            "summary": (
                f"Rule-based analysis detected a {trend['strength'].lower()} "
                f"{trend['direction'].lower()} trend with {momentum['signal'].lower()} "
                f"momentum. Recommendation: {rec['action']}. "
                f"For accurate support/resistance levels and pattern names, "
                f"add Claude API credits."
            ),
            "_fallback": True,
        }

    except Exception as e:
        return {"error": f"Fallback analysis failed: {e}"}