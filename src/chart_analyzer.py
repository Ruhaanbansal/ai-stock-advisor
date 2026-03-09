# =============================================================
# chart_analyzer.py — AI-Powered Chart Screenshot Analysis
# =============================================================
#
# Uses Claude claude-sonnet-4-20250514 vision to analyse stock chart
# screenshots and extract:
#   - Chart patterns (H&S, Double Top/Bottom, etc.)
#   - Trend direction & momentum
#   - Support & Resistance levels
#   - Buy / Sell / Hold recommendation
#
# Supports: PNG, JPG, JPEG, WEBP screenshots from any platform
# (Zerodha, Groww, TradingView, Google Finance, etc.)
# =============================================================

import os
import base64
import json
import streamlit as st
from io import BytesIO


# ─────────────────────────────────────────────────────────────
# API Key
# ─────────────────────────────────────────────────────────────

def get_api_key() -> str | None:
    """Get Claude API key from Streamlit secrets or env."""
    # Streamlit Cloud secrets
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    # Local .env
    key = os.getenv("ANTHROPIC_API_KEY")
    return key if key else None


# ─────────────────────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────────────────────

def _image_to_base64(image_bytes: bytes, media_type: str) -> str:
    """Convert image bytes to base64 string."""
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def _get_media_type(filename: str) -> str:
    """Infer media type from filename."""
    ext = filename.lower().split(".")[-1]
    return {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
    }.get(ext, "image/png")


# ─────────────────────────────────────────────────────────────
# Analysis Prompt
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are an expert technical analyst specialising in Indian stock markets (NSE/BSE).

Analyse this stock chart screenshot carefully and provide a detailed technical analysis.

Return your analysis as a valid JSON object with EXACTLY this structure:
{
  "stock_name": "name/ticker if visible, else 'Unknown'",
  "timeframe": "timeframe visible on chart (e.g. 1D, 1W, 15min)",
  "current_price": "price if visible, else null",

  "trend": {
    "direction": "Bullish / Bearish / Sideways",
    "strength": "Strong / Moderate / Weak",
    "description": "2-3 sentence description of the trend"
  },

  "momentum": {
    "signal": "Bullish / Bearish / Neutral",
    "rsi_visible": true or false,
    "rsi_value": number or null,
    "macd_visible": true or false,
    "macd_signal": "Bullish crossover / Bearish crossover / Neutral / null",
    "description": "Brief momentum description"
  },

  "patterns": [
    {
      "name": "pattern name (e.g. Head and Shoulders, Double Top, Bull Flag)",
      "type": "Continuation / Reversal",
      "reliability": "High / Medium / Low",
      "description": "What the pattern suggests"
    }
  ],

  "support_resistance": {
    "key_support": ["price level 1", "price level 2"],
    "key_resistance": ["price level 1", "price level 2"],
    "description": "Brief S/R analysis"
  },

  "candlestick_patterns": [
    {
      "name": "pattern name (e.g. Doji, Hammer, Engulfing)",
      "signal": "Bullish / Bearish / Neutral",
      "location": "where on chart (recent / at support / at resistance)"
    }
  ],

  "recommendation": {
    "action": "Strong Buy / Buy / Hold / Sell / Strong Sell",
    "confidence": number between 0 and 100,
    "entry_zone": "price range for entry if applicable",
    "stop_loss": "suggested stop loss level",
    "target": "price target if applicable",
    "rationale": "3-4 sentence explanation of the recommendation"
  },

  "risk_factors": ["risk 1", "risk 2", "risk 3"],

  "summary": "2-3 sentence overall summary a retail investor would understand"
}

Important rules:
- Return ONLY the JSON object, no markdown, no backticks, no preamble
- If something is not visible in the chart, use null
- Be specific with price levels if visible
- Base analysis purely on what is visible in the chart
- Consider Indian market context (NSE/BSE stocks trade in INR)"""


# ─────────────────────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────────────────────

def analyse_chart(
    image_bytes: bytes,
    filename:    str,
    extra_context: str = "",
) -> dict:
    """
    Send chart image to Claude claude-sonnet-4-20250514 for technical analysis.

    Args:
        image_bytes:   Raw image bytes from uploaded file
        filename:      Original filename (used to detect media type)
        extra_context: Optional user-provided context (stock name, etc.)

    Returns:
        dict with full analysis or {"error": "message"} on failure
    """
    api_key = get_api_key()
    if not api_key:
        # No API key — use rule-based fallback silently
        from src.chart_analyzer_fallback import analyse_chart_fallback  # noqa
        try:
            from chart_analyzer_fallback import analyse_chart_fallback
        except ImportError:
            from src.chart_analyzer_fallback import analyse_chart_fallback
        result = analyse_chart_fallback(image_bytes, filename)
        result["_fallback"] = True
        result["_fallback_reason"] = "No ANTHROPIC_API_KEY found — using rule-based analysis."
        return result

    media_type = _get_media_type(filename)
    img_b64    = _image_to_base64(image_bytes, media_type)

    # Build prompt with optional context
    prompt = ANALYSIS_PROMPT
    if extra_context.strip():
        prompt = (
            f"Additional context provided by user: {extra_context}\n\n"
            + ANALYSIS_PROMPT
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":  "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw_text = message.content[0].text.strip()

        # Clean any accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError as e:
        return {"error": f"Could not parse AI response: {e}"}
    except Exception as e:
        err = str(e)
        if "529" in err or "overload" in err.lower():
            return {"error": "Claude API is overloaded. Please try again in a moment."}
        # Credit exhausted or auth error → use fallback
        if ("credit" in err.lower() or "balance" in err.lower()
                or "401" in err or "402" in err):
            try:
                from chart_analyzer_fallback import analyse_chart_fallback
            except ImportError:
                from src.chart_analyzer_fallback import analyse_chart_fallback
            result = analyse_chart_fallback(image_bytes, filename)
            result["_fallback"] = True
            result["_fallback_reason"] = (
                "Claude API credits exhausted or key invalid. "
                "Showing rule-based analysis instead."
            )
            return result
        return {"error": f"Analysis failed: {err}"}


# ─────────────────────────────────────────────────────────────
# Helper: recommendation color & icon
# ─────────────────────────────────────────────────────────────

def rec_style(action: str) -> tuple[str, str]:
    """Return (color, icon) for a recommendation action."""
    styles = {
        "Strong Buy":  ("#00d4aa", "🚀"),
        "Buy":         ("#4caf50", "📈"),
        "Hold":        ("#ffb347", "⏸️"),
        "Sell":        ("#ff7043", "📉"),
        "Strong Sell": ("#ff6b6b", "🔴"),
    }
    return styles.get(action, ("#7a8299", "⚪"))


def trend_icon(direction: str) -> str:
    return {"Bullish": "📈", "Bearish": "📉", "Sideways": "↔️"}.get(direction, "❓")


def pattern_type_color(ptype: str) -> str:
    return "#00d4aa" if ptype == "Continuation" else "#ff6b6b"