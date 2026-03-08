# =============================================================
# sentiment.py — Financial News Sentiment
# =============================================================
#
# Two-tier approach:
#   Tier 1 (with NEWS_API_KEY): Fetch real headlines via NewsAPI
#           + analyse with FinBERT for financial-grade accuracy
#   Tier 2 (no API key):        Use yfinance's built-in news feed
#           + analyse with VADER (no API key needed, works offline)
#
# This means sentiment always works, even without setup.
# =============================================================

import os
import numpy as np
import streamlit as st

from src.config import NEWS_PAGE_SIZE, BULLISH_THRESHOLD, BEARISH_THRESHOLD

# ── Lazy-loaded singletons ────────────────────────────────────
_newsapi_client = None
_finbert        = None
_vader          = None


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _get_newsapi():
    global _newsapi_client
    if _newsapi_client is None:
        api_key = os.getenv("NEWS_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from newsapi import NewsApiClient
            _newsapi_client = NewsApiClient(api_key=api_key)
        except Exception:
            return None
    return _newsapi_client


@st.cache_resource(show_spinner=False)
def _get_finbert():
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Check whether NewsAPI key is configured
# ─────────────────────────────────────────────────────────────

def has_news_api_key() -> bool:
    return bool(os.getenv("NEWS_API_KEY", "").strip())


# ─────────────────────────────────────────────────────────────
# Fetch Headlines
# ─────────────────────────────────────────────────────────────

def _fetch_from_newsapi(stock: str) -> list[str]:
    """Fetch headlines via NewsAPI (requires API key)."""
    client = _get_newsapi()
    if client is None:
        return []
    try:
        company  = stock.split(".")[0]
        query    = f"{company} stock OR {company} earnings OR {company} NSE"
        articles = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=NEWS_PAGE_SIZE,
        )
        return [a["title"] for a in articles.get("articles", []) if a.get("title")]
    except Exception:
        return []


def _fetch_from_yfinance(stock: str) -> list[str]:
    """
    Fetch headlines from yfinance's built-in news feed.
    No API key required — works for any valid ticker.
    """
    try:
        import yfinance as yf
        ticker  = yf.Ticker(stock)
        news    = ticker.news or []
        return [
            item.get("content", {}).get("title", "") or item.get("title", "")
            for item in news[:NEWS_PAGE_SIZE]
            if item.get("content", {}).get("title") or item.get("title")
        ]
    except Exception:
        return []


def get_stock_news(stock: str) -> tuple[list[str], str]:
    """
    Return (headlines, source) where source is 'newsapi' or 'yfinance'.
    Tries NewsAPI first, falls back to yfinance automatically.
    """
    if has_news_api_key():
        headlines = _fetch_from_newsapi(stock)
        if headlines:
            return headlines, "newsapi"

    # Fallback — always works
    headlines = _fetch_from_yfinance(stock)
    return headlines, "yfinance"


# ─────────────────────────────────────────────────────────────
# Sentiment Scoring
# ─────────────────────────────────────────────────────────────

def _score_with_finbert(headlines: list[str]) -> float:
    finbert = _get_finbert()
    if finbert is None:
        return 0.0
    scores = []
    for h in headlines:
        try:
            result = finbert(h[:512])[0]
            label  = result["label"].lower()
            score  = result["score"]
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def _score_with_vader(headlines: list[str]) -> float:
    vader = _get_vader()
    if vader is None:
        return 0.0
    scores = [vader.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def get_news_sentiment(stock: str) -> tuple[float, str, list[str], str]:
    """
    Returns (sentiment_score, label, headlines, source_info).

    source_info is a human-readable string describing what was used,
    e.g. "NewsAPI + FinBERT" or "yfinance + VADER (no API key set)"
    """
    headlines, news_source = get_stock_news(stock)

    if not headlines:
        return 0.0, "Neutral", [], "No news available"

    # Choose sentiment model based on news source and availability
    if news_source == "newsapi" and _get_finbert() is not None:
        score       = _score_with_finbert(headlines)
        model_label = "NewsAPI + FinBERT"
    else:
        score       = _score_with_vader(headlines)
        model_label = "yfinance news + VADER"

    if score > BULLISH_THRESHOLD:
        label = "Bullish"
    elif score < BEARISH_THRESHOLD:
        label = "Bearish"
    else:
        label = "Neutral"

    return round(score, 4), label, headlines, model_label