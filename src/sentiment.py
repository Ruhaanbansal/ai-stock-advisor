# =============================================================
# sentiment.py — News Sentiment Analysis
# =============================================================
#
# Tier 1: FinBERT (ProsusAI/finbert) — finance-specific BERT model
#          runs on CPU locally, no API key needed (~440MB, cached)
# Tier 2: VADER — lightweight fallback, always available
#
# Sources:
#   - NewsAPI (if NEWS_API_KEY set)
#   - yfinance .news (always available, no API key)
# =============================================================

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────
# Ticker → Company name (for news search)
# ─────────────────────────────────────────────────────────────

def _clean_ticker(stock: str) -> str:
    """Extract plain company name from ticker for news search."""
    return stock.replace(".NS", "").replace(".BO", "").replace("_", " ")


def _format_ticker_query(stock: str, name: str = None) -> str:
    """Build a simpler query for higher hit rate on NewsAPI."""
    if name:
        # Standardize "Corporation/Limited/etc" removal if needed, or just use name
        short_name = name.split(' ')[0].split('-')[0]
        return f'"{name}" OR "{short_name} stock"'
    
    clean = _clean_ticker(stock)
    return f'"{clean}" AND "India"'


# ─────────────────────────────────────────────────────────────
# FinBERT — Finance-tuned BERT model (Tier 1)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_finbert():
    """
    Load ProsusAI/finbert from HuggingFace.
    Returns pipeline or None if transformers not installed.
    """
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1  # -1 = CPU
        pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
            max_length=512,
            truncation=True,
        )
        return pipe
    except ImportError:
        return None
    except Exception:
        return None


def _score_with_finbert(headlines: list[str]) -> list[float]:
    """
    Score headlines with FinBERT.
    Returns list of scores in [-1, +1]:
      positive → +score, negative → -score, neutral → 0
    """
    pipe = _load_finbert()
    if pipe is None or not headlines:
        return []

    scores = []
    try:
        results = pipe(headlines[:30], batch_size=8, truncation=True)
        for r in results:
            label = r["label"].lower()
            conf  = float(r["score"])
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
    except Exception:
        return []
    return scores


# ─────────────────────────────────────────────────────────────
# VADER — Lightweight fallback (Tier 2)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_vader():
    """Load VADER SentimentIntensityAnalyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def _score_with_vader(headlines: list[str]) -> list[float]:
    """Score headlines with VADER. Returns compound scores [-1, +1]."""
    analyzer = _load_vader()
    if analyzer is None or not headlines:
        return []
    return [analyzer.polarity_scores(h)["compound"] for h in headlines]


# ─────────────────────────────────────────────────────────────
# Score headlines with best available model
# ─────────────────────────────────────────────────────────────

def _score_headlines(headlines: list[str]) -> tuple[list[float], str]:
    """
    Try FinBERT first, fall back to VADER.
    Returns (scores, model_name).
    """
    finbert_scores = _score_with_finbert(headlines)
    if finbert_scores:
        return finbert_scores, "FinBERT"

    vader_scores = _score_with_vader(headlines)
    if vader_scores:
        return vader_scores, "VADER"

    return [], "None"


# ─────────────────────────────────────────────────────────────
# News Fetching
# ─────────────────────────────────────────────────────────────

def _fetch_from_newsapi(stock: str, api_key: str, name: str = None, n: int = 20) -> list[dict]:
    """Fetch news from NewsAPI using params for safe encoding and 30-day window."""
    try:
        import requests
        query    = _format_ticker_query(stock, name=name)
        # Expand window to 30 days for better coverage, especially for smaller stocks
        from_dt  = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url      = "https://newsapi.org/v2/everything"
        
        params = {
            "q":        query,
            "from":     from_dt,
            "sortBy":   "relevancy",
            "pageSize": n,
            "language": "en",
            "apiKey":   api_key
        }
        
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
            
        data = r.json()
        if data.get("status") != "ok":
            return []
        return [
            {
                "title":       a.get("title", "").strip(),
                "description": a.get("description", "").strip(),
                "url":         a.get("url", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source":      a.get("source", {}).get("name", "Market News"),
            }
            for a in data.get("articles", [])
            if a.get("title") and a.get("title").strip()
        ]
    except Exception:
        return []


def _fetch_from_yfinance(stock: str, n: int = 20) -> list[dict]:
    """
    Fetch news from yfinance .news property.
    Handles both the old flat schema and the new nested `content` schema
    introduced in yfinance >=0.2.50.
    """
    try:
        ticker     = yf.Ticker(stock)
        news_items = ticker.news or []
        result     = []
        for item in news_items[:n]:
            # ── New schema (yfinance >=0.2.50) ───────────────
            # item = {"id": "...", "content": {"title": ..., "summary": ...,
            #          "canonicalUrl": {"url": ...}, "provider": {"displayName": ...}}}
            content = item.get("content") or {}
            if content:
                title = (content.get("title") or "").strip()
                if not title:
                    continue
                # URL may be nested under canonicalUrl or clickThroughUrl
                url = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or ""
                )
                description = (content.get("summary") or content.get("description") or "").strip()
                provider    = content.get("provider") or {}
                source      = (provider.get("displayName") or provider.get("name") or "Market News").strip()
                published   = content.get("pubDate") or content.get("displayTime") or ""
            else:
                # ── Old flat schema ───────────────────────────
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                url         = item.get("link") or item.get("url") or ""
                description = (item.get("summary") or item.get("description") or "").strip()
                source      = (item.get("publisher") or item.get("source") or "Market News").strip()
                published   = item.get("providerPublishTime") or ""

            result.append({
                "title":       title,
                "description": description,
                "url":         url,
                "publishedAt": published,
                "source":      source,
            })
        return result
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Main Public API
# ─────────────────────────────────────────────────────────────

def has_news_api_key() -> bool:
    """Check if NewsAPI key is set in environment."""
    return bool(os.getenv("NEWS_API_KEY", ""))


@st.cache_data(ttl=1800, show_spinner=False)
def analyse_sentiment(stock: str, company_name: str = None) -> dict:
    """
    Fetch news and analyse sentiment for a given stock.

    Returns:
        {
          "score":           float (-1 to +1),
          "label":           "Bullish" | "Neutral" | "Bearish",
          "num_articles":    int,
          "articles":        list[dict],
          "model":           "FinBERT" | "VADER" | "None",
          "article_scores":  list[float],
          "score_std":       float,
        }
    """
    news_api_key = os.getenv("NEWS_API_KEY", "").strip().replace('"', '').replace("'", "")

    # ── Fetch headlines ───────────────────────────────────────
    articles = []
    if news_api_key:
        articles = _fetch_from_newsapi(stock, news_api_key, name=company_name)
    
    # If NewsAPI is empty OR failed, try yfinance
    if not articles:
        articles = _fetch_from_yfinance(stock)
        
    # If still empty, try yfinance with base name search (no .NS)
    if not articles and company_name:
        articles = _fetch_from_yfinance(_clean_ticker(stock))

    if not articles:
        return {
            "score": 0.0, "label": "Neutral", "num_articles": 0,
            "articles": [], "model": "None", "article_scores": [],
            "score_std": 0.0,
        }

    # ── Score each headline ───────────────────────────────────
    headlines = [
        f"{a['title']}. {a.get('description', '')}".strip()
        for a in articles
    ]
    headlines = [re.sub(r'\s+', ' ', h)[:512] for h in headlines if h]

    scores, model_name = _score_headlines(headlines)

    if not scores:
        return {
            "score": 0.0, "label": "Neutral",
            "num_articles": len(articles),
            "articles":     articles,
            "model":        "None",
            "article_scores": [],
            "score_std":    0.0,
        }

    # ── Aggregate with outlier trim ───────────────────────────
    scores_arr   = np.array(scores)
    trimmed_mean = float(np.mean(
        scores_arr[
            (scores_arr >= np.percentile(scores_arr, 10)) &
            (scores_arr <= np.percentile(scores_arr, 90))
        ]
    )) if len(scores_arr) > 4 else float(np.mean(scores_arr))

    std = float(np.std(scores_arr))

    # ── Label ─────────────────────────────────────────────────
    if trimmed_mean > 0.1:
        label = "Bullish"
    elif trimmed_mean < -0.1:
        label = "Bearish"
    else:
        label = "Neutral"

    # ── Attach scores to articles for display ─────────────────
    for i, art in enumerate(articles[:len(scores)]):
        art["_score"] = round(float(scores[i]), 3)

    return {
        "score":         round(trimmed_mean, 4),
        "label":         label,
        "num_articles":  len(articles),
        "articles":      articles,
        "model":         model_name,
        "article_scores": [round(s, 3) for s in scores],
        "score_std":     round(std, 3),
    }