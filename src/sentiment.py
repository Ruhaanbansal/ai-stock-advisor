from newsapi import NewsApiClient
from transformers import pipeline
import numpy as np


# ---------------------------------------
# News API Setup
# ---------------------------------------

API_KEY = "dc2e10a2668b43de97031e332dab40b7"

newsapi = NewsApiClient(api_key=API_KEY)


# ---------------------------------------
# FinBERT Sentiment Model
# ---------------------------------------

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


# ---------------------------------------
# Fetch Stock News
# ---------------------------------------

def get_stock_news(stock):

    company = stock.split(".")[0]

    query = f"{company} stock OR {company} company OR {company} earnings"

    articles = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    print("Fetching news for:", query)
    print("Articles found:", len(articles["articles"]))

    headlines = []

    for article in articles["articles"]:
        headlines.append(article["title"])

    return headlines

# ---------------------------------------
# FinBERT Sentiment Analysis
# ---------------------------------------

def get_news_sentiment(stock):

    headlines = get_stock_news(stock)

    if len(headlines) == 0:
        return 0, "Neutral", []

    scores = []

    for headline in headlines:

        result = sentiment_model(headline)[0]

        label = result["label"]
        score = result["score"]

        if label == "positive":
            scores.append(score)

        elif label == "negative":
            scores.append(-score)

        else:
            scores.append(0)

    sentiment_score = np.mean(scores)

    if sentiment_score > 0.2:
        sentiment_label = "Bullish"

    elif sentiment_score < -0.2:
        sentiment_label = "Bearish"

    else:
        sentiment_label = "Neutral"

    return sentiment_score, sentiment_label, headlines