
import sys
import os
sys.path.append(os.getcwd())

try:
    from src.sentiment import analyse_sentiment, has_news_api_key
    print("Import successful!")
    print(f"analyse_sentiment: {analyse_sentiment}")
    print(f"has_news_api_key: {has_news_api_key}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
