import yfinance as yf
import pandas as pd


def load_stock_data(stock, period="6mo"):
    """
    Download stock data from Yahoo Finance
    """

    data = yf.download(stock, period=period, auto_adjust=True)

    if data.empty:
        return None

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def get_close_prices(data):
    """
    Extract clean close price series
    """

    close_prices = data["Close"]

    close_prices = pd.to_numeric(close_prices, errors="coerce")

    close_prices = close_prices.dropna()

    return close_prices


def load_portfolio_data(stocks, period="1y"):
    """
    Download multiple stocks for portfolio analysis
    """

    data = yf.download(stocks, period=period, auto_adjust=True)

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]

    return prices


def calculate_returns(price_data):
    """
    Calculate daily returns
    """

    returns = price_data.pct_change().dropna()

    return returns