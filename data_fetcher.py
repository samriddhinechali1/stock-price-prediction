import yfinance as yf
import pandas as pd
from datetime import datetime

end_date = pd.Timestamp.today().normalize()

def fetch_stock_data(ticker, start_date, end_date):

    """
    Fetch historical data from Yahoo
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.columns = [col[0] for col in stock_data.columns]

    df = stock_data.reset_index()
    df.set_index("Date", inplace=True)

    return df

def get_real_time_data(ticker, period="1d", interval="1m"):
    """
    Fetches real-time (latest available) stock data from Yahoo Finance.
    :param symbol: Stock symbol (ticker)
    :return: DataFrame containing stock data with the latest closing price
    """
    data = yf.download(ticker, end=end_date, period=period, interval=interval)  
    data.columns = [col[0] for col in data.columns]
    df = data.reset_index()
    df.set_index("Datetime", inplace=True)
    return df

def fetch_data(ticker, period, interval):
    stock_data = yf.Ticker(ticker)
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history