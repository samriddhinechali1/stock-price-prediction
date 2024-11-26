import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import yfinance as yf
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from data_fetcher import fetch_stock_data

# Function to fetch the stock history
def fetch_stock_data(ticker, period, interval):
    stock_data = yf.Ticker(ticker)

    stock_history = stock_data.history(period=period, interval=interval)

    return stock_history


def get_periods_intervals():
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods

def  get_stock_predictions(ticker):

    # Pull data
    try: 
        stock_data = yf.Ticker(ticker)

        # Extract data for last 1yr with 1d interval
        history = stock_data.history(period="2y", interval="1d")

        # Clean data to keep only the Close column
        stock_data_close = history[['Close']]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq('D', method='ffill')

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        cutoff = int(len(stock_data_close)) * 0.8

        train_data = stock_data_close.iloc[: int(len(stock_data_close) * 0.8) + 1]  # 90%
        test_data = stock_data_close.iloc[int(len(stock_data_close) * 0.8) :]  # 10%

        # Build model
        model = AutoReg(train_data['Close'], 250).fit(cov_type="HC0")

        # Predict for test data

        predictions = model.predict(
            start = test_data.index[0], end= test_data.index[-1], dynamic=True
        )

        # Predict 90 days into the future
        forecast = model.predict(
            start= test_data.index[0],
            end= test_data.index[-1] + dt.timedelta(days=90),
            dynamic=True
        )

        return train_data, test_data, forecast, predictions
    except:
        return None, None, None, None