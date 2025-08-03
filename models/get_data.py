# ticker.py
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def get_historical_data(ticker, start_date=None, end_date=None):
    try:
        if start_date is None:
            start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        if end_date is None:
            end_date = datetime.datetime.now()
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        return data
    except Exception as e:
        raise Exception(f"Error fetching data for ticker {ticker}: {str(e)}")

def get_columns(data: pd.DataFrame):
    return list(data.columns)

def get_last_price(data: pd.DataFrame, column_name: str):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    return data[column_name].iloc[-1]

def plot_data(data: pd.DataFrame, ticker: str, column_name: str):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    
    plt.figure(figsize=(10, 6))
    data[column_name].plot()
    plt.ylabel(column_name)
    plt.xlabel('Date')
    plt.title(f'Historical data for {ticker} - {column_name}')
    plt.legend([ticker], loc='best')
    plt.tight_layout()
    return plt
