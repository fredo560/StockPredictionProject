import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker_symbol = "AMZN"

end_date = datetime.today()  # Today's date
start_date = end_date - timedelta(days=365)

data = yf.download(ticker_symbol, 
                   start=start_date.strftime('%Y-%m-%d'), 
                   end=end_date.strftime('%Y-%m-%d'), 
                   interval='1d')


print("Amazon (AMZN) - Historical Data (Past ~1 Year)")
print(data.head())
print("\n...\n")
print(data.tail())

data.to_csv("amazon_historical_data.csv")
