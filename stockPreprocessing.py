import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

CSV_PATH = r"C:\Users\alfre\Desktop\stockProject\amazon_historical_data.csv"
PROCESSED_DIR = r"C:\Users\alfre\Desktop\stockProject"


os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading and preprocessing data...")
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Target: Next-day % change
df['Next_Close'] = df['Close'].shift(-1)
df['Target'] = (df['Next_Close'] - df['Close']) / df['Close'] * 100
df = df.dropna(subset=['Target'])

# Feature Engineering (same as before)
df['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

df['SMA_10'] = df['Close'].rolling(10).mean()
df['SMA_30'] = df['Close'].rolling(30).mean()
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

for lag in [1, 3, 5]:
    df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)

df = df.dropna().reset_index(drop=True)

# Select features
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'HL_Pct', 'OC_Pct',
    'SMA_10', 'SMA_30',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI',
    'Volume_Ratio',
    'Return_Lag_1', 'Return_Lag_3', 'Return_Lag_5'
]

print(df.head())
print(df.tail())
print(df.columns)

df.to_csv(os.path.join(PROCESSED_DIR, "processed_amazon_data.csv"), index=False)
