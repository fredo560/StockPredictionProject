# predict_amzn_clean.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta
import time

# ================================
# 1. DOWNLOAD DATA FROM YFINANCE
# ================================
print("Downloading AMZN data from yfinance...")
ticker = "AMZN"
end_date   = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=70)).strftime('%Y-%m-%d')

time.sleep(5)
df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
print(df.head())

# ================================
# 2. REMOVE FIRST 3 ROWS & RENAME COLUMNS
# ================================
if len(df) <= 3:
    raise ValueError("Not enough data after download.")

# Drop first 3 rows (in case of junk headers)
df = df.iloc[3:].copy()

# Reset index and rename columns explicitly
df = df.reset_index()
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

print(f"Cleaned data: {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")

print(df.head())

# ================================
# 3. DATA CLEANING
# ================================
# Drop invalid price or volume
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]
df = df[df['Volume'] > 0]

if len(df) < 40:
    raise ValueError("Not enough valid data after cleaning.")

# ================================
# 4. FEATURE ENGINEERING
# ================================
def safe_rolling(s, window):
    return s.rolling(window=window, min_periods=1).mean()

df['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

df['SMA_10'] = safe_rolling(df['Close'], 10)
df['SMA_30'] = safe_rolling(df['Close'], 30)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = safe_rolling(gain, 14)
avg_loss = safe_rolling(loss, 14)
rs = avg_gain / (avg_loss + 1e-10)
df['RSI'] = 100 - (100 / (1 + rs))

df['Volume_SMA_10'] = safe_rolling(df['Volume'], 10)
df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_10'] + 1e-10)

for lag in [1, 3, 5]:
    df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)

# Drop rows with any NaN
df = df.dropna().reset_index(drop=True)
print(f"After features & NaN drop: {len(df)} rows")

if len(df) < 20:
    raise ValueError("Not enough data for sequence length 20.")

# ================================
# 5. FEATURES & SCALING
# ================================
feature_cols = [
    "High", "Low", "Open", "Volume",
    "HL_Pct", "OC_Pct", "SMA_10", "SMA_30",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "Volume_SMA_10", "Volume_Ratio",
    "Return_Lag_1", "Return_Lag_3", "Return_Lag_5"
]

X_raw = df[feature_cols].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

# Target scaler (for Close)
scaler_y = MinMaxScaler()
y_raw = df[['Close']].values
scaler_y.fit(y_raw)

# ================================
# 6. SEQUENCE CREATION
# ================================
SEQ_LENGTH = 20

def create_sequences(X, y_scaler, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X) - 1):  # leave last row for prediction
        Xs.append(X[i-seq_len:i])
        ys.append(y_scaler.transform([[df['Close'].iloc[i]]])[0][0])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_scaled, scaler_y, SEQ_LENGTH)

# Latest sequence for prediction
latest_seq = X_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)

print(f"Training on {X_train.shape[0]} sequences")

# ================================
# 7. LSTM MODEL (Hardcoded Best Hyperparams)
# ================================
model = keras.Sequential([
    layers.LSTM(96, return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])),
    layers.Dropout(0.5),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(25, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
print("Training LSTM...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# ================================
# 8. PREDICT NEXT DAY
# ================================
y_pred_scaled = model.predict(latest_seq, verbose=0)[0][0]
y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]

last_close = df['Close'].iloc[-1]
last_date = df['Date'].iloc[-1]
next_date = last_date + pd.Timedelta(days=1)

change_dollar = y_pred - last_close
change_pct = change_dollar / last_close * 100

print(f"\nLast Close ({last_date.date()}): ${last_close:.2f}")
print(f"Predicted ({next_date.date()}): ${y_pred:.2f}")
print(f"Change: {change_dollar:+.2f} ({change_pct:+.2f}%)")

# ================================
# 9. SAVE PREDICTION
# ================================
pred_df = pd.DataFrame([{
    "Date": next_date.strftime('%Y-%m-%d'),
    "Ticker": ticker,
    "Last_Close": round(last_close, 2),
    "Predicted_Next_Close": round(y_pred, 2),
    "Predicted_Change_$": round(change_dollar, 2),
    "Predicted_Change_%": round(change_pct, 2)
}])

import os

csv_file = "lstmPrediction.csv"

if os.path.exists(csv_file):
    # Append without writing header
    pred_df.to_csv(csv_file, mode='a', header=False, index=False)
    print(f"\nAppended prediction to {csv_file}")
else:
    # Create new file with header
    pred_df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"\nCreated and saved: {csv_file}")


