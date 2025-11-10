import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Import the Regressor instead of the Classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset (rest of the loading code is fine)
df = pd.read_csv('processed_amazon_data.csv', parse_dates=['Date'])
df = df.sort_values("Date").reset_index(drop=True)

features_col = [
    "High", "Low", "Open", "Volume",
    "HL_Pct", "OC_Pct", "SMA_10", "SMA_30",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "Volume_SMA_10", "Volume_Ratio",
    "Return_Lag_1", "Return_Lag_3", "Return_Lag_5"
]

X_raw = df[features_col].values
y_raw = df['Close'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Change this line to use RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)

# The fit method will now work with continuous y_train values
reg.fit(X_train, y_train)

# Use the regressor's predict method
y_pred_scaled = reg.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
mse = np.mean((y_test_original - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual Close Prices', color='blue')
plt.plot(y_pred, label='Predicted Close Prices', color='red')
plt.title('Random Forest Regression: Actual vs Predicted Close Prices')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.show()
