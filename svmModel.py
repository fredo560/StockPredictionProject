# ================================
#  SVM – Predict Next-Day Close Price
#  (Scikit-learn + Hyperparameter Tuning + Plots + Save All)
# ================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
CSV_PATH = r"C:\Users\alfre\Desktop\stockProject\processed_amazon_data.csv"

# Create timestamped results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = r"C:\Users\alfre\Desktop\stockProject"
RESULTS_DIR = os.path.join(RESULTS_ROOT, f"svm_close_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Features (exclude current Close to avoid leakage)
feature_cols = [
    'High', 'Low', 'Open', 'Volume',
    'HL_Pct', 'OC_Pct', 'SMA_10', 'SMA_30',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI',
    'Volume_SMA_10', 'Volume_Ratio',
    'Return_Lag_1', 'Return_Lag_3', 'Return_Lag_5'
]

X_raw = df[feature_cols].values
y = df['Close'].values  # Target: next-day close price

print(f"Features: {X_raw.shape}, Target: {y.shape}")

# ----------------------------------------------------------------------
# 3. SCALING
# ----------------------------------------------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Save scalers
joblib.dump(scaler_X, os.path.join(RESULTS_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(RESULTS_DIR, "scaler_y.pkl"))
print(f"Scalers saved in: {RESULTS_DIR}")

# ----------------------------------------------------------------------
# 4. TRAIN-TEST SPLIT (Time Series Aware)
# ----------------------------------------------------------------------
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING (Grid Search with TimeSeriesSplit)
# ----------------------------------------------------------------------
print("\nStarting hyperparameter tuning with GridSearchCV...")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

svr = SVR()
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_svr = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# ----------------------------------------------------------------------
# 6. FINAL MODEL TRAINING (on full training data)
# ----------------------------------------------------------------------
print("Retraining best model on full training set...")
final_svr = SVR(**grid_search.best_params_)
final_svr.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 7. PREDICTIONS
# ----------------------------------------------------------------------
y_train_pred_scaled = final_svr.predict(X_train)
y_test_pred_scaled = final_svr.predict(X_test)

# Inverse transform
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# ----------------------------------------------------------------------
# 8. EVALUATION METRICS
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MSE  : {mse:,.4f}")
    print(f"  RMSE : {rmse:,.4f}")
    print(f"  MAE  : {mae:,.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

train_metrics = evaluate(y_train_true, y_train_pred, "TRAIN")
test_metrics = evaluate(y_test_true, y_test_pred, "TEST")

# ----------------------------------------------------------------------
# 9. PLOTS
# ----------------------------------------------------------------------
# --- Plot 1: Actual vs Predicted (Test Set) ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_true, label='Actual Close', alpha=0.8, color='blue')
plt.plot(y_test_pred, label='Predicted Close', alpha=0.8, color='red')
plt.title('SVM: Actual vs Predicted Next-Day Close Price (Test Set)')
plt.xlabel('Time Steps')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
pred_plot_path = os.path.join(RESULTS_DIR, "svm_actual_vs_predicted.png")
plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved: {pred_plot_path}")

# --- Plot 2: Residuals ---
residuals = y_test_true - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('SVM: Residuals (Actual - Predicted)')
plt.xlabel('Predicted Close Price ($)')
plt.ylabel('Residual ($)')
plt.grid(True, alpha=0.3)
resid_plot_path = os.path.join(RESULTS_DIR, "svm_residuals.png")
plt.savefig(resid_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Residuals plot saved: {resid_plot_path}")

# ----------------------------------------------------------------------
# 10. SAVE MODEL & RESULTS
# ----------------------------------------------------------------------
# Save model
model_path = os.path.join(RESULTS_DIR, "best_svm_close.pkl")
joblib.dump(final_svr, model_path)
print(f"Model saved: {model_path}")

# Save predictions
pred_df = pd.DataFrame({
    'Date': df['Date'].iloc[split_idx:].reset_index(drop=True),
    'Actual_Close': y_test_true,
    'Predicted_Close': y_test_pred,
    'Residual': residuals
})
pred_csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"Predictions saved: {pred_csv_path}")

# Save stats
with open(os.path.join(RESULTS_DIR, "evaluation_stats.txt"), "w") as f:
    f.write("SVM Next-Day Close Price Prediction\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    f.write("TRAIN METRICS:\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:,.4f}\n")
    f.write("\nTEST METRICS:\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:,.4f}\n")

print(f"Stats saved: {RESULTS_DIR}/evaluation_stats.txt")

# Save feature list
with open(os.path.join(RESULTS_DIR, "features.txt"), "w") as f:
    for col in feature_cols:
        f.write(col + "\n")

# ----------------------------------------------------------------------
# 11. SUMMARY
# ----------------------------------------------------------------------
print("SVM MODEL TRAINING COMPLETE")
print("="*60)
print(f"All results saved in:\n   {RESULTS_DIR}")
print("\nFiles created:")
print("   - best_svm_close.pkl")
print("   - scaler_X.pkl, scaler_y.pkl")
print("   - predictions.csv")
print("   - svm_actual_vs_predicted.png")
print("   - svm_residuals.png")
print("   - evaluation_stats.txt")
print("   - features.txt")
