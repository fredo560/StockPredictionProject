# ================================
#  CNN-LSTM Hybrid – Predict Next-Day Close Price
#  (Keras + Conv1D + LSTM + Keras Tuner + Full Save + Plots)
# ================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
CSV_PATH = r"C:\Users\alfre\Desktop\stockProject\processed_amazon_data.csv"

# Timestamped results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = r"C:\Users\alfre\Desktop\stockProject"
RESULTS_DIR = os.path.join(RESULTS_ROOT, f"cnn_lstm_close_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_TRIALS = 50
EPOCHS_TUNE = 50
EPOCHS_FINAL = 120
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Features (exclude current Close)
feature_cols = [
    'High', 'Low', 'Open', 'Volume',
    'HL_Pct', 'OC_Pct', 'SMA_10', 'SMA_30',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI',
    'Volume_SMA_10', 'Volume_Ratio',
    'Return_Lag_1', 'Return_Lag_3', 'Return_Lag_5'
]

X_raw = df[feature_cols].values
y_raw = df['Close'].values.reshape(-1, 1)  # Target: next-day Close

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw).flatten()

# Save scalers
joblib.dump(scaler_X, os.path.join(RESULTS_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(RESULTS_DIR, "scaler_y.pkl"))
print(f"Scalers saved in: {RESULTS_DIR}")

# ----------------------------------------------------------------------
# 3. SEQUENCE CREATOR
# ----------------------------------------------------------------------
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ----------------------------------------------------------------------
# 4. CNN-LSTM MODEL BUILDER (Keras Tuner)
# ----------------------------------------------------------------------
def build_cnn_lstm(hp):
    seq_len = hp.Int('seq_length', 20, 120, step=10)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
    
    split = int(len(X_seq) * 0.8)
    X_tr, X_va = X_seq[:split], X_seq[split:]
    y_tr, y_va = y_seq[:split], y_seq[split:]

    inputs = keras.Input(shape=(seq_len, len(feature_cols)))

    # --- CNN Feature Extractor ---
    x = layers.Conv1D(
        filters=hp.Int('conv_filters', 32, 128, step=16),
        kernel_size=hp.Choice('kernel_size', [3, 5]),
        padding='same',
        activation='relu'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(hp.Float('dropout_conv', 0.1, 0.5, step=0.1))(x)

    x = layers.Conv1D(
        filters=hp.Int('conv_filters_2', 16, 64, step=16),
        kernel_size=hp.Choice('kernel_size_2', [3, 5]),
        padding='same',
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # --- LSTM Sequence Model ---
    x = layers.LSTM(
        units=hp.Int('lstm_units', 32, 128, step=16),
        return_sequences=False
    )(x)
    x = layers.Dropout(hp.Float('dropout_lstm', 0.1, 0.5, step=0.1))(x)

    # --- Dense Head ---
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout_dense', 0.1, 0.5, step=0.1))(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model, X_tr, y_tr, X_va, y_va, seq_len

# ----------------------------------------------------------------------
# 5. CUSTOM TUNER
# ----------------------------------------------------------------------
class CNNLSTMTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model, X_tr, y_tr, X_va, y_va, _ = self.hypermodel.build(hp)
        
        hist = model.fit(
            X_tr, y_tr,
            epochs=EPOCHS_TUNE,
            batch_size=BATCH_SIZE,
            validation_data=(X_va, y_va),
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )
        val_loss = min(hist.history['val_loss'])
        self.oracle.update_trial(trial.trial_id, {'val_loss': val_loss})
        return val_loss

# ----------------------------------------------------------------------
# 6. HYPERPARAMETER SEARCH
# ----------------------------------------------------------------------
print("\nStarting CNN-LSTM hyperparameter search...")
tuner = CNNLSTMTuner(
    hypermodel=build_cnn_lstm,
    objective='val_loss',
    max_trials=MAX_TRIALS,
    directory=RESULTS_DIR,
    project_name='cnn_lstm_tune',
    overwrite=True
)

tuner.search_space_summary()
tuner.search()

# ----------------------------------------------------------------------
# 7. RETRAIN BEST MODEL ON FULL DATA
# ----------------------------------------------------------------------
print("\nRetraining best CNN-LSTM on full dataset...")
best_hp = tuner.get_best_hyperparameters(1)[0]
best_seq = best_hp.get('seq_length')

print(f"Best seq_length     : {best_seq}")
print(f"Conv Filters        : {best_hp.get('conv_filters')}, {best_hp.get('conv_filters_2')}")
print(f"LSTM Units          : {best_hp.get('lstm_units')}")
print(f"Dropouts (Conv/LSTM/Dense) : {best_hp.get('dropout_conv'):.2f}, {best_hp.get('dropout_lstm'):.2f}, {best_hp.get('dropout_dense'):.2f}")
print(f"Learning rate       : {best_hp.get('lr'):.2e}")

# Full sequences
X_full, y_full = create_sequences(X_scaled, y_scaled, best_seq)

# Build best model
final_model = build_cnn_lstm(best_hp)[0]

# Final training
history = final_model.fit(
    X_full, y_full,
    epochs=EPOCHS_FINAL,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, "best_cnn_lstm.keras"),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
)

# ----------------------------------------------------------------------
# 8. SAVE MODEL (.keras & .h5)
# ----------------------------------------------------------------------
keras_path = os.path.join(RESULTS_DIR, "best_cnn_lstm.keras")
h5_path = os.path.join(RESULTS_DIR, "best_cnn_lstm.h5")
final_model.save(keras_path)
final_model.save(h5_path)
print(f"Model saved → {keras_path}")
print(f"Model saved → {h5_path}")

# ----------------------------------------------------------------------
# 9. PREDICTIONS & METRICS
# ----------------------------------------------------------------------
y_pred_scaled = final_model.predict(X_full)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_full.reshape(-1, 1))

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\nFinal MSE : {mse:,.4f}")
print(f"RMSE      : {rmse:,.4f}")
print(f"MAE       : {mae:,.4f}")
print(f"R²        : {r2:.4f}")

# ----------------------------------------------------------------------
# 10. PLOTS
# ----------------------------------------------------------------------
# --- Actual vs Predicted ---
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='Actual Close', alpha=0.8, color='blue')
plt.plot(y_pred, label='Predicted Close', alpha=0.8, color='orange')
plt.title('CNN-LSTM: Actual vs Predicted Next-Day Close Price')
plt.xlabel('Time Steps')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
pred_plot = os.path.join(RESULTS_DIR, "cnn_lstm_actual_vs_pred.png")
plt.savefig(pred_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved: {pred_plot}")

# --- Residuals ---
residuals = y_true.flatten() - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title('CNN-LSTM: Residuals (Actual - Predicted)')
plt.xlabel('Predicted Close ($)')
plt.ylabel('Residual ($)')
plt.grid(True, alpha=0.3)
resid_plot = os.path.join(RESULTS_DIR, "cnn_lstm_residuals.png")
plt.savefig(resid_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Residuals plot saved: {resid_plot}")

# --- Loss Curves ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN-LSTM: Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
loss_plot = os.path.join(RESULTS_DIR, "cnn_lstm_loss_curves.png")
plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Loss curve saved: {loss_plot}")

# ----------------------------------------------------------------------
# 11. SAVE SEQUENCES & STATS
# ----------------------------------------------------------------------
seq_dir = os.path.join(RESULTS_DIR, f"seq_{best_seq}")
os.makedirs(seq_dir, exist_ok=True)

np.save(os.path.join(seq_dir, "X.npy"), X_full)
np.save(os.path.join(seq_dir, "y.npy"), y_full)

with open(os.path.join(seq_dir, "hyperparameters.txt"), "w") as f:
    f.write(f"seq_length: {best_seq}\n")
    f.write(f"conv_filters: {best_hp.get('conv_filters')}, {best_hp.get('conv_filters_2')}\n")
    f.write(f"kernel_sizes: {best_hp.get('kernel_size')}, {best_hp.get('kernel_size_2')}\n")
    f.write(f"lstm_units: {best_hp.get('lstm_units')}\n")
    f.write(f"dropout_conv: {best_hp.get('dropout_conv')}\n")
    f.write(f"dropout_lstm: {best_hp.get('dropout_lstm')}\n")
    f.write(f"dropout_dense: {best_hp.get('dropout_dense')}\n")
    f.write(f"learning_rate: {best_hp.get('lr')}\n")
    f.write(f"mse: {mse:,.4f}\n")
    f.write(f"rmse: {rmse:,.4f}\n")
    f.write(f"mae: {mae:,.4f}\n")
    f.write(f"r2: {r2:.4f}\n")

with open(os.path.join(seq_dir, "features.txt"), "w") as f:
    for c in feature_cols:
        f.write(c + "\n")

# Save predictions
pred_df = pd.DataFrame({
    'Actual': y_true.flatten(),
    'Predicted': y_pred.flatten(),
    'Residual': residuals
})
pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# ----------------------------------------------------------------------
# 12. FINAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("CNN-LSTM TRAINING COMPLETE")
print("="*60)
print(f"All results saved in:\n   {RESULTS_DIR}")
print("\nFiles:")
print("   best_cnn_lstm.keras")
print("   best_cnn_lstm.h5")
print("   scaler_X.pkl, scaler_y.pkl")
print("   predictions.csv")
print("   cnn_lstm_actual_vs_pred.png")
print("   cnn_lstm_residuals.png")
print("   cnn_lstm_loss_curves.png")
print("   seq_XX/ (X.npy, y.npy, hyperparameters, features)")
print("="*60)