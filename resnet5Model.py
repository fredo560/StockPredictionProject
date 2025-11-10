# ================================
#  ResNet-5 – Predict Next-Day Close Price
#  (Keras + Residual Blocks + Keras Tuner + Full Save)
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
RESULTS_DIR = os.path.join(RESULTS_ROOT, f"resnet5_close_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_TRIALS = 50
EPOCHS_TUNE = 50
EPOCHS_FINAL = 8
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
# 4. RESNET-5 BLOCK
# ----------------------------------------------------------------------
def residual_block(x, filters, kernel_size=3, dropout=0.0):
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match shortcut dimensions
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    return x

# ----------------------------------------------------------------------
# 5. MODEL BUILDER (Keras Tuner)
# ----------------------------------------------------------------------
def build_resnet5(hp):
    seq_len = hp.Int('seq_length', 20, 120, step=10)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
    
    split = int(len(X_seq) * 0.8)
    X_tr, X_va = X_seq[:split], X_seq[split:]
    y_tr, y_va = y_seq[:split], y_seq[split:]

    inputs = keras.Input(shape=(seq_len, len(feature_cols)))
    
    x = inputs
    filters = hp.Int('filters', 32, 128, step=16)
    dropout = hp.Float('dropout', 0.1, 0.5, step=0.1)
    
    # 5 Residual Blocks
    for _ in range(5):
        x = residual_block(x, filters, dropout=dropout)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
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
# 6. CUSTOM TUNER
# ----------------------------------------------------------------------
class ResNetTuner(kt.BayesianOptimization):
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
# 7. HYPERPARAMETER SEARCH
# ----------------------------------------------------------------------
print("\nStarting ResNet-5 hyperparameter search...")
tuner = ResNetTuner(
    hypermodel=build_resnet5,
    objective='val_loss',
    max_trials=MAX_TRIALS,
    directory=RESULTS_DIR,
    project_name='resnet5_tune',
    overwrite=True
)

tuner.search_space_summary()
tuner.search()

# ----------------------------------------------------------------------
# 8. RETRAIN BEST MODEL ON FULL DATA
# ----------------------------------------------------------------------
print("\nRetraining best ResNet-5 on full dataset...")
best_hp = tuner.get_best_hyperparameters(1)[0]
best_seq = best_hp.get('seq_length')

print(f"Best seq_length : {best_seq}")
print(f"Filters         : {best_hp.get('filters')}")
print(f"Dropout         : {best_hp.get('dropout'):.2f}")
print(f"Learning rate   : {best_hp.get('lr'):.2e}")

# Full sequences
X_full, y_full = create_sequences(X_scaled, y_scaled, best_seq)

# Build best model
final_model = build_resnet5(best_hp)[0]

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
            os.path.join(RESULTS_DIR, "best_resnet5.keras"),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
)

# ----------------------------------------------------------------------
# 9. SAVE MODEL (.keras & .h5)
# ----------------------------------------------------------------------
keras_path = os.path.join(RESULTS_DIR, "best_resnet5.keras")
h5_path = os.path.join(RESULTS_DIR, "best_resnet5.h5")
final_model.save(keras_path)
final_model.save(h5_path)
print(f"Model saved → {keras_path}")
print(f"Model saved → {h5_path}")

# ----------------------------------------------------------------------
# 10. PREDICTIONS & METRICS
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
# 11. PLOTS
# ----------------------------------------------------------------------
# --- Actual vs Predicted ---
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='Actual Close', alpha=0.8)
plt.plot(y_pred, label='Predicted Close', alpha=0.8)
plt.title('ResNet-5: Actual vs Predicted Next-Day Close Price')
plt.xlabel('Time Steps')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
pred_plot = os.path.join(RESULTS_DIR, "resnet5_actual_vs_pred.png")
plt.savefig(pred_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved: {pred_plot}")

# --- Residuals ---
residuals = y_true.flatten() - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('ResNet-5: Residuals')
plt.xlabel('Predicted Close ($)')
plt.ylabel('Residual ($)')
plt.grid(True, alpha=0.3)
resid_plot = os.path.join(RESULTS_DIR, "resnet5_residuals.png")
plt.savefig(resid_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Residuals plot saved: {resid_plot}")

# --- Loss Curves ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ResNet-5: Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
loss_plot = os.path.join(RESULTS_DIR, "resnet5_loss_curves.png")
plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Loss curve saved: {loss_plot}")

# ----------------------------------------------------------------------
# 12. SAVE SEQUENCES & STATS
# ----------------------------------------------------------------------
seq_dir = os.path.join(RESULTS_DIR, f"seq_{best_seq}")
os.makedirs(seq_dir, exist_ok=True)

np.save(os.path.join(seq_dir, "X.npy"), X_full)
np.save(os.path.join(seq_dir, "y.npy"), y_full)

with open(os.path.join(seq_dir, "hyperparameters.txt"), "w") as f:
    f.write(f"seq_length: {best_seq}\n")
    f.write(f"filters: {best_hp.get('filters')}\n")
    f.write(f"dropout: {best_hp.get('dropout')}\n")
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
# 13. FINAL SUMMARY
# ----------------------------------------------------------------------

print("RESNET-5 TRAINING COMPLETE")
print("="*60)
print(f"All results saved in:\n   {RESULTS_DIR}")
print("\nFiles:")
print("   best_resnet5.keras")
print("   best_resnet5.h5")
print("   scaler_X.pkl, scaler_y.pkl")
print("   predictions.csv")
print("   resnet5_actual_vs_pred.png")
print("   resnet5_residuals.png")
print("   resnet5_loss_curves.png")
print("   seq_XX/ (X.npy, y.npy, hyperparameters, features)")
