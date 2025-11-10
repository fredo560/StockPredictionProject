# ================================
#  LSTM – Predict Next-Day Close Price
#  (Keras + Keras-Tuner + training/validation curves)
# ================================

import os
import joblib
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
CSV_PATH = r"C:\Users\alfre\Desktop\stockProject\processed_amazon_data.csv"

# ---- create a timestamped results folder --------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = r"C:\Users\alfre\Desktop\stockProject"
RESULTS_DIR = os.path.join(RESULTS_ROOT, f"lstm_close_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_TRIALS   = 50
EPOCHS_TUNE  = 50
EPOCHS_FINAL = 120
BATCH_SIZE   = 32


# ----------------------------------------------------------------------
# 2. LOAD & PREPARE DATA
# ----------------------------------------------------------------------
print("\nLoading data …")
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---- features (Close is removed – we will predict it) -----------------
feature_cols = [
    "High", "Low", "Open", "Volume",
    "HL_Pct", "OC_Pct", "SMA_10", "SMA_30",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "Volume_SMA_10", "Volume_Ratio",
    "Return_Lag_1", "Return_Lag_3", "Return_Lag_5"
]

X_raw = df[feature_cols].values
y_raw = df["Close"].values.reshape(-1, 1)          # <-- TARGET = next-day Close

# ---- scaling ---------------------------------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw).flatten()   # 1-D for sequences

# ---- save scalers ----------------------------------------------------
joblib.dump(scaler_X, os.path.join(RESULTS_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(RESULTS_DIR, "scaler_y.pkl"))
print(f"Scalers saved in {RESULTS_DIR}")

print(f"X shape: {X_scaled.shape}   y shape: {y_scaled.shape}")


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
# 4. MODEL BUILDER (used by Keras-Tuner)
# ----------------------------------------------------------------------
def build_model(hp):
    seq_len = hp.Int("seq_length", min_value=20, max_value=120, step=10)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)

    split = int(len(X_seq) * 0.8)
    X_tr, X_va = X_seq[:split], X_seq[split:]
    y_tr, y_va = y_seq[:split], y_seq[split:]

    model = keras.Sequential([
        layers.LSTM(
            hp.Int("units_1", 32, 128, step=16),
            return_sequences=True,
            input_shape=(seq_len, X_seq.shape[2])
        ),
        layers.Dropout(hp.Float("drop_1", 0.1, 0.5, step=0.1)),
        layers.LSTM(hp.Int("units_2", 16, 64, step=16)),
        layers.Dropout(hp.Float("drop_2", 0.1, 0.5, step=0.1)),
        layers.Dense(25, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-2, sampling="log")
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model, X_tr, y_tr, X_va, y_va, seq_len


# ----------------------------------------------------------------------
# 5. CUSTOM TUNER (early-stop inside each trial)
# ----------------------------------------------------------------------
class LSTMTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model, X_tr, y_tr, X_va, y_va, _ = self.hypermodel.build(hp)

        hist = model.fit(
            X_tr, y_tr,
            epochs=EPOCHS_TUNE,
            batch_size=BATCH_SIZE,
            validation_data=(X_va, y_va),
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10,
                                                    restore_best_weights=True)]
        )
        val_loss = min(hist.history["val_loss"])
        self.oracle.update_trial(trial.trial_id, {"val_loss": val_loss})
        return val_loss


# ----------------------------------------------------------------------
# 6. HYPER-PARAMETER SEARCH
# ----------------------------------------------------------------------
print("\nStarting hyper-parameter search …")
tuner = LSTMTuner(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=MAX_TRIALS,
    directory=RESULTS_DIR,
    project_name="tune_close",
    overwrite=True
)

tuner.search_space_summary()
tuner.search()


# ----------------------------------------------------------------------
# 7. RETRIEVE BEST HYPER-PARAMETERS & RE-TRAIN ON FULL DATA
# ----------------------------------------------------------------------
print("\nTuning finished – retraining best model …")
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_seq = best_hp.get("seq_length")

print(f"Best seq_length : {best_seq}")
print(f"Units          : {best_hp.get('units_1')}, {best_hp.get('units_2')}")
print(f"Dropout        : {best_hp.get('drop_1'):.2f}, {best_hp.get('drop_2'):.2f}")
print(f"Learning rate  : {best_hp.get('lr'):.2e}")

# ---- full sequences ---------------------------------------------------
X_full, y_full = create_sequences(X_scaled, y_scaled, best_seq)

# ---- build the *exact* best model (no train/val split needed) ----------
best_model = build_model(best_hp)[0]      # only the model object

# ---- final training ---------------------------------------------------
history = best_model.fit(
    X_full, y_full,
    epochs=EPOCHS_FINAL,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, "best_lstm_close.keras"),
            save_best_only=True,
            monitor="val_loss"
        )
    ]
)


# ----------------------------------------------------------------------
# 8. SAVE MODEL (both .keras & .h5)
# ----------------------------------------------------------------------
keras_path = os.path.join(RESULTS_DIR, "best_lstm_close.keras")
h5_path    = os.path.join(RESULTS_DIR, "best_lstm_close.h5")

best_model.save(keras_path)
best_model.save(h5_path)
print(f"\nModel saved → {keras_path}")
print(f"Model saved → {h5_path}")


# ----------------------------------------------------------------------
# 9. PREDICTIONS & METRICS
# ----------------------------------------------------------------------
y_pred_scaled = best_model.predict(X_full)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_full.reshape(-1, 1))

mse = mean_squared_error(y_true, y_pred)
print(f"\nMSE (Close price) : {mse:,.4f}")


# ----------------------------------------------------------------------
# 10. PLOTS
# ----------------------------------------------------------------------
# ---- 10a) Actual vs Predicted Close ----------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_true, label="Actual Close", alpha=0.8)
plt.plot(y_pred, label="Predicted Close", alpha=0.8)
plt.title("LSTM – Next-Day Close Price")
plt.xlabel("Time step")
plt.ylabel("Close Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
close_plot = os.path.join(RESULTS_DIR, "close_actual_vs_pred.png")
plt.savefig(close_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"Close-plot saved → {close_plot}")

# ---- 10b) Training / Validation loss curves -------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)
loss_plot = os.path.join(RESULTS_DIR, "loss_curves.png")
plt.savefig(loss_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"Loss-curve plot saved → {loss_plot}")


# ----------------------------------------------------------------------
# 11. SAVE SEQUENCES + HYPER-PARAMETERS
# ----------------------------------------------------------------------
seq_dir = os.path.join(RESULTS_DIR, f"seq_{best_seq}")
os.makedirs(seq_dir, exist_ok=True)

np.save(os.path.join(seq_dir, "X.npy"), X_full)
np.save(os.path.join(seq_dir, "y.npy"), y_full)

with open(os.path.join(seq_dir, "hyperparameters.txt"), "w") as f:
    f.write(f"seq_length: {best_seq}\n")
    f.write(f"units_1: {best_hp.get('units_1')}\n")
    f.write(f"units_2: {best_hp.get('units_2')}\n")
    f.write(f"dropout_1: {best_hp.get('drop_1')}\n")
    f.write(f"dropout_2: {best_hp.get('drop_2')}\n")
    f.write(f"learning_rate: {best_hp.get('lr')}\n")
    f.write(f"mse: {mse:,.4f}\n")

with open(os.path.join(seq_dir, "features.txt"), "w") as f:
    for c in feature_cols:
        f.write(c + "\n")

print(f"\nAll artifacts are in:\n   {RESULTS_DIR}")
print("  seq_{best_seq}/  (X.npy, y.npy, hyperparams, features)")