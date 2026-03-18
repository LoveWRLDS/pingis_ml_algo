"""
train_cnn.py — Phase 2: 1D CNN for on-device TFLite inference

Trains a small 1D Convolutional Neural Network on raw windowed IMU data.
Exports the trained model to .tflite for React Native on-device inference.

Run this when you have 500+ labeled samples (200+ per class).

Usage:
    python skills/pingis-stroke-detection/scripts/train_cnn.py

Output:
    data/models/stroke_cnn.tflite   — on-device model
    data/models/cnn_scaler_mean.npy — normalization mean (bake into app)
    data/models/cnn_scaler_std.npy  — normalization std (bake into app)

Dependencies:
    pip install tensorflow numpy pandas scikit-learn matplotlib
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

RAW_DIR = Path("data/raw")
MODEL_DIR = Path("data/models")

WINDOW_SAMPLES = 40   # 800ms at 50Hz
N_CHANNELS = 6        # accel_x/y/z + gyro_x/y/z (no mag)
TARGET_LABELS = ["hit", "swing_miss", "idle"]
EPOCHS = 50
BATCH_SIZE = 16


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_windows() -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load raw session JSON files and return (X, y, label_encoder)."""
    session_files = sorted(RAW_DIR.glob("*.json"))
    if not session_files:
        print(f"No session files in {RAW_DIR}/. Run data collection first.")
        sys.exit(1)

    all_windows = []
    all_labels = []

    for filepath in session_files:
        with open(filepath) as f:
            data = json.load(f)
        events = data if isinstance(data, list) else [data]

        for event in events:
            label = event.get("label")
            if label not in TARGET_LABELS:
                continue

            samples = event.get("samples", [])
            if len(samples) < WINDOW_SAMPLES:
                continue

            center = len(samples) // 2
            half = WINDOW_SAMPLES // 2
            window_samples = samples[center - half: center + half]

            try:
                window = np.array([
                    [s["accel_x"], s["accel_y"], s["accel_z"],
                     s["gyro_x"], s["gyro_y"], s["gyro_z"]]
                    for s in window_samples
                ], dtype=float)
            except KeyError:
                continue

            if window.shape != (WINDOW_SAMPLES, N_CHANNELS):
                continue

            all_windows.append(window)
            all_labels.append(label)

    if len(all_windows) < 30:
        print(f"Only {len(all_windows)} windows found. Need at least 30.")
        sys.exit(1)

    X = np.array(all_windows)  # (N, 40, 6)
    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    print(f"\nLoaded {len(X)} windows")
    for label, count in zip(*np.unique(all_labels, return_counts=True)):
        print(f"  {label}: {count}")

    return X, y, le


def normalize(X_train, X_test):
    """Normalize per channel using training set statistics."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, 6)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean.squeeze(), std.squeeze()


# ── Model architecture ────────────────────────────────────────────────────────

def build_model(n_classes: int) -> keras.Model:
    """Small 1D CNN for time-series classification."""
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SAMPLES, N_CHANNELS)),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_n, X_test_n, mean, std = normalize(X_train, X_test)

    model = build_model(n_classes=len(le.classes_))
    model.summary()

    history = model.fit(
        X_train_n, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ],
        verbose=1,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test_n), axis=1)
    print("\n── Test Set Results ──────────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Plot training curves + confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=axes[2], colorbar=False)
    axes[2].set_title("Confusion Matrix (test)")

    plt.tight_layout()
    plt.show()

    return model, mean, std


# ── Export ────────────────────────────────────────────────────────────────────

def export_tflite(model, mean, std):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save normalization params (bake these into the React Native app)
    np.save(MODEL_DIR / "cnn_scaler_mean.npy", mean)
    np.save(MODEL_DIR / "cnn_scaler_std.npy", std)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = MODEL_DIR / "stroke_cnn.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = tflite_path.stat().st_size / 1024
    print(f"\n✓ TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")
    print(f"✓ Normalization mean saved: {MODEL_DIR}/cnn_scaler_mean.npy")
    print(f"✓ Normalization std saved:  {MODEL_DIR}/cnn_scaler_std.npy")
    print("\nNext steps:")
    print("  1. Copy stroke_cnn.tflite to your React Native app's assets/")
    print("  2. Hardcode the mean/std arrays from cnn_scaler_mean.npy into your inference code")
    print("  3. Use react-native-fast-tflite to run inference on-device")


def main():
    X, y, le = load_raw_windows()
    model, mean, std = train_model(X, y, le)
    export_tflite(model, mean, std)

    # Print label mapping for reference in the app
    print("\nLabel mapping (index → class name):")
    for i, name in enumerate(le.classes_):
        print(f"  {i} → {name}")


if __name__ == "__main__":
    main()
