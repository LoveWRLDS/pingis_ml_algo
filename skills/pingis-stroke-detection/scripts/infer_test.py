"""
infer_test.py — Offline inference test

Run a saved model against a session file to verify predictions
before deploying to the mobile app.

Usage:
    # Test Random Forest model:
    python skills/pingis-stroke-detection/scripts/infer_test.py rf data/raw/session_2026-03-18_001.json

    # Test TFLite model:
    python skills/pingis-stroke-detection/scripts/infer_test.py tflite data/raw/session_2026-03-18_001.json
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path

WINDOW_SAMPLES = 40
N_CHANNELS = 6
MODEL_DIR = Path("data/models")


# ── Feature extraction (must match preprocess.py exactly) ────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract same features as preprocess.py. Returns 1D feature vector."""
    channels = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    features = []

    for i in range(6):
        x = window[:, i]
        features.extend([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.ptp(x),
            np.sqrt(np.mean(x ** 2)),
        ])

    accel_mag = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    features.extend([
        np.mean(accel_mag),
        np.max(accel_mag),
        np.sqrt(np.mean(accel_mag ** 2)),
        np.std(gyro_mag),
        np.max(gyro_mag),
    ])

    return np.array(features)


def samples_to_window(samples: list[dict]) -> np.ndarray | None:
    """Convert sample list to (WINDOW_SAMPLES, 6) array."""
    if len(samples) < WINDOW_SAMPLES:
        return None
    center = len(samples) // 2
    half = WINDOW_SAMPLES // 2
    window_samples = samples[center - half: center + half]
    try:
        return np.array([
            [s["accel_x"], s["accel_y"], s["accel_z"],
             s["gyro_x"], s["gyro_y"], s["gyro_z"]]
            for s in window_samples
        ], dtype=float)
    except KeyError:
        return None


# ── Random Forest inference ───────────────────────────────────────────────────

def infer_rf(session_file: Path):
    clf = joblib.load(MODEL_DIR / "rf_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")

    with open(session_file) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]

    print(f"\nRandom Forest inference on {session_file.name}")
    print(f"{'True Label':<15} {'Predicted':<15} {'Confidence':<12} {'Match'}")
    print("─" * 55)

    correct = 0
    total = 0
    for event in events:
        true_label = event.get("label", "?")
        window = samples_to_window(event.get("samples", []))
        if window is None:
            continue

        features = extract_features(window).reshape(1, -1)
        features_s = scaler.transform(features)
        proba = clf.predict_proba(features_s)[0]
        pred_idx = np.argmax(proba)
        pred_label = le.classes_[pred_idx]
        confidence = proba[pred_idx]
        match = "✓" if pred_label == true_label else "✗"

        if pred_label == true_label:
            correct += 1
        total += 1

        print(f"{true_label:<15} {pred_label:<15} {confidence:.3f}        {match}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")


# ── TFLite inference ──────────────────────────────────────────────────────────

def infer_tflite(session_file: Path):
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    tflite_path = MODEL_DIR / "stroke_cnn.tflite"
    mean = np.load(MODEL_DIR / "cnn_scaler_mean.npy")
    std = np.load(MODEL_DIR / "cnn_scaler_std.npy")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Label order from training (must match train_cnn.py LabelEncoder output)
    label_names = ["hit", "idle", "swing_miss"]  # alphabetical — LabelEncoder default
    print(f"(Assuming label order: {label_names} — verify against train_cnn.py output)")

    with open(session_file) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]

    print(f"\nTFLite inference on {session_file.name}")
    print(f"{'True Label':<15} {'Predicted':<15} {'Confidence':<12} {'Match'}")
    print("─" * 55)

    correct = 0
    total = 0
    for event in events:
        true_label = event.get("label", "?")
        window = samples_to_window(event.get("samples", []))
        if window is None:
            continue

        # Normalize
        window_n = (window - mean) / (std + 1e-8)
        input_tensor = window_n.astype(np.float32)[np.newaxis, ...]  # (1, 40, 6)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_idx = int(np.argmax(output))
        pred_label = label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx)
        confidence = float(output[pred_idx])
        match = "✓" if pred_label == true_label else "✗"

        if pred_label == true_label:
            correct += 1
        total += 1

        print(f"{true_label:<15} {pred_label:<15} {confidence:.3f}        {match}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer_test.py [rf|tflite] <session_file.json>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    session_file = Path(sys.argv[2])

    if not session_file.exists():
        print(f"File not found: {session_file}")
        sys.exit(1)

    if mode == "rf":
        infer_rf(session_file)
    elif mode == "tflite":
        infer_tflite(session_file)
    else:
        print(f"Unknown mode: {mode}. Use 'rf' or 'tflite'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
