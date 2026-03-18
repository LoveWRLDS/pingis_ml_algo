"""
serve_api.py — Flask inference API for real-time stroke detection

Runs a local HTTP server that accepts IMU windows and returns predictions.
Use this to test the trained model from your React Native app before
switching to on-device TFLite inference.

Usage:
    python skills/pingis-stroke-detection/scripts/serve_api.py

The server runs on http://localhost:5000

Endpoints:
    POST /predict          — predict label for a window of IMU samples
    GET  /health           — check server + model status
    GET  /labels           — get list of class names

Request format (POST /predict):
    {
      "samples": [
        {"accel_x": 412, "accel_y": -88, "accel_z": 980,
         "gyro_x": 12.3, "gyro_y": -4.1, "gyro_z": 88.0,
         "ts_ms": 1711234567890},
        ...
      ]
    }
    Minimum 20 samples, ideally 40 (800ms at 50Hz).

Response:
    {
      "label": "hit",
      "confidence": 0.91,
      "probabilities": {"hit": 0.91, "swing_miss": 0.07, "idle": 0.02}
    }

React Native integration:
    const response = await fetch('http://192.168.x.x:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: last40Samples }),
    });
    const result = await response.json();
    if (result.label === 'hit' && result.confidence > 0.8) { ... }

    Replace 192.168.x.x with your laptop's IP on the same WiFi network.
"""

import sys
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("data/models")
WINDOW_SAMPLES = 40
HOST = "0.0.0.0"   # listen on all interfaces so phone can reach it over WiFi
PORT = 5000

app = Flask(__name__)

# Loaded at startup
clf = None
scaler = None
le = None


# ── Feature extraction (must match preprocess.py exactly) ────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract same features as preprocess.py. Returns 1D feature vector."""
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
    gyro_mag  = np.linalg.norm(window[:, 3:6], axis=1)

    features.extend([
        np.mean(accel_mag),
        np.max(accel_mag),
        np.sqrt(np.mean(accel_mag ** 2)),
        np.std(gyro_mag),
        np.max(gyro_mag),
    ])

    return np.array(features)


def samples_to_window(samples: list[dict]) -> np.ndarray | None:
    """Convert list of sample dicts to (N, 6) float array. Returns None if too few."""
    if len(samples) < 10:
        return None

    rows = []
    for s in samples:
        try:
            rows.append([
                float(s["accel_x"]), float(s["accel_y"]), float(s["accel_z"]),
                float(s["gyro_x"]),  float(s["gyro_y"]),  float(s["gyro_z"]),
            ])
        except (KeyError, ValueError, TypeError):
            return None

    window = np.array(rows, dtype=float)

    # Crop or pad to WINDOW_SAMPLES
    if len(window) >= WINDOW_SAMPLES:
        # Take the center WINDOW_SAMPLES
        center = len(window) // 2
        half = WINDOW_SAMPLES // 2
        window = window[center - half: center + half]
    else:
        # Pad with zeros at end (not ideal but functional)
        pad = np.zeros((WINDOW_SAMPLES - len(window), 6))
        window = np.vstack([window, pad])

    return window


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": clf is not None,
        "model_type": "RandomForest" if clf is not None else None,
        "classes": list(le.classes_) if le is not None else None,
    })


@app.route("/labels", methods=["GET"])
def labels():
    if le is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({"labels": list(le.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded. Run train_rf.py first."}), 503

    data = request.get_json(silent=True)
    if not data or "samples" not in data:
        return jsonify({"error": "Request must have a 'samples' field"}), 400

    samples = data["samples"]
    window = samples_to_window(samples)
    if window is None:
        return jsonify({"error": f"Need at least 10 valid samples, got {len(samples)}"}), 400

    features = extract_features(window).reshape(1, -1)
    features_s = scaler.transform(features)

    proba = clf.predict_proba(features_s)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    probabilities = {label: float(p) for label, p in zip(le.classes_, proba)}

    return jsonify({
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
    })


# ── Startup ───────────────────────────────────────────────────────────────────

def load_model():
    global clf, scaler, le

    rf_path = MODEL_DIR / "rf_classifier.pkl"
    scaler_path = MODEL_DIR / "feature_scaler.pkl"
    le_path = MODEL_DIR / "label_encoder.pkl"

    if not rf_path.exists():
        print(f"⚠  Model not found at {rf_path}")
        print("   Run train_rf.py first to train a model.")
        print("   The server will still start — /predict will return 503 until you train.\n")
        return

    clf = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    print(f"✓  Loaded model: {rf_path}")
    print(f"   Classes: {list(le.classes_)}")


def print_startup_info():
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"

    print("═" * 55)
    print("  PINGIS INFERENCE API")
    print("═" * 55)
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{local_ip}:{PORT}")
    print()
    print("  → Use the network URL in your React Native app")
    print("    (phone and laptop must be on same WiFi)")
    print()
    print("  Endpoints:")
    print(f"    GET  /health   — model status")
    print(f"    POST /predict  — classify a stroke window")
    print("═" * 55)


if __name__ == "__main__":
    load_model()
    print_startup_info()
    app.run(host=HOST, port=PORT, debug=False)
