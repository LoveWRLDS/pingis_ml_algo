# Model Selection Reference

## Decision Guide

| Condition | Use |
|-----------|-----|
| < 200 samples | Random Forest on engineered features |
| 200-500 samples | Random Forest, consider CNN if accuracy plateaus |
| 500+ samples | 1D CNN → TFLite for on-device inference |
| Need to debug features | Random Forest (has feature_importances_) |
| Production mobile (offline) | 1D CNN TFLite |

---

## Phase 1: Random Forest

**Why start here:**
- 5 lines of scikit-learn code
- Works reliably with 50-200 samples
- `feature_importances_` tells you which sensor axes and features matter most
- No GPU required — trains in seconds on a laptop
- Interpretable: you can understand *why* it classifies something as a hit

**Expected accuracy** with 200 balanced samples: 85-95%

**Deployment**: Flask/FastAPI server API on your laptop or a VM. React Native app sends 800ms window JSON to `POST /predict`, receives `{label, confidence}`.

**Limitations**:
- Cannot do on-device TFLite inference natively (needs ONNX conversion or manual port)
- Accuracy plateaus faster than CNN with more data

**Code**: `scripts/train_rf.py`

---

## Phase 2: 1D Convolutional Neural Network

**Why use this next:**
- Reads raw time-series directly — no manual feature engineering
- Learns patterns you didn't think of (e.g., subtle pre-impact motion cues)
- Exports to `.tflite` for on-device mobile inference (no network needed)
- Relatively fast to train even on CPU (~5-15 minutes for 500 samples)
- Well-documented — hundreds of tutorials for time-series classification

**Architecture** (start small, scale up if needed):
```
Input: (40, 6)  ← 40 time steps, 6 channels (accel + gyro)
Conv1D(32, kernel_size=3, activation='relu')
Conv1D(64, kernel_size=3, activation='relu')
GlobalAveragePooling1D()
Dense(32, activation='relu')
Dropout(0.3)
Dense(3, activation='softmax')  ← hit / swing_miss / idle
```

**Expected accuracy** with 500+ balanced samples: 90-97%

**Code**: `scripts/train_cnn.py`

---

## What NOT to use (and why)

### LSTM / GRU
- Requires more data (1000+ per class) to outperform CNN
- Harder to debug
- Slower to train without GPU
- Harder to convert to TFLite
- Only beats CNN when temporal dependencies span multiple seconds — not your case

### Transformer (e.g., ViT for time series)
- Massively overengineered for this task
- Requires thousands of samples
- Use only if CNN plateaus and you have 5000+ samples

### Raw threshold rules (e.g., "if accel > 500, then hit")
- Brittle — different players, paddle weights, grip styles change thresholds
- Cannot distinguish forehand from backhand
- Use only as a sanity check baseline to compare against

---

## Evaluation Metrics

Always use these, never accuracy alone:

```
              precision    recall  f1-score   support
         hit       0.91      0.88      0.89        40
  swing_miss       0.87      0.93      0.90        40
        idle       0.98      0.95      0.96        40
    accuracy                           0.92       120
```

**Key terms for a beginner:**
- **Precision**: Of everything the model called "hit", how many were actually hits? (avoids false alarms)
- **Recall**: Of all actual hits, how many did the model catch? (avoids missing hits)
- **F1-score**: Harmonic mean of precision and recall — the main number to optimize
- **Confusion matrix**: Shows which classes get confused with which

**Acceptable starting targets:**
- Hit detection F1 > 0.85 before proceeding to stroke type classification
- Aim for recall > 0.90 for `hit` (missing a hit is worse than a false alarm)

---

## On-device Inference: TFLite

When the CNN is trained, export with:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('data/models/stroke_cnn.tflite', 'wb') as f:
    f.write(tflite_model)
```

React Native package: `react-native-fast-tflite`

Input must be normalized with the same `StandardScaler` used during training.
Store mean/std arrays in the app (bake them in from `scaler.mean_` and `scaler.scale_`).

Model size estimate: ~50-200KB for this architecture. Entirely acceptable to bundle in the app.
