# Feature Engineering Reference

## Window Parameters
- **Window size**: 40 samples (~800ms at 50Hz)
- **Channels**: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z (skip mag for ML — too noisy)
- **Phase 1**: Extract hand-crafted features below → ~42 features total
- **Phase 2**: Feed raw 40×6 window directly into 1D CNN

---

## Phase 1 Features (Random Forest input)

### Per-axis time-domain features
Computed for each of the 6 channels (accel_x/y/z, gyro_x/y/z):

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| Mean | `mean(x)` | Offset / gravity component |
| Std | `std(x)` | Overall energy / variability |
| Min | `min(x)` | Minimum excursion |
| Max | `max(x)` | Maximum excursion |
| Peak-to-peak | `max(x) - min(x)` | Range of motion |
| RMS | `sqrt(mean(x²))` | Signal energy — **spikes on hit contact** |

6 channels × 6 features = **36 features**

### Magnitude features (channel-independent)
Computed on the vector magnitude, removing orientation dependence:

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| Accel magnitude RMS | `mean(sqrt(ax²+ay²+az²))` | Total acceleration energy |
| Accel magnitude peak | `max(sqrt(ax²+ay²+az²))` | Peak impact force |
| Gyro magnitude std | `std(sqrt(gx²+gy²+gz²))` | Total angular velocity variation |

= **3 features**

### Cross-axis features (add in iteration 2)
| Feature | Formula | Why it matters |
|---------|---------|----------------|
| Accel-Gyro correlation | `corr(accel_mag, gyro_mag)` | Forehand vs backhand have different coupling |
| Gyro Z sign at peak | `sign(gz[argmax(accel_mag)])` | Rotation direction differs by stroke type |

= **2 features** (optional, add when investigating forehand/backhand)

**Total Phase 1: ~42 features**

---

## Phase 2 Features (1D CNN — no manual features needed)

Input tensor shape: `(40, 6)` — 40 time steps, 6 channels (accel + gyro, no mag)

Normalize each channel to zero mean, unit variance using `StandardScaler` fitted on training data only. Save the scaler as `data/models/feature_scaler.pkl` — must be applied identically at inference time.

---

## The Key Physical Signal: Hit Impulse

Ball contact with paddle creates a 2-5ms mechanical impulse transmitted through the handle.

At 50Hz (20ms per sample), this appears as:
- **1-2 sample spike** in accelerometer magnitude, ~2-4× higher than swing motion
- Gyroscope also spikes but ~1 sample later (handle rotates after impact)
- High-frequency content (5-20Hz band) in accelerometer — detectable via FFT

**Most discriminating features:**
1. `accel_magnitude_peak` — highest for hit, moderate for swing
2. `accel_rms` per axis — elevated around contact
3. `gyro_std` per axis — roughly similar (swing dominates both hit and miss)

**Forehand vs Backhand discrimination:**
1. `gyro_z` sign at peak accel — forehand: positive yaw rotation, backhand: negative (device-orientation-dependent, verify empirically)
2. Temporal ordering of `accel_x` vs `accel_y` peaks — trajectory differs per stroke direction

---

## Frequency Domain Features (Phase 1, iteration 2)

Apply FFT to each channel's 40-sample window. Compute energy in bands:

| Band | Range | Signal |
|------|-------|--------|
| Low | 0-5Hz | Gross arm motion (similar for hit and miss) |
| Mid | 5-20Hz | Wrist snap + contact vibration **(hit > miss here)** |
| High | 20+ Hz | Noise floor |

```python
import numpy as np

def fft_band_energy(signal, fs=50.0, low=5.0, high=20.0):
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(signal))
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(fft_mag[mask] ** 2)
```

Add one `fft_band_energy_5_20hz` feature per accel channel = 3 extra features.

---

## Feature Scaling

Always apply `StandardScaler` from scikit-learn:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use fit from train only!
```

Save the fitted scaler alongside the model — must be loaded and applied at inference time.
