# Pitfalls

These are the main failure modes uncovered while iterating on the `skatingbergs` project. Treat them as warnings, not supported alternatives.

## 1. Do Not Upgrade The AHRS Before Validating Inputs

A more advanced AHRS can perform worse than the baseline complementary filter if any of these are still uncertain:

- gyro scaling
- accelerometer scaling
- magnetometer sign or scale
- timing model
- sensor synchronization

In this project, more complicated fusion attempts became less stable before the input assumptions were nailed down.

## 2. Do Not Reintroduce Magnetometer Fusion Blindly

Magnetometer behavior was the easiest path to destabilizing orientation:

- wrong sign assumptions
- wrong scale assumptions
- unvalidated calibration
- incorrect axis remaps

The validated parser truth is:

```kotlin
x = -rawX / 10f
y = -rawY / 10f
z = -rawZ / 10f
```

If another app changes that mapping, it is no longer matching this baseline.

## 3. Do Not Assume Sensor Timestamps Drive The Baseline

The current project uses `receivedAtMs` for estimator `dt`, not the 24-bit sensor timestamp. If another app switches to a sensor-timestamp sync model, it may improve or worsen behavior, but it is no longer the same baseline.

## 4. Do Not Confuse Pose Calibration With Sensor Calibration

`Calibrate Pose` only stores the current quaternion as a reference orientation for display.

It does not:

- change BLE parsing
- remove gyro bias
- recalibrate accelerometer scale
- calibrate the magnetometer

## 5. Do Not Treat The Mesh Orientation As Sensor Truth

If the object appears rotated incorrectly:

- first verify the debug quaternion and raw sensor values
- then fix the renderer-side mesh correction

Do not compensate by changing BLE parsing or orientation math unless the sensor data is actually wrong.

## 6. Do Not Hide Debugging Too Early

The fastest way to verify integration is to keep:

- raw accel/gyro/mag values
- display quaternion
- debug Euler angles
- rolling charts

Removing debug views too early makes parser and axis mistakes harder to isolate.

## 7. Do Not Promise Better-Than-Baseline Accuracy

This skill is for reproducing the current validated baseline behavior. It does not include:

- a tuned production-grade AHRS
- full soft-iron ellipsoid fitting
- advanced timestamp synchronization
- guaranteed exact angle metrology
