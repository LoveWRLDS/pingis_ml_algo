# Orientation Baseline

Use this reference to reproduce the current validated orientation behavior from the `skatingbergs` baseline.

## Core Contracts

- Represent orientation internally as quaternions.
- Use Euler angles only for debug display.
- Keep the estimator and pose calibration independent from the renderer.

## `ImuSample`

Use an `ImuSample` equivalent to:

```kotlin
data class ImuSample(
    val accelerometer: Vector3f = Vector3f.Zero,
    val gyroscopeDps: Vector3f = Vector3f.Zero,
    val magnetometerUt: Vector3f = Vector3f.Zero,
    val sensorTimestamp: Int = 0,
    val receivedAtMs: Long = System.currentTimeMillis()
)
```

## Estimator Type

The baseline estimator is a quaternion complementary filter, not a full AHRS.

Keep these behaviors exact:

- integrate gyro each update
- compute a measured orientation from accel + mag
- slerp predicted orientation toward measured orientation
- enforce quaternion sign continuity

## Estimator Parameters

Use the current baseline constants:

- first-frame `dt = 1 / 50`
- later `dt = clamp((receivedAtMs - lastReceivedAtMs) / 1000, 0.001, 0.05)`
- accel low-pass alpha: `0.15`
- mag low-pass alpha: `0.10`
- correction alpha: `1 - exp(-dt / 0.35)`
- gyro integration deadband: `angularSpeed < 1e-5 -> no delta rotation`

## Frame Assumptions

The baseline estimator currently uses the parsed vectors directly:

```kotlin
accel = sample.accelerometer
gyro = sample.gyroscopeDps
mag = sample.magnetometerUt
```

There is no extra remap layer in the current estimator. If another app needs to match the baseline behavior, preserve this.

## Gyro Integration

Convert gyro to radians/sec inside the estimator:

```kotlin
val gyroRad = Vector3f(
    x = Math.toRadians(gyroDps.x.toDouble()).toFloat(),
    y = Math.toRadians(gyroDps.y.toDouble()).toFloat(),
    z = Math.toRadians(gyroDps.z.toDouble()).toFloat()
)
```

Integrate by axis-angle:

```kotlin
val angularSpeed = gyroRad.magnitude()
val delta = Quaternionf.fromAxisAngle(gyroRad / angularSpeed, angularSpeed * dt)
val predicted = (current * delta).normalized()
```

## Measured Orientation

Derive measured orientation from gravity and horizontal magnetic field:

1. Normalize accel to get `up`.
2. Remove the `up` component from magnetometer to get horizontal magnetic field.
3. Build `east = horizontalMag.cross(up).normalized()`.
4. Build `north = up.cross(east).normalized()`.
5. Build a rotation matrix from `east`, `north`, and `up`.
6. Convert the matrix to a quaternion and invert it to get the sensor orientation.

If any normalized vector collapses to zero, fall back to the gyro-predicted orientation.

## Complementary Correction

Blend predicted orientation toward measured orientation:

```kotlin
val orientation = predicted.slerp(measured, correctionAlpha)
```

Then keep quaternion continuity:

```kotlin
if (orientation.dot(previous) < 0f) {
    orientation = orientation.negated()
}
```

## Pose Calibration

Pose calibration is a display-space reference offset, not a sensor recalibration.

### Calibrate

- Store the current absolute orientation as `referenceOrientation`.
- Set display orientation to identity immediately.

### Apply

Render:

```kotlin
displayOrientation = inverse(referenceOrientation) * absoluteOrientation
```

If no reference exists:

```kotlin
displayOrientation = absoluteOrientation
```

### Reset Reference

- Clear `referenceOrientation`
- Keep the estimator state intact
- Resume displaying the absolute orientation directly

## Rolling Chart Baseline

Use a rolling 10 second window of emitted `ImuSample`s:

```text
append new sample -> drop any sample with receivedAtMs < latestReceivedAtMs - 10_000
```

Charts are optional in the destination app, but recommended during integration because they expose parser mistakes, axis mistakes, and calibration mistakes quickly.

## Acceptance Examples

- Hold still: the 3D object should remain stable with small jitter only.
- Press calibrate pose while holding a neutral pose: the displayed quaternion becomes identity and the object treats that pose as zero.
- Rotate around one axis after calibration: the object should rotate relative to that calibrated pose.
- Replace the shoe renderer with another mesh: orientation behavior should remain the same if the renderer consumes the same display quaternion.
