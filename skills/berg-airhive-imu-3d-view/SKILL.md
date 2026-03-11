---
name: berg-airhive-imu-3d-view
description: Integrate BERG AirHive BLE IMU sensors into an app, parse the streamed accelerometer/gyroscope/magnetometer packets, reproduce the skatingbergs project's baseline quaternion orientation behavior, add pose calibration, plot live IMU charts, and drive any 3D object from the same orientation pipeline. Use when building or porting BERG AirHive sensor support, matching the current baseline accuracy/behavior, or separating the sensor/orientation stack from a project-specific 3D model.
---

# BERG AirHive IMU 3D View

Use this skill to reproduce the current validated BERG AirHive sensor pipeline from the `skatingbergs` project in another app. Treat the baseline as a sensor contract plus an orientation contract:

- `BLE stream -> parsed IMU sample -> orientation estimator -> display quaternion -> renderer`
- Keep the sensor protocol exact.
- Keep the quaternion math exact.
- Treat the rendered object as replaceable.

This skill matches the current project baseline. It does not promise a more advanced AHRS, full magnetometer compensation, or better-than-baseline angular metrology.

## Quick Start

1. Read [references/sensor-protocol.md](references/sensor-protocol.md) to implement scanning, GATT subscription, and packet parsing.
2. Read [references/orientation-baseline.md](references/orientation-baseline.md) to reproduce the current quaternion estimator, pose calibration, and chart behavior.
3. Read [references/renderer-adapter.md](references/renderer-adapter.md) to connect the output quaternion to any 3D object.
4. Read [references/pitfalls.md](references/pitfalls.md) before changing fusion, timing, or magnetometer handling.

## Default Workflow

### 1. Implement the sensor contract

Implement the BERG AirHive BLE service and the three notification characteristics exactly as documented in [references/sensor-protocol.md](references/sensor-protocol.md).

Use the validated parser truth for magnetometer values:

```kotlin
Vector3f(
    x = -rawX / 10f,
    y = -rawY / 10f,
    z = -rawZ / 10f
)
```

### 2. Implement the baseline orientation stack

Define these public contracts in the target app:

```kotlin
data class ImuSample(
    val accelerometer: Vector3f,
    val gyroscopeDps: Vector3f,
    val magnetometerUt: Vector3f,
    val sensorTimestamp: Int,
    val receivedAtMs: Long
)

interface OrientationEstimator {
    fun reset()
    fun update(sample: ImuSample): Quaternionf
}

interface PoseCalibration {
    fun calibrate(currentQuaternion: Quaternionf)
    fun reset()
    fun apply(absoluteQuaternion: Quaternionf): Quaternionf
}

interface RendererAdapter {
    fun render(displayQuaternion: Quaternionf)
}
```

Then reproduce the baseline estimator and pose-calibration behavior from [references/orientation-baseline.md](references/orientation-baseline.md).

### 3. Connect the quaternion to the target renderer

Do not bake the renderer shape into the estimator. Pass the display quaternion into a renderer adapter and let the target stack convert that quaternion into the engine's transform representation. See [references/renderer-adapter.md](references/renderer-adapter.md).

### 4. Keep the debug path

Even if the destination app hides them later, implement raw sensor readouts and rolling charts during integration. The baseline behavior is easier to verify when the app shows:

- raw accel, gyro, and mag vectors
- sensor timestamp
- display quaternion
- display Euler angles for debugging only
- a rolling 10 second chart window

## Baseline Boundaries

Keep these baseline rules unless the user explicitly asks to change them:

- Use the current complementary filter as the reference estimator.
- Use `receivedAtMs` for `dt`, not the sensor timestamp.
- Emit one `ImuSample` every time any characteristic updates, using the latest known values from the other sensors.
- Keep pose calibration as a pure reference quaternion offset.
- Keep the renderer object-agnostic.

Do not silently upgrade the pipeline to a different AHRS or a different timing model.
