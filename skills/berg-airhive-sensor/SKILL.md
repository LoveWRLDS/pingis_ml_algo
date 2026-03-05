---
name: berg-airhive-sensor
description: "Exact implementation instructions for connecting to, streaming data from, and calibrating BERG AirHive BLE sensors (9-axis IMU with accelerometer + gyroscope). Use this skill when working with: AirHive, BLE sensor connection, Bluetooth Low Energy scanning, accelerometer/gyroscope data parsing, Madgwick AHRS sensor fusion, 6-point accelerometer calibration, IMU orientation tracking, pitch/roll/yaw calculation, or any task involving the BERG hardware platform."
---

# BERG AirHive Sensor Integration

Follow these instructions exactly when implementing BLE connection, data streaming, or calibration for the BERG AirHive sensors. Do not deviate from the specified UUIDs, byte formats, or mathematical formulas.

## 1. BLE Connection & Scanning

### 1.1 Required Android Permissions

```
ACCESS_FINE_LOCATION, ACCESS_COARSE_LOCATION
BLUETOOTH_SCAN, BLUETOOTH_CONNECT          (Android 12+ / API 31+)
FOREGROUND_SERVICE                          (Android 9+ / API 28+)
FOREGROUND_SERVICE_CONNECTED_DEVICE         (Android 14+ / API 34+)
```

### 1.2 Scanning

Use `BluetoothLeScanner`. Filter out anonymous devices — Android MAC cloaking floods results otherwise:

```kotlin
val name = result.scanRecord?.deviceName ?: device.name ?: "Unknown"
if (name != "Unknown") {
    // Add device to UI list
}
```

### 1.3 GATT Connection

The BLE connection **must** run inside a **Foreground Service** (`BleService`) to survive Doze mode and backgrounding.

**UUIDs (do not change):**

| Purpose | UUID |
|---|---|
| Service | `07c80000-07c8-07c8-07c8-07c807c807c8` |
| Accelerometer Characteristic | `07c80001-07c8-07c8-07c8-07c807c807c8` |
| Gyroscope Characteristic | `07c80004-07c8-07c8-07c8-07c807c807c8` |
| CCCD Descriptor | `00002902-0000-1000-8000-00805f9b34fb` |

**Connection flow:**

1. `device.connectGatt(context, false, gattCallback)`
2. `onConnectionStateChange` → `STATE_CONNECTED` → call `gatt.discoverServices()`. On `STATE_DISCONNECTED` → auto-reconnect with `gatt.connect()`.
3. `onServicesDiscovered`:
   - Request high priority: `gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)`
   - Find both Accel and Gyro characteristics
   - Enable notifications: `gatt.setCharacteristicNotification(char, true)`
   - Write CCCD descriptor: `descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE; gatt.writeDescriptor(descriptor)`
   - **Queue Gyro descriptor write until Accel descriptor write completes** (`onDescriptorWrite` callback) — overlapping writes will fail silently.

## 2. Data Parsing

Both payloads are **9 bytes, BIG_ENDIAN**.

### 2.1 Accelerometer

| Bytes | Field | Type |
|---|---|---|
| 0–1 | X axis | Int16 signed |
| 2–3 | Y axis | Int16 signed |
| 4–5 | Z axis | Int16 signed |
| 6–8 | Timestamp | UInt24 unsigned |

```kotlin
val buffer = ByteBuffer.wrap(data).order(ByteOrder.BIG_ENDIAN)
val x = buffer.short.toInt()
val y = buffer.short.toInt()
val z = buffer.short.toInt()
val b0 = data[6].toInt() and 0xFF
val b1 = data[7].toInt() and 0xFF
val b2 = data[8].toInt() and 0xFF
val ts = (b0 shl 16) or (b1 shl 8) or b2
```

### 2.2 Gyroscope

Same byte layout. Max range: ±2000 dps.

```kotlin
val buffer = ByteBuffer.wrap(data).order(ByteOrder.BIG_ENDIAN)
val gyroX = buffer.short.toFloat()
val gyroY = buffer.short.toFloat()
val gyroZ = buffer.short.toFloat()
```

### 2.3 Sensor Fusion (Madgwick AHRS)

Feed Accel + Gyro into a **Madgwick AHRS** algorithm to compute Pitch, Roll, Yaw.

**Critical axis remapping** (PCB orientation differs from algorithm expectations):

| Algorithm Input | Raw Source |
|---|---|
| AccelX | `rawX` |
| AccelY | `-rawZ` |
| AccelZ | `rawY` |
| GyroX | `rawGyroX` |
| GyroY | `-rawGyroZ` |
| GyroZ | `rawGyroY` |

Convert Gyro values to **radians/second** before feeding into Madgwick.

## 3. 6-Point Calibration

Normalizes per-sensor hardware variations by computing bias (offset) and gain (scale).

### 3.1 The 6 Poses

Guide the user to hold the sensor still in each orientation. Target gravity vector (Norm = 1000 mg):

| Step | Orientation | Target (X, Y, Z) |
|---|---|---|
| 1 | +X Up | `(1000, 0, 0)` |
| 2 | −X Up | `(-1000, 0, 0)` |
| 3 | +Y Up | `(0, 1000, 0)` |
| 4 | −Y Up | `(0, -1000, 0)` |
| 5 | +Z Up (flat) | `(0, 0, 1000)` |
| 6 | −Z Up (upside down) | `(0, 0, -1000)` |

### 3.2 Stability & Direction Gates

Use a sliding window of ~50 samples (~500ms at high freq). Both gates must pass before capturing:

- **Direction gate:** Dot product between sensor's normalized average vector and target vector must be `> 0.866` (within ~30° of perfect axis).
- **Stability gate:** StdDev of X, Y, Z within window must be `< 30f`. Magnitude deviation from 1g must be `< 200f`.

### 3.3 Capturing

Once stable, capture exactly **1 second** of raw data. Compute a **10% Trimmed Mean** for each axis (discards outliers).

### 3.4 Computing Bias & Scale

After collecting 6 trimmed means:

```kotlin
// Bias (offset)
val bx = (axPos + axNeg) / 2f
val by = (ayPos + ayNeg) / 2f
val bz = (azPos + azNeg) / 2f

// Scale (gain)
val sx = 2000f / (axPos - axNeg)
val sy = 2000f / (ayPos - ayNeg)
val sz = 2000f / (azPos - azNeg)
```

### 3.5 Applying Calibration

Apply to all future raw readings:

```kotlin
val calibratedX = scaleX * (rawX - biasX)
val calibratedY = scaleY * (rawY - biasY)
val calibratedZ = scaleZ * (rawZ - biasZ)
```

Pass calibrated values downstream into graphing and pose state machines.
