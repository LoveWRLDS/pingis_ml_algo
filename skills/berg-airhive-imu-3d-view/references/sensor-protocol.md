# Sensor Protocol

Use this reference to reproduce the BERG AirHive BLE and packet parsing contract exactly.

## BLE UUIDs

- Service: `07c80000-07c8-07c8-07c8-07c807c807c8`
- Accelerometer characteristic: `07c80001-07c8-07c8-07c8-07c807c807c8`
- Alternate accelerometer characteristic: `07c80203-07c8-07c8-07c8-07c807c807c8`
- Gyroscope characteristic: `07c80004-07c8-07c8-07c8-07c807c807c8`
- Magnetometer characteristic: `07c80010-07c8-07c8-07c8-07c807c807c8`
- CCCD: `00002902-0000-1000-8000-00805f9b34fb`

## Android BLE Baseline

Use Android BLE as the primary concrete example, because that is the validated implementation.

### Scan behavior

- Use `BluetoothLeScanner`.
- Filter out devices whose resolved name is `"Unknown"`.
- Keep a list of discovered named devices sorted by RSSI.

Example name resolution:

```kotlin
val name = result.scanRecord?.deviceName ?: result.device.name ?: "Unknown"
if (name == "Unknown") return
```

### Connection behavior

- Maintain the connection through a foreground service or equivalent background-stable transport owner.
- Connect with `connectGatt(context, false, callback)` or the LE transport overload on newer Android.
- On `STATE_CONNECTED`, call `discoverServices()`.
- On disconnect, attempt reconnect while the connection is still allowed by the app state.
- Request high connection priority after services are discovered.

### Notification behavior

- Subscribe to accelerometer, gyroscope, and magnetometer notifications.
- Enable local notifications with `setCharacteristicNotification`.
- Queue CCCD writes one at a time. Do not overlap descriptor writes.

## Packet Layout

All three streams use the same 9-byte payload format and big-endian axis values.

- Bytes `0..1`: signed `Int16` X
- Bytes `2..3`: signed `Int16` Y
- Bytes `4..5`: signed `Int16` Z
- Bytes `6..8`: unsigned 24-bit timestamp

Example parser:

```kotlin
val buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
val x = buffer.short.toFloat()
val y = buffer.short.toFloat()
val z = buffer.short.toFloat()

val b0 = payload[6].toInt() and 0xFF
val b1 = payload[7].toInt() and 0xFF
val b2 = payload[8].toInt() and 0xFF
val timestamp = (b0 shl 16) or (b1 shl 8) or b2
```

## Unit Contract

### Accelerometer

- Parse as signed big-endian vector.
- Keep the values as the baseline app uses them.
- The baseline app treats these values as already usable physical-ish values and charts them directly.

### Gyroscope

- Parse as signed big-endian vector.
- Treat values as degrees per second in the baseline app.
- Convert to radians per second only inside the orientation estimator.

### Magnetometer

This mapping is user-verified and must be treated as the canonical parser truth for reproducing the current project behavior:

```kotlin
Vector3f(
    x = -rawX / 10f,
    y = -rawY / 10f,
    z = -rawZ / 10f
)
```

Implications:

- invert all three raw magnetometer axes
- divide by `10` to express the baseline values in microtesla
- do not add further axis swaps in the parser

## Sample Emission Model

The baseline app does not synchronize packets across sensor timestamps. It emits a new `ImuSample` every time any one characteristic changes:

- update the latest vector for the reporting characteristic
- keep the last known values for the other two sensors
- emit a single combined sample immediately

That means the baseline sample model is:

```text
notification -> update one sensor vector -> emit full sample with latest-known accel/gyro/mag
```

If the goal is to match the current project behavior, preserve this model.

## Acceptance Examples

- A still sensor should produce near-zero gyro values.
- A level still sensor should show accelerometer magnitude close to `1000 mg`-style gravity magnitude in the baseline app.
- A still calibrated magnetic field magnitude around `40-50 uT` is plausible in normal environments.
- A correct magnetometer parser should not require any further sign inversion downstream if the rest of the pipeline matches this baseline.
