package com.app.skatingbergs

data class BleDiscoveredDevice(
    val name: String,
    val address: String,
    val rssi: Int
)

enum class BleConnectionState {
    Idle,
    Scanning,
    Connecting,
    Connected,
    Reconnecting,
    Disconnected,
    Error
}

data class ImuSample(
    val accelerometer: Vector3f = Vector3f.Zero,
    val gyroscopeDps: Vector3f = Vector3f.Zero,
    val magnetometerUt: Vector3f = Vector3f.Zero,
    val sensorTimestamp: Int = 0,
    val receivedAtMs: Long = System.currentTimeMillis()
)

data class OrientationSnapshot(
    val absoluteOrientation: Quaternionf = Quaternionf.Identity,
    val displayOrientation: Quaternionf = Quaternionf.Identity
) {
    val displayEulerDegrees: Vector3f
        get() = displayOrientation.toEulerDegrees()
}

data class MainUiState(
    val hasPermissions: Boolean = false,
    val isScanning: Boolean = false,
    val devices: List<BleDiscoveredDevice> = emptyList(),
    val selectedDevice: BleDiscoveredDevice? = null,
    val connectionState: BleConnectionState = BleConnectionState.Idle,
    val latestSample: ImuSample = ImuSample(),
    val chartSamples: List<ImuSample> = emptyList(),
    val orientation: OrientationSnapshot = OrientationSnapshot(),
    val logs: List<String> = emptyList()
)
