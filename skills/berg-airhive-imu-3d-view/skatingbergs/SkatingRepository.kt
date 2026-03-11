package com.app.skatingbergs

import android.annotation.SuppressLint
import android.bluetooth.BluetoothManager
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.content.Intent
import android.os.Build
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

object SkatingRepository {
    private lateinit var appContext: Context

    private val orientationEstimator: OrientationEstimator = ComplementaryOrientationEstimator()
    private var referenceOrientation: Quaternionf? = null
    private var scanCallback: ScanCallback? = null

    private val _hasPermissions = MutableStateFlow(false)
    val hasPermissions: StateFlow<Boolean> = _hasPermissions.asStateFlow()

    private val _isScanning = MutableStateFlow(false)
    val isScanning: StateFlow<Boolean> = _isScanning.asStateFlow()

    private val _devices = MutableStateFlow<List<BleDiscoveredDevice>>(emptyList())
    val devices: StateFlow<List<BleDiscoveredDevice>> = _devices.asStateFlow()

    private val _selectedDevice = MutableStateFlow<BleDiscoveredDevice?>(null)
    val selectedDevice: StateFlow<BleDiscoveredDevice?> = _selectedDevice.asStateFlow()

    private val _connectionState = MutableStateFlow(BleConnectionState.Idle)
    val connectionState: StateFlow<BleConnectionState> = _connectionState.asStateFlow()

    private val _latestSample = MutableStateFlow(ImuSample())
    val latestSample: StateFlow<ImuSample> = _latestSample.asStateFlow()

    private val _chartSamples = MutableStateFlow<List<ImuSample>>(emptyList())
    val chartSamples: StateFlow<List<ImuSample>> = _chartSamples.asStateFlow()

    private val _orientation = MutableStateFlow(OrientationSnapshot())
    val orientation: StateFlow<OrientationSnapshot> = _orientation.asStateFlow()

    private val _logs = MutableStateFlow<List<String>>(emptyList())
    val logs: StateFlow<List<String>> = _logs.asStateFlow()

    fun initialize(context: Context) {
        if (!::appContext.isInitialized) {
            appContext = context.applicationContext
        }
    }

    fun setPermissionsGranted(granted: Boolean) {
        _hasPermissions.value = granted
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        val scanner = bluetoothManager().adapter?.bluetoothLeScanner ?: return
        stopScan()
        _devices.value = emptyList()
        _isScanning.value = true
        _connectionState.value = BleConnectionState.Scanning

        scanCallback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val device = result.device
                val name = result.scanRecord?.deviceName ?: device.name ?: "Unknown"
                if (name == "Unknown") return

                val discovered = BleDiscoveredDevice(
                    name = name,
                    address = device.address,
                    rssi = result.rssi
                )
                _devices.update { existing ->
                    (existing.filterNot { it.address == discovered.address } + discovered)
                        .sortedByDescending { it.rssi }
                }
            }
        }

        scanner.startScan(scanCallback)
        addLog("Started BLE scan")
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        val callback = scanCallback ?: return
        bluetoothManager().adapter?.bluetoothLeScanner?.stopScan(callback)
        scanCallback = null
        _isScanning.value = false
        if (_connectionState.value == BleConnectionState.Scanning) {
            _connectionState.value = BleConnectionState.Idle
        }
        addLog("Stopped BLE scan")
    }

    fun connectToDevice(device: BleDiscoveredDevice) {
        initialize(appContext)
        stopScan()
        _selectedDevice.value = device
        _connectionState.value = BleConnectionState.Connecting
        addLog("Connecting to ${device.name} (${device.address})")

        val intent = Intent(appContext, BleService::class.java).apply {
            action = BleService.ACTION_CONNECT
            putExtra(BleService.EXTRA_DEVICE_NAME, device.name)
            putExtra(BleService.EXTRA_DEVICE_ADDRESS, device.address)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            appContext.startForegroundService(intent)
        } else {
            appContext.startService(intent)
        }
    }

    fun disconnect() {
        if (!::appContext.isInitialized) return
        val intent = Intent(appContext, BleService::class.java).apply {
            action = BleService.ACTION_DISCONNECT
        }
        appContext.startService(intent)
        _connectionState.value = BleConnectionState.Disconnected
        addLog("Disconnect requested")
    }

    fun calibrateCurrentPose() {
        referenceOrientation = _orientation.value.absoluteOrientation.normalized()
        _orientation.update { current ->
            current.copy(displayOrientation = Quaternionf.Identity)
        }
        addLog("Calibrated current pose as identity orientation")
    }

    fun resetCalibration() {
        referenceOrientation = null
        val absolute = _orientation.value.absoluteOrientation
        _orientation.value = OrientationSnapshot(
            absoluteOrientation = absolute,
            displayOrientation = absolute
        )
        addLog("Cleared pose calibration")
    }

    internal fun onServiceStateChanged(state: BleConnectionState) {
        _connectionState.value = state
    }

    internal fun onSensorSample(sample: ImuSample) {
        _latestSample.value = sample
        _chartSamples.update { existing ->
            val updated = existing + sample
            val cutoff = sample.receivedAtMs - 10_000L
            updated.filter { it.receivedAtMs >= cutoff }
        }

        val absolute = orientationEstimator.update(sample)
        val display = referenceOrientation
            ?.inverse()
            ?.times(absolute)
            ?.normalized()
            ?: absolute

        _orientation.value = OrientationSnapshot(
            absoluteOrientation = absolute,
            displayOrientation = display
        )
    }

    internal fun addLog(message: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault())
            .format(java.util.Date())
        _logs.update { existing ->
            listOf("[$timestamp] $message") + existing.take(199)
        }
    }

    private fun bluetoothManager(): BluetoothManager =
        appContext.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
}
