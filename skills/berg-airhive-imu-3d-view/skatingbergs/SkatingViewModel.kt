package com.app.skatingbergs

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

class SkatingViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = SkatingRepository

    val uiState = combine(
        repository.hasPermissions,
        repository.isScanning,
        repository.devices,
        repository.selectedDevice,
        repository.connectionState,
        repository.latestSample,
        repository.chartSamples,
        repository.orientation,
        repository.logs
    ) { values ->
        val hasPermissions = values[0] as Boolean
        val isScanning = values[1] as Boolean
        val devices = values[2] as List<BleDiscoveredDevice>
        val selectedDevice = values[3] as BleDiscoveredDevice?
        val connectionState = values[4] as BleConnectionState
        val latestSample = values[5] as ImuSample
        val chartSamples = values[6] as List<ImuSample>
        val orientation = values[7] as OrientationSnapshot
        val logs = values[8] as List<String>
        MainUiState(
            hasPermissions = hasPermissions,
            isScanning = isScanning,
            devices = devices,
            selectedDevice = selectedDevice,
            connectionState = connectionState,
            latestSample = latestSample,
            chartSamples = chartSamples,
            orientation = orientation,
            logs = logs
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = MainUiState()
    )

    init {
        repository.initialize(application)
    }

    fun onPermissionsChanged(granted: Boolean) {
        repository.setPermissionsGranted(granted)
    }

    fun toggleScan() {
        if (uiState.value.isScanning) {
            repository.stopScan()
        } else {
            repository.startScan()
        }
    }

    fun connect(device: BleDiscoveredDevice) {
        repository.connectToDevice(device)
    }

    fun disconnect() {
        repository.disconnect()
    }

    fun calibratePose() {
        repository.calibrateCurrentPose()
    }

    fun resetCalibration() {
        repository.resetCalibration()
    }
}
