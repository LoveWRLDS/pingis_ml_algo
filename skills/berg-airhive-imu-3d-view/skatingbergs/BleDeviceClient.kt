package com.app.skatingbergs

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.content.Context
import android.os.Build
import java.util.ArrayDeque

@SuppressLint("MissingPermission")
class BleDeviceClient(
    private val context: Context,
    private val device: BluetoothDevice,
    private val onStateChanged: (BleConnectionState) -> Unit,
    private val onSample: (ImuSample) -> Unit,
    private val onLog: (String) -> Unit
) {
    private var gatt: BluetoothGatt? = null
    private var allowReconnect = true
    private val pendingDescriptorWrites = ArrayDeque<BluetoothGattDescriptor>()

    private var latestAccel = Vector3f.Zero
    private var latestGyro = Vector3f.Zero
    private var latestMag = Vector3f.Zero
    private var latestTimestamp = 0

    private val callback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    onStateChanged(BleConnectionState.Connected)
                    onLog("Connected to ${device.address}")
                    gatt.discoverServices()
                }

                BluetoothProfile.STATE_DISCONNECTED -> {
                    onLog("Disconnected from ${device.address} (status=$status)")
                    if (allowReconnect) {
                        onStateChanged(BleConnectionState.Reconnecting)
                        gatt.connect()
                    } else {
                        onStateChanged(BleConnectionState.Disconnected)
                    }
                }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                onStateChanged(BleConnectionState.Error)
                onLog("Service discovery failed with status=$status")
                return
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)
            }

            pendingDescriptorWrites.clear()
            queueNotification(gatt, BleUuids.accelerometer, BleUuids.accelerometerAlt)
            queueNotification(gatt, BleUuids.gyroscope)
            queueNotification(gatt, BleUuids.magnetometer)
            writeNextDescriptor(gatt)
        }

        override fun onDescriptorWrite(gatt: BluetoothGatt?, descriptor: BluetoothGattDescriptor?, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                writeNextDescriptor(gatt)
            } else {
                onLog("Descriptor write failed for ${descriptor?.characteristic?.uuid} with status=$status")
            }
        }

        @Deprecated("Deprecated in Java")
        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            handleNotification(characteristic.uuid, characteristic.value)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleNotification(characteristic.uuid, value)
        }
    }

    fun connect() {
        allowReconnect = true
        onStateChanged(BleConnectionState.Connecting)
        gatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE)
        } else {
            device.connectGatt(context, false, callback)
        }
    }

    fun disconnect() {
        allowReconnect = false
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        onStateChanged(BleConnectionState.Disconnected)
    }

    private fun handleNotification(characteristicUuid: java.util.UUID, payload: ByteArray) {
        val packet = BlePacketParser.parse(characteristicUuid, payload) ?: return
        latestTimestamp = packet.sensorTimestamp

        when (packet) {
            is ParsedBlePacket.Accelerometer -> latestAccel = packet.value
            is ParsedBlePacket.Gyroscope -> latestGyro = packet.value
            is ParsedBlePacket.Magnetometer -> latestMag = packet.value
        }

        onSample(
            ImuSample(
                accelerometer = latestAccel,
                gyroscopeDps = latestGyro,
                magnetometerUt = latestMag,
                sensorTimestamp = latestTimestamp
            )
        )
    }

    private fun queueNotification(gatt: BluetoothGatt, primaryUuid: java.util.UUID, secondaryUuid: java.util.UUID? = null) {
        val characteristic = gatt.services
            .flatMap { it.characteristics }
            .firstOrNull { it.uuid == primaryUuid || it.uuid == secondaryUuid }
            ?: return

        gatt.setCharacteristicNotification(characteristic, true)
        val descriptor = characteristic.getDescriptor(BleUuids.cccd) ?: return
        descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
        pendingDescriptorWrites.add(descriptor)
    }

    private fun writeNextDescriptor(gatt: BluetoothGatt?) {
        if (pendingDescriptorWrites.isEmpty()) return
        val next = pendingDescriptorWrites.removeFirst()
        gatt?.writeDescriptor(next)
    }
}
