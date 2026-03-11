package com.app.skatingbergs

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

class BleService : Service() {
    private var bleClient: BleDeviceClient? = null
    private var currentAddress: String? = null

    override fun onCreate() {
        super.onCreate()
        SkatingRepository.initialize(this)
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Waiting for BLE device"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CONNECT -> {
                val address = intent.getStringExtra(EXTRA_DEVICE_ADDRESS) ?: return START_NOT_STICKY
                val name = intent.getStringExtra(EXTRA_DEVICE_NAME).orEmpty()
                connect(address, name)
            }

            ACTION_DISCONNECT -> {
                disconnectAndStop()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        bleClient?.disconnect()
        bleClient = null
        currentAddress = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun connect(address: String, name: String) {
        if (currentAddress == address && bleClient != null) return

        bleClient?.disconnect()
        currentAddress = address
        SkatingRepository.onServiceStateChanged(BleConnectionState.Connecting)
        updateNotification("Connecting to $name")

        val adapter = (getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
        val device = adapter.getRemoteDevice(address)
        bleClient = BleDeviceClient(
            context = this,
            device = device,
            onStateChanged = { state ->
                SkatingRepository.onServiceStateChanged(state)
                updateNotification(
                    when (state) {
                        BleConnectionState.Connected -> "Streaming IMU data from $name"
                        BleConnectionState.Reconnecting -> "Reconnecting to $name"
                        BleConnectionState.Connecting -> "Connecting to $name"
                        else -> "BLE disconnected"
                    }
                )
            },
            onSample = SkatingRepository::onSensorSample,
            onLog = SkatingRepository::addLog
        )
        bleClient?.connect()
    }

    private fun disconnectAndStop() {
        bleClient?.disconnect()
        bleClient = null
        currentAddress = null
        SkatingRepository.onServiceStateChanged(BleConnectionState.Disconnected)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun updateNotification(contentText: String) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, buildNotification(contentText))
    }

    private fun buildNotification(contentText: String) =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle("Figure Skating BLE")
            .setContentText(contentText)
            .setOngoing(true)
            .setContentIntent(
                PendingIntent.getActivity(
                    this,
                    0,
                    Intent(this, MainActivity::class.java),
                    PendingIntent.FLAG_IMMUTABLE
                )
            )
            .build()

    companion object {
        const val ACTION_CONNECT = "com.app.skatingbergs.action.CONNECT"
        const val ACTION_DISCONNECT = "com.app.skatingbergs.action.DISCONNECT"
        const val EXTRA_DEVICE_NAME = "extra_device_name"
        const val EXTRA_DEVICE_ADDRESS = "extra_device_address"

        private const val CHANNEL_ID = "figure_skating_ble"
        private const val NOTIFICATION_ID = 7
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Figure Skating BLE",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
}
