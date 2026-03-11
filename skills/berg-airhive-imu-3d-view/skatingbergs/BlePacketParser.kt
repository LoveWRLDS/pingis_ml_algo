package com.app.skatingbergs

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

object BleUuids {
    val service: UUID = UUID.fromString("07c80000-07c8-07c8-07c8-07c807c807c8")
    val accelerometer: UUID = UUID.fromString("07c80001-07c8-07c8-07c8-07c807c807c8")
    val accelerometerAlt: UUID = UUID.fromString("07c80203-07c8-07c8-07c8-07c807c807c8")
    val gyroscope: UUID = UUID.fromString("07c80004-07c8-07c8-07c8-07c807c807c8")
    val magnetometer: UUID = UUID.fromString("07c80010-07c8-07c8-07c8-07c807c807c8")
    val cccd: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
}

sealed interface ParsedBlePacket {
    val sensorTimestamp: Int

    data class Accelerometer(
        val value: Vector3f,
        override val sensorTimestamp: Int
    ) : ParsedBlePacket

    data class Gyroscope(
        val value: Vector3f,
        override val sensorTimestamp: Int
    ) : ParsedBlePacket

    data class Magnetometer(
        val value: Vector3f,
        override val sensorTimestamp: Int
    ) : ParsedBlePacket
}

object BlePacketParser {
    fun parse(characteristicUuid: UUID, payload: ByteArray): ParsedBlePacket? {
        if (payload.size < 9) return null
        val vector = parseVector(payload)
        val timestamp = parseTimestamp(payload)

        return when (characteristicUuid) {
            BleUuids.accelerometer, BleUuids.accelerometerAlt ->
                ParsedBlePacket.Accelerometer(vector, timestamp)

            BleUuids.gyroscope ->
                ParsedBlePacket.Gyroscope(vector, timestamp)

            BleUuids.magnetometer ->
                ParsedBlePacket.Magnetometer(
                    // User-verified magnetometer truth:
                    // invert X/Y/Z, then scale by 0.1 so values are in uT.
                    value = Vector3f(
                        x = -vector.x / 10f,
                        y = -vector.y / 10f,
                        z = -vector.z / 10f
                    ),
                    sensorTimestamp = timestamp
                )

            else -> null
        }
    }

    private fun parseVector(payload: ByteArray): Vector3f {
        val buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        return Vector3f(
            x = buffer.short.toFloat(),
            y = buffer.short.toFloat(),
            z = buffer.short.toFloat()
        )
    }

    private fun parseTimestamp(payload: ByteArray): Int {
        val b0 = payload[6].toInt() and 0xFF
        val b1 = payload[7].toInt() and 0xFF
        val b2 = payload[8].toInt() and 0xFF
        return (b0 shl 16) or (b1 shl 8) or b2
    }
}
