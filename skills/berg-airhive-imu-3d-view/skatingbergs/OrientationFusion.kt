package com.app.skatingbergs

import kotlin.math.exp

interface OrientationEstimator {
    fun reset()
    fun update(sample: ImuSample): Quaternionf
}

class ComplementaryOrientationEstimator : OrientationEstimator {
    private var orientation = Quaternionf.Identity
    private var lastSampleTimeMs: Long? = null
    private var filteredAccel = Vector3f.Zero
    private var filteredMag = Vector3f.Zero

    override fun reset() {
        orientation = Quaternionf.Identity
        lastSampleTimeMs = null
        filteredAccel = Vector3f.Zero
        filteredMag = Vector3f.Zero
    }

    override fun update(sample: ImuSample): Quaternionf {
        val previous = orientation
        val accel = remapAccel(sample.accelerometer)
        val gyroRad = remapGyro(sample.gyroscopeDps).let {
            Vector3f(
                x = Math.toRadians(it.x.toDouble()).toFloat(),
                y = Math.toRadians(it.y.toDouble()).toFloat(),
                z = Math.toRadians(it.z.toDouble()).toFloat()
            )
        }
        val mag = remapMag(sample.magnetometerUt)

        val dt = lastSampleTimeMs
            ?.let { ((sample.receivedAtMs - it) / 1000f).coerceIn(0.001f, 0.05f) }
            ?: (1f / 50f)
        lastSampleTimeMs = sample.receivedAtMs

        val predicted = integrateGyro(orientation, gyroRad, dt)
        val accelNorm = accel.normalized()
        val magNorm = mag.normalized()

        if (accelNorm == Vector3f.Zero || magNorm == Vector3f.Zero) {
            orientation = predicted
            return orientation
        }

        filteredAccel = if (filteredAccel == Vector3f.Zero) accelNorm else filteredAccel.lerp(accelNorm, 0.15f)
        filteredMag = if (filteredMag == Vector3f.Zero) magNorm else filteredMag.lerp(magNorm, 0.10f)

        val measured = measuredOrientation(filteredAccel.normalized(), filteredMag.normalized())
        val correctionAlpha = 1f - exp((-dt / 0.35f).toDouble()).toFloat()
        orientation = predicted.slerp(measured, correctionAlpha)
        if (orientation.dot(previous) < 0f) {
            orientation = orientation.negated()
        }
        return orientation
    }

    private fun integrateGyro(current: Quaternionf, gyroRad: Vector3f, dt: Float): Quaternionf {
        val angularSpeed = gyroRad.magnitude()
        if (angularSpeed < 1e-5f) return current
        val delta = Quaternionf.fromAxisAngle(gyroRad / angularSpeed, angularSpeed * dt)
        return (current * delta).normalized()
    }

    private fun measuredOrientation(accel: Vector3f, magnetometer: Vector3f): Quaternionf {
        val up = accel.normalized()
        val horizontalMag = (magnetometer - up * magnetometer.dot(up)).normalized()
        if (horizontalMag == Vector3f.Zero) return orientation

        val east = horizontalMag.cross(up).normalized()
        val north = up.cross(east).normalized()
        if (east == Vector3f.Zero || north == Vector3f.Zero) return orientation

        val sensorFromWorld = Quaternionf.fromRotationMatrix(
            m00 = east.x, m01 = north.x, m02 = up.x,
            m10 = east.y, m11 = north.y, m12 = up.y,
            m20 = east.z, m21 = north.z, m22 = up.z
        )
        return sensorFromWorld.inverse().normalized()
    }

    private fun remapAccel(raw: Vector3f): Vector3f =
        // Use the sensor/body axes directly so rotations map naturally:
        // X -> roll, Y -> pitch, Z -> yaw after pose calibration.
        raw

    private fun remapGyro(raw: Vector3f): Vector3f =
        raw

    private fun remapMag(raw: Vector3f): Vector3f =
        // Magnetometer is already aligned to the accel/gyro body frame in the packet parser.
        raw
}
