package com.app.skatingbergs

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class Vector3f(
    val x: Float,
    val y: Float,
    val z: Float
) {
    operator fun plus(other: Vector3f) = Vector3f(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Vector3f) = Vector3f(x - other.x, y - other.y, z - other.z)
    operator fun times(scale: Float) = Vector3f(x * scale, y * scale, z * scale)
    operator fun div(scale: Float) = Vector3f(x / scale, y / scale, z / scale)

    fun dot(other: Vector3f): Float = x * other.x + y * other.y + z * other.z

    fun cross(other: Vector3f): Vector3f = Vector3f(
        x = y * other.z - z * other.y,
        y = z * other.x - x * other.z,
        z = x * other.y - y * other.x
    )

    fun magnitude(): Float = sqrt(dot(this))

    fun normalized(): Vector3f {
        val mag = magnitude()
        return if (mag < 1e-6f) Zero else this / mag
    }

    fun lerp(target: Vector3f, alpha: Float): Vector3f =
        Vector3f(
            x = x + (target.x - x) * alpha,
            y = y + (target.y - y) * alpha,
            z = z + (target.z - z) * alpha
        )

    fun isFinite(): Boolean = x.isFinite() && y.isFinite() && z.isFinite()

    companion object {
        val Zero = Vector3f(0f, 0f, 0f)
    }
}

data class Quaternionf(
    val w: Float,
    val x: Float,
    val y: Float,
    val z: Float
) {
    operator fun times(other: Quaternionf): Quaternionf = Quaternionf(
        w = w * other.w - x * other.x - y * other.y - z * other.z,
        x = w * other.x + x * other.w + y * other.z - z * other.y,
        y = w * other.y - x * other.z + y * other.w + z * other.x,
        z = w * other.z + x * other.y - y * other.x + z * other.w
    )

    fun normalized(): Quaternionf {
        val norm = sqrt(w * w + x * x + y * y + z * z)
        return if (norm < 1e-6f) Identity else Quaternionf(w / norm, x / norm, y / norm, z / norm)
    }

    fun inverse(): Quaternionf {
        val normSq = w * w + x * x + y * y + z * z
        return if (normSq < 1e-6f) Identity else Quaternionf(w / normSq, -x / normSq, -y / normSq, -z / normSq)
    }

    fun negated(): Quaternionf = Quaternionf(-w, -x, -y, -z)

    fun dot(other: Quaternionf): Float = w * other.w + x * other.x + y * other.y + z * other.z

    fun slerp(target: Quaternionf, alpha: Float): Quaternionf {
        var end = target
        var cosTheta = dot(end)
        if (cosTheta < 0f) {
            end = Quaternionf(-end.w, -end.x, -end.y, -end.z)
            cosTheta = -cosTheta
        }
        if (cosTheta > 0.9995f) {
            return Quaternionf(
                w = w + (end.w - w) * alpha,
                x = x + (end.x - x) * alpha,
                y = y + (end.y - y) * alpha,
                z = z + (end.z - z) * alpha
            ).normalized()
        }

        val theta0 = kotlin.math.acos(cosTheta)
        val theta = theta0 * alpha
        val sinTheta = sin(theta)
        val sinTheta0 = sin(theta0)
        val s0 = cos(theta) - cosTheta * sinTheta / sinTheta0
        val s1 = sinTheta / sinTheta0
        return Quaternionf(
            w = w * s0 + end.w * s1,
            x = x * s0 + end.x * s1,
            y = y * s0 + end.y * s1,
            z = z * s0 + end.z * s1
        ).normalized()
    }

    fun toEulerDegrees(): Vector3f {
        val pitch = asin((2f * (w * y - z * x)).coerceIn(-1f, 1f))
        val roll = atan2(2f * (w * x + y * z), 1f - 2f * (x * x + y * y))
        val yaw = atan2(2f * (w * z + x * y), 1f - 2f * (y * y + z * z))
        return Vector3f(
            x = Math.toDegrees(roll.toDouble()).toFloat(),
            y = Math.toDegrees(pitch.toDouble()).toFloat(),
            z = Math.toDegrees(yaw.toDouble()).toFloat()
        )
    }

    fun toMatrix4(): FloatArray {
        val q = normalized()
        val xx = q.x * q.x
        val yy = q.y * q.y
        val zz = q.z * q.z
        val xy = q.x * q.y
        val xz = q.x * q.z
        val yz = q.y * q.z
        val wx = q.w * q.x
        val wy = q.w * q.y
        val wz = q.w * q.z

        return floatArrayOf(
            1f - 2f * (yy + zz), 2f * (xy + wz), 2f * (xz - wy), 0f,
            2f * (xy - wz), 1f - 2f * (xx + zz), 2f * (yz + wx), 0f,
            2f * (xz + wy), 2f * (yz - wx), 1f - 2f * (xx + yy), 0f,
            0f, 0f, 0f, 1f
        )
    }

    companion object {
        val Identity = Quaternionf(1f, 0f, 0f, 0f)

        fun fromAxisAngle(axis: Vector3f, angleRad: Float): Quaternionf {
            val normalizedAxis = axis.normalized()
            if (!normalizedAxis.isFinite() || normalizedAxis == Vector3f.Zero || abs(angleRad) < 1e-6f) {
                return Identity
            }
            val half = angleRad / 2f
            val sinHalf = sin(half)
            return Quaternionf(
                w = cos(half),
                x = normalizedAxis.x * sinHalf,
                y = normalizedAxis.y * sinHalf,
                z = normalizedAxis.z * sinHalf
            ).normalized()
        }

        fun fromRotationMatrix(
            m00: Float, m01: Float, m02: Float,
            m10: Float, m11: Float, m12: Float,
            m20: Float, m21: Float, m22: Float
        ): Quaternionf {
            val trace = m00 + m11 + m22
            return if (trace > 0f) {
                val s = sqrt(trace + 1f) * 2f
                Quaternionf(
                    w = 0.25f * s,
                    x = (m21 - m12) / s,
                    y = (m02 - m20) / s,
                    z = (m10 - m01) / s
                )
            } else if (m00 > m11 && m00 > m22) {
                val s = sqrt(1f + m00 - m11 - m22) * 2f
                Quaternionf(
                    w = (m21 - m12) / s,
                    x = 0.25f * s,
                    y = (m01 + m10) / s,
                    z = (m02 + m20) / s
                )
            } else if (m11 > m22) {
                val s = sqrt(1f + m11 - m00 - m22) * 2f
                Quaternionf(
                    w = (m02 - m20) / s,
                    x = (m01 + m10) / s,
                    y = 0.25f * s,
                    z = (m12 + m21) / s
                )
            } else {
                val s = sqrt(1f + m22 - m00 - m11) * 2f
                Quaternionf(
                    w = (m10 - m01) / s,
                    x = (m02 + m20) / s,
                    y = (m12 + m21) / s,
                    z = 0.25f * s
                )
            }.normalized()
        }
    }
}
