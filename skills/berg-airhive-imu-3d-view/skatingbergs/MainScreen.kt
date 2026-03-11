package com.app.skatingbergs

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.max

@Composable
fun MainScreen(
    uiState: MainUiState,
    onScanToggle: () -> Unit,
    onConnect: (BleDiscoveredDevice) -> Unit,
    onDisconnect: () -> Unit,
    onCalibratePose: () -> Unit,
    onResetCalibration: () -> Unit,
    onRequestPermissions: () -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF3F6FB))
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Figure Skating IMU",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "BLE streaming, fused orientation, and a live 3D shoe reference view",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF546274)
            )
        }

        if (!uiState.hasPermissions) {
            item {
                SectionCard(title = "Permissions") {
                    Text("Bluetooth, location, and foreground-service permissions are required before scanning.")
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onRequestPermissions) {
                        Text("Grant Permissions")
                    }
                }
            }
        }

        item {
            BleControlsCard(
                uiState = uiState,
                onScanToggle = onScanToggle,
                onConnect = onConnect,
                onDisconnect = onDisconnect
            )
        }

        item {
            CalibrationCard(
                uiState = uiState,
                onCalibratePose = onCalibratePose,
                onResetCalibration = onResetCalibration
            )
        }

        item {
            SectionCard(title = "3D Shoe View") {
                ShoeViewport(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(1.4f)
                        .border(1.dp, Color(0xFFD1DAE6), RoundedCornerShape(16.dp))
                        .background(Color(0xFFE9F0F8), RoundedCornerShape(16.dp)),
                    orientation = uiState.orientation.displayOrientation
                )
                Spacer(modifier = Modifier.height(8.dp))
                val euler = uiState.orientation.displayEulerDegrees
                Text(
                    text = "Display orientation (deg): roll=${euler.x.format1()} pitch=${euler.y.format1()} yaw=${euler.z.format1()}",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        item {
            DebugCard(uiState)
        }

        item {
            SensorChartCard(
                title = "Accelerometer",
                subtitle = "Raw big-endian X/Y/Z stream from BLE",
                samples = uiState.chartSamples,
                vectorSelector = { it.accelerometer },
                minimumScale = 1500f
            )
        }

        item {
            SensorChartCard(
                title = "Gyroscope",
                subtitle = "Raw X/Y/Z in degrees per second",
                samples = uiState.chartSamples,
                vectorSelector = { it.gyroscopeDps },
                minimumScale = 500f
            )
        }

        item {
            SensorChartCard(
                title = "Magnetometer",
                subtitle = "Raw chip axes from sensor_calibration_guide.md",
                samples = uiState.chartSamples,
                vectorSelector = { it.magnetometerUt },
                minimumScale = 4900f
            )
        }

        item {
            SectionCard(title = "Debug Log") {
                if (uiState.logs.isEmpty()) {
                    Text("No BLE log entries yet.", color = Color(0xFF6A788A))
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 220.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        uiState.logs.take(20).forEach { log ->
                            Text(
                                text = log,
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun BleControlsCard(
    uiState: MainUiState,
    onScanToggle: () -> Unit,
    onConnect: (BleDiscoveredDevice) -> Unit,
    onDisconnect: () -> Unit
) {
    SectionCard(title = "BLE Connection") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onScanToggle, enabled = uiState.hasPermissions) {
                Text(if (uiState.isScanning) "Stop Scan" else "Start Scan")
            }
            OutlinedButton(onClick = onDisconnect, enabled = uiState.connectionState != BleConnectionState.Idle) {
                Text("Disconnect")
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Text("State: ${uiState.connectionState}")
        uiState.selectedDevice?.let { device ->
            Text("Selected: ${device.name} (${device.address})", style = MaterialTheme.typography.bodySmall)
        }
        Spacer(modifier = Modifier.height(8.dp))
        if (uiState.devices.isEmpty()) {
            Text("No named BLE devices discovered yet.", color = Color(0xFF6A788A))
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                uiState.devices.forEach { device ->
                    Card(colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFD))) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(device.name, fontWeight = FontWeight.SemiBold)
                                Text(device.address, style = MaterialTheme.typography.bodySmall)
                                Text("RSSI ${device.rssi} dBm", style = MaterialTheme.typography.bodySmall, color = Color(0xFF6A788A))
                            }
                            Button(onClick = { onConnect(device) }, enabled = uiState.hasPermissions) {
                                Text("Connect")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CalibrationCard(
    uiState: MainUiState,
    onCalibratePose: () -> Unit,
    onResetCalibration: () -> Unit
) {
    SectionCard(title = "Calibration") {
        Text("Capture the current fused pose as the identity orientation for the 3D shoe.")
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onCalibratePose,
                enabled = uiState.connectionState == BleConnectionState.Connected || uiState.connectionState == BleConnectionState.Reconnecting
            ) {
                Text("Calibrate Pose")
            }
            OutlinedButton(onClick = onResetCalibration) {
                Text("Reset Reference")
            }
        }
    }
}

@Composable
private fun DebugCard(uiState: MainUiState) {
    val sample = uiState.latestSample
    val quaternion = uiState.orientation.displayOrientation

    SectionCard(title = "Raw Debug Values") {
        Text("Accel: ${sample.accelerometer.formatTriplet()}")
        Text("Gyro: ${sample.gyroscopeDps.formatTriplet()}")
        Text("Mag: ${sample.magnetometerUt.formatTriplet()}")
        Text("Sensor timestamp: ${sample.sensorTimestamp}")
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "Quaternion: w=${quaternion.w.format3()} x=${quaternion.x.format3()} y=${quaternion.y.format3()} z=${quaternion.z.format3()}",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun SensorChartCard(
    title: String,
    subtitle: String,
    samples: List<ImuSample>,
    vectorSelector: (ImuSample) -> Vector3f,
    minimumScale: Float
) {
    SectionCard(title = title) {
        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = Color(0xFF6A788A))
        Spacer(modifier = Modifier.height(8.dp))
        ChartLegend()
        Spacer(modifier = Modifier.height(4.dp))

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(180.dp)
                .background(Color.White, RoundedCornerShape(12.dp))
                .border(1.dp, Color(0xFFD9E1EA), RoundedCornerShape(12.dp))
                .padding(8.dp)
        ) {
            if (samples.size < 2) return@Canvas

            val latestTime = samples.last().receivedAtMs
            val windowStart = latestTime - 10_000L
            val visible = samples.filter { it.receivedAtMs >= windowStart }
            if (visible.size < 2) return@Canvas

            val maxAbs = visible.fold(minimumScale) { acc, sample ->
                val vector = vectorSelector(sample)
                max(acc, maxOf(kotlin.math.abs(vector.x), kotlin.math.abs(vector.y), kotlin.math.abs(vector.z)))
            }
            val centerY = size.height / 2f
            val scaleY = if (maxAbs <= 0f) 1f else centerY / maxAbs

            drawLine(Color(0xFFCED7E3), Offset(0f, centerY), Offset(size.width, centerY), 1.5f)

            drawSeries(visible, windowStart, latestTime, centerY, scaleY, vectorSelector, { it.x }, Color(0xFFE34A4A))
            drawSeries(visible, windowStart, latestTime, centerY, scaleY, vectorSelector, { it.y }, Color(0xFF2FA66A))
            drawSeries(visible, windowStart, latestTime, centerY, scaleY, vectorSelector, { it.z }, Color(0xFF3C74E6))
        }
    }
}

@Composable
private fun ChartLegend() {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        LegendChip("X", Color(0xFFE34A4A))
        LegendChip("Y", Color(0xFF2FA66A))
        LegendChip("Z", Color(0xFF3C74E6))
        Text("Last 10 seconds", style = MaterialTheme.typography.bodySmall, color = Color(0xFF6A788A))
    }
}

@Composable
private fun LegendChip(label: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(color, RoundedCornerShape(100))
        )
        Spacer(modifier = Modifier.size(6.dp))
        Text(label, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun SectionCard(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFDFEFF)),
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            content()
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSeries(
    samples: List<ImuSample>,
    windowStart: Long,
    latestTime: Long,
    centerY: Float,
    scaleY: Float,
    vectorSelector: (ImuSample) -> Vector3f,
    axisSelector: (Vector3f) -> Float,
    color: Color
) {
    var previous: Offset? = null
    samples.forEach { sample ->
        val x = ((sample.receivedAtMs - windowStart).toFloat() / (latestTime - windowStart).coerceAtLeast(1L).toFloat()) * size.width
        val y = centerY - axisSelector(vectorSelector(sample)) * scaleY
        val current = Offset(x, y)
        previous?.let { drawLine(color = color, start = it, end = current, strokeWidth = 3f) }
        previous = current
    }
}

private fun Float.format1(): String = String.format("%.1f", this)
private fun Float.format3(): String = String.format("%.3f", this)
private fun Vector3f.formatTriplet(): String = "x=${x.format1()} y=${y.format1()} z=${z.format1()}"
