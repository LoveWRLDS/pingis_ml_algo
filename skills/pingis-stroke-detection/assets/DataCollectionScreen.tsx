/**
 * DataCollectionScreen.tsx
 *
 * React Native screen for recording labeled IMU training data from BERG AirHive.
 *
 * Flow:
 *  1. Connect to AirHive sensor via BLE
 *  2. Stream ImuSamples into a 3-second circular buffer
 *  3. Player selects stroke type (forehand / backhand)
 *  4. Player taps HIT at moment of ball contact, or MISS after a swing without contact
 *  5. App slices the buffer: 500ms before + 500ms after the tap timestamp
 *  6. Saves labeled event to AsyncStorage as JSON
 *  7. After session: export all events to a file for transfer to laptop
 *
 * Integration:
 *  - Requires react-native-ble-plx for BLE connectivity
 *  - Requires @react-native-async-storage/async-storage for local storage
 *  - Sensor protocol from skills/berg-airhive-imu-3d-view/references/sensor-protocol.md
 *
 * IMPORTANT: This is a template/reference component.
 * Adapt BLE UUID constants and ImuSample type to match your actual project's types.
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
} from 'react-native';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ImuSample {
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;   // degrees/second
  gyro_y: number;
  gyro_z: number;
  mag_x: number;    // microtesla (-rawX/10)
  mag_y: number;
  mag_z: number;
  ts_ms: number;    // receivedAtMs — wall clock
}

interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

type StrokeType = 'forehand' | 'backhand';

// ── Constants ─────────────────────────────────────────────────────────────────

const BUFFER_DURATION_MS = 3000;     // Keep 3 seconds of history
const CAPTURE_BEFORE_MS = 500;       // ms before label tap to include
const CAPTURE_AFTER_MS = 500;        // ms after label tap to include
const SAMPLE_RATE_HZ = 50;

// BLE UUIDs — from sensor-protocol.md
const BLE_SERVICE_UUID = '07c80000-07c8-07c8-07c8-07c807c807c8';
const BLE_ACCEL_UUID   = '07c80001-07c8-07c8-07c8-07c807c807c8';
const BLE_GYRO_UUID    = '07c80004-07c8-07c8-07c8-07c807c807c8';
const BLE_MAG_UUID     = '07c80010-07c8-07c8-07c8-07c807c807c8';

// ── Sensor buffer ─────────────────────────────────────────────────────────────

/**
 * Extracts a window of samples centered on tapTimestampMs.
 * Returns samples from (tapTimestampMs - CAPTURE_BEFORE_MS) to
 * (tapTimestampMs + CAPTURE_AFTER_MS).
 */
function extractWindow(buffer: ImuSample[], tapTimestampMs: number): ImuSample[] {
  const from = tapTimestampMs - CAPTURE_BEFORE_MS;
  const to = tapTimestampMs + CAPTURE_AFTER_MS;
  return buffer.filter(s => s.ts_ms >= from && s.ts_ms <= to);
}

/**
 * Trims buffer to keep only the last BUFFER_DURATION_MS milliseconds.
 */
function trimBuffer(buffer: ImuSample[]): ImuSample[] {
  if (buffer.length === 0) return buffer;
  const cutoff = buffer[buffer.length - 1].ts_ms - BUFFER_DURATION_MS;
  return buffer.filter(s => s.ts_ms > cutoff);
}

// ── Main Component ────────────────────────────────────────────────────────────

export function DataCollectionScreen() {
  const [isConnected, setIsConnected] = useState(false);
  const [strokeType, setStrokeType] = useState<StrokeType>('forehand');
  const [eventCount, setEventCount] = useState(0);
  const [lastLabel, setLastLabel] = useState<string | null>(null);
  const [sampleCount, setSampleCount] = useState(0);

  const sampleBuffer = useRef<ImuSample[]>([]);
  const sessionEvents = useRef<LabeledEvent[]>([]);
  const latestAccel = useRef({ x: 0, y: 0, z: 0 });
  const latestGyro = useRef({ x: 0, y: 0, z: 0 });
  const latestMag = useRef({ x: 0, y: 0, z: 0 });

  // ── BLE simulation hook (replace with actual BLE integration) ──────────────
  // In a real app, subscribe to BLE notifications from the AirHive sensor.
  // Each characteristic notification fires onSensorSample with the latest ImuSample.
  //
  // Wire up like this (pseudocode):
  //   bleDevice.monitorCharacteristic(BLE_ACCEL_UUID, (rawBytes) => {
  //     const { x, y, z } = parsePacket(rawBytes);  // raw Int16
  //     latestAccel.current = { x, y, z };
  //     onSensorSample();  // emit combined sample
  //   });
  //
  // See BleDeviceClient.kt in skatingbergs for the full pattern.

  const onSensorSample = useCallback(() => {
    const sample: ImuSample = {
      accel_x: latestAccel.current.x,
      accel_y: latestAccel.current.y,
      accel_z: latestAccel.current.z,
      gyro_x: latestGyro.current.x,
      gyro_y: latestGyro.current.y,
      gyro_z: latestGyro.current.z,
      mag_x: latestMag.current.x,
      mag_y: latestMag.current.y,
      mag_z: latestMag.current.z,
      ts_ms: Date.now(),
    };

    sampleBuffer.current = trimBuffer([...sampleBuffer.current, sample]);
    setSampleCount(sampleBuffer.current.length);
  }, []);

  // ── Label capture ──────────────────────────────────────────────────────────

  const captureEvent = useCallback(
    (label: LabeledEvent['label']) => {
      const tapTimestamp = Date.now();
      const window = extractWindow(sampleBuffer.current, tapTimestamp);

      const minSamples = Math.floor((SAMPLE_RATE_HZ * (CAPTURE_BEFORE_MS + CAPTURE_AFTER_MS)) / 1000);
      if (window.length < minSamples / 2) {
        Alert.alert(
          'Too few samples',
          `Only ${window.length} samples captured. Make sure sensor is streaming.`
        );
        return;
      }

      const event: LabeledEvent = {
        label,
        stroke_type: label === 'idle' ? 'unknown' : strokeType,
        recorded_at: new Date().toISOString(),
        samples: window,
      };

      sessionEvents.current.push(event);
      setEventCount(sessionEvents.current.length);
      setLastLabel(`${label} (${window.length} samples)`);
    },
    [strokeType]
  );

  const onHit = useCallback(() => captureEvent('hit'), [captureEvent]);
  const onMiss = useCallback(() => captureEvent('swing_miss'), [captureEvent]);
  const onIdle = useCallback(() => captureEvent('idle'), [captureEvent]);

  // ── Session export ─────────────────────────────────────────────────────────

  const exportSession = useCallback(async () => {
    if (sessionEvents.current.length === 0) {
      Alert.alert('No data', 'Record some events first.');
      return;
    }

    const date = new Date().toISOString().slice(0, 10);
    const sessionData = JSON.stringify(sessionEvents.current, null, 2);

    // In a real app: write to device filesystem with react-native-fs
    // or upload to S3 via AWS Amplify.
    // Example with react-native-fs:
    //   import RNFS from 'react-native-fs';
    //   const path = `${RNFS.DocumentDirectoryPath}/session_${date}_${Date.now()}.json`;
    //   await RNFS.writeFile(path, sessionData, 'utf8');

    // For now: log to console so you can copy from Metro output
    console.log('SESSION_DATA_START');
    console.log(sessionData);
    console.log('SESSION_DATA_END');

    Alert.alert(
      'Session exported',
      `${sessionEvents.current.length} events logged to console.\n\nCopy from Metro output and save as data/raw/session_${date}_001.json`
    );
  }, []);

  const clearSession = useCallback(() => {
    Alert.alert('Clear session?', 'All recorded events will be lost.', [
      { text: 'Cancel' },
      {
        text: 'Clear',
        onPress: () => {
          sessionEvents.current = [];
          setEventCount(0);
          setLastLabel(null);
        },
      },
    ]);
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────

  const labelCounts = sessionEvents.current.reduce<Record<string, number>>(
    (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
    {}
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <Text style={styles.title}>Pingis Data Collection</Text>

      {/* Connection status */}
      <View style={[styles.statusBadge, isConnected ? styles.connected : styles.disconnected]}>
        <Text style={styles.statusText}>
          {isConnected ? `● Connected  |  Buffer: ${sampleCount} samples` : '○ Disconnected — connect AirHive'}
        </Text>
      </View>

      {/* Stroke type selector */}
      <Text style={styles.sectionLabel}>Stroke Type</Text>
      <View style={styles.row}>
        {(['forehand', 'backhand'] as StrokeType[]).map(type => (
          <TouchableOpacity
            key={type}
            style={[styles.typeBtn, strokeType === type && styles.typeBtnActive]}
            onPress={() => setStrokeType(type)}
          >
            <Text style={[styles.typeBtnText, strokeType === type && styles.typeBtnTextActive]}>
              {type.toUpperCase()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Main action buttons */}
      <Text style={styles.sectionLabel}>Label Event</Text>

      <TouchableOpacity style={[styles.actionBtn, styles.hitBtn]} onPress={onHit} activeOpacity={0.7}>
        <Text style={styles.actionBtnText}>HIT</Text>
        <Text style={styles.actionBtnSub}>tap at moment of ball contact</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.actionBtn, styles.missBtn]} onPress={onMiss} activeOpacity={0.7}>
        <Text style={styles.actionBtnText}>MISS</Text>
        <Text style={styles.actionBtnSub}>swing without ball contact</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.actionBtn, styles.idleBtn]} onPress={onIdle} activeOpacity={0.7}>
        <Text style={styles.actionBtnText}>IDLE</Text>
        <Text style={styles.actionBtnSub}>at rest / between strokes</Text>
      </TouchableOpacity>

      {/* Last event feedback */}
      {lastLabel && (
        <Text style={styles.feedback}>Last: {lastLabel}</Text>
      )}

      {/* Session stats */}
      <View style={styles.statsBox}>
        <Text style={styles.statsTitle}>Session ({eventCount} events)</Text>
        {Object.entries(labelCounts).map(([label, count]) => (
          <Text key={label} style={styles.statsRow}>
            {label}: {count}
          </Text>
        ))}
        {eventCount === 0 && <Text style={styles.statsHint}>No events recorded yet</Text>}
      </View>

      {/* Export / Clear */}
      <View style={styles.row}>
        <TouchableOpacity style={[styles.secondaryBtn, styles.exportBtn]} onPress={exportSession}>
          <Text style={styles.secondaryBtnText}>Export Session</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.secondaryBtn, styles.clearBtn]} onPress={clearSession}>
          <Text style={styles.secondaryBtnText}>Clear</Text>
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      <View style={styles.instructionBox}>
        <Text style={styles.instructionTitle}>How to record:</Text>
        <Text style={styles.instruction}>1. Select stroke type before each stroke</Text>
        <Text style={styles.instruction}>2. Perform the stroke</Text>
        <Text style={styles.instruction}>3. If ball contact: tap HIT immediately</Text>
        <Text style={styles.instruction}>4. If no contact: tap MISS after swing</Text>
        <Text style={styles.instruction}>5. Sit still 2s, tap IDLE between strokes</Text>
        <Text style={styles.instruction}>6. Target: 20+ hits + 20+ misses + 20+ idle</Text>
      </View>
    </ScrollView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 40 },
  title: { color: '#fff', fontSize: 22, fontWeight: '700', marginBottom: 16 },

  statusBadge: { borderRadius: 8, padding: 10, marginBottom: 20 },
  connected: { backgroundColor: '#0d2d1a' },
  disconnected: { backgroundColor: '#2d0d0d' },
  statusText: { color: '#aaa', fontSize: 13 },

  sectionLabel: { color: '#666', fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 8, marginTop: 16 },

  row: { flexDirection: 'row', gap: 12, marginBottom: 4 },

  typeBtn: { flex: 1, padding: 14, borderRadius: 8, borderWidth: 1, borderColor: '#333', alignItems: 'center' },
  typeBtnActive: { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  typeBtnText: { color: '#666', fontWeight: '600', fontSize: 14 },
  typeBtnTextActive: { color: '#4a9eff' },

  actionBtn: { borderRadius: 12, padding: 22, marginBottom: 12, alignItems: 'center' },
  hitBtn: { backgroundColor: '#0d3d1a' },
  missBtn: { backgroundColor: '#3d0d0d' },
  idleBtn: { backgroundColor: '#1a1a2d' },
  actionBtnText: { color: '#fff', fontSize: 20, fontWeight: '700', letterSpacing: 2 },
  actionBtnSub: { color: '#666', fontSize: 11, marginTop: 4 },

  feedback: { color: '#4a9eff', textAlign: 'center', fontSize: 13, marginVertical: 8 },

  statsBox: { backgroundColor: '#1a1a1a', borderRadius: 10, padding: 16, marginTop: 12 },
  statsTitle: { color: '#fff', fontWeight: '600', marginBottom: 8 },
  statsRow: { color: '#aaa', fontSize: 14, marginBottom: 2 },
  statsHint: { color: '#444', fontSize: 13, fontStyle: 'italic' },

  secondaryBtn: { flex: 1, padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 12 },
  exportBtn: { backgroundColor: '#1a2d1a' },
  clearBtn: { backgroundColor: '#2d1a1a' },
  secondaryBtnText: { color: '#aaa', fontWeight: '600' },

  instructionBox: { marginTop: 24, backgroundColor: '#111', borderRadius: 10, padding: 16 },
  instructionTitle: { color: '#555', fontWeight: '600', marginBottom: 8 },
  instruction: { color: '#444', fontSize: 13, marginBottom: 3 },
});
