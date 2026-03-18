// @ts-nocheck
// This file is a reference template. Module-resolution errors are expected here
// because there is no node_modules in skills/assets/. Copy this file into your
// React Native project where react-native-ble-plx and react-native-fs are installed.

/**
 * DataCollectionScreen.tsx
 *
 * React Native screen for recording labeled IMU training data from BERG AirHive.
 * Uses react-native-ble-plx for BLE and react-native-fs for file storage.
 *
 * Installation:
 *   npm install react-native-ble-plx react-native-fs
 *   npx pod-install   # iOS only
 *
 * Android permissions required in AndroidManifest.xml:
 *   <uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
 *   <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
 *   <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
 *
 * iOS permissions required in Info.plist:
 *   NSBluetoothAlwaysUsageDescription
 *
 * After recording sessions, copy JSON files from the device:
 *   Android: /sdcard/Android/data/<package>/files/pingis_sessions/
 *   iOS:     App Documents folder (accessible via Finder)
 *
 * Then on your laptop:
 *   python skills/pingis-stroke-detection/scripts/preprocess.py
 *   python skills/pingis-stroke-detection/scripts/train_rf.py
 */

import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import { BleManager, Device, BleError, Characteristic } from 'react-native-ble-plx';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ImuSample {
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  mag_x: number;
  mag_y: number;
  mag_z: number;
  ts_ms: number;
}

interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

type StrokeType = 'forehand' | 'backhand';
type ConnectionState = 'idle' | 'scanning' | 'connecting' | 'connected' | 'error';

// ── BLE Constants ─────────────────────────────────────────────────────────────

const SERVICE_UUID   = '07C80000-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID     = '07C80001-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID_ALT = '07C80203-07C8-07C8-07C8-07C807C807C8';
const GYRO_UUID      = '07C80004-07C8-07C8-07C8-07C807C807C8';
const MAG_UUID       = '07C80010-07C8-07C8-07C8-07C807C807C8';

// ── Data constants ────────────────────────────────────────────────────────────

const BUFFER_MS   = 3000;   // circular buffer size
const BEFORE_MS   = 500;    // ms before label tap
const AFTER_MS    = 500;    // ms after label tap (wait before saving)

const SESSION_DIR = `${RNFS.ExternalDirectoryPath ?? RNFS.DocumentDirectoryPath}/pingis_sessions`;

// ── BLE packet parser (mirrors BlePacketParser.kt exactly) ───────────────────

interface ParsedVector {
  type: 'accel' | 'gyro' | 'mag';
  x: number;
  y: number;
  z: number;
}

function parsePacket(uuid: string, base64Data: string): ParsedVector | null {
  const bytes = Buffer.from(base64Data, 'base64');
  if (bytes.length < 9) return null;

  // Big-endian signed Int16
  const x = bytes.readInt16BE(0);
  const y = bytes.readInt16BE(2);
  const z = bytes.readInt16BE(4);

  const uuidUpper = uuid.toUpperCase();

  if (uuidUpper === ACCEL_UUID || uuidUpper === ACCEL_UUID_ALT) {
    return { type: 'accel', x, y, z };
  }
  if (uuidUpper === GYRO_UUID) {
    return { type: 'gyro', x, y, z };
  }
  if (uuidUpper === MAG_UUID) {
    // Canonical transform: invert + divide by 10 → microtesla
    return { type: 'mag', x: -x / 10, y: -y / 10, z: -z / 10 };
  }
  return null;
}

// ── Latest sensor values (mutable, not React state for performance) ───────────

function makeLatestValues() {
  return {
    accel: { x: 0, y: 0, z: 0 },
    gyro:  { x: 0, y: 0, z: 0 },
    mag:   { x: 0, y: 0, z: 0 },
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

export function DataCollectionScreen() {
  const [connState, setConnState] = useState<ConnectionState>('idle');
  const [strokeType, setStrokeType] = useState<StrokeType>('forehand');
  const [events, setEvents] = useState<LabeledEvent[]>([]);
  const [sampleHz, setSampleHz] = useState(0);
  const [lastFeedback, setLastFeedback] = useState<string | null>(null);

  const bleManager = useRef(new BleManager()).current;
  const deviceRef = useRef<Device | null>(null);
  const buffer = useRef<ImuSample[]>([]);
  const latest = useRef(makeLatestValues());
  const sampleCountRef = useRef(0);
  const hzIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── BLE permissions ────────────────────────────────────────────────────────

  const requestPermissions = useCallback(async (): Promise<boolean> => {
    if (Platform.OS !== 'android') return true;

    const api = Platform.Version as number;
    if (api >= 31) {
      const results = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);
      return Object.values(results).every(r => r === PermissionsAndroid.RESULTS.GRANTED);
    } else {
      const result = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
      );
      return result === PermissionsAndroid.RESULTS.GRANTED;
    }
  }, []);

  // ── Sample emission (mirrors SkatingRepository.onSensorSample) ────────────

  const emitSample = useCallback(() => {
    const now = Date.now();
    const l = latest.current;
    const sample: ImuSample = {
      accel_x: l.accel.x,
      accel_y: l.accel.y,
      accel_z: l.accel.z,
      gyro_x:  l.gyro.x,
      gyro_y:  l.gyro.y,
      gyro_z:  l.gyro.z,
      mag_x:   l.mag.x,
      mag_y:   l.mag.y,
      mag_z:   l.mag.z,
      ts_ms:   now,
    };

    buffer.current.push(sample);
    sampleCountRef.current += 1;

    // Trim buffer to BUFFER_MS
    const cutoff = now - BUFFER_MS;
    let i = 0;
    while (i < buffer.current.length && buffer.current[i].ts_ms < cutoff) i++;
    if (i > 0) buffer.current = buffer.current.slice(i);
  }, []);

  // ── BLE notification handler ───────────────────────────────────────────────

  const handleNotification = useCallback(
    (_error: BleError | null, characteristic: Characteristic | null) => {
      if (!characteristic?.value || !characteristic.uuid) return;

      const parsed = parsePacket(characteristic.uuid, characteristic.value);
      if (!parsed) return;

      const l = latest.current;
      if (parsed.type === 'accel') {
        l.accel = { x: parsed.x, y: parsed.y, z: parsed.z };
      } else if (parsed.type === 'gyro') {
        l.gyro = { x: parsed.x, y: parsed.y, z: parsed.z };
      } else if (parsed.type === 'mag') {
        l.mag = { x: parsed.x, y: parsed.y, z: parsed.z };
      }

      emitSample();
    },
    [emitSample]
  );

  // ── Connect ────────────────────────────────────────────────────────────────

  const connect = useCallback(async () => {
    const granted = await requestPermissions();
    if (!granted) {
      Alert.alert('Permission denied', 'Bluetooth permission is required.');
      return;
    }

    setConnState('scanning');

    bleManager.startDeviceScan(
      [SERVICE_UUID],
      { allowDuplicates: false },
      async (error, device) => {
        if (error) {
          setConnState('error');
          Alert.alert('Scan error', error.message);
          return;
        }
        if (!device) return;

        bleManager.stopDeviceScan();
        setConnState('connecting');

        try {
          const connected = await device.connect();
          await connected.discoverAllServicesAndCharacteristics();

          deviceRef.current = connected;
          setConnState('connected');

          // Subscribe to all sensor characteristics
          for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
            try {
              connected.monitorCharacteristicForService(
                SERVICE_UUID, uuid, handleNotification
              );
            } catch (_) { /* UUID may not exist on this device */ }
          }

          // Start Hz counter
          let lastCount = 0;
          hzIntervalRef.current = setInterval(() => {
            const delta = sampleCountRef.current - lastCount;
            lastCount = sampleCountRef.current;
            setSampleHz(delta);
          }, 1000);

          connected.onDisconnected(() => {
            setConnState('idle');
            if (hzIntervalRef.current) clearInterval(hzIntervalRef.current);
          });
        } catch (e: any) {
          setConnState('error');
          Alert.alert('Connection failed', e.message);
        }
      }
    );
  }, [bleManager, handleNotification, requestPermissions]);

  const disconnect = useCallback(() => {
    deviceRef.current?.cancelConnection();
    bleManager.stopDeviceScan();
    setConnState('idle');
    if (hzIntervalRef.current) clearInterval(hzIntervalRef.current);
  }, [bleManager]);

  // ── Label capture ──────────────────────────────────────────────────────────

  const captureEvent = useCallback(
    (label: LabeledEvent['label']) => {
      const tapMs = Date.now();

      // Wait AFTER_MS then slice the window
      setTimeout(() => {
        const snap = [...buffer.current];
        const window = snap.filter(
          s => s.ts_ms >= tapMs - BEFORE_MS && s.ts_ms <= tapMs + AFTER_MS
        );

        if (window.length < 10) {
          setLastFeedback(`⚠ Too few samples (${window.length}) — is sensor streaming?`);
          return;
        }

        const event: LabeledEvent = {
          label,
          stroke_type: label === 'idle' ? 'unknown' : strokeType,
          recorded_at: new Date(tapMs).toISOString(),
          samples: window,
        };

        setEvents(prev => {
          const next = [...prev, event];
          setLastFeedback(
            `${label.toUpperCase()} (${event.stroke_type}) — ${window.length} samples`
          );
          return next;
        });
      }, AFTER_MS + 50); // wait for trailing samples to arrive
    },
    [strokeType]
  );

  // ── Save session ───────────────────────────────────────────────────────────

  const saveSession = useCallback(async () => {
    if (events.length === 0) {
      Alert.alert('No data', 'Record some events first.');
      return;
    }

    try {
      await RNFS.mkdir(SESSION_DIR);

      const date = new Date().toISOString().slice(0, 10);
      let n = 1;
      let filePath: string;

      do {
        filePath = `${SESSION_DIR}/session_${date}_${String(n).padStart(3, '0')}.json`;
        n++;
      } while (await RNFS.exists(filePath));

      await RNFS.writeFile(filePath, JSON.stringify(events, null, 2), 'utf8');

      Alert.alert(
        'Session saved',
        `${events.length} events saved to:\n${filePath}\n\nTransfer to laptop and run preprocess.py`,
        [{ text: 'OK', onPress: () => setEvents([]) }]
      );
    } catch (e: any) {
      Alert.alert('Save failed', e.message);
    }
  }, [events]);

  // ── Cleanup ────────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      bleManager.destroy();
      if (hzIntervalRef.current) clearInterval(hzIntervalRef.current);
    };
  }, [bleManager]);

  // ── Render ─────────────────────────────────────────────────────────────────

  const labelCounts = events.reduce<Record<string, number>>(
    (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
    {}
  );

  const connLabel: Record<ConnectionState, string> = {
    idle:       '○  Disconnected',
    scanning:   '⟳  Scanning...',
    connecting: '⟳  Connecting...',
    connected:  `●  Connected  |  ${sampleHz}Hz`,
    error:      '✕  Error',
  };

  return (
    <ScrollView style={s.root} contentContainerStyle={s.content}>
      <Text style={s.title}>Pingis Data Collection</Text>

      {/* Connection */}
      <View style={[s.statusBar, connState === 'connected' ? s.statusOk : s.statusOff]}>
        <Text style={s.statusText}>{connLabel[connState]}</Text>
        <TouchableOpacity
          style={s.connBtn}
          onPress={connState === 'connected' ? disconnect : connect}
        >
          <Text style={s.connBtnText}>
            {connState === 'connected' ? 'Disconnect' : 'Connect'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Stroke type selector */}
      <Text style={s.sectionLabel}>Stroke Type</Text>
      <View style={s.row}>
        {(['forehand', 'backhand'] as StrokeType[]).map(t => (
          <TouchableOpacity
            key={t}
            style={[s.typeBtn, strokeType === t && s.typeBtnOn]}
            onPress={() => setStrokeType(t)}
          >
            <Text style={[s.typeBtnTxt, strokeType === t && s.typeBtnTxtOn]}>
              {t.toUpperCase()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Action buttons */}
      <Text style={s.sectionLabel}>Label Event</Text>

      <TouchableOpacity
        style={[s.actionBtn, s.hitBtn]}
        onPress={() => captureEvent('hit')}
        activeOpacity={0.7}
      >
        <Text style={s.actionBtnTxt}>HIT</Text>
        <Text style={s.actionBtnSub}>tap at moment of ball contact</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[s.actionBtn, s.missBtn]}
        onPress={() => captureEvent('swing_miss')}
        activeOpacity={0.7}
      >
        <Text style={s.actionBtnTxt}>MISS</Text>
        <Text style={s.actionBtnSub}>full swing without ball contact</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[s.actionBtn, s.idleBtn]}
        onPress={() => captureEvent('idle')}
        activeOpacity={0.7}
      >
        <Text style={s.actionBtnTxt}>IDLE</Text>
        <Text style={s.actionBtnSub}>at rest / between strokes (2–3 sec)</Text>
      </TouchableOpacity>

      {/* Feedback */}
      {lastFeedback && <Text style={s.feedback}>{lastFeedback}</Text>}

      {/* Session stats */}
      <View style={s.statsBox}>
        <Text style={s.statsTitle}>Session ({events.length} events)</Text>
        <Text style={s.statsRow}>hit: {labelCounts.hit ?? 0}</Text>
        <Text style={s.statsRow}>swing_miss: {labelCounts.swing_miss ?? 0}</Text>
        <Text style={s.statsRow}>idle: {labelCounts.idle ?? 0}</Text>
        {events.length === 0 && (
          <Text style={s.statsHint}>Target: 20+ hits, 20+ misses, 20+ idle</Text>
        )}
      </View>

      {/* Save / Clear */}
      <View style={s.row}>
        <TouchableOpacity style={[s.secBtn, s.saveBtn]} onPress={saveSession}>
          <Text style={s.secBtnTxt}>Save Session</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[s.secBtn, s.clearBtn]}
          onPress={() =>
            Alert.alert('Clear?', 'Discard all unsaved events?', [
              { text: 'Cancel' },
              { text: 'Clear', onPress: () => setEvents([]) },
            ])
          }
        >
          <Text style={s.secBtnTxt}>Clear</Text>
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      <View style={s.guide}>
        <Text style={s.guideTitle}>Instructions</Text>
        <Text style={s.guideLine}>1. Tap Connect to find your AirHive</Text>
        <Text style={s.guideLine}>2. Select Forehand or Backhand before each stroke</Text>
        <Text style={s.guideLine}>3. Perform stroke → tap HIT at ball contact</Text>
        <Text style={s.guideLine}>4. For a swing without contact → tap MISS</Text>
        <Text style={s.guideLine}>5. Sit still 2s → tap IDLE between strokes</Text>
        <Text style={s.guideLine}>6. Tap Save Session when done</Text>
        <Text style={s.guideLine}>7. Transfer JSON to laptop → run preprocess.py</Text>
      </View>
    </ScrollView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  root:          { flex: 1, backgroundColor: '#0d0d0d' },
  content:       { padding: 20, paddingBottom: 50 },
  title:         { color: '#fff', fontSize: 22, fontWeight: '700', marginBottom: 16 },

  statusBar:     { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderRadius: 8, padding: 12, marginBottom: 20 },
  statusOk:      { backgroundColor: '#0d2d1a' },
  statusOff:     { backgroundColor: '#1a1a1a' },
  statusText:    { color: '#aaa', fontSize: 13, flex: 1 },
  connBtn:       { backgroundColor: '#222', borderRadius: 6, paddingHorizontal: 14, paddingVertical: 6 },
  connBtnText:   { color: '#aaa', fontSize: 13 },

  sectionLabel:  { color: '#555', fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 8, marginTop: 16 },
  row:           { flexDirection: 'row', gap: 12 },

  typeBtn:       { flex: 1, padding: 14, borderRadius: 8, borderWidth: 1, borderColor: '#333', alignItems: 'center' },
  typeBtnOn:     { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  typeBtnTxt:    { color: '#555', fontWeight: '600', fontSize: 14 },
  typeBtnTxtOn:  { color: '#4a9eff' },

  actionBtn:     { borderRadius: 12, padding: 22, marginBottom: 12, alignItems: 'center' },
  hitBtn:        { backgroundColor: '#0d3d1a' },
  missBtn:       { backgroundColor: '#3d0d0d' },
  idleBtn:       { backgroundColor: '#1a1a2d' },
  actionBtnTxt:  { color: '#fff', fontSize: 20, fontWeight: '700', letterSpacing: 2 },
  actionBtnSub:  { color: '#666', fontSize: 11, marginTop: 4 },

  feedback:      { color: '#4a9eff', textAlign: 'center', fontSize: 13, marginVertical: 10 },

  statsBox:      { backgroundColor: '#141414', borderRadius: 10, padding: 16, marginTop: 12 },
  statsTitle:    { color: '#fff', fontWeight: '600', marginBottom: 8 },
  statsRow:      { color: '#888', fontSize: 14, marginBottom: 3 },
  statsHint:     { color: '#444', fontSize: 12, fontStyle: 'italic', marginTop: 4 },

  secBtn:        { flex: 1, padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 12 },
  saveBtn:       { backgroundColor: '#0d2d0d' },
  clearBtn:      { backgroundColor: '#2d0d0d' },
  secBtnTxt:     { color: '#aaa', fontWeight: '600' },

  guide:         { marginTop: 24, backgroundColor: '#111', borderRadius: 10, padding: 16 },
  guideTitle:    { color: '#555', fontWeight: '600', marginBottom: 8 },
  guideLine:     { color: '#444', fontSize: 13, marginBottom: 4 },
});
