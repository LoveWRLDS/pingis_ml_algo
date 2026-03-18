/**
 * CalibrationScreen.tsx
 *
 * 1. Ansluter till AirHive via BLE
 * 2. Instruerar användaren att lägga sensorn still på bordet
 * 3. Detekterar automatiskt stabilitet: gyro_mag < 5 °/s i ≥150 samples (~3s)
 * 4. Beräknar gravity-baseline (medel accel) och gyro-bias (medel gyro)
 * 5. Skickar CalibrationData + Device-instansen vidare (undviker reconnect)
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  PermissionsAndroid,
  Platform,
  Alert,
} from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import type { Device, BleError, Characteristic } from 'react-native-ble-plx';
import type { PlayerSetup, CalibrationData, ImuSample } from './types';

// ── BLE UUIDs ─────────────────────────────────────────────────────────────────

const SERVICE_UUID   = '07C80000-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID     = '07C80001-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID_ALT = '07C80203-07C8-07C8-07C8-07C807C807C8';
const GYRO_UUID      = '07C80004-07C8-07C8-07C8-07C807C807C8';
const MAG_UUID       = '07C80010-07C8-07C8-07C8-07C807C807C8';

// ── Kalibreringskonstanter ────────────────────────────────────────────────────

const STABLE_SAMPLES_NEEDED = 150;   // ~3s vid 50Hz
const GYRO_STABLE_THRESHOLD = 5.0;   // °/s — under detta = sensor är still

// ── Paketparser (identisk med DataCollectionScreen) ───────────────────────────

function parsePacket(
  uuid: string,
  base64Data: string,
): { type: 'accel' | 'gyro' | 'mag'; x: number; y: number; z: number } | null {
  const binaryStr = atob(base64Data);
  if (binaryStr.length < 9) return null;
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const x = view.getInt16(0, false);
  const y = view.getInt16(2, false);
  const z = view.getInt16(4, false);
  const u = uuid.toUpperCase();
  if (u === ACCEL_UUID || u === ACCEL_UUID_ALT) return { type: 'accel', x, y, z };
  if (u === GYRO_UUID) return { type: 'gyro', x, y, z };
  if (u === MAG_UUID) return { type: 'mag', x: -x / 10, y: -y / 10, z: -z / 10 };
  return null;
}

// ── Typer ─────────────────────────────────────────────────────────────────────

type ConnState = 'idle' | 'scanning' | 'connecting' | 'connected';
type CalState  = 'waiting' | 'measuring' | 'done';

interface Props {
  setup: PlayerSetup;
  onCalibrated: (cal: CalibrationData, device: Device) => void;
  onBack: () => void;
}

// ── Komponent ─────────────────────────────────────────────────────────────────

const bleManager = new BleManager();

export function CalibrationScreen({ setup, onCalibrated, onBack }: Props) {
  const [connState, setConnState] = useState<ConnState>('idle');
  const [calState, setCalState] = useState<CalState>('waiting');
  const [stableCount, setStableCount] = useState(0);
  const [calibration, setCalibration] = useState<CalibrationData | null>(null);
  const [sampleHz, setSampleHz] = useState(0);

  const deviceRef = useRef<Device | null>(null);
  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro:  { x: 0, y: 0, z: 0 },
    mag:   { x: 0, y: 0, z: 0 },
  });

  // Stabila samples-buffer för kalibrering
  const stableBufferRef = useRef<ImuSample[]>([]);
  const consecutiveStableRef = useRef(0);

  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const calDoneRef = useRef(false);

  // ── BLE-behörigheter ────────────────────────────────────────────────────────

  const requestPermissions = useCallback(async () => {
    if (Platform.OS !== 'android') return true;
    const api = Platform.Version as number;
    if (api >= 31) {
      const results = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);
      return Object.values(results).every(r => r === PermissionsAndroid.RESULTS.GRANTED);
    }
    const r = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
    );
    return r === PermissionsAndroid.RESULTS.GRANTED;
  }, []);

  // ── Stabilitetsdetektering + kalibrering ────────────────────────────────────

  const processSampleForCalibration = useCallback((sample: ImuSample) => {
    if (calDoneRef.current) return;

    const gyroMag = Math.sqrt(
      sample.gyro_x ** 2 + sample.gyro_y ** 2 + sample.gyro_z ** 2,
    );

    if (gyroMag < GYRO_STABLE_THRESHOLD) {
      consecutiveStableRef.current += 1;
      stableBufferRef.current.push(sample);

      // Håll bara de senaste STABLE_SAMPLES_NEEDED
      if (stableBufferRef.current.length > STABLE_SAMPLES_NEEDED) {
        stableBufferRef.current.shift();
      }

      setStableCount(consecutiveStableRef.current);

      if (consecutiveStableRef.current >= STABLE_SAMPLES_NEEDED) {
        // Kalibreringen klar — beräkna medelvärden
        calDoneRef.current = true;
        const buf = stableBufferRef.current;
        const n = buf.length;

        const gravX = buf.reduce((s, x) => s + x.accel_x, 0) / n;
        const gravY = buf.reduce((s, x) => s + x.accel_y, 0) / n;
        const gravZ = buf.reduce((s, x) => s + x.accel_z, 0) / n;
        const biasX = buf.reduce((s, x) => s + x.gyro_x, 0) / n;
        const biasY = buf.reduce((s, x) => s + x.gyro_y, 0) / n;
        const biasZ = buf.reduce((s, x) => s + x.gyro_z, 0) / n;

        const cal: CalibrationData = {
          gravity:   { x: gravX, y: gravY, z: gravZ },
          gyro_bias: { x: biasX, y: biasY, z: biasZ },
        };
        setCalibration(cal);
        setCalState('done');
      }
    } else {
      // Rörelse detekterad — nollställ räknaren
      consecutiveStableRef.current = 0;
      stableBufferRef.current = [];
      setStableCount(0);
      if (calState !== 'measuring') setCalState('measuring');
    }
  }, [calState]);

  // ── BLE-notifikation ────────────────────────────────────────────────────────

  const handleNotification = useCallback(
    (_err: BleError | null, char: Characteristic | null) => {
      if (!char?.value || !char.uuid) return;
      const parsed = parsePacket(char.uuid, char.value);
      if (!parsed) return;

      const l = latestRef.current;
      if (parsed.type === 'accel') l.accel = parsed;
      else if (parsed.type === 'gyro') l.gyro = parsed;
      else if (parsed.type === 'mag') l.mag = parsed;

      sampleCountRef.current += 1;

      const sample: ImuSample = {
        accel_x: l.accel.x, accel_y: l.accel.y, accel_z: l.accel.z,
        gyro_x:  l.gyro.x,  gyro_y:  l.gyro.y,  gyro_z:  l.gyro.z,
        mag_x:   l.mag.x,   mag_y:   l.mag.y,   mag_z:   l.mag.z,
        ts_ms: Date.now(),
      };
      processSampleForCalibration(sample);
    },
    [processSampleForCalibration],
  );

  // ── Anslut ──────────────────────────────────────────────────────────────────

  const connect = useCallback(async () => {
    if (!(await requestPermissions())) {
      Alert.alert('Behörighet nekad', 'Bluetooth-behörighet krävs.');
      return;
    }

    setConnState('scanning');

    bleManager.startDeviceScan(null, { allowDuplicates: false }, async (error, device) => {
      if (error) {
        setConnState('idle');
        Alert.alert('Skanningsfel', error.message);
        return;
      }
      if (!device) return;

      const name = device.name ?? '';
      if (!name.toLowerCase().includes('airhive') && !name.toLowerCase().includes('berg')) return;

      bleManager.stopDeviceScan();
      setConnState('connecting');

      try {
        const connected = await device.connect();
        await connected.discoverAllServicesAndCharacteristics();
        deviceRef.current = connected;
        setConnState('connected');
        setCalState('measuring');

        for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
          try {
            connected.monitorCharacteristicForService(SERVICE_UUID, uuid, handleNotification);
          } catch (_) {}
        }

        let last = 0;
        hzTimerRef.current = setInterval(() => {
          setSampleHz(sampleCountRef.current - last);
          last = sampleCountRef.current;
        }, 1000);

        connected.onDisconnected(() => {
          setConnState('idle');
          setSampleHz(0);
          if (hzTimerRef.current) clearInterval(hzTimerRef.current);
        });
      } catch (e: any) {
        setConnState('idle');
        Alert.alert('Anslutning misslyckades', e.message);
      }
    });
  }, [handleNotification, requestPermissions]);

  useEffect(() => {
    return () => {
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
    };
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────

  const progressPct = Math.min(
    (consecutiveStableRef.current / STABLE_SAMPLES_NEEDED) * 100,
    100,
  );

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={onBack} style={s.backBtn}>
          <Text style={s.backTxt}>← Tillbaka</Text>
        </TouchableOpacity>
        <Text style={s.playerTxt}>
          {setup.name} · {setup.handedness === 'right' ? 'Höger' : 'Vänster'}hand
        </Text>
      </View>

      <View style={s.content}>
        <Text style={s.title}>Kalibrering</Text>

        {/* Anslutningsstatus */}
        {connState !== 'connected' && (
          <>
            <Text style={s.instruction}>
              Anslut din AirHive-sensor för att börja.
            </Text>
            <TouchableOpacity
              style={[s.connectBtn, connState !== 'idle' && s.connectBtnBusy]}
              onPress={connect}
              disabled={connState !== 'idle'}
            >
              <Text style={s.connectBtnTxt}>
                {connState === 'idle' ? 'Anslut AirHive' : connState === 'scanning' ? 'Skannar...' : 'Kopplar...'}
              </Text>
            </TouchableOpacity>
          </>
        )}

        {/* Kalibreringsinstruktion */}
        {connState === 'connected' && calState !== 'done' && (
          <>
            <View style={s.instrBox}>
              <Text style={s.instrTitle}>Lägg sensorn still</Text>
              <Text style={s.instrText}>
                Placera AirHive-sensorn plant och orörlig på bordet.{'\n'}
                Håll den still tills kalibreringen är klar (~3 sekunder).
              </Text>
            </View>

            {/* Progress-bar */}
            <View style={s.progressBg}>
              <View style={[s.progressFill, { width: `${progressPct}%` as any }]} />
            </View>
            <Text style={s.progressTxt}>
              {stableCount < STABLE_SAMPLES_NEEDED
                ? `Mäter stabilitet... ${stableCount}/${STABLE_SAMPLES_NEEDED} samples`
                : 'Klar!'}
            </Text>

            <Text style={s.hzTxt}>{sampleHz} Hz</Text>
          </>
        )}

        {/* Kalibrering klar */}
        {calState === 'done' && calibration && (
          <>
            <View style={s.doneBox}>
              <Text style={s.doneTick}>✓</Text>
              <Text style={s.doneTitle}>Kalibrering klar!</Text>
              <Text style={s.doneDetail}>
                Gravitation: x={calibration.gravity.x.toFixed(0)}  y={calibration.gravity.y.toFixed(0)}  z={calibration.gravity.z.toFixed(0)}
              </Text>
              <Text style={s.doneDetail}>
                Gyro-bias: x={calibration.gyro_bias.x.toFixed(2)}  y={calibration.gyro_bias.y.toFixed(2)}  z={calibration.gyro_bias.z.toFixed(2)} °/s
              </Text>
            </View>

            <TouchableOpacity
              style={s.startBtn}
              onPress={() => deviceRef.current && onCalibrated(calibration, deviceRef.current)}
              activeOpacity={0.7}
            >
              <Text style={s.startBtnTxt}>Börja spela in →</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  root:          { flex: 1, backgroundColor: '#0d0d0d' },
  header:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, paddingTop: 20 },
  backBtn:       { padding: 4 },
  backTxt:       { color: '#555', fontSize: 14 },
  playerTxt:     { color: '#444', fontSize: 12 },
  content:       { flex: 1, padding: 24, justifyContent: 'center' },
  title:         { color: '#fff', fontSize: 28, fontWeight: '800', marginBottom: 28 },

  instruction:   { color: '#666', fontSize: 15, marginBottom: 20, lineHeight: 22 },
  connectBtn:    { backgroundColor: '#0d2d0d', borderRadius: 12, padding: 18, alignItems: 'center' },
  connectBtnBusy:{ backgroundColor: '#1a1a1a' },
  connectBtnTxt: { color: '#2ecc71', fontWeight: '700', fontSize: 16 },

  instrBox:      { backgroundColor: '#111', borderRadius: 12, padding: 20, marginBottom: 24 },
  instrTitle:    { color: '#fff', fontWeight: '700', fontSize: 16, marginBottom: 10 },
  instrText:     { color: '#666', fontSize: 14, lineHeight: 22 },

  progressBg:    { height: 8, backgroundColor: '#1a1a1a', borderRadius: 4, marginBottom: 10, overflow: 'hidden' },
  progressFill:  { height: '100%', backgroundColor: '#2ecc71', borderRadius: 4 },
  progressTxt:   { color: '#555', fontSize: 13, textAlign: 'center', marginBottom: 8 },
  hzTxt:         { color: '#333', fontSize: 11, textAlign: 'center' },

  doneBox:       { backgroundColor: '#0d2d1a', borderRadius: 14, padding: 24, alignItems: 'center', marginBottom: 28 },
  doneTick:      { fontSize: 40, marginBottom: 8 },
  doneTitle:     { color: '#2ecc71', fontSize: 22, fontWeight: '800', marginBottom: 12 },
  doneDetail:    { color: '#555', fontSize: 12, fontFamily: 'monospace', marginBottom: 3 },

  startBtn:      { backgroundColor: '#0d2d0d', borderRadius: 12, padding: 20, alignItems: 'center' },
  startBtnTxt:   { color: '#2ecc71', fontWeight: '800', fontSize: 18, letterSpacing: 1 },
});
