/**
 * CalibrationScreen.tsx
 *
 * 1. Ansluter till AirHive via BLE
 * 2. Visar en visuell guide: lägg sensorn plant på bordet
 * 3. Detekterar automatiskt stabilitet: gyro_mag < 5 °/s i ≥150 samples (~3s)
 * 4. Beräknar gravity-baseline (medel accel) och gyro-bias (medel gyro)
 * 5. Visar bekräftelse med gravitationsvektor-indikator
 * 6. Skickar CalibrationData + Device-instansen vidare (undviker reconnect)
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
  ScrollView,
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

const STABLE_SAMPLES_NEEDED = 150;
const GYRO_STABLE_THRESHOLD = 5.0;

// ── Paketparser ───────────────────────────────────────────────────────────────

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

// ── Visuell sensor-på-bord diagram ────────────────────────────────────────────

function SensorOnTableDiagram() {
  return (
    <View style={d.wrap}>
      {/* Sensor ovanifrån */}
      <View style={d.sensor}>
        <View style={d.sensorLed} />
        <Text style={d.sensorText}>AirHive</Text>
        <Text style={d.sensorSub}>displayen uppåt</Text>
      </View>

      {/* Pil nedåt = gravitation */}
      <View style={d.arrowCol}>
        <View style={d.arrowLine} />
        <View style={d.arrowHead} />
        <Text style={d.arrowLabel}>gravitation</Text>
      </View>

      {/* Bords-yta */}
      <View style={d.tableRow}>
        <View style={d.tableLine} />
        <Text style={d.tableLabel}>BORD</Text>
        <View style={d.tableLine} />
      </View>
    </View>
  );
}

// ── Gravitationsvektor-bekräftelse ────────────────────────────────────────────

function GravityIndicator({ gravity }: { gravity: { x: number; y: number; z: number } }) {
  const axes = [
    { label: 'X', value: gravity.x },
    { label: 'Y', value: gravity.y },
    { label: 'Z', value: gravity.z },
  ];
  const maxAbs = Math.max(...axes.map(a => Math.abs(a.value)));
  const dominant = axes.reduce((a, b) =>
    Math.abs(a.value) > Math.abs(b.value) ? a : b,
  );

  return (
    <View style={g.wrap}>
      <Text style={g.title}>GRAVITATIONSVEKTOR</Text>
      {axes.map(({ label, value }) => {
        const pct = maxAbs > 0 ? Math.abs(value) / maxAbs : 0;
        const isDom = label === dominant.label;
        return (
          <View key={label} style={g.row}>
            <Text style={[g.axisLabel, isDom && g.axisLabelDom]}>{label}</Text>
            <View style={g.barBg}>
              <View
                style={[
                  g.barFill,
                  { width: `${pct * 100}%` as any },
                  isDom ? g.barDom : g.barOther,
                ]}
              />
            </View>
            <Text style={[g.valText, isDom && g.valTextDom]}>
              {value.toFixed(0)}
            </Text>
          </View>
        );
      })}
      <Text style={g.hint}>
        {dominant.label}-axeln dominerar — gravitationen ser korrekt ut
      </Text>
    </View>
  );
}

// ── Typer ─────────────────────────────────────────────────────────────────────

type ConnState = 'idle' | 'scanning' | 'connecting' | 'connected';
type CalState  = 'waiting' | 'measuring' | 'done';

interface Props {
  setup: PlayerSetup;
  onCalibrated: (cal: CalibrationData, device: Device) => void;
  onBack: () => void;
}

const bleManager = new BleManager();

// ── Komponent ─────────────────────────────────────────────────────────────────

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

  const stableBufferRef = useRef<ImuSample[]>([]);
  const consecutiveStableRef = useRef(0);
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const calDoneRef = useRef(false);
  const calStateRef = useRef<CalState>('waiting');

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

  // ── Stabilitetsdetektering ───────────────────────────────────────────────────

  const processSampleForCalibration = useCallback((sample: ImuSample) => {
    if (calDoneRef.current) return;

    const gyroMag = Math.sqrt(
      sample.gyro_x ** 2 + sample.gyro_y ** 2 + sample.gyro_z ** 2,
    );

    if (gyroMag < GYRO_STABLE_THRESHOLD) {
      consecutiveStableRef.current += 1;
      stableBufferRef.current.push(sample);
      if (stableBufferRef.current.length > STABLE_SAMPLES_NEEDED) {
        stableBufferRef.current.shift();
      }
      setStableCount(consecutiveStableRef.current);

      if (consecutiveStableRef.current >= STABLE_SAMPLES_NEEDED) {
        calDoneRef.current = true;
        const buf = stableBufferRef.current;
        const n = buf.length;

        const cal: CalibrationData = {
          gravity: {
            x: buf.reduce((s, x) => s + x.accel_x, 0) / n,
            y: buf.reduce((s, x) => s + x.accel_y, 0) / n,
            z: buf.reduce((s, x) => s + x.accel_z, 0) / n,
          },
          gyro_bias: {
            x: buf.reduce((s, x) => s + x.gyro_x, 0) / n,
            y: buf.reduce((s, x) => s + x.gyro_y, 0) / n,
            z: buf.reduce((s, x) => s + x.gyro_z, 0) / n,
          },
        };
        setCalibration(cal);
        setCalState('done');
        calStateRef.current = 'done';
      }
    } else {
      consecutiveStableRef.current = 0;
      stableBufferRef.current = [];
      setStableCount(0);
      if (calStateRef.current !== 'measuring') {
        setCalState('measuring');
        calStateRef.current = 'measuring';
      }
    }
  }, []);

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
        calStateRef.current = 'measuring';

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
    (stableCount / STABLE_SAMPLES_NEEDED) * 100,
    100,
  );

  return (
    <ScrollView style={s.root} contentContainerStyle={s.content}>
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

      <Text style={s.title}>Kalibrering</Text>

      {/* ── STEG 1: Anslut ── */}
      {connState !== 'connected' && (
        <View style={s.stepBox}>
          <Text style={s.stepNum}>STEG 1 AV 2</Text>
          <Text style={s.stepTitle}>Anslut sensorn</Text>
          <Text style={s.stepText}>
            Se till att AirHive är påslagen och nära telefonen.
          </Text>
          <TouchableOpacity
            style={[s.connectBtn, connState !== 'idle' && s.connectBtnBusy]}
            onPress={connect}
            disabled={connState !== 'idle'}
          >
            <Text style={s.connectBtnTxt}>
              {connState === 'idle'
                ? 'Anslut AirHive'
                : connState === 'scanning'
                ? 'Skannar efter sensor...'
                : 'Kopplar upp...'}
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {/* ── STEG 2: Lägg på bordet ── */}
      {connState === 'connected' && calState !== 'done' && (
        <View style={s.stepBox}>
          <Text style={s.stepNum}>STEG 2 AV 2  ·  {sampleHz} Hz</Text>
          <Text style={s.stepTitle}>Lägg sensorn plant på bordet</Text>
          <Text style={s.stepText}>
            Placera AirHive-sensorn med <Text style={s.stepEmphasis}>displayen uppåt</Text>, plant och orörlig på bordet.{'\n'}
            Håll den still tills mätningen är klar (~3 sekunder).
          </Text>

          {/* Visuell guide */}
          <SensorOnTableDiagram />

          {/* Progress */}
          <View style={s.progressBg}>
            <View style={[s.progressFill, { width: `${progressPct}%` as any }]} />
          </View>
          <Text style={s.progressTxt}>
            {stableCount < STABLE_SAMPLES_NEEDED
              ? `Mäter stabilitet... ${stableCount} / ${STABLE_SAMPLES_NEEDED}`
              : 'Klar!'}
          </Text>
          <Text style={s.hintTxt}>
            Rör du sensorn nollställs räknaren — håll helt still.
          </Text>
        </View>
      )}

      {/* ── Kalibrering klar ── */}
      {calState === 'done' && calibration && (
        <>
          <View style={s.doneBox}>
            <Text style={s.doneTick}>✓</Text>
            <Text style={s.doneTitle}>Kalibrering klar!</Text>
            <Text style={s.doneSubtitle}>
              Granska att gravitationsvektorn ser rimlig ut nedan:
            </Text>
          </View>

          <GravityIndicator gravity={calibration.gravity} />

          <View style={s.biasBox}>
            <Text style={s.biasTitle}>GYRO-BIAS (°/s)</Text>
            <Text style={s.biasTxt}>
              X {calibration.gyro_bias.x.toFixed(2)}  ·  Y {calibration.gyro_bias.y.toFixed(2)}  ·  Z {calibration.gyro_bias.z.toFixed(2)}
            </Text>
          </View>

          <TouchableOpacity
            style={s.startBtn}
            onPress={() =>
              deviceRef.current && onCalibrated(calibration, deviceRef.current)
            }
            activeOpacity={0.7}
          >
            <Text style={s.startBtnTxt}>Ser bra ut — Börja spela in →</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={s.recalBtn}
            onPress={() => {
              calDoneRef.current = false;
              consecutiveStableRef.current = 0;
              stableBufferRef.current = [];
              setStableCount(0);
              setCalibration(null);
              setCalState('measuring');
              calStateRef.current = 'measuring';
            }}
          >
            <Text style={s.recalTxt}>Kalibrera om</Text>
          </TouchableOpacity>
        </>
      )}
    </ScrollView>
  );
}

// ── SensorOnTable styles ──────────────────────────────────────────────────────

const d = StyleSheet.create({
  wrap:       { alignItems: 'center', marginVertical: 20 },
  sensor: {
    width: 140,
    height: 80,
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#4a9eff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  sensorLed:  { width: 10, height: 10, borderRadius: 5, backgroundColor: '#2ecc71', marginBottom: 6 },
  sensorText: { color: '#4a9eff', fontWeight: '700', fontSize: 13 },
  sensorSub:  { color: '#778', fontSize: 10, marginTop: 2 },

  arrowCol:   { alignItems: 'center', marginBottom: 4 },
  arrowLine:  { width: 2, height: 20, backgroundColor: '#555' },
  arrowHead: {
    width: 0, height: 0,
    borderLeftWidth: 6, borderRightWidth: 6, borderTopWidth: 10,
    borderLeftColor: 'transparent', borderRightColor: 'transparent', borderTopColor: '#555',
    marginBottom: 2,
  },
  arrowLabel: { color: '#778', fontSize: 10 },

  tableRow:   { flexDirection: 'row', alignItems: 'center', gap: 8 },
  tableLine:  { flex: 1, height: 2, backgroundColor: '#444' },
  tableLabel: { color: '#aaa', fontSize: 11, fontWeight: '600', letterSpacing: 2 },
});

// ── GravityIndicator styles ───────────────────────────────────────────────────

const g = StyleSheet.create({
  wrap:         { backgroundColor: '#111', borderRadius: 12, padding: 16, marginBottom: 16 },
  title:        { color: '#666', fontSize: 10, letterSpacing: 2, marginBottom: 12 },
  row:          { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  axisLabel:    { color: '#666', fontSize: 13, fontWeight: '700', width: 20 },
  axisLabelDom: { color: '#2ecc71' },
  barBg:        { flex: 1, height: 8, backgroundColor: '#1a1a1a', borderRadius: 4, marginHorizontal: 10, overflow: 'hidden' },
  barFill:      { height: '100%', borderRadius: 4 },
  barDom:       { backgroundColor: '#2ecc71' },
  barOther:     { backgroundColor: '#2a2a2a' },
  valText:      { color: '#666', fontSize: 11, width: 52, textAlign: 'right', fontFamily: 'monospace' },
  valTextDom:   { color: '#2ecc71' },
  hint:         { color: '#2ecc71', fontSize: 12, marginTop: 8, textAlign: 'center' },
});

// ── Main styles ───────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 40 },

  header:    { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 },
  backBtn:   { padding: 4 },
  backTxt:   { color: '#888', fontSize: 14 },
  playerTxt: { color: '#666', fontSize: 12 },

  title:     { color: '#fff', fontSize: 26, fontWeight: '800', marginBottom: 20 },

  stepBox:      { backgroundColor: '#111', borderRadius: 14, padding: 20, marginBottom: 16 },
  stepNum:      { color: '#555', fontSize: 10, letterSpacing: 2, marginBottom: 6 },
  stepTitle:    { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 10 },
  stepText:     { color: '#888', fontSize: 14, lineHeight: 22 },
  stepEmphasis: { color: '#fff', fontWeight: '700' },

  connectBtn:     { backgroundColor: '#0d2d0d', borderRadius: 12, padding: 18, alignItems: 'center', marginTop: 16 },
  connectBtnBusy: { backgroundColor: '#1a1a1a' },
  connectBtnTxt:  { color: '#2ecc71', fontWeight: '700', fontSize: 16 },

  progressBg:   { height: 8, backgroundColor: '#1a1a1a', borderRadius: 4, marginTop: 20, marginBottom: 10, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#2ecc71', borderRadius: 4 },
  progressTxt:  { color: '#aaa', fontSize: 13, textAlign: 'center' },
  hintTxt:      { color: '#666', fontSize: 12, textAlign: 'center', marginTop: 8 },

  doneBox:      { backgroundColor: '#0d2d1a', borderRadius: 14, padding: 20, alignItems: 'center', marginBottom: 16 },
  doneTick:     { fontSize: 40, marginBottom: 6 },
  doneTitle:    { color: '#2ecc71', fontSize: 22, fontWeight: '800', marginBottom: 4 },
  doneSubtitle: { color: '#888', fontSize: 13 },

  biasBox:   { backgroundColor: '#111', borderRadius: 10, padding: 14, marginBottom: 16 },
  biasTitle: { color: '#555', fontSize: 10, letterSpacing: 2, marginBottom: 6 },
  biasTxt:   { color: '#888', fontSize: 12, fontFamily: 'monospace' },

  startBtn:    { backgroundColor: '#0d2d0d', borderRadius: 12, padding: 20, alignItems: 'center', marginBottom: 12 },
  startBtnTxt: { color: '#2ecc71', fontWeight: '800', fontSize: 17, letterSpacing: 1 },

  recalBtn:  { padding: 14, alignItems: 'center' },
  recalTxt:  { color: '#666', fontSize: 14 },
});
