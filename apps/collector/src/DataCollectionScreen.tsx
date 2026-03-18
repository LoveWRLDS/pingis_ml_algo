/**
 * DataCollectionScreen.tsx — Pingis träningsdatainsamlare
 *
 * Kopplar upp mot BERG AirHive via BLE, streamer IMU-data i en
 * cirkulär buffer, och sparar märkta events (hit/miss/idle) till fil.
 *
 * Filerna sparas till:
 *   Android: /sdcard/Android/data/com.collectorapp/files/pingis_sessions/
 *
 * Kopiera till laptopen med ADB:
 *   adb pull /sdcard/Android/data/com.collectorapp/files/pingis_sessions/ ./data/raw/
 *
 * Sedan:
 *   python skills/pingis-stroke-detection/scripts/preprocess.py
 *   python skills/pingis-stroke-detection/scripts/train_rf.py
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  PermissionsAndroid,
  Platform,
  StatusBar,
} from 'react-native';
import { BleManager } from 'react-native-ble-plx';
import type { Device, BleError, Characteristic } from 'react-native-ble-plx';
import RNFS from 'react-native-fs';

// ── BLE UUIDs (från sensor-protocol.md) ──────────────────────────────────────

const SERVICE_UUID   = '07C80000-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID     = '07C80001-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID_ALT = '07C80203-07C8-07C8-07C8-07C807C807C8';
const GYRO_UUID      = '07C80004-07C8-07C8-07C8-07C807C807C8';
const MAG_UUID       = '07C80010-07C8-07C8-07C8-07C807C807C8';

// ── Konstanter ────────────────────────────────────────────────────────────────

const BUFFER_MS = 3000;
const BEFORE_MS = 500;
const AFTER_MS  = 500;
const SESSION_DIR = `${RNFS.ExternalDirectoryPath ?? RNFS.DocumentDirectoryPath}/pingis_sessions`;

// ── Typer ─────────────────────────────────────────────────────────────────────

interface ImuSample {
  accel_x: number; accel_y: number; accel_z: number;
  gyro_x: number;  gyro_y: number;  gyro_z: number;
  mag_x: number;   mag_y: number;   mag_z: number;
  ts_ms: number;
}

interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

type StrokeType = 'forehand' | 'backhand';
type ConnState = 'idle' | 'scanning' | 'connecting' | 'connected' | 'error';

// ── BLE-paketparser (speglar BlePacketParser.kt exakt) ────────────────────────

function parsePacket(
  uuid: string,
  base64Data: string,
): { type: 'accel' | 'gyro' | 'mag'; x: number; y: number; z: number } | null {
  // Avkoda base64 → bytes utan Buffer-polyfill
  const binaryStr = atob(base64Data);
  if (binaryStr.length < 9) return null;

  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }

  const view = new DataView(bytes.buffer);
  const x = view.getInt16(0, false); // big-endian
  const y = view.getInt16(2, false);
  const z = view.getInt16(4, false);

  const u = uuid.toUpperCase();
  if (u === ACCEL_UUID || u === ACCEL_UUID_ALT) {
    return { type: 'accel', x, y, z };
  }
  if (u === GYRO_UUID) {
    return { type: 'gyro', x, y, z };
  }
  if (u === MAG_UUID) {
    // Kanonisk transform: invertera + dela på 10 → mikrotesla
    return { type: 'mag', x: -x / 10, y: -y / 10, z: -z / 10 };
  }
  return null;
}

// ── Komponent ─────────────────────────────────────────────────────────────────

const bleManager = new BleManager();

export function DataCollectionScreen() {
  const [connState, setConnState] = useState<ConnState>('idle');
  const [strokeType, setStrokeType] = useState<StrokeType>('forehand');
  const [events, setEvents] = useState<LabeledEvent[]>([]);
  const [sampleHz, setSampleHz] = useState(0);
  const [feedback, setFeedback] = useState<string | null>(null);

  const deviceRef = useRef<Device | null>(null);
  const bufferRef = useRef<ImuSample[]>([]);
  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro:  { x: 0, y: 0, z: 0 },
    mag:   { x: 0, y: 0, z: 0 },
  });
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
      return Object.values(results).every(
        r => r === PermissionsAndroid.RESULTS.GRANTED,
      );
    }
    const r = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
    );
    return r === PermissionsAndroid.RESULTS.GRANTED;
  }, []);

  // ── BLE-notifikationshanterare ──────────────────────────────────────────────

  const handleNotification = useCallback(
    (_err: BleError | null, char: Characteristic | null) => {
      if (!char?.value || !char.uuid) return;
      const parsed = parsePacket(char.uuid, char.value);
      if (!parsed) return;

      const l = latestRef.current;
      if (parsed.type === 'accel') l.accel = parsed;
      else if (parsed.type === 'gyro') l.gyro = parsed;
      else if (parsed.type === 'mag') l.mag = parsed;

      const now = Date.now();
      const sample: ImuSample = {
        accel_x: l.accel.x, accel_y: l.accel.y, accel_z: l.accel.z,
        gyro_x:  l.gyro.x,  gyro_y:  l.gyro.y,  gyro_z:  l.gyro.z,
        mag_x:   l.mag.x,   mag_y:   l.mag.y,   mag_z:   l.mag.z,
        ts_ms: now,
      };

      bufferRef.current.push(sample);
      sampleCountRef.current += 1;

      // Trimma buffern
      const cutoff = now - BUFFER_MS;
      let i = 0;
      while (i < bufferRef.current.length && bufferRef.current[i].ts_ms < cutoff) i++;
      if (i > 0) bufferRef.current = bufferRef.current.slice(i);
    },
    [],
  );

  // ── Anslut ──────────────────────────────────────────────────────────────────

  const connect = useCallback(async () => {
    if (!(await requestPermissions())) {
      Alert.alert('Behörighet nekad', 'Bluetooth-behörighet krävs.');
      return;
    }

    setConnState('scanning');
    setFeedback('Letar efter AirHive...');

    bleManager.startDeviceScan(
      null,
      { allowDuplicates: false },
      async (error, device) => {
        if (error) {
          setConnState('error');
          setFeedback(`Skanningsfel: ${error.message}`);
          return;
        }
        if (!device) return;

        const name = device.name ?? '';
        if (
          !name.toLowerCase().includes('airhive') &&
          !name.toLowerCase().includes('berg')
        ) return;

        bleManager.stopDeviceScan();
        setConnState('connecting');
        setFeedback(`Hittade: ${name} — kopplar...`);

        try {
          const connected = await device.connect();
          await connected.discoverAllServicesAndCharacteristics();
          deviceRef.current = connected;
          setConnState('connected');
          setFeedback(null);

          for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
            try {
              connected.monitorCharacteristicForService(
                SERVICE_UUID, uuid, handleNotification,
              );
            } catch (_) {}
          }

          // Hz-räknare
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
          setConnState('error');
          setFeedback(`Anslutning misslyckades: ${e.message}`);
        }
      },
    );
  }, [handleNotification, requestPermissions]);

  const disconnect = useCallback(() => {
    deviceRef.current?.cancelConnection();
    bleManager.stopDeviceScan();
    setConnState('idle');
    setSampleHz(0);
    if (hzTimerRef.current) clearInterval(hzTimerRef.current);
  }, []);

  // ── Märk event ──────────────────────────────────────────────────────────────

  const captureEvent = useCallback(
    (label: LabeledEvent['label']) => {
      const tapMs = Date.now();

      setTimeout(() => {
        const snap = [...bufferRef.current];
        const window = snap.filter(
          s => s.ts_ms >= tapMs - BEFORE_MS && s.ts_ms <= tapMs + AFTER_MS,
        );

        if (window.length < 10) {
          setFeedback(`⚠ Bara ${window.length} samples — är sensorn ansluten och streamar?`);
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
          setFeedback(
            `✓ ${label.toUpperCase()} (${event.stroke_type}) — ${window.length} samples`,
          );
          return next;
        });
      }, AFTER_MS + 50);
    },
    [strokeType],
  );

  // ── Spara session ───────────────────────────────────────────────────────────

  const saveSession = useCallback(async () => {
    if (events.length === 0) {
      Alert.alert('Ingen data', 'Spela in events först.');
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
        '✓ Session sparad',
        `${events.length} events sparade.\n\nFil: ${filePath}\n\nKopiera till laptop:\nadb pull /sdcard/Android/data/com.collectorapp/files/pingis_sessions/ ./data/raw/`,
        [{ text: 'OK', onPress: () => setEvents([]) }],
      );
    } catch (e: any) {
      Alert.alert('Fel', `Kunde inte spara: ${e.message}`);
    }
  }, [events]);

  // ── Rensa ───────────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      bleManager.destroy();
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
    };
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────

  const counts = events.reduce<Record<string, number>>(
    (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
    {},
  );

  const connLabels: Record<ConnState, string> = {
    idle:       '○  Frånkopplad',
    scanning:   '⟳  Skannar...',
    connecting: '⟳  Kopplar upp...',
    connected:  `●  Ansluten  ${sampleHz}Hz`,
    error:      '✕  Fel',
  };

  return (
    <ScrollView style={s.root} contentContainerStyle={s.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />
      <Text style={s.title}>Pingis Datainsamling</Text>

      {/* Anslutningsstatus */}
      <View style={[s.bar, connState === 'connected' ? s.barOn : s.barOff]}>
        <Text style={s.barTxt}>{connLabels[connState]}</Text>
        <TouchableOpacity
          style={s.connBtn}
          onPress={connState === 'connected' ? disconnect : connect}
        >
          <Text style={s.connBtnTxt}>
            {connState === 'connected' ? 'Koppla från' : 'Anslut'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Feedback-rad */}
      {feedback && <Text style={s.feedbackTxt}>{feedback}</Text>}

      {/* Slag-typ */}
      <Text style={s.sectionLabel}>SLAG-TYP</Text>
      <View style={s.row}>
        {(['forehand', 'backhand'] as StrokeType[]).map(t => (
          <TouchableOpacity
            key={t}
            style={[s.typeBtn, strokeType === t && s.typeBtnOn]}
            onPress={() => setStrokeType(t)}
          >
            <Text style={[s.typeTxt, strokeType === t && s.typeTxtOn]}>
              {t === 'forehand' ? 'FOREHAND' : 'BACKHAND'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Märkningsknappar */}
      <Text style={s.sectionLabel}>MÄRK EVENT</Text>

      <TouchableOpacity
        style={[s.bigBtn, s.hitBtn]}
        onPress={() => captureEvent('hit')}
        activeOpacity={0.7}
      >
        <Text style={s.bigBtnTxt}>HIT</Text>
        <Text style={s.bigBtnSub}>tryck exakt vid bollkontakt</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[s.bigBtn, s.missBtn]}
        onPress={() => captureEvent('swing_miss')}
        activeOpacity={0.7}
      >
        <Text style={s.bigBtnTxt}>MISS</Text>
        <Text style={s.bigBtnSub}>sving utan bollkontakt</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[s.bigBtn, s.idleBtn]}
        onPress={() => captureEvent('idle')}
        activeOpacity={0.7}
      >
        <Text style={s.bigBtnTxt}>IDLE</Text>
        <Text style={s.bigBtnSub}>vila / mellan slag (håll stilla 2 sek)</Text>
      </TouchableOpacity>

      {/* Sessions-statistik */}
      <View style={s.statsBox}>
        <Text style={s.statsTitle}>Session — {events.length} events</Text>
        <View style={s.statsRow}>
          <Stat label="HIT"  value={counts.hit ?? 0}        color="#2ecc71" />
          <Stat label="MISS" value={counts.swing_miss ?? 0} color="#e74c3c" />
          <Stat label="IDLE" value={counts.idle ?? 0}       color="#95a5a6" />
        </View>
        {events.length === 0 && (
          <Text style={s.statsHint}>Mål: 20+ hits + 20+ misses + 20+ idle</Text>
        )}
      </View>

      {/* Spara / Rensa */}
      <View style={s.row}>
        <TouchableOpacity style={[s.secBtn, s.saveBtn]} onPress={saveSession}>
          <Text style={s.secBtnTxt}>Spara session</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[s.secBtn, s.clearBtn]}
          onPress={() =>
            Alert.alert('Rensa?', 'Alla osparade events tas bort.', [
              { text: 'Avbryt' },
              { text: 'Rensa', style: 'destructive', onPress: () => setEvents([]) },
            ])
          }
        >
          <Text style={s.secBtnTxt}>Rensa</Text>
        </TouchableOpacity>
      </View>

      {/* Guide */}
      <View style={s.guide}>
        <Text style={s.guideTit}>Hur du spelar in:</Text>
        <Text style={s.guideLine}>1. Tryck Anslut — appen hittar din AirHive</Text>
        <Text style={s.guideLine}>2. Välj Forehand eller Backhand</Text>
        <Text style={s.guideLine}>3. Gör ett slag → tryck HIT vid bollkontakt</Text>
        <Text style={s.guideLine}>4. Sving utan träff → tryck MISS</Text>
        <Text style={s.guideLine}>5. Stå still 2 sek → tryck IDLE</Text>
        <Text style={s.guideLine}>6. Tryck Spara session när du är klar</Text>
        <Text style={[s.guideLine, { marginTop: 8, color: '#555' }]}>
          Kopiera till laptop med ADB:{'\n'}
          adb pull /sdcard/Android/data/{'\n'}com.collectorapp/files/pingis_sessions/ ./data/raw/
        </Text>
      </View>
    </ScrollView>
  );
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <View style={s.statItem}>
      <Text style={[s.statValue, { color }]}>{value}</Text>
      <Text style={s.statLabel}>{label}</Text>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  root:       { flex: 1, backgroundColor: '#0d0d0d' },
  content:    { padding: 20, paddingBottom: 50 },
  title:      { color: '#fff', fontSize: 24, fontWeight: '700', marginBottom: 16 },

  bar:        { flexDirection: 'row', alignItems: 'center', borderRadius: 10, padding: 12, marginBottom: 4 },
  barOn:      { backgroundColor: '#0d2d1a' },
  barOff:     { backgroundColor: '#1a1a1a' },
  barTxt:     { color: '#aaa', fontSize: 13, flex: 1 },
  connBtn:    { backgroundColor: '#222', borderRadius: 6, paddingHorizontal: 14, paddingVertical: 7 },
  connBtnTxt: { color: '#ccc', fontSize: 13, fontWeight: '600' },

  feedbackTxt: { color: '#4a9eff', fontSize: 13, textAlign: 'center', marginVertical: 8 },

  sectionLabel: { color: '#444', fontSize: 10, letterSpacing: 2, marginTop: 20, marginBottom: 8 },
  row:          { flexDirection: 'row', gap: 12 },

  typeBtn:    { flex: 1, padding: 14, borderRadius: 8, borderWidth: 1, borderColor: '#2a2a2a', alignItems: 'center' },
  typeBtnOn:  { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  typeTxt:    { color: '#444', fontWeight: '700', fontSize: 13 },
  typeTxtOn:  { color: '#4a9eff' },

  bigBtn:     { borderRadius: 14, padding: 24, marginBottom: 12, alignItems: 'center' },
  hitBtn:     { backgroundColor: '#0d2d1a' },
  missBtn:    { backgroundColor: '#2d0d0d' },
  idleBtn:    { backgroundColor: '#15152a' },
  bigBtnTxt:  { color: '#fff', fontSize: 22, fontWeight: '800', letterSpacing: 3 },
  bigBtnSub:  { color: '#555', fontSize: 12, marginTop: 5 },

  statsBox:   { backgroundColor: '#141414', borderRadius: 12, padding: 16, marginTop: 8 },
  statsTitle: { color: '#888', fontWeight: '600', marginBottom: 12 },
  statsRow:   { flexDirection: 'row', justifyContent: 'space-around' },
  statsHint:  { color: '#333', fontSize: 12, fontStyle: 'italic', marginTop: 10, textAlign: 'center' },
  statItem:   { alignItems: 'center' },
  statValue:  { fontSize: 28, fontWeight: '800' },
  statLabel:  { color: '#555', fontSize: 11, marginTop: 2 },

  secBtn:     { flex: 1, padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 12 },
  saveBtn:    { backgroundColor: '#0d2d0d' },
  clearBtn:   { backgroundColor: '#2d0d0d' },
  secBtnTxt:  { color: '#aaa', fontWeight: '700' },

  guide:      { marginTop: 24, backgroundColor: '#111', borderRadius: 12, padding: 16 },
  guideTit:   { color: '#555', fontWeight: '700', marginBottom: 10 },
  guideLine:  { color: '#383838', fontSize: 13, marginBottom: 5 },
});
