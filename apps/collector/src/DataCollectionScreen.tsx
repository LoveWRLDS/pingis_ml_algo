/**
 * DataCollectionScreen.tsx
 *
 * Tar emot en redan ansluten BLE-enhet + kalibrering + spelarinformation.
 * Streamar IMU-data, låter användaren märka events, och sparar till
 * /sdcard/Download/pingis_sessions/ (synlig i Filer-appen).
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  StatusBar,
} from 'react-native';
import type { Device, BleError, Characteristic } from 'react-native-ble-plx';
import RNFS from 'react-native-fs';
import type {
  ImuSample,
  LabeledEvent,
  PlayerSetup,
  CalibrationData,
  SessionFile,
} from './types';

// ── BLE UUIDs ─────────────────────────────────────────────────────────────────

const SERVICE_UUID   = '07C80000-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID     = '07C80001-07C8-07C8-07C8-07C807C807C8';
const ACCEL_UUID_ALT = '07C80203-07C8-07C8-07C8-07C807C807C8';
const GYRO_UUID      = '07C80004-07C8-07C8-07C8-07C807C807C8';
const MAG_UUID       = '07C80010-07C8-07C8-07C8-07C807C807C8';

const APP_VERSION  = '1.0';
const BUFFER_MS    = 3000;
const BEFORE_MS    = 500;
const AFTER_MS     = 500;
const SESSION_DIR  = `${RNFS.DownloadDirectoryPath}/pingis_sessions`;

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

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  setup: PlayerSetup;
  calibration: CalibrationData;
  device: Device;
  onDone: () => void;
}

type StrokeType = 'forehand' | 'backhand';

// ── Komponent ─────────────────────────────────────────────────────────────────

export function DataCollectionScreen({ setup, calibration, device, onDone }: Props) {
  const [strokeType, setStrokeType] = useState<StrokeType>('forehand');
  const [events, setEvents] = useState<LabeledEvent[]>([]);
  const [sampleHz, setSampleHz] = useState(0);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(true);

  const bufferRef = useRef<ImuSample[]>([]);
  const latestRef = useRef({
    accel: { x: 0, y: 0, z: 0 },
    gyro:  { x: 0, y: 0, z: 0 },
    mag:   { x: 0, y: 0, z: 0 },
  });
  const sampleCountRef = useRef(0);
  const hzTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

      const now = Date.now();
      const sample: ImuSample = {
        accel_x: l.accel.x, accel_y: l.accel.y, accel_z: l.accel.z,
        gyro_x:  l.gyro.x,  gyro_y:  l.gyro.y,  gyro_z:  l.gyro.z,
        mag_x:   l.mag.x,   mag_y:   l.mag.y,   mag_z:   l.mag.z,
        ts_ms: now,
      };

      bufferRef.current.push(sample);
      sampleCountRef.current += 1;

      const cutoff = now - BUFFER_MS;
      let i = 0;
      while (i < bufferRef.current.length && bufferRef.current[i].ts_ms < cutoff) i++;
      if (i > 0) bufferRef.current = bufferRef.current.slice(i);
    },
    [],
  );

  // ── Starta BLE-prenumerationer på den vidarebefordrade device-instansen ──────

  useEffect(() => {
    for (const uuid of [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]) {
      try {
        device.monitorCharacteristicForService(SERVICE_UUID, uuid, handleNotification);
      } catch (_) {}
    }

    let last = 0;
    hzTimerRef.current = setInterval(() => {
      setSampleHz(sampleCountRef.current - last);
      last = sampleCountRef.current;
    }, 1000);

    const unsub = device.onDisconnected(() => setIsConnected(false));

    return () => {
      if (hzTimerRef.current) clearInterval(hzTimerRef.current);
      unsub.remove();
    };
  }, [device, handleNotification]);

  // ── Event-märkning ──────────────────────────────────────────────────────────

  const captureEvent = useCallback(
    (label: LabeledEvent['label']) => {
      const tapMs = Date.now();

      setTimeout(() => {
        const snap = [...bufferRef.current];
        const window = snap.filter(
          s => s.ts_ms >= tapMs - BEFORE_MS && s.ts_ms <= tapMs + AFTER_MS,
        );

        if (window.length < 10) {
          setFeedback(`⚠ Bara ${window.length} samples — kontrollera anslutningen`);
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
          setFeedback(`✓ ${label.toUpperCase()} (${event.stroke_type}) — ${window.length} samples`);
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

      const sessionData: SessionFile = {
        session_meta: {
          player_name: setup.name,
          handedness: setup.handedness,
          calibration_accel: calibration.gravity,
          calibration_gyro_bias: calibration.gyro_bias,
          session_date: new Date().toISOString(),
          app_version: APP_VERSION,
        },
        events,
      };

      await RNFS.writeFile(filePath, JSON.stringify(sessionData, null, 2), 'utf8');

      const counts = events.reduce<Record<string, number>>(
        (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
        {},
      );

      Alert.alert(
        '✓ Session sparad',
        `${events.length} events sparade\n` +
        `hit: ${counts.hit ?? 0}  miss: ${counts.swing_miss ?? 0}  idle: ${counts.idle ?? 0}\n\n` +
        `Fil: Download/pingis_sessions/${filePath.split('/').pop()}\n\n` +
        `Öppna Filer-appen → Downloads/pingis_sessions/ för att hitta filen.`,
        [
          { text: 'Ny session', onPress: onDone },
          { text: 'Fortsätt spela in', onPress: () => setEvents([]) },
        ],
      );
    } catch (e: any) {
      Alert.alert('Fel', `Kunde inte spara: ${e.message}`);
    }
  }, [events, setup, calibration, onDone]);

  // ── Render ──────────────────────────────────────────────────────────────────

  const counts = events.reduce<Record<string, number>>(
    (acc, e) => ({ ...acc, [e.label]: (acc[e.label] ?? 0) + 1 }),
    {},
  );

  return (
    <ScrollView style={s.root} contentContainerStyle={s.content}>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d0d" />

      {/* Header */}
      <View style={s.header}>
        <View>
          <Text style={s.playerName}>{setup.name}</Text>
          <Text style={s.playerMeta}>
            {setup.handedness === 'right' ? 'Höger' : 'Vänster'}hand  ·  Kalibrerad ✓
          </Text>
        </View>
        <View style={[s.hzBadge, !isConnected && s.hzBadgeOff]}>
          <Text style={s.hzTxt}>{isConnected ? `${sampleHz}Hz` : 'Bortkopplad'}</Text>
        </View>
      </View>

      {/* Varning om bortkopplad */}
      {!isConnected && (
        <View style={s.warnBox}>
          <Text style={s.warnTxt}>⚠ Sensorn kopplades från. Spara din session och anslut igen.</Text>
        </View>
      )}

      {/* Feedback */}
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
        <Text style={s.bigBtnSub}>vila / stå still 2 sekunder</Text>
      </TouchableOpacity>

      {/* Statistik */}
      <View style={s.statsBox}>
        <Text style={s.statsTitle}>Session — {events.length} events</Text>
        <View style={s.statsRow}>
          <Stat label="HIT"  value={counts.hit ?? 0}        color="#2ecc71" />
          <Stat label="MISS" value={counts.swing_miss ?? 0} color="#e74c3c" />
          <Stat label="IDLE" value={counts.idle ?? 0}       color="#95a5a6" />
        </View>
        {events.length === 0 && (
          <Text style={s.statsHint}>Mål: minst 20 av varje</Text>
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
            Alert.alert('Rensa?', 'Alla osparade events raderas.', [
              { text: 'Avbryt' },
              { text: 'Rensa', style: 'destructive', onPress: () => setEvents([]) },
            ])
          }
        >
          <Text style={s.secBtnTxt}>Rensa</Text>
        </TouchableOpacity>
      </View>

      {/* Fil-info */}
      <View style={s.fileBox}>
        <Text style={s.fileTit}>Var sparas filerna?</Text>
        <Text style={s.fileTxt}>
          Filer-appen → Downloads → pingis_sessions/{'\n'}
          Ladda upp därifrån till Google Drive eller skicka via mail.
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

  header:     { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 },
  playerName: { color: '#fff', fontSize: 20, fontWeight: '700' },
  playerMeta: { color: '#3a3a3a', fontSize: 12, marginTop: 2 },
  hzBadge:    { backgroundColor: '#0d2d1a', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  hzBadgeOff: { backgroundColor: '#2d0d0d' },
  hzTxt:      { color: '#2ecc71', fontSize: 12, fontWeight: '600' },

  warnBox:    { backgroundColor: '#2d1a00', borderRadius: 8, padding: 12, marginBottom: 12 },
  warnTxt:    { color: '#e67e22', fontSize: 13 },

  feedbackTxt: { color: '#4a9eff', fontSize: 13, textAlign: 'center', marginBottom: 10 },

  sectionLabel: { color: '#333', fontSize: 10, letterSpacing: 2, marginTop: 18, marginBottom: 8 },
  row:          { flexDirection: 'row', gap: 12 },

  typeBtn:    { flex: 1, padding: 14, borderRadius: 8, borderWidth: 1, borderColor: '#1e1e1e', alignItems: 'center' },
  typeBtnOn:  { borderColor: '#4a9eff', backgroundColor: '#0d1f33' },
  typeTxt:    { color: '#333', fontWeight: '700', fontSize: 12 },
  typeTxtOn:  { color: '#4a9eff' },

  bigBtn:     { borderRadius: 14, padding: 24, marginBottom: 12, alignItems: 'center' },
  hitBtn:     { backgroundColor: '#0d2d1a' },
  missBtn:    { backgroundColor: '#2d0d0d' },
  idleBtn:    { backgroundColor: '#15152a' },
  bigBtnTxt:  { color: '#fff', fontSize: 22, fontWeight: '800', letterSpacing: 3 },
  bigBtnSub:  { color: '#444', fontSize: 12, marginTop: 5 },

  statsBox:   { backgroundColor: '#111', borderRadius: 12, padding: 16, marginTop: 8 },
  statsTitle: { color: '#666', fontWeight: '600', marginBottom: 12 },
  statsRow:   { flexDirection: 'row', justifyContent: 'space-around' },
  statsHint:  { color: '#2a2a2a', fontSize: 12, fontStyle: 'italic', marginTop: 10, textAlign: 'center' },
  statItem:   { alignItems: 'center' },
  statValue:  { fontSize: 28, fontWeight: '800' },
  statLabel:  { color: '#444', fontSize: 11, marginTop: 2 },

  secBtn:     { flex: 1, padding: 14, borderRadius: 8, alignItems: 'center', marginTop: 12 },
  saveBtn:    { backgroundColor: '#0d2d0d' },
  clearBtn:   { backgroundColor: '#2d0d0d' },
  secBtnTxt:  { color: '#aaa', fontWeight: '700' },

  fileBox:    { marginTop: 20, backgroundColor: '#0d0d0d', borderWidth: 1, borderColor: '#1a1a1a', borderRadius: 10, padding: 14 },
  fileTit:    { color: '#333', fontWeight: '600', marginBottom: 6 },
  fileTxt:    { color: '#2a2a2a', fontSize: 12, lineHeight: 18 },
});
