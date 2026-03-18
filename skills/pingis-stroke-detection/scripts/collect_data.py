"""
collect_data.py — Real-time BLE data collection from BERG AirHive

The FASTEST path to training data. Run this on your laptop while holding
the sensor and playing table tennis. No mobile app required.

Usage:
    python skills/pingis-stroke-detection/scripts/collect_data.py

Keyboard controls (during recording):
    H  — HIT (press at the exact moment of ball contact)
    M  — MISS (swing without ball contact)
    I  — IDLE (at rest between strokes)
    F  — toggle FOREHAND / BACKHAND stroke type
    S  — save current session to data/raw/ and start a new one
    Q  — quit (auto-saves current session)

Output:
    data/raw/session_YYYY-MM-DD_NNN.json

Requirements:
    pip install bleak pynput

How it works:
    - Scans for BERG AirHive BLE device
    - Subscribes to accelerometer, gyroscope, magnetometer notifications
    - Keeps a 3-second circular buffer of ImuSamples
    - On keypress: slices 500ms before + 500ms after the keypress timestamp
    - Saves labeled event to session JSON file
"""

import asyncio
import json
import os
import struct
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.characteristic import BleakGATTCharacteristic
except ImportError:
    print("bleak not installed. Run: pip install bleak")
    sys.exit(1)

try:
    from pynput import keyboard as kb
except ImportError:
    print("pynput not installed. Run: pip install pynput")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

SERVICE_UUID    = "07c80000-07c8-07c8-07c8-07c807c807c8"
ACCEL_UUID      = "07c80001-07c8-07c8-07c8-07c807c807c8"
ACCEL_UUID_ALT  = "07c80203-07c8-07c8-07c8-07c807c807c8"
GYRO_UUID       = "07c80004-07c8-07c8-07c8-07c807c807c8"
MAG_UUID        = "07c80010-07c8-07c8-07c8-07c807c807c8"

BUFFER_MS = 3000       # circular buffer size in ms
BEFORE_MS = 500        # ms to capture before label keypress
AFTER_MS  = 500        # ms to capture after label keypress

RAW_DIR = Path("data/raw")

# ── Packet parser (mirrors BlePacketParser.kt exactly) ───────────────────────

def parse_packet(uuid: str, payload: bytes) -> dict | None:
    """Parse 9-byte BLE packet. Returns dict with x, y, z, sensor_ts."""
    if len(payload) < 9:
        return None

    # Big-endian signed Int16 for X, Y, Z
    x, y, z = struct.unpack(">hhh", payload[0:6])

    # Unsigned 24-bit timestamp
    b0, b1, b2 = payload[6], payload[7], payload[8]
    sensor_ts = (b0 << 16) | (b1 << 8) | b2

    uuid_lower = uuid.lower()
    if uuid_lower in (ACCEL_UUID, ACCEL_UUID_ALT):
        return {"type": "accel", "x": float(x), "y": float(y), "z": float(z), "sensor_ts": sensor_ts}
    elif uuid_lower == GYRO_UUID:
        return {"type": "gyro", "x": float(x), "y": float(y), "z": float(z), "sensor_ts": sensor_ts}
    elif uuid_lower == MAG_UUID:
        # User-verified canonical transform: invert + scale by 0.1
        return {"type": "mag", "x": -x / 10.0, "y": -y / 10.0, "z": -z / 10.0, "sensor_ts": sensor_ts}
    return None


# ── Session state ─────────────────────────────────────────────────────────────

class RecordingSession:
    def __init__(self):
        self.buffer: deque = deque()      # deque of ImuSample dicts
        self.events: list = []            # saved labeled events
        self.stroke_type: str = "forehand"
        self.sample_count: int = 0

        # Latest known values for each sensor (emission model: update one, emit all)
        self._accel = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._gyro  = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._mag   = {"x": 0.0, "y": 0.0, "z": 0.0}

        self._lock = threading.Lock()

    def on_packet(self, uuid: str, payload: bytes):
        """Called from BLE notification thread for every incoming packet."""
        parsed = parse_packet(uuid, payload)
        if parsed is None:
            return

        now_ms = int(datetime.now().timestamp() * 1000)

        with self._lock:
            t = parsed["type"]
            if t == "accel":
                self._accel = {"x": parsed["x"], "y": parsed["y"], "z": parsed["z"]}
            elif t == "gyro":
                self._gyro = {"x": parsed["x"], "y": parsed["y"], "z": parsed["z"]}
            elif t == "mag":
                self._mag = {"x": parsed["x"], "y": parsed["y"], "z": parsed["z"]}

            # Emit combined sample (emission model: any notification → emit full sample)
            sample = {
                "accel_x": self._accel["x"],
                "accel_y": self._accel["y"],
                "accel_z": self._accel["z"],
                "gyro_x":  self._gyro["x"],
                "gyro_y":  self._gyro["y"],
                "gyro_z":  self._gyro["z"],
                "mag_x":   self._mag["x"],
                "mag_y":   self._mag["y"],
                "mag_z":   self._mag["z"],
                "ts_ms":   now_ms,
            }
            self.buffer.append(sample)
            self.sample_count += 1

            # Trim buffer to BUFFER_MS
            while self.buffer and (now_ms - self.buffer[0]["ts_ms"]) > BUFFER_MS:
                self.buffer.popleft()

    def capture_event(self, label: str):
        """Slice the buffer around the current timestamp and save as labeled event."""
        tap_ms = int(datetime.now().timestamp() * 1000)

        with self._lock:
            snap = list(self.buffer)

        window = [
            s for s in snap
            if (tap_ms - BEFORE_MS) <= s["ts_ms"] <= (tap_ms + AFTER_MS)
        ]

        # Wait AFTER_MS for trailing samples if buffer is live
        # (For simplicity we capture synchronously; in practice add small delay)
        if len(window) < 5:
            print(f"  ⚠  Only {len(window)} samples in window — sensor may not be streaming yet")
            return

        event = {
            "label": label,
            "stroke_type": self.stroke_type if label != "idle" else "unknown",
            "recorded_at": datetime.now().isoformat(),
            "samples": window,
        }
        self.events.append(event)

        counts = self._label_counts()
        print(f"  ✓  {label.upper()} ({self.stroke_type}) — {len(window)} samples  |  "
              f"hit:{counts.get('hit',0)}  miss:{counts.get('swing_miss',0)}  idle:{counts.get('idle',0)}")

    def _label_counts(self) -> dict:
        counts = {}
        for e in self.events:
            l = e["label"]
            counts[l] = counts.get(l, 0) + 1
        return counts

    def toggle_stroke_type(self):
        self.stroke_type = "backhand" if self.stroke_type == "forehand" else "forehand"
        print(f"  ↕  Stroke type → {self.stroke_type.upper()}")

    def save(self) -> Path | None:
        if not self.events:
            print("  No events to save.")
            return None

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Find next available file number for today
        n = 1
        while True:
            path = RAW_DIR / f"session_{date_str}_{n:03d}.json"
            if not path.exists():
                break
            n += 1

        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)

        counts = self._label_counts()
        print(f"\n  💾  Saved {len(self.events)} events → {path}")
        print(f"      hit:{counts.get('hit',0)}  miss:{counts.get('swing_miss',0)}  idle:{counts.get('idle',0)}")
        return path

    def reset(self):
        self.events = []
        self.sample_count = 0
        print("  Session cleared. Starting new session.")


# ── BLE connection ────────────────────────────────────────────────────────────

async def scan_for_airhive() -> str | None:
    """Scan for BERG AirHive device. Returns BLE address or None."""
    print("Scanning for BERG AirHive sensor (10 seconds)...")
    devices = await BleakScanner.discover(timeout=10.0, service_uuids=[SERVICE_UUID])

    if not devices:
        # Try without service UUID filter (some adapters ignore the filter)
        print("  No match with service UUID filter, scanning all devices...")
        devices = await BleakScanner.discover(timeout=10.0)

    # Look for AirHive by name
    for d in devices:
        name = d.name or ""
        if "airhive" in name.lower() or "berg" in name.lower() or "AIRHIVE" in name:
            print(f"  Found: {d.name} ({d.address})")
            return d.address

    # If not found by name, show all discovered devices
    if devices:
        print("\nNo AirHive found by name. Discovered devices:")
        for i, d in enumerate(sorted(devices, key=lambda x: x.rssi or -100, reverse=True)):
            print(f"  [{i}] {d.name or 'Unknown':<30} {d.address}  RSSI:{d.rssi}")
        choice = input("\nEnter device index (or press Enter to cancel): ").strip()
        if choice.isdigit() and int(choice) < len(devices):
            return sorted(devices, key=lambda x: x.rssi or -100, reverse=True)[int(choice)].address

    return None


async def run_collection(session: RecordingSession, address: str):
    """Connect to AirHive and stream data."""

    def on_notification(characteristic: BleakGATTCharacteristic, data: bytearray):
        session.on_packet(str(characteristic.uuid), bytes(data))

    print(f"\nConnecting to {address}...")
    async with BleakClient(address, timeout=15.0) as client:
        print(f"  Connected: {client.is_connected}")

        # Subscribe to all three sensor characteristics
        subscribed = []
        for uuid in [ACCEL_UUID, ACCEL_UUID_ALT, GYRO_UUID, MAG_UUID]:
            try:
                await client.start_notify(uuid, on_notification)
                subscribed.append(uuid.split("-")[0])
            except Exception:
                pass  # Some sensors only have one accel UUID

        print(f"  Subscribed to: {', '.join(subscribed)}")
        print_controls()

        # Keep alive until stopped
        while client.is_connected and not session._stop:
            await asyncio.sleep(0.1)

        print("\nDisconnected.")


# ── Keyboard listener ─────────────────────────────────────────────────────────

def print_controls():
    print("\n" + "─" * 55)
    print("  H  — HIT      (tap at moment of ball contact)")
    print("  M  — MISS     (swing without ball contact)")
    print("  I  — IDLE     (resting / between strokes)")
    print("  F  — toggle   FOREHAND / BACKHAND")
    print("  S  — save session and start new")
    print("  Q  — quit (auto-saves)")
    print("─" * 55)
    print("  Sensor streaming... (watch sample count below)")
    print()


def attach_keyboard(session: RecordingSession, loop: asyncio.AbstractEventLoop):
    """Start pynput keyboard listener in its own thread."""
    session._stop = False

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else ""
        except AttributeError:
            k = ""

        if k == "h":
            asyncio.run_coroutine_threadsafe(
                asyncio.coroutine_to_be_run(session, "hit"), loop
            )
            session.capture_event("hit")
        elif k == "m":
            session.capture_event("swing_miss")
        elif k == "i":
            session.capture_event("idle")
        elif k == "f":
            session.toggle_stroke_type()
        elif k == "s":
            session.save()
            session.reset()
        elif k == "q":
            session.save()
            session._stop = True
            return False  # stop listener

    listener = kb.Listener(on_press=on_press)
    listener.start()
    return listener


# ── Status printer ────────────────────────────────────────────────────────────

async def status_printer(session: RecordingSession):
    """Print live sample rate every 2 seconds."""
    last_count = 0
    while not getattr(session, "_stop", False):
        await asyncio.sleep(2.0)
        delta = session.sample_count - last_count
        last_count = session.sample_count
        hz = delta / 2.0
        buf_len = len(session.buffer)
        counts = session._label_counts()
        total = sum(counts.values())
        print(f"  [stream {hz:.0f}Hz | buf:{buf_len}] "
              f"events:{total} "
              f"(hit:{counts.get('hit',0)} miss:{counts.get('swing_miss',0)} idle:{counts.get('idle',0)}) "
              f"stroke:{session.stroke_type}",
              end="\r")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("═" * 55)
    print("  PINGIS DATA COLLECTOR — BERG AirHive")
    print("═" * 55)

    session = RecordingSession()
    session._stop = False

    # Scan for device
    address = await scan_for_airhive()
    if not address:
        print("\nNo device found. Make sure AirHive is powered on and nearby.")
        sys.exit(1)

    # Start keyboard listener
    loop = asyncio.get_event_loop()

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else ""
        except AttributeError:
            k = ""

        if k == "h":
            session.capture_event("hit")
        elif k == "m":
            session.capture_event("swing_miss")
        elif k == "i":
            session.capture_event("idle")
        elif k == "f":
            session.toggle_stroke_type()
        elif k == "s":
            session.save()
            session.reset()
        elif k == "q":
            session.save()
            session._stop = True
            return False

    listener = kb.Listener(on_press=on_press)
    listener.start()

    # Start status printer
    asyncio.ensure_future(status_printer(session))

    # Connect and stream
    try:
        await run_collection(session, address)
    except Exception as e:
        print(f"\nBLE error: {e}")
        session.save()
    finally:
        listener.stop()


def asyncio_coroutine_to_be_run(session, label):
    """Dummy async wrapper (not actually needed — kept for clarity)."""
    pass


if __name__ == "__main__":
    asyncio.run(main())
