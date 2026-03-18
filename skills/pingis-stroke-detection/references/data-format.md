# Data Format Reference

## Session JSON Schema

Each labeled event is saved as one JSON object (one line in a JSONL file, or one entry in a session array).

```json
{
  "label": "hit",
  "stroke_type": "forehand",
  "recorded_at": "2026-03-18T14:23:01.123Z",
  "samples": [
    {
      "accel_x": 412,
      "accel_y": -88,
      "accel_z": 980,
      "gyro_x": 12.3,
      "gyro_y": -4.1,
      "gyro_z": 88.0,
      "mag_x": -4.2,
      "mag_y": 3.1,
      "mag_z": -1.8,
      "ts_ms": 1711234567890
    }
  ]
}
```

### Field definitions

| Field | Type | Unit | Notes |
|-------|------|------|-------|
| `accel_x/y/z` | int | raw Int16 | Direct from sensor, no conversion |
| `gyro_x/y/z` | float | degrees/second | Direct from sensor |
| `mag_x/y/z` | float | microtesla | Already converted: `-rawX/10` etc. |
| `ts_ms` | int | milliseconds | `receivedAtMs` from ImuSample — wall clock time |

### Session file naming
`data/raw/session_YYYY-MM-DD_NNN.json`

Example: `data/raw/session_2026-03-18_001.json`

NNN is a zero-padded counter per day. Increment for each new recording session.

---

## Label Taxonomy

### Primary label (`label` field)

| Value | Meaning |
|-------|---------|
| `hit` | Ball contact confirmed — player hit the ball |
| `swing_miss` | Full stroke motion with no ball contact |
| `idle` | Sensor at rest or slow movement between strokes |

### Stroke type (`stroke_type` field)

| Value | Meaning |
|-------|---------|
| `forehand` | Forehand stroke (leave null/empty if not applicable) |
| `backhand` | Backhand stroke |
| `unknown` | Not recorded or uncertain |

**Note:** Only label `stroke_type` when `label` is `hit`. For `swing_miss` and `idle`, set `stroke_type` to `"unknown"`.

---

## Window Specification

- **Window size**: 800ms = ~40 samples at 50Hz
- **Centering**: The labeled contact event is at the center of the window (sample index 20)
- **Saved samples**: 500ms before the label tap + 500ms after = ~50 samples total
  - The preprocessing script crops to exactly 40 samples centered on the event
- **Minimum samples**: If fewer than 40 samples are available, discard the event

---

## Minimum Dataset Requirements

| Phase | Samples needed | Expected accuracy |
|-------|---------------|-------------------|
| First prototype (RF) | 50 per class | ~75-85% |
| Good baseline (RF) | 200 per class | ~85-93% |
| CNN training | 500+ per class | ~90-97% |
| Production | 1000+ per class, multiple sessions | >95% |

For hit detection: classes are `hit`, `swing_miss`, `idle`.
For stroke type: classes are `forehand_hit`, `backhand_hit`.

---

## How to Record Labeled Data

1. Wear/hold the AirHive sensor in your normal pingpong grip
2. Open the DataCollectionScreen in the app
3. Select stroke type (forehand / backhand) before each stroke
4. Perform the stroke:
   - If ball contact: tap **HIT** at the moment of contact
   - If swing without contact: tap **MISS** after the swing
5. Sit still for 2-3 seconds between strokes to capture **IDLE** samples
6. The app auto-saves after each session

**Target per session**: 20-30 hits + 20-30 misses + 20-30 idle = ~70-90 labeled events
**Goal before first training run**: 3-4 sessions = ~200-300 total events
