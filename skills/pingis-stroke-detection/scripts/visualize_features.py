"""
visualize_features.py — Plot raw sensor streams to verify labeled data

Run this BEFORE training any model. You should be able to see the difference
between hits and misses visually. If you can't see it, the model won't learn it.

Usage:
    python skills/pingis-stroke-detection/scripts/visualize_features.py
    python skills/pingis-stroke-detection/scripts/visualize_features.py data/raw/session_2026-03-18_001.json

Output:
    Interactive matplotlib plots (close each to continue)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RAW_DIR = Path("data/raw")
SAMPLE_RATE = 50


def load_session(filepath: Path) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def plot_event(event: dict, ax_rows, title_prefix: str = ""):
    """Plot one labeled event's sensor streams on provided axes."""
    samples = event.get("samples", [])
    if not samples:
        return

    n = len(samples)
    t = np.arange(n) / SAMPLE_RATE * 1000  # ms

    accel = np.array([[s["accel_x"], s["accel_y"], s["accel_z"]] for s in samples])
    gyro = np.array([[s["gyro_x"], s["gyro_y"], s["gyro_z"]] for s in samples])
    accel_mag = np.linalg.norm(accel, axis=1)

    label = event.get("label", "?")
    stroke = event.get("stroke_type", "")
    color = {"hit": "green", "swing_miss": "red", "idle": "gray"}.get(label, "blue")

    title = f"{title_prefix}{label}"
    if stroke and stroke != "unknown":
        title += f" ({stroke})"

    axes = ax_rows
    axes[0].plot(t, accel[:, 0], color="C0", lw=1, label="X")
    axes[0].plot(t, accel[:, 1], color="C1", lw=1, label="Y")
    axes[0].plot(t, accel[:, 2], color="C2", lw=1, label="Z")
    axes[0].set_title(title, color=color, fontweight="bold")
    axes[0].set_ylabel("Accel (raw)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, gyro[:, 0], color="C0", lw=1, label="X")
    axes[1].plot(t, gyro[:, 1], color="C1", lw=1, label="Y")
    axes[1].plot(t, gyro[:, 2], color="C2", lw=1, label="Z")
    axes[1].set_ylabel("Gyro (°/s)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, accel_mag, color=color, lw=1.5)
    axes[2].set_ylabel("‖Accel‖")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(True, alpha=0.3)

    # Mark the center (where contact was labeled)
    center_t = t[len(t) // 2]
    for ax in axes:
        ax.axvline(center_t, color="orange", lw=1, linestyle="--", alpha=0.7)


def compare_label_groups(events: list[dict]):
    """Show a comparison grid: hits vs misses side by side."""
    hits = [e for e in events if e.get("label") == "hit"][:4]
    misses = [e for e in events if e.get("label") == "swing_miss"][:4]
    idles = [e for e in events if e.get("label") == "idle"][:2]

    print(f"\nShowing: {len(hits)} hits, {len(misses)} misses, {len(idles)} idle")

    # Plot accel magnitude overlay for hits vs misses
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Accelerometer Magnitude — Hit vs Swing Miss\n(orange line = labeled contact moment)", fontsize=12)

    for event in hits:
        samples = event["samples"]
        t = np.arange(len(samples)) / SAMPLE_RATE * 1000
        accel = np.array([[s["accel_x"], s["accel_y"], s["accel_z"]] for s in samples])
        axes[0].plot(t, np.linalg.norm(accel, axis=1), alpha=0.6, color="green", lw=1)
    axes[0].axvline(len(hits[0]["samples"]) // 2 / SAMPLE_RATE * 1000 if hits else 400,
                    color="orange", lw=2, linestyle="--", label="contact moment")
    axes[0].set_title(f"HITS (n={len(hits)})", color="green", fontweight="bold")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("‖Accel‖ (raw)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for event in misses:
        samples = event["samples"]
        t = np.arange(len(samples)) / SAMPLE_RATE * 1000
        accel = np.array([[s["accel_x"], s["accel_y"], s["accel_z"]] for s in samples])
        axes[1].plot(t, np.linalg.norm(accel, axis=1), alpha=0.6, color="red", lw=1)
    axes[1].axvline(len(misses[0]["samples"]) // 2 / SAMPLE_RATE * 1000 if misses else 400,
                    color="orange", lw=2, linestyle="--", label="labeled moment")
    axes[1].set_title(f"SWING MISSES (n={len(misses)})", color="red", fontweight="bold")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("‖Accel‖ (raw)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Individual event detail (first 4 events total)
    all_sample_events = (hits[:2] + misses[:2])
    if not all_sample_events:
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Individual Event Detail (3 rows: accel xyz, gyro xyz, accel magnitude)", fontsize=11)
    gs = gridspec.GridSpec(3, len(all_sample_events))

    for col, event in enumerate(all_sample_events):
        ax_accel = fig.add_subplot(gs[0, col])
        ax_gyro = fig.add_subplot(gs[1, col])
        ax_mag = fig.add_subplot(gs[2, col])
        plot_event(event, [ax_accel, ax_gyro, ax_mag])

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        session_files = [filepath]
    else:
        session_files = sorted(RAW_DIR.glob("*.json"))

    if not session_files:
        print(f"No session files found in {RAW_DIR}/")
        print("Record some training data first using the DataCollectionScreen app.")
        return

    all_events = []
    for filepath in session_files:
        print(f"Loading: {filepath.name}")
        events = load_session(filepath)
        print(f"  {len(events)} labeled events")
        for e in events:
            print(f"    {e.get('label', '?')} {e.get('stroke_type', '')} — {len(e.get('samples', []))} samples")
        all_events.extend(events)

    print(f"\nTotal events: {len(all_events)}")

    label_counts = {}
    for e in all_events:
        l = e.get("label", "unknown")
        label_counts[l] = label_counts.get(l, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    compare_label_groups(all_events)


if __name__ == "__main__":
    main()
