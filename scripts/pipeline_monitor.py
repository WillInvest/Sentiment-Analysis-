"""Live monitor for scripts/pipeline.py.

Polls results/pipeline_status.json and renders a refreshing summary.
Run in a second terminal:

    python scripts/pipeline_monitor.py
    python scripts/pipeline_monitor.py --once     # print once and exit
    python scripts/pipeline_monitor.py --interval 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = ROOT / "results/pipeline_status.json"

STAGE_ORDER = [
    "data_prep", "chunk_features", "baseline",
    "train_regressor", "train_classifier", "metrics", "hp_sweep",
]
ICON = {"done": "✓", "in_progress": "▸", "error": "✗", None: "·"}


def parse_iso(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def humanize(seconds: float) -> str:
    if seconds < 60:    return f"{seconds:.0f}s"
    if seconds < 3600:  return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def render(status: dict) -> str:
    lines = []
    started = parse_iso(status.get("started"))
    if started:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        lines.append(f"pipeline started: {status['started']}   total elapsed: {humanize(elapsed)}")
    else:
        lines.append("pipeline status:")
    lines.append("─" * 78)

    stages = status.get("stages", {})
    for name in STAGE_ORDER:
        s = stages.get(name, {})
        st = s.get("status")
        icon = ICON.get(st, "·")
        head = f"  {icon} {name:<18}"
        if st is None:
            lines.append(head + "(pending)")
            continue
        if st == "done":
            extras = []
            if "rows" in s:        extras.append(f"rows={s['rows']:,}")
            if "chunks" in s:      extras.append(f"chunks={s['chunks']:,}")
            if "articles" in s:    extras.append(f"articles={s['articles']:,}")
            if "summary_path" in s: extras.append(s["summary_path"])
            lines.append(head + "done   " + "  ".join(extras))
            continue
        if st == "error":
            lines.append(head + f"ERROR: {s.get('error','?')}")
            continue
        # in_progress
        bits = []
        if "phase" in s:           bits.append(f"phase={s['phase']}")
        if "current_window" in s:  bits.append(f"window={s['current_window']}")
        if "epoch" in s:
            bits.append(f"ep={s['epoch']}")
        if "train_loss" in s:      bits.append(f"train={s['train_loss']}")
        if "val_loss" in s:        bits.append(f"val={s['val_loss']}")
        if "best_val" in s:        bits.append(f"best={s['best_val']}")
        if "progress" in s:        bits.append(f"progress={s['progress']}")
        if "n_train" in s:
            bits.append(f"n_tr={s['n_train']}")
            bits.append(f"n_va={s['n_val']}")
            bits.append(f"n_te={s['n_test']}")
        upd = parse_iso(s.get("updated"))
        if upd:
            age = (datetime.now(timezone.utc) - upd).total_seconds()
            bits.append(f"upd={humanize(age)} ago")
        lines.append(head + "running   " + "  ".join(bits))

    lines.append("─" * 78)
    lines.append(f"status file: {STATUS_PATH}")
    return "\n".join(lines)


def load_status():
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


CLEAR = "\x1b[2J\x1b[H"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0, help="poll seconds")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()

    while True:
        status = load_status()
        if status is None:
            text = f"(no status file yet at {STATUS_PATH} — start the pipeline)"
        else:
            text = render(status)
        if args.once:
            print(text)
            return
        sys.stdout.write(CLEAR + text + "\n")
        sys.stdout.flush()
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    main()
