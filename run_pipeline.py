#!/usr/bin/env python3
"""Pipeline runner with live progress display.

Shows a persistent progress header at the top of the terminal using
ANSI scroll regions, while subprocess output (including tqdm bars)
scrolls normally below. A background thread polls progress.json
every 2 seconds to keep the header current.

Usage: python run_pipeline.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


# ─── Pipeline stages ───

STAGES = [
    {
        "name": "Data Preparation",
        "script": "scripts/data_preparation.py",
        "key": "data_preparation",
        "type": "simple",
        "expected": 1,
    },
    {
        "name": "Extract Embeddings",
        "script": "scripts/extract_embeddings.py",
        "key": "embeddings",
        "type": "dict",
        "expected_keys": ["bert", "finbert", "roberta"],
    },
    {
        "name": "Pretrained Sentiment",
        "script": "scripts/pretrained_sentiment.py",
        "key": "pretrained_sentiment",
        "type": "dict",
        "expected_keys": ["finbert", "roberta", "llama"],
    },
    {
        "name": "Fine-tuning",
        "script": "scripts/finetune.py",
        "key": "finetune",
        "type": "dict",
        "expected_keys": None,  # computed below
    },
    {
        "name": "Evaluation",
        "script": "scripts/evaluate.py",
        "key": "evaluate",
        "type": "simple",
        "expected": 1,
    },
    {
        "name": "Market Trend",
        "script": "scripts/market_trend.py",
        "key": "market_trend",
        "type": "simple",
        "expected": 1,
    },
]


def build_finetune_keys():
    """Expected fine-tune run keys from config."""
    encoders = ["bert", "finbert", "roberta"]
    windows = ["window_1", "window_2", "window_3"]
    horizons = [1, 3, 5, 10, 30]
    keys = []
    for e in encoders:
        for w in windows:
            for h in horizons:
                keys.append(f"{e}_separate_{h}d_{w}")
            keys.append(f"{e}_single_{w}")
    keys.append("predictions")
    return keys


# ─── Progress helpers ───

PROGRESS_FILE = Path("results/progress.json")


def read_progress():
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def stage_progress(progress, stage):
    """Return (done, total, current_activity) for a stage."""
    if stage["type"] == "simple":
        status = progress.get(stage["key"], "pending")
        done = 1 if status == "done" else 0
        activity = None if status == "done" else status
        return done, 1, activity
    sub = progress.get(stage["key"], {})
    if not isinstance(sub, dict):
        return (1 if sub == "done" else 0), 1, sub
    expected = stage.get("expected_keys") or list(sub.keys()) or ["?"]
    total = len(expected)
    done = sum(1 for k in expected if sub.get(k) == "done")
    # find what's in progress
    activity = next((k for k, v in sub.items() if v == "in_progress"), None)
    return done, total, activity


def overall_progress(progress):
    """Return (done, total) across all stages."""
    done_all, total_all = 0, 0
    for s in STAGES:
        d, t, _ = stage_progress(progress, s)
        done_all += d
        total_all += t
    return done_all, total_all


# ─── Display ───

HEADER_ROWS = 8  # lines reserved at top


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def bar(done, total, width=30):
    if total == 0:
        return "░" * width
    f = int(width * done / total)
    return "█" * f + "░" * (width - f)


def render_header(stage_idx, progress, start_time):
    """Build the header lines."""
    elapsed = time.time() - start_time
    try:
        tw = os.get_terminal_size().columns
    except OSError:
        tw = 80
    w = min(tw, 72)
    sep = "─" * w

    d_all, t_all = overall_progress(progress)
    pct = d_all / t_all * 100 if t_all else 0

    if stage_idx < len(STAGES):
        cs = STAGES[stage_idx]
        sd, st, act = stage_progress(progress, cs)
        stage_line = f"  Stage {stage_idx+1}/{len(STAGES)}:  {cs['name']}"
        sub_bar = f"  Sub-tasks  {bar(sd, st, 25)}  {sd}/{st}"
        act_str = f"  Active     ⟳ {act}" if act and act not in ("done", "pending") else ""
    else:
        stage_line = "  All stages complete!"
        sub_bar = ""
        act_str = ""

    lines = [
        sep,
        f"  Overall    {bar(d_all, t_all)}  {pct:5.1f}%   ⏱  {fmt_time(elapsed)}",
        stage_line,
    ]
    if sub_bar:
        lines.append(sub_bar)
    if act_str:
        lines.append(act_str)

    last = progress.get("last_updated")
    if last:
        lines.append(f"  Updated    {last[:19]}")

    err = progress.get("last_error")
    if err:
        lines.append(f"  ⚠ Error    {str(err)[:w-12]}")

    lines.append(sep)

    # Pad/truncate to exactly HEADER_ROWS lines
    while len(lines) < HEADER_ROWS:
        lines.insert(-1, "")
    lines = lines[:HEADER_ROWS]

    return lines


def setup_scroll_region():
    """Reserve top rows for the header; output scrolls below."""
    try:
        rows = os.get_terminal_size().lines
    except OSError:
        rows = 40
    # Clear screen, move to top
    sys.stderr.write("\033[2J\033[H")
    # Print blank header space
    for _ in range(HEADER_ROWS):
        sys.stderr.write("\n")
    # Set scroll region to below the header
    sys.stderr.write(f"\033[{HEADER_ROWS + 1};{rows}r")
    # Move cursor into the scroll region
    sys.stderr.write(f"\033[{HEADER_ROWS + 1};1H")
    sys.stderr.flush()


def restore_terminal():
    """Reset scroll region to full terminal."""
    try:
        rows = os.get_terminal_size().lines
    except OSError:
        rows = 40
    sys.stderr.write(f"\033[1;{rows}r")
    sys.stderr.write(f"\033[{rows};1H")
    sys.stderr.flush()


def write_header(lines):
    """Overwrite the reserved header area without moving the scroll cursor."""
    sys.stderr.write("\0337")  # save cursor
    for i, line in enumerate(lines):
        sys.stderr.write(f"\033[{i+1};1H\033[2K{line}")
    sys.stderr.write("\0338")  # restore cursor
    sys.stderr.flush()


def set_title(text):
    """Set terminal window/tab title (xterm escape)."""
    sys.stderr.write(f"\033]0;{text}\007")
    sys.stderr.flush()


# ─── Runner ───

def main():
    # Build finetune expected keys
    ft_keys = build_finetune_keys()
    for s in STAGES:
        if s["name"] == "Fine-tuning":
            s["expected_keys"] = ft_keys

    start_time = time.time()
    current_stage = [0]
    stop_event = threading.Event()

    # Detect whether we have a real terminal
    is_tty = sys.stderr.isatty()

    if is_tty:
        setup_scroll_region()

    # Header updater thread
    def updater():
        while not stop_event.is_set():
            try:
                prog = read_progress()
                if is_tty:
                    lines = render_header(current_stage[0], prog, start_time)
                    write_header(lines)
                # Always update terminal title
                d, t = overall_progress(prog)
                pct = d / t * 100 if t else 0
                si = current_stage[0]
                sn = STAGES[si]["name"] if si < len(STAGES) else "Done"
                set_title(f"[{si+1}/{len(STAGES)} {pct:.0f}%] {sn} — {fmt_time(time.time()-start_time)}")
            except Exception:
                pass
            stop_event.wait(2)

    t = threading.Thread(target=updater, daemon=True)
    t.start()

    exit_code = 0

    try:
        for i, stage in enumerate(STAGES):
            current_stage[0] = i

            # Check if already done
            prog = read_progress()
            if stage["type"] == "simple" and prog.get(stage["key"]) == "done":
                print(f"✓ {stage['name']} — already done, skipping")
                continue
            if stage["type"] == "dict":
                ek = stage.get("expected_keys", [])
                sub = prog.get(stage["key"], {})
                if isinstance(sub, dict) and all(sub.get(k) == "done" for k in ek):
                    print(f"✓ {stage['name']} — already done, skipping")
                    continue

            print(f"\n{'='*50}")
            print(f"  Starting: {stage['name']}")
            print(f"{'='*50}\n")

            result = subprocess.run(
                [sys.executable, stage["script"]],
            )

            if result.returncode != 0:
                print(f"\n✗ {stage['name']} FAILED (exit code {result.returncode})")
                print("  Pipeline is resumable — fix the issue and re-run.")
                exit_code = result.returncode
                break

            print(f"\n✓ {stage['name']} — done")

        current_stage[0] = len(STAGES)
        # One last header update
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Pipeline is resumable — just re-run.")
        exit_code = 130
    finally:
        stop_event.set()
        t.join(timeout=3)
        if is_tty:
            restore_terminal()
        set_title("")  # clear title

        # Final summary
        prog = read_progress()
        elapsed = time.time() - start_time
        d, total = overall_progress(prog)

        print()
        print("=" * 50)
        print(f"  Elapsed:  {fmt_time(elapsed)}")
        print(f"  Progress: {d}/{total} sub-tasks")
        if exit_code == 0 and d == total:
            print("  Status:   ✓ ALL COMPLETE")
            print()
            print("  Next: open notebooks/results_and_analysis.ipynb")
        elif exit_code == 0:
            print("  Status:   Partial (some stages were skipped?)")
        else:
            print("  Status:   Stopped (resumable — re-run this script)")
        print("=" * 50)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
