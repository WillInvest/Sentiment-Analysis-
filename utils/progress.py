"""Progress tracker for pipeline resumability.

Reads/writes results/progress.json. All scripts call update_progress()
after completing a unit of work, and check_progress() before starting
to skip completed work.
"""

import json
from pathlib import Path
from datetime import datetime

PROGRESS_FILE = Path("results/progress.json")


def _load() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "data_preparation": "pending",
        "embeddings": {},
        "pretrained_sentiment": {},
        "finetune": {},
        "evaluate": "pending",
        "market_trend": "pending",
        "last_updated": None,
        "last_error": None,
    }


def _save(state: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(state, indent=2))


def check_progress(stage: str, key: str | None = None) -> str:
    """Return status: 'pending', 'in_progress', 'done', or 'error'."""
    state = _load()
    if key is None:
        return state.get(stage, "pending")
    return state.get(stage, {}).get(key, "pending")


def update_progress(stage: str, key: str | None = None, status: str = "done", error: str | None = None) -> None:
    """Update progress for a stage/key."""
    state = _load()
    if key is None:
        state[stage] = status
    else:
        if stage not in state or not isinstance(state[stage], dict):
            state[stage] = {}
        state[stage][key] = status
    state["last_error"] = error
    _save(state)


def get_full_progress() -> dict:
    """Return the full progress state."""
    return _load()


def print_progress_summary() -> None:
    """Print human-readable progress summary."""
    state = _load()
    print("=" * 60)
    print("PIPELINE PROGRESS")
    print("=" * 60)
    print(f"Last updated: {state.get('last_updated', 'never')}")
    print(f"Last error: {state.get('last_error', 'none')}")
    print()
    print(f"Data preparation: {state.get('data_preparation', 'pending')}")
    print()
    print("Embeddings:")
    for k, v in state.get("embeddings", {}).items():
        print(f"  {k}: {v}")
    print()
    print("Pretrained sentiment:")
    for k, v in state.get("pretrained_sentiment", {}).items():
        print(f"  {k}: {v}")
    print()
    print("Fine-tuning:")
    for k, v in state.get("finetune", {}).items():
        print(f"  {k}: {v}")
    print()
    print(f"Evaluation: {state.get('evaluate', 'pending')}")
    print(f"Market trend: {state.get('market_trend', 'pending')}")
    print("=" * 60)
