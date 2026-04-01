# Autonomous Improvement Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-cron autonomous system where a cloud agent iteratively improves the sentiment pipeline's predictive performance, and a server worker trains experiments and reports results.

**Architecture:** Cloud agent (Anthropic scheduled trigger, every 2 hours) pushes experiment branches to GitHub. Server worker (crontab, every 10 minutes) polls for new experiments, trains in a git worktree, pushes results back. Agent scores results, merges improvements, logs to an experiment journal, and detects plateau to transition to trading strategy.

**Tech Stack:** Python 3.10, PyTorch, bash, git, Claude Code scheduled triggers

**Spec:** `docs/superpowers/specs/2026-04-01-autonomous-improvement-agent-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `utils/composite_score.py` (create) | Compute normalized composite score from metrics JSON files |
| `utils/progress.py` (modify) | Add `reset_stages()` for selective pipeline re-running |
| `docs/experiment_journal.md` (create) | Agent's persistent memory — baseline, experiment log, plateau counter |
| `server_worker.sh` (create) | Server-side cron script — poll, train in worktree, push results |
| `docs/agent_prompt.md` (create) | Cloud agent's system prompt for scheduled trigger |
| `CLAUDE.md` (modify) | Document the autonomous agent system |

---

## Task 1: Composite Score Module

**Files:**
- Create: `utils/composite_score.py`
- Create: `tests/test_composite_score.py`

- [ ] **Step 0: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 1: Write test for composite_score()**

```python
# tests/test_composite_score.py
import pytest
from utils.composite_score import composite_score


def test_baseline_score():
    """Composite score for current baseline should be ~0.44."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.0039,
             "learned_accuracy_mean": 0.5617, "learned_f1_mean": 0.538},
            {"model_type": "pretrained", "r2_mean": -0.01,
             "learned_accuracy_mean": 0.50, "learned_f1_mean": 0.49},
        ]
    }
    market_trend = [
        {"ols_r2": 0.3764, "pearson_p": 0.0447},
        {"ols_r2": 0.0234, "pearson_p": 0.31},
    ]
    score = composite_score(eval_results, market_trend)
    assert 0.40 < score < 0.50, f"Expected ~0.44, got {score}"


def test_negative_r2_clamped_to_zero():
    """Negative R² should be clamped to 0, not drag score down."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": -0.05,
             "learned_accuracy_mean": 0.50, "learned_f1_mean": 0.45},
        ]
    }
    market_trend = [{"ols_r2": 0.10, "pearson_p": 0.20}]
    score = composite_score(eval_results, market_trend)
    assert score >= 0.0


def test_perfect_score_near_one():
    """A hypothetically perfect model should score near 1.0."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.05,
             "learned_accuracy_mean": 0.95, "learned_f1_mean": 0.95},
        ]
    }
    market_trend = [{"ols_r2": 0.90, "pearson_p": 0.001}]
    score = composite_score(eval_results, market_trend)
    assert 0.90 < score <= 1.0, f"Expected near 1.0, got {score}"


def test_score_increases_with_better_metrics():
    """Improving any metric should increase the score."""
    base = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.004,
             "learned_accuracy_mean": 0.56, "learned_f1_mean": 0.54},
        ]
    }
    trend = [{"ols_r2": 0.38, "pearson_p": 0.045}]
    base_score = composite_score(base, trend)

    improved = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.01,
             "learned_accuracy_mean": 0.58, "learned_f1_mean": 0.56},
        ]
    }
    improved_score = composite_score(improved, trend)
    assert improved_score > base_score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_composite_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.composite_score'`

- [ ] **Step 3: Implement composite_score.py**

```python
# utils/composite_score.py
"""Composite score for evaluating overall pipeline quality.

Combines stock-level and market-level metrics into a single [0, 1] number.
Used by the autonomous agent to decide whether an experiment improved things.
"""

import json
from pathlib import Path

import pandas as pd

# Normalization ceiling for stock-level R²
# R²=0.05 on individual stock returns from news = exceptional
_R2_CEILING = 0.05

# Component weights (sum to 1.0)
_WEIGHTS = {
    "stock_r2": 0.25,
    "stock_acc": 0.15,
    "stock_f1": 0.15,
    "market_r2": 0.30,
    "market_sig": 0.15,
}


def composite_score(eval_results: dict, market_trend: list) -> float:
    """Compute composite score in [0, 1]. Higher is better."""
    aggregated = pd.DataFrame(eval_results["aggregated"])
    trend_df = pd.DataFrame(market_trend)

    ft = aggregated[aggregated["model_type"] == "finetuned"]

    stock_r2 = min(1.0, max(0.0, ft["r2_mean"].max()) / _R2_CEILING)
    stock_acc = ft["learned_accuracy_mean"].max()
    stock_f1 = ft["learned_f1_mean"].max()

    market_r2 = trend_df["ols_r2"].max()
    best_idx = trend_df["ols_r2"].idxmax()
    best_p = trend_df.loc[best_idx, "pearson_p"]
    market_sig = max(0.0, 1.0 - best_p)

    score = (
        _WEIGHTS["stock_r2"] * stock_r2
        + _WEIGHTS["stock_acc"] * stock_acc
        + _WEIGHTS["stock_f1"] * stock_f1
        + _WEIGHTS["market_r2"] * market_r2
        + _WEIGHTS["market_sig"] * market_sig
    )
    return round(score, 4)


def compute_from_files(
    eval_path: str = "results/metrics/evaluation_results.json",
    trend_path: str = "results/metrics/market_trend.json",
) -> float:
    """Compute composite score from result files on disk."""
    eval_results = json.loads(Path(eval_path).read_text())
    market_trend = json.loads(Path(trend_path).read_text())
    return composite_score(eval_results, market_trend)


def print_score() -> None:
    """Print composite score and component breakdown."""
    eval_results = json.loads(Path("results/metrics/evaluation_results.json").read_text())
    market_trend = json.loads(Path("results/metrics/market_trend.json").read_text())

    aggregated = pd.DataFrame(eval_results["aggregated"])
    trend_df = pd.DataFrame(market_trend)
    ft = aggregated[aggregated["model_type"] == "finetuned"]

    raw_r2 = ft["r2_mean"].max()
    raw_acc = ft["learned_accuracy_mean"].max()
    raw_f1 = ft["learned_f1_mean"].max()
    raw_mkt_r2 = trend_df["ols_r2"].max()
    best_p = trend_df.loc[trend_df["ols_r2"].idxmax(), "pearson_p"]

    score = composite_score(eval_results, market_trend)

    print("=" * 50)
    print("COMPOSITE SCORE BREAKDOWN")
    print("=" * 50)
    print(f"  Stock R²:      {raw_r2:.4f} (norm: {min(1.0, max(0, raw_r2) / _R2_CEILING):.3f}) × {_WEIGHTS['stock_r2']}")
    print(f"  Stock Acc:     {raw_acc:.4f} × {_WEIGHTS['stock_acc']}")
    print(f"  Stock F1:      {raw_f1:.4f} × {_WEIGHTS['stock_f1']}")
    print(f"  Market R²:     {raw_mkt_r2:.4f} × {_WEIGHTS['market_r2']}")
    print(f"  Market Sig:    {1 - best_p:.4f} (p={best_p:.4f}) × {_WEIGHTS['market_sig']}")
    print(f"  {'─' * 40}")
    print(f"  COMPOSITE:     {score:.4f}")
    print("=" * 50)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_composite_score.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Verify with real data**

Run: `python3 -c "from utils.composite_score import print_score; print_score()"`
Expected: Score breakdown printing ~0.44

- [ ] **Step 6: Commit**

```bash
git add utils/composite_score.py tests/test_composite_score.py
git commit -m "feat: add composite score module for autonomous agent evaluation"
```

---

## Task 2: Progress Reset Function

**Files:**
- Modify: `utils/progress.py` (add `reset_stages()`)
- Create: `tests/test_progress_reset.py`

- [ ] **Step 1: Write test for reset_stages()**

```python
# tests/test_progress_reset.py
import json
import pytest
from pathlib import Path
from utils.progress import reset_stages


@pytest.fixture
def mock_progress(tmp_path, monkeypatch):
    """Use a temp progress file."""
    p = tmp_path / "progress.json"
    monkeypatch.setattr("utils.progress.PROGRESS_FILE", p)
    state = {
        "data_preparation": "done",
        "embeddings": {"bert": "done", "finbert": "done"},
        "finetune": {"bert_separate_1d_split": "done", "bert_single_split": "done"},
        "evaluate": "done",
        "market_trend": "done",
        "last_updated": "2026-01-01T00:00:00",
        "last_error": None,
    }
    p.write_text(json.dumps(state))
    return p


def test_reset_single_stage(mock_progress):
    reset_stages(["evaluate"])
    state = json.loads(mock_progress.read_text())
    assert state["evaluate"] == "pending"
    assert state["data_preparation"] == "done"  # untouched


def test_reset_dict_stage(mock_progress):
    reset_stages(["finetune"])
    state = json.loads(mock_progress.read_text())
    assert state["finetune"] == {}
    assert state["embeddings"]["bert"] == "done"  # untouched


def test_reset_multiple_stages(mock_progress):
    reset_stages(["finetune", "evaluate", "market_trend"])
    state = json.loads(mock_progress.read_text())
    assert state["finetune"] == {}
    assert state["evaluate"] == "pending"
    assert state["market_trend"] == "pending"
    assert state["data_preparation"] == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_progress_reset.py -v`
Expected: FAIL — `ImportError: cannot import name 'reset_stages'`

- [ ] **Step 3: Add reset_stages() to utils/progress.py**

Append to `utils/progress.py` before `print_progress_summary()`:

```python
def reset_stages(stages: list[str]) -> None:
    """Reset specified pipeline stages to pending.

    For dict-type stages (embeddings, finetune, etc.), clears all keys.
    For string-type stages (evaluate, market_trend), sets to 'pending'.
    """
    state = _load()
    for stage in stages:
        if isinstance(state.get(stage), dict):
            state[stage] = {}
        else:
            state[stage] = "pending"
    _save(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_progress_reset.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add utils/progress.py tests/test_progress_reset.py
git commit -m "feat: add reset_stages() for selective pipeline re-running"
```

---

## Task 3: Experiment Journal

**Files:**
- Create: `docs/experiment_journal.md`

The journal is seeded with the current baseline metrics. The composite score must be computed from the real data (Task 1 must be complete).

- [ ] **Step 1: Compute the exact baseline score**

Run: `python3 -c "from utils.composite_score import print_score; print_score()"`

Note the exact score value for use in the journal.

- [ ] **Step 2: Create the experiment journal**

Create `docs/experiment_journal.md` with the baseline metrics filled in from the actual score output:

```markdown
# Experiment Journal

## Status
- **Phase:** 1 (Improving Prediction)
- **Baseline Score:** <SCORE from step 1>
- **Current Best Score:** <SCORE from step 1>
- **Plateau Counter:** 0
- **Last Experiment:** (none)

## Baseline Metrics (2026-04-01)
- Stock R² (best): 0.0039 (FinBERT, single, r_30d)
- Stock Accuracy (best): 56.17% (FinBERT, single, r_30d)
- Stock F1 (best): 0.538 (FinBERT, single, r_30d)
- Market R² (best): 0.376 (FinBERT, separate, r_30d)
- Market p-value (best): 0.045

## Experiment Log
```

- [ ] **Step 3: Commit**

```bash
git add docs/experiment_journal.md
git commit -m "docs: seed experiment journal with baseline metrics"
```

---

## Task 4: Server Worker Script

**Files:**
- Create: `server_worker.sh`
- Create: `tests/test_server_worker.sh`

- [ ] **Step 1: Find the conda init path**

Run on the server to find the correct conda.sh path:
```bash
find /home/fao -maxdepth 4 -name "conda.sh" -path "*/profile.d/*" 2>/dev/null
```

If no conda is found, the server uses system Python (check with `which python3`). In that case, remove the conda lines from the script.

- [ ] **Step 2: Create server_worker.sh**

```bash
#!/bin/bash
# server_worker.sh — Poll for new experiment branches and run training
# Uses git worktree to avoid corrupting main working directory
# Uses lockfile to prevent concurrent runs

set -euo pipefail

REPO_DIR="/home/fao/projects/sentiment"
LOCK_FILE="$REPO_DIR/.server_worker.lock"
WORKTREE_DIR="$REPO_DIR/.worktrees"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
TIMEOUT_HOURS=3

cd "$REPO_DIR"

# Lockfile: prevent concurrent cron runs
if [ -f "$LOCK_FILE" ]; then
    lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE") ))
    if [ $lock_age -gt $(( TIMEOUT_HOURS * 3600 )) ]; then
        log " Stale lock detected (${lock_age}s old), removing"
        rm -f "$LOCK_FILE"
    else
        log " Another worker is running (lock age: ${lock_age}s), exiting"
        exit 0
    fi
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

# System Python — no conda activation needed on this server
# If conda is needed in the future, add: source <conda.sh path> && conda activate sentiment

git fetch origin 2>/dev/null

mkdir -p "$WORKTREE_DIR"

FOUND_WORK=false

# Find experiment/* branches without matching results/* branches
for ref in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/experiment/); do
    name="${ref#origin/experiment/}"

    # Skip if results already exist for this experiment
    if git for-each-ref --format='%(refname:short)' refs/remotes/origin/results/ 2>/dev/null | grep -q "origin/results/$name"; then
        continue
    fi

    FOUND_WORK=true
    log " Processing: experiment/$name"

    wt="$WORKTREE_DIR/$name"

    # Clean up any leftover worktree from a previous crashed run
    if [ -d "$wt" ]; then
        git worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"
    fi

    # Create worktree for this experiment (isolates from main)
    git worktree add "$wt" "$ref" --detach

    # Symlink gitignored data directories into worktree so pipeline can find them
    mkdir -p "$wt/results"
    ln -sf "$REPO_DIR/data" "$wt/data"
    ln -sf "$REPO_DIR/results/embeddings" "$wt/results/embeddings"
    ln -sf "$REPO_DIR/results/checkpoints" "$wt/results/checkpoints"

    pushd "$wt" > /dev/null

    STATUS="success"
    ERROR_MSG=""

    # Run full pipeline — each script checks progress.json and skips completed stages
    for script in scripts/data_preparation.py scripts/extract_embeddings.py scripts/pretrained_sentiment.py scripts/finetune.py scripts/evaluate.py scripts/market_trend.py; do
        if [ -f "$script" ]; then
            log " Running: $script"
            if ! python3 "$script" 2>&1; then
                STATUS="failed"
                ERROR_MSG="$script failed"
                log " ERROR: $ERROR_MSG"
                break
            fi
        fi
    done

    # Create results branch and push (even on failure, with status)
    git checkout -b "results/$name"

    mkdir -p results/metrics
    echo "{\"status\": \"$STATUS\", \"error\": \"$ERROR_MSG\", \"timestamp\": \"$(date -Iseconds)\"}" \
        > results/metrics/experiment_status.json

    git add results/
    git commit -m "results: $name ($STATUS)" --allow-empty

    if ! git push origin "results/$name" 2>&1; then
        log " ERROR: Failed to push results/$name"
    fi

    popd > /dev/null

    # Cleanup worktree
    git worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"

    log " Completed: experiment/$name ($STATUS)"
done

if [ "$FOUND_WORK" = false ]; then
    log " No new experiments found"
fi
```

- [ ] **Step 3: Make executable**

```bash
chmod +x server_worker.sh
```

- [ ] **Step 4: Test the script in dry-run mode (no experiment branches exist)**

```bash
bash server_worker.sh
```

Expected: `No new experiments found` and exits cleanly. No errors.

- [ ] **Step 5: Verify .gitignore excludes lock and worktree files**

Check `.gitignore` contains entries for `.server_worker.lock` and `.worktrees/`. If not, add them:

```
.server_worker.lock
.worktrees/
results/server_worker.log
```

- [ ] **Step 6: Commit**

```bash
git add server_worker.sh .gitignore
git commit -m "feat: add server worker script for automated training"
```

---

## Task 5: Agent Prompt

**Files:**
- Create: `docs/agent_prompt.md`

This is the most critical file — it tells the cloud agent exactly what to do on each run. The prompt must be self-contained (the agent has no conversation history between runs).

- [ ] **Step 1: Create docs/agent_prompt.md**

```markdown
# Autonomous Improvement Agent — Prompt

You are an autonomous ML research agent improving a financial sentiment analysis pipeline.
Your goal: maximize the composite prediction score. You run on a schedule (every 2 hours).

## Your Environment

- **Repo:** WillInvest/Sentiment-Analysis- (you have push access)
- **Server:** A school server with GPU runs `server_worker.sh` every 10 minutes to train your experiments
- **Communication:** You push `experiment/*` branches, the server trains and pushes `results/*` branches

## Every Run — Follow This Exact Loop

### 1. Orient

```bash
git pull origin main
```

Read these files:
- `docs/experiment_journal.md` — your memory across runs
- `results/metrics/evaluation_results.json` — current model metrics
- `results/metrics/market_trend.json` — market correlation results

Compute current composite score:
```bash
python3 -c "from utils.composite_score import print_score; print_score()"
```

Check the plateau counter in the journal. If plateau >= 5, skip to Phase 2 instructions at the bottom.

### 2. Check Previous Experiment

Read the journal to find the last experiment. If its outcome is PENDING:

```bash
git fetch origin
git branch -r | grep "results/"
```

**If `results/<last-experiment-name>` exists:**
- Check out that branch and read `results/metrics/experiment_status.json`
- If status is `"failed"`: log FAILURE in journal, increment plateau counter
- If status is `"success"`:
  - Compute the new composite score from the results metrics
  - If new score > current best score: merge `experiment/<name>` into main, reset plateau to 0, log SUCCESS
  - If not improved: log NO IMPROVEMENT, increment plateau counter
- Delete both `experiment/*` and `results/*` branches (local and remote)

**If no results branch exists:**
- Check when the experiment branch was pushed (git log the branch)
- If more than 6 hours ago: mark as TIMEOUT, increment plateau, delete branch
- Otherwise: training is still running — update journal and exit this run

### 3. Analyze & Hypothesize

Read the FULL experiment journal. Then:

1. Identify which composite score component is weakest
2. Review what has been tried before (successes AND failures)
3. Pick ONE improvement to try — do NOT repeat a failed approach unless you have a specific new reason
4. Write your hypothesis in the journal

Consider these categories (but decide for yourself):
- **Config changes:** more HP configs, different search ranges, learning rate schedules
- **Architecture:** residual connections, batch norm, ensemble methods, gradient boosting (XGBoost)
- **Features:** mean-pool embeddings, multi-encoder features, metadata features
- **Data:** expand to 3 rolling windows, per-sector models, data augmentation
- **Evaluation:** different split strategies, robustness checks

### 4. Implement

```bash
# Get next experiment number from journal
git checkout -b experiment/<NNN>-<short-name> main
```

Make your code changes. Key rules:
- Reset the relevant progress.json keys so the server re-runs affected stages
- Use `from utils.progress import reset_stages; reset_stages(["finetune", "evaluate", "market_trend"])` in a small setup script or inline in the modified script
- Test that your changes don't have syntax errors: `python3 -c "import scripts.finetune"` (etc.)
- Commit and push:

```bash
git add scripts/ configs/ utils/ results/progress.json
git commit -m "experiment: <description of what you changed>"
git push origin experiment/<NNN>-<short-name>
```

### 5. Update Journal on Main

```bash
git checkout main
```

Append the new experiment entry to `docs/experiment_journal.md`:

```markdown
### EXP-<NNN>: <title> (<current timestamp>)
- **Hypothesis:** <what you think will improve and why>
- **What changed:** <files modified>
- **Branch:** experiment/<NNN>-<short-name>
- **Result:** PENDING
- **Outcome:** PENDING
```

Update the Status section: set Last Experiment to this one.

```bash
git add docs/experiment_journal.md
git commit -m "journal: start EXP-<NNN> <short-name>"
git push origin main
```

## Phase 2 — Trading Strategy

When plateau >= 5, switch to building a trading strategy:

1. Write `docs/improvement_summary.md` summarizing all Phase 1 experiments
2. Update journal status to Phase 2
3. Build `scripts/backtest.py` using the best model's monthly sentiment signal
4. Strategy: long SPX when monthly sentiment > threshold, cash otherwise
5. Optimize threshold on validation data, backtest on test data
6. Report: Sharpe ratio, max drawdown, win rate, equity curve
7. Each backtest experiment follows the same branch/merge pattern
8. After 5 consecutive Sharpe improvements plateau, write final report and stop

## Important Rules

- ONE experiment per run. Do not try multiple things at once.
- Always read the full journal before deciding what to try.
- Never force-push. Never delete main.
- If something is unclear, be conservative — make a small, safe change.
- Commit messages start with "experiment:" for experiment branches.
- The server uses `python3` (system Python, no conda needed).
```

- [ ] **Step 2: Commit**

```bash
git add docs/agent_prompt.md
git commit -m "docs: add agent prompt for scheduled trigger"
```

---

## Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append autonomous agent section to CLAUDE.md**

Add after the existing content:

```markdown

## Autonomous Improvement Agent

An autonomous "Scientist Agent" runs on a schedule to iteratively improve pipeline performance.

### Architecture
- **Cloud agent** (Anthropic scheduled trigger, every 2h): analyzes results, implements experiments, pushes branches
- **Server worker** (`server_worker.sh`, crontab every 10min): polls for experiment branches, trains, pushes results

### Branch Convention
- `experiment/<NNN>-<name>` — code changes (pushed by cloud agent)
- `results/<NNN>-<name>` — training results (pushed by server worker)
- Only merged to `main` if composite score improves

### Key Files
- `utils/composite_score.py` — composite score computation
- `docs/experiment_journal.md` — agent's persistent memory
- `docs/agent_prompt.md` — the scheduled trigger's instructions
- `server_worker.sh` — server-side training worker

### Composite Score
Run: `python3 -c "from utils.composite_score import print_score; print_score()"`

### Server Worker
The crontab entry (add manually):
```
*/10 * * * * cd /home/fao/projects/sentiment && bash server_worker.sh >> results/server_worker.log 2>&1
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add autonomous agent section to CLAUDE.md"
```

---

## Task 7: Set Up Crontab on Server

**No files to create — this is a manual server configuration step.**

- [ ] **Step 1: Add server_worker.sh to crontab**

```bash
crontab -e
```

Add this line:
```
*/10 * * * * cd /home/fao/projects/sentiment && bash server_worker.sh >> results/server_worker.log 2>&1
```

- [ ] **Step 2: Verify crontab is saved**

```bash
crontab -l | grep server_worker
```

Expected: The line you just added.

- [ ] **Step 3: Verify log file location is gitignored**

Check that `results/server_worker.log` won't be committed. Add to `.gitignore` if needed:
```
results/server_worker.log
```

- [ ] **Step 4: Commit .gitignore if changed**

```bash
git add .gitignore
git commit -m "chore: gitignore server worker log"
```

---

## Task 8: Create Scheduled Trigger

**This uses Claude Code's `/schedule` feature to create the recurring cloud agent.**

- [ ] **Step 1: Push all committed work to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: Create the scheduled trigger**

Use the Claude Code `/schedule` command to create a trigger with:
- **Schedule:** Every 2 hours (cron: `0 */2 * * *`)
- **Repo:** WillInvest/Sentiment-Analysis-
- **Prompt:** The contents of `docs/agent_prompt.md`

- [ ] **Step 3: Verify the trigger is listed**

Use `/schedule list` to confirm the trigger exists and shows the correct schedule.

---

## Task 9: End-to-End Smoke Test

**Verify the full loop works before leaving it autonomous.**

- [ ] **Step 1: Manually create a test experiment branch**

```bash
git checkout -b experiment/000-smoke-test main
# Make a trivial change — e.g., add a comment to configs/experiment.yaml
echo "# smoke test" >> configs/experiment.yaml
python3 -c "from utils.progress import reset_stages; reset_stages(['finetune', 'evaluate', 'market_trend'])"
git add -A
git commit -m "experiment: smoke test"
git push origin experiment/000-smoke-test
git checkout main
```

- [ ] **Step 2: Run server_worker.sh manually**

```bash
bash server_worker.sh 2>&1 | tail -20
```

Expected:
- Picks up `experiment/000-smoke-test`
- Creates worktree
- Runs pipeline (should reuse cached embeddings, re-run finetune/evaluate/market_trend)
- Pushes `results/000-smoke-test` branch

- [ ] **Step 3: Verify results branch exists**

```bash
git fetch origin
git branch -r | grep results/000-smoke-test
```

Expected: `origin/results/000-smoke-test`

- [ ] **Step 4: Check experiment_status.json**

```bash
git show origin/results/000-smoke-test:results/metrics/experiment_status.json
```

Expected: `{"status": "success", ...}`

- [ ] **Step 5: Clean up smoke test branches**

```bash
git push origin --delete experiment/000-smoke-test
git push origin --delete results/000-smoke-test
```

- [ ] **Step 6: Commit any fixes discovered during smoke test**

If any fixes were needed, commit them:
```bash
git add -A
git commit -m "fix: address issues found in smoke test"
git push origin main
```
