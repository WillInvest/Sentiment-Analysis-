# Autonomous Improvement Agent — Experiment Design

## Overview

An autonomous "Scientist Agent" system that iteratively improves the financial sentiment analysis pipeline's predictive performance, then builds a trading strategy once performance plateaus. The system uses two decoupled crons communicating through GitHub branches.

## Goals

1. **Phase 1 — Improve prediction:** Self-directed ML experimentation to maximize a composite performance score across stock-level and market-level metrics
2. **Phase 2 — Trading strategy:** Once improvement plateaus, use the best model to build and backtest a monthly SPX trading strategy

## System Architecture

### Components

```
┌─────────────────────────────────────┐
│  CLOUD AGENT (Anthropic scheduled)  │
│  Runs every 2 hours                 │
│                                     │
│  1. Check for completed results     │
│  2. Score & merge/reject            │
│  3. Analyze, hypothesize            │
│  4. Implement next experiment       │
│  5. Push experiment/* branch        │
│  6. Update journal, commit          │
└──────────────┬──────────────────────┘
               │ GitHub repo
               │ WillInvest/Sentiment-Analysis-
┌──────────────┴──────────────────────┐
│  SERVER CRON (school server)        │
│  Polls every 10 minutes             │
│                                     │
│  1. git fetch                       │
│  2. Check for new experiment/*      │
│     branches (no matching results/) │
│  3. Checkout, run training pipeline │
│  4. Push results/* branch           │
└─────────────────────────────────────┘
```

### Branch Convention

| Branch pattern | Purpose | Created by |
|---------------|---------|------------|
| `main` | Stable baseline, only updated on successful experiments | Cloud agent (merge) |
| `experiment/<name>` | Code changes for one experiment | Cloud agent |
| `results/<name>` | Training results for that experiment | Server worker |

### Communication Flow

1. Cloud agent pushes `experiment/foo` branch with code changes
2. Server cron detects new branch (no matching `results/foo`), checks it out, trains
3. Server pushes `results/foo` branch with updated `results/metrics/*.json`
4. Cloud agent detects `results/foo`, compares composite score vs. main
5. If improved: agent merges `experiment/foo` → main, deletes both branches
6. If not: agent logs failure, deletes both branches

## Cloud Agent — Run Loop

### Step 1: Orient

```
git pull origin main
Read docs/experiment_journal.md
Read results/metrics/evaluation_results.json
Read results/metrics/market_trend.json
Compute current composite score on main
Check plateau counter from journal
```

If plateau >= 5, transition to Phase 2 (see below).

### Step 2: Check Previous Experiment

```
If last experiment was pending:
    git fetch origin
    If results/<last-experiment> branch exists:
        Read results/metrics/experiment_status.json from results branch
        If status == "failed":
            Increment plateau counter
            Log FAILURE in journal with error message
        Else:
            Copy results/metrics/*.json from results branch
            Compute new composite score
            If new > baseline:
                Merge experiment/<name> → main (fast-forward or --no-ff)
                Reset plateau counter to 0
                Log SUCCESS in journal
            Else:
                Increment plateau counter
                Log NO IMPROVEMENT in journal with score delta
        Delete experiment/* and results/* branches (both local and remote)
    Else:
        Check staleness (>6 hours since experiment pushed?)
        If stale: mark TIMEOUT, plateau += 1, delete branch, continue
        Else: training still in progress — skip this run, exit
```

### Step 3: Analyze & Hypothesize

The agent examines current metrics to identify the weakest component:

```
Read experiment journal (past successes, failures, insights)
Identify weakest composite score component
Consider what hasn't been tried yet
Pick ONE improvement to try
Write hypothesis and rationale to journal
```

The agent is self-directed — it decides what to try based on analysis of results and past experiments. It must not repeat a previously failed approach unless it has a specific reason to believe the outcome will differ.

### Step 4: Implement

```
Create branch: experiment/<NNN>-<short-name>  (e.g., experiment/001-rolling-windows)
Modify code (scripts/, configs/, utils/)
Include journal update in the experiment branch (not main)
Commit with message: "experiment: <description>"
Push to origin
```

Branch names are prefixed with a zero-padded sequence number to prevent collisions and maintain order.

### Step 5: Update Journal on Main

```
On main branch:
  Append hypothesis entry to journal (outcome: PENDING)
  Commit: "journal: start EXP-NNN <short-name>"
  Push main
```

The journal is updated on main with the hypothesis before training starts (survives crashes). The outcome is filled in during Step 2 of the *next* run, after results arrive.

### Staleness Timeout

If an experiment has been pending for more than 6 hours (3 consecutive cloud agent runs with no results branch), the agent:
1. Marks it as TIMEOUT in the journal
2. Increments plateau counter
3. Deletes the experiment branch
4. Proceeds to the next experiment

## Server Worker — `server_worker.sh`

### Location & Schedule

- **Path:** `/home/fao/projects/sentiment/server_worker.sh`
- **Crontab:** `*/10 * * * * cd /home/fao/projects/sentiment && bash server_worker.sh >> results/server_worker.log 2>&1`

### Behavior

```bash
#!/bin/bash
# Poll for new experiment branches and run training
# Uses git worktree to avoid corrupting main working directory
# Uses lockfile to prevent concurrent runs

set -euo pipefail

REPO_DIR="/home/fao/projects/sentiment"
LOCK_FILE="$REPO_DIR/.server_worker.lock"
WORKTREE_DIR="$REPO_DIR/.worktrees"
TIMEOUT_HOURS=3

cd "$REPO_DIR"

# Lockfile: prevent concurrent cron runs
if [ -f "$LOCK_FILE" ]; then
    lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE") ))
    if [ $lock_age -gt $(( TIMEOUT_HOURS * 3600 )) ]; then
        echo "[$(date)] Stale lock detected (${lock_age}s old), removing"
        rm -f "$LOCK_FILE"
    else
        echo "[$(date)] Another worker is running (lock age: ${lock_age}s), exiting"
        exit 0
    fi
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

source ~/miniforge3/etc/profile.d/conda.sh  # adjust path if needed
conda activate sentiment

git fetch origin

mkdir -p "$WORKTREE_DIR"

# Find experiment/* branches without matching results/* branches
for ref in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/experiment/); do
    name=${ref#origin/experiment/}
    
    # Skip if results already exist
    if git for-each-ref --format='%(refname:short)' refs/remotes/origin/results/ | grep -q "origin/results/$name"; then
        continue
    fi
    
    echo "[$(date)] Processing: experiment/$name"
    
    wt="$WORKTREE_DIR/$name"
    
    # Create worktree for this experiment (isolates from main)
    git worktree add "$wt" "origin/experiment/$name" --detach
    
    # Run pipeline in worktree
    pushd "$wt" > /dev/null
    
    STATUS="success"
    ERROR_MSG=""
    
    # Run full pipeline (each script checks progress.json and skips completed stages)
    # The experiment branch should have reset relevant progress keys
    for script in scripts/data_preparation.py scripts/extract_embeddings.py scripts/pretrained_sentiment.py scripts/finetune.py scripts/evaluate.py scripts/market_trend.py; do
        if [ -f "$script" ]; then
            echo "[$(date)] Running: $script"
            if ! python "$script" 2>&1; then
                STATUS="failed"
                ERROR_MSG="$script failed"
                break
            fi
        fi
    done
    
    # Create results branch and push (even on failure, with status)
    git checkout -b "results/$name"
    
    # Write status file so cloud agent knows what happened
    echo "{\"status\": \"$STATUS\", \"error\": \"$ERROR_MSG\", \"timestamp\": \"$(date -Iseconds)\"}" \
        > results/metrics/experiment_status.json
    
    git add results/metrics/ results/predictions/ results/figures/ results/training_logs/
    git commit -m "results: $name ($STATUS)"
    git push origin "results/$name"
    
    popd > /dev/null
    
    # Cleanup worktree
    git worktree remove "$wt" --force
    
    echo "[$(date)] Completed: experiment/$name ($STATUS)"
done
```

### Error Handling

- **Lockfile** prevents concurrent cron runs from corrupting git state. Stale locks (>3 hours) are automatically cleaned.
- **Git worktree** isolates experiment training from the main working directory — your local checkout is never disturbed.
- **Failure reporting:** If training fails, the server still pushes a `results/*` branch with `experiment_status.json` containing `"status": "failed"` and the error message. The cloud agent reads this and logs the failure instead of waiting forever.
- **Timeout:** If a lock is older than 3 hours, the next cron run assumes the previous one crashed and proceeds.
- All output logged to `results/server_worker.log`

### Conda Environment

The `sentiment` conda env already exists on this server. If it needs to be recreated:

```bash
conda env create -f environment.yml
```

## Composite Score

### Formula

```python
def composite_score(eval_results, market_trend):
    """
    Single number in [0, 1] summarizing model quality.
    All components normalized to [0, 1] before weighting.
    Higher is better.
    """
    aggregated = pd.DataFrame(eval_results["aggregated"])
    trend_df = pd.DataFrame(market_trend)
    
    # Stock-level: best fine-tuned model across all configs
    ft = aggregated[aggregated["model_type"] == "finetuned"]
    stock_r2 = min(1.0, max(0, ft["r2_mean"].max()) / 0.05)  # normalize: 0.05 = perfect
    stock_acc = ft["learned_accuracy_mean"].max()               # already [0, 1]
    stock_f1 = ft["learned_f1_mean"].max()                      # already [0, 1]
    
    # Market-level: best model's correlation
    market_r2 = trend_df["ols_r2"].max()                        # already [0, 1]
    best_p = trend_df.loc[trend_df["ols_r2"].idxmax(), "pearson_p"]
    market_sig = max(0, 1 - best_p)                             # p=0.01→0.99
    
    score = (
        0.25 * stock_r2 +       # 0.004/0.05 = 0.078
        0.15 * stock_acc +       # ~0.56
        0.15 * stock_f1 +        # ~0.54
        0.30 * market_r2 +       # ~0.38
        0.15 * market_sig        # ~0.955
    )
    return round(score, 4)
```

**Normalization:** Stock R² is divided by 0.05 (a ceiling — achieving R²=0.05 on individual stock returns from news would be exceptional). This keeps all components in [0, 1] and the final score in [0, 1].

### Current Baseline

| Component | Raw Value | Normalized | Weighted |
|-----------|-----------|------------|----------|
| stock_r2 | 0.0039 | 0.078 | 0.0195 |
| stock_acc | 0.5617 | 0.5617 | 0.0843 |
| stock_f1 | 0.5380 | 0.5380 | 0.0807 |
| market_r2 | 0.3764 | 0.3764 | 0.1129 |
| market_sig | 0.9553 | 0.9553 | 0.1433 |
| **Total** | | | **≈ 0.4407** |

### Implementation

- File: `utils/composite_score.py`
- Called by cloud agent after each experiment
- Also callable standalone: `python -c "from utils.composite_score import print_score; print_score()"`

## Experiment Journal

### Location

`docs/experiment_journal.md`

### Format

```markdown
# Experiment Journal

## Status
- **Phase:** 1 (Improving Prediction)
- **Baseline Score:** 0.4407
- **Current Best Score:** 0.4407
- **Plateau Counter:** 0
- **Last Experiment:** (none)

## Baseline Metrics (2026-04-01)
- Stock R² (best): 0.0039 (FinBERT, single, r_30d)
- Stock Accuracy (best): 56.17% (FinBERT, single, r_30d)
- Stock F1 (best): 0.538 (FinBERT, single, r_30d)
- Market R² (best): 0.376 (FinBERT, separate, r_30d)
- Market p-value (best): 0.045

## Experiment Log

### EXP-001: <title> (<timestamp>)
- **Hypothesis:** <what we think will improve and why>
- **What changed:** <files modified, what was done>
- **Branch:** experiment/<name>
- **Result:** Composite X.XXXX → X.XXXX (+/-X.XXXX)
- **Outcome:** MERGED ✓ / REJECTED ✗ | Plateau: N
- **Insight:** <what we learned, informs future experiments>
```

### Rules

- Agent appends hypothesis at start of run (survives crashes)
- Agent appends result/outcome after evaluation
- Journal is committed to main after every update
- Agent must read full journal before hypothesizing (avoid repeats)

## Phase 2 — Trading Strategy

### Trigger

Plateau counter reaches 5 (five consecutive experiments with no composite score improvement).

### Transition

1. Agent writes `docs/improvement_summary.md`:
   - Starting vs. final composite score
   - All experiments (successes and failures)
   - Best configuration per metric
   - Recommendations for future work
2. Agent updates journal: `## Phase 1 Complete`
3. Agent's scheduled prompt switches to trading strategy mode

### Trading Strategy Design

- **Signal:** Monthly aggregated sentiment from best model
- **Universe:** S&P 500 index (SPX)
- **Strategy:** Long SPX when sentiment > threshold, cash otherwise
- **Threshold:** Optimized on validation windows
- **Backtest:** Walk-forward on rolling windows, out-of-sample on test period
- **Output:**
  - `scripts/backtest.py` — strategy implementation
  - `results/strategy/backtest_results.json` — performance metrics
  - `results/strategy/equity_curve.png` — monthly equity curve
  - `results/strategy/trade_log.csv` — entry/exit dates, returns

### Strategy Metrics

- Annualized Sharpe ratio
- Maximum drawdown
- Win rate (% of months with positive return)
- Total return vs. buy-and-hold benchmark
- Information ratio vs. SPX

### Phase 2 Plateau

5 consecutive strategy experiments with no Sharpe ratio improvement → agent stops and writes final report.

## Candidate Improvement Ideas (Non-Exhaustive)

The agent is self-directed, but here are plausible improvements it might try:

### Low-Hanging Fruit
- Expand to 3 rolling windows (per original spec)
- Increase HP search from 25 to 50+ configs
- Try different FC architectures (residual connections, batch norm)
- Learning rate scheduling (cosine, step decay)

### Feature Engineering
- Sentence-level embeddings (mean pool all tokens, not just [CLS])
- Combine multiple encoder embeddings as features
- Add article metadata features (publication time, ticker sector)
- TF-IDF or keyword features alongside embeddings

### Model Architecture
- Attention-weighted embedding aggregation
- Ensemble of best models per horizon
- Gradient boosting on embeddings (XGBoost/LightGBM)
- Multi-task learning with auxiliary objectives

### Data & Evaluation
- Data augmentation (paraphrase, back-translation)
- Different train/val/test splits for robustness
- Per-sector models instead of one model for all tickers
- Longer lookback windows for training

## Infrastructure

### GitHub Repository

- **URL:** https://github.com/WillInvest/Sentiment-Analysis-
- **Main branch:** `main`
- **Branch protection:** None required (agent is the only committer)

### Server (School Server)

- **Path:** `/home/fao/projects/sentiment`
- **Conda env:** `sentiment`
- **GPU:** Available (CUDA)
- **Crontab:** `*/10 * * * *` for `server_worker.sh`

### Cloud Agent (Anthropic Scheduled Trigger)

- **Schedule:** Every 2 hours
- **Prompt:** See `docs/agent_prompt.md`
- **Repo:** WillInvest/Sentiment-Analysis-

## Files to Create

| File | Purpose |
|------|---------|
| `utils/composite_score.py` | Compute and print composite score |
| `docs/experiment_journal.md` | Agent's persistent memory |
| `server_worker.sh` | Server-side poll + train script |
| `docs/agent_prompt.md` | The scheduled trigger's system prompt |

## Files to Modify

| File | Change |
|------|--------|
| `utils/progress.py` | Add `reset_stages()` function for selective re-running |
| `CLAUDE.md` | Document the autonomous agent system |
