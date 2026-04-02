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
        log "Stale lock detected (${lock_age}s old), removing"
        rm -f "$LOCK_FILE"
    else
        log "Another worker is running (lock age: ${lock_age}s), exiting"
        exit 0
    fi
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

# System Python — no conda activation needed on this server

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
    log "Processing: experiment/$name"

    wt="$WORKTREE_DIR/$name"

    # Clean up any leftover worktree from a previous crashed run
    if [ -d "$wt" ]; then
        git worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"
    fi

    # Create worktree for this experiment (isolates from main)
    git worktree add "$wt" "$ref" --detach

    # Symlink gitignored data directories into worktree so pipeline can find them
    ln -sf "$REPO_DIR/data" "$wt/data"
    # Create results subdirs and symlink heavy cached data
    mkdir -p "$wt/results/metrics" "$wt/results/predictions" "$wt/results/figures" "$wt/results/training_logs"
    ln -sf "$REPO_DIR/results/embeddings" "$wt/results/embeddings"
    ln -sf "$REPO_DIR/results/checkpoints" "$wt/results/checkpoints"
    # Only copy progress.json if the experiment branch doesn't have its own
    # (experiment branches typically commit a reset progress.json)
    if [ ! -f "$wt/results/progress.json" ]; then
        cp "$REPO_DIR/results/progress.json" "$wt/results/progress.json" 2>/dev/null || true
    fi

    pushd "$wt" > /dev/null

    STATUS="success"
    ERROR_MSG=""

    # Run full pipeline — each script checks progress.json and skips completed stages
    for script in scripts/data_preparation.py scripts/extract_embeddings.py scripts/pretrained_sentiment.py scripts/finetune.py scripts/evaluate.py scripts/market_trend.py; do
        if [ -f "$script" ]; then
            log "Running: $script"
            if ! python3 "$script" 2>&1; then
                STATUS="failed"
                ERROR_MSG="$script failed"
                log "ERROR: $ERROR_MSG"
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
        log "ERROR: Failed to push results/$name"
    fi

    popd > /dev/null

    # Cleanup worktree
    git worktree remove "$wt" --force 2>/dev/null || rm -rf "$wt"

    log "Completed: experiment/$name ($STATUS)"
done

if [ "$FOUND_WORK" = false ]; then
    log "No new experiments found"
fi
