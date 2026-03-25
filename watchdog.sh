#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pipeline Watchdog                                           ║
# ║                                                              ║
# ║  Runs the pipeline with live progress display.               ║
# ║  If it fails, sends the error log to Claude Code CLI to      ║
# ║  diagnose and fix, then retries.                             ║
# ║  Repeats up to MAX_RETRIES times.                            ║
# ║                                                              ║
# ║  Usage: ./watchdog.sh [max_retries]                          ║
# ║         ./watchdog.sh 10    # try up to 10 times             ║
# ╚═══════════════════════════════════════════════════════════════╝

set -u

MAX_RETRIES="${1:-5}"       # default 5 retries, override: ./watchdog.sh 10
LOG_DIR="results/watchdog"
PIPELINE_CMD="python3 run_pipeline.py"

mkdir -p "$LOG_DIR"

# ─── Helpers ───

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/watchdog.log"; }

extract_error_context() {
    local logfile="$1"
    {
        echo "=== LAST 80 LINES ==="
        tail -80 "$logfile"
        echo ""
        echo "=== TRACEBACK LINES ==="
        grep -n -B2 -A5 -E "Traceback|Error|Exception|FAILED" "$logfile" | tail -60
    } 2>/dev/null
}

# ─── Main loop ───

attempt=0

while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    run_log="$LOG_DIR/run_${attempt}.log"

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  Attempt $attempt/$MAX_RETRIES"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Pipeline log → $run_log"
    log ""

    # Run pipeline with LIVE output (tee to both terminal and log file)
    # Progress bar renders in terminal, full output captured in log
    $PIPELINE_CMD 2>&1 | tee "$run_log"
    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        log ""
        log "╔═══════════════════════════════════════════╗"
        log "║  ✓ Pipeline completed on attempt $attempt!       ║"
        log "╚═══════════════════════════════════════════╝"
        log ""

        python3 -c "from utils.progress import print_progress_summary; print_progress_summary()" 2>/dev/null
        exit 0
    fi

    log ""
    log "✗ Pipeline failed (exit code $exit_code)"

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        log ""
        log "╔═══════════════════════════════════════════╗"
        log "║  Max retries ($MAX_RETRIES) reached. Giving up.   ║"
        log "╚═══════════════════════════════════════════╝"
        log "Last log: $run_log"
        exit 1
    fi

    # Extract error context for Claude
    error_context=$(extract_error_context "$run_log")
    fix_log="$LOG_DIR/fix_${attempt}.log"

    log ""
    log "🔧 Sending error to Claude Code for auto-fix..."
    log ""

    # Build the prompt for Claude
    prompt="$(cat <<PROMPT
The financial sentiment analysis pipeline failed on attempt $attempt.
Working directory: /home/fao/projects/sentiment

Here is the error output:

$error_context

Instructions:
1. Read the failing script to understand the code around the error
2. Identify the root cause
3. Fix the code (edit the file directly)
4. Do NOT re-run the pipeline — just fix the bug and exit
5. Be concise — state what you found and what you fixed

The pipeline is resumable — it will retry from where it left off after your fix.
PROMPT
)"

    # Call Claude Code in non-interactive mode
    echo "$prompt" | claude -p \
        --allowedTools "Read Edit Write Grep Glob Bash" \
        > "$fix_log" 2>&1

    claude_exit=$?

    if [ "$claude_exit" -ne 0 ]; then
        log "⚠ Claude Code exited with code $claude_exit"
        log "Claude output saved to: $fix_log"
        log "Retrying pipeline anyway..."
    else
        # Show what Claude fixed
        log "Claude's fix:"
        cat "$fix_log" | head -10
        log "(full output: $fix_log)"
    fi

    log ""
    log "Retrying in 5 seconds..."
    sleep 5
done
