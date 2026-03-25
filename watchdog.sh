#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║  Pipeline Watchdog                                           ║
# ║                                                              ║
# ║  Runs the pipeline. If it fails, sends the error log to      ║
# ║  Claude Code CLI to diagnose and fix, then retries.          ║
# ║  Repeats up to MAX_RETRIES times.                            ║
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
    # Get the last 80 lines (enough for traceback + context)
    # Plus grep for ERROR, Exception, Traceback lines with surrounding context
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

    log "━━━ Attempt $attempt/$MAX_RETRIES ━━━"
    log "Running pipeline → $run_log"

    # Run pipeline, capture all output
    $PIPELINE_CMD > "$run_log" 2>&1
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        log "✓ Pipeline completed successfully on attempt $attempt!"
        log "Check results/ for output."

        # Show final progress
        python3 -c "from utils.progress import print_progress_summary; print_progress_summary()" 2>/dev/null
        exit 0
    fi

    log "✗ Pipeline failed (exit code $exit_code)"

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        log "Max retries ($MAX_RETRIES) reached. Giving up."
        log "Last log: $run_log"
        exit 1
    fi

    # Extract error context for Claude
    error_context=$(extract_error_context "$run_log")
    fix_log="$LOG_DIR/fix_${attempt}.log"

    log "Sending error to Claude Code for diagnosis..."

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
        log "Claude output: $fix_log"
        log "Retrying pipeline anyway..."
    else
        log "Claude applied a fix. See: $fix_log"
    fi

    # Brief pause before retry
    log "Retrying in 5 seconds..."
    sleep 5
done
