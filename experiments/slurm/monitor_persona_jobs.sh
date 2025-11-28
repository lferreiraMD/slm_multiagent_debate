#!/bin/bash

# Monitor persona diversity experiments in real-time
# Usage: bash monitor_persona_jobs.sh [interval_seconds]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=${1:-30}  # Default: refresh every 30 seconds

# Load job IDs if available
MATH_JOB_ID=""
GSM_JOB_ID=""
BIOGRAPHY_JOB_ID=""
MMLU_JOB_ID=""

if [ -f "$SCRIPT_DIR/logs/last_math_job_id.txt" ]; then
    MATH_JOB_ID=$(cat "$SCRIPT_DIR/logs/last_math_job_id.txt")
fi
if [ -f "$SCRIPT_DIR/logs/last_gsm_job_id.txt" ]; then
    GSM_JOB_ID=$(cat "$SCRIPT_DIR/logs/last_gsm_job_id.txt")
fi
if [ -f "$SCRIPT_DIR/logs/last_biography_job_id.txt" ]; then
    BIOGRAPHY_JOB_ID=$(cat "$SCRIPT_DIR/logs/last_biography_job_id.txt")
fi
if [ -f "$SCRIPT_DIR/logs/last_mmlu_job_id.txt" ]; then
    MMLU_JOB_ID=$(cat "$SCRIPT_DIR/logs/last_mmlu_job_id.txt")
fi

# Function to count job states
count_jobs() {
    local job_id=$1
    local pending=$(squeue -j "$job_id" -h -t PENDING 2>/dev/null | wc -l | tr -d ' ')
    local running=$(squeue -j "$job_id" -h -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
    local total=$((pending + running))
    echo "$pending,$running,$total"
}

# Function to get completion count from log files
count_completed() {
    local task=$1
    local job_id=$2
    local completed=0

    if [ -d "$SCRIPT_DIR/logs" ]; then
        # Count log files with "Exit code: 0"
        completed=$(grep -l "Exit code: 0" "$SCRIPT_DIR/logs/persona_${task}_${job_id}_"*.out 2>/dev/null | wc -l | tr -d ' ')
    fi

    echo "$completed"
}

echo "=================================================="
echo "Persona Diversity Experiment Monitor"
echo "=================================================="
echo "Refresh interval: ${INTERVAL}s (Ctrl+C to stop)"
echo "Job IDs:"
echo "  Math:      $MATH_JOB_ID"
echo "  GSM:       $GSM_JOB_ID"
echo "  Biography: $BIOGRAPHY_JOB_ID"
echo "  MMLU:      $MMLU_JOB_ID"
echo "=================================================="
echo ""

# Continuous monitoring loop
while true; do
    clear
    echo "=================================================="
    echo "Persona Diversity Experiment Status"
    echo "Updated: $(date)"
    echo "=================================================="
    echo ""

    # Overall queue status
    echo "User Queue Summary:"
    squeue -u $USER --format="%.10i %.12j %.10T %.10M %.6D" 2>/dev/null || echo "  No jobs in queue"
    echo ""

    # Detailed status per task
    echo "Task-Specific Progress:"
    echo "----------------------------------------"

    if [ -n "$MATH_JOB_ID" ]; then
        math_stats=$(count_jobs "$MATH_JOB_ID")
        math_pending=$(echo "$math_stats" | cut -d',' -f1)
        math_running=$(echo "$math_stats" | cut -d',' -f2)
        math_active=$(echo "$math_stats" | cut -d',' -f3)
        math_completed=$(count_completed "math" "$MATH_JOB_ID")
        math_total=60
        math_done=$((math_completed + math_active))
        math_pct=$((math_done * 100 / math_total))

        echo "Math ($MATH_JOB_ID):"
        echo "  Completed: $math_completed/60 | Running: $math_running | Pending: $math_pending | Progress: ${math_pct}%"
    fi

    if [ -n "$GSM_JOB_ID" ]; then
        gsm_stats=$(count_jobs "$GSM_JOB_ID")
        gsm_pending=$(echo "$gsm_stats" | cut -d',' -f1)
        gsm_running=$(echo "$gsm_stats" | cut -d',' -f2)
        gsm_active=$(echo "$gsm_stats" | cut -d',' -f3)
        gsm_completed=$(count_completed "gsm" "$GSM_JOB_ID")
        gsm_total=60
        gsm_done=$((gsm_completed + gsm_active))
        gsm_pct=$((gsm_done * 100 / gsm_total))

        echo "GSM ($GSM_JOB_ID):"
        echo "  Completed: $gsm_completed/60 | Running: $gsm_running | Pending: $gsm_pending | Progress: ${gsm_pct}%"
    fi

    if [ -n "$BIOGRAPHY_JOB_ID" ]; then
        bio_stats=$(count_jobs "$BIOGRAPHY_JOB_ID")
        bio_pending=$(echo "$bio_stats" | cut -d',' -f1)
        bio_running=$(echo "$bio_stats" | cut -d',' -f2)
        bio_active=$(echo "$bio_stats" | cut -d',' -f3)
        bio_completed=$(count_completed "biography" "$BIOGRAPHY_JOB_ID")
        bio_total=60
        bio_done=$((bio_completed + bio_active))
        bio_pct=$((bio_done * 100 / bio_total))

        echo "Biography ($BIOGRAPHY_JOB_ID):"
        echo "  Completed: $bio_completed/60 | Running: $bio_running | Pending: $bio_pending | Progress: ${bio_pct}%"
    fi

    if [ -n "$MMLU_JOB_ID" ]; then
        mmlu_stats=$(count_jobs "$MMLU_JOB_ID")
        mmlu_pending=$(echo "$mmlu_stats" | cut -d',' -f1)
        mmlu_running=$(echo "$mmlu_stats" | cut -d',' -f2)
        mmlu_active=$(echo "$mmlu_stats" | cut -d',' -f3)
        mmlu_completed=$(count_completed "mmlu" "$MMLU_JOB_ID")
        mmlu_total=60
        mmlu_done=$((mmlu_completed + mmlu_active))
        mmlu_pct=$((mmlu_done * 100 / mmlu_total))

        echo "MMLU ($MMLU_JOB_ID):"
        echo "  Completed: $mmlu_completed/60 | Running: $mmlu_running | Pending: $mmlu_pending | Progress: ${mmlu_pct}%"
    fi

    echo ""
    echo "Overall Progress:"
    total_completed=$((${math_completed:-0} + ${gsm_completed:-0} + ${bio_completed:-0} + ${mmlu_completed:-0}))
    total_running=$((${math_running:-0} + ${gsm_running:-0} + ${bio_running:-0} + ${mmlu_running:-0}))
    total_pending=$((${math_pending:-0} + ${gsm_pending:-0} + ${bio_pending:-0} + ${mmlu_pending:-0}))
    overall_pct=$((total_completed * 100 / 240))

    echo "  Completed: $total_completed/240 (${overall_pct}%)"
    echo "  Running:   $total_running"
    echo "  Pending:   $total_pending"

    echo ""
    echo "=================================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Next refresh in ${INTERVAL}s..."

    sleep "$INTERVAL"
done
