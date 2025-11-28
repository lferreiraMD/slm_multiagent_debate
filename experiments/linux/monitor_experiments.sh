#!/bin/bash

# Monitor running Linux persona diversity experiments
# Shows real-time progress for all tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=${1:-5}  # Default: refresh every 5 seconds

echo "=================================================="
echo "Linux Experiment Monitor"
echo "=================================================="
echo "Refresh interval: ${INTERVAL}s (Ctrl+C to stop)"
echo "=================================================="
echo ""

# Function to count completions
count_completions() {
    local task=$1
    local log_dir="$SCRIPT_DIR/logs/$task"

    if [ ! -d "$log_dir" ]; then
        echo "0,0,0"
        return
    fi

    local total_logs=$(ls "$log_dir"/job_*.out 2>/dev/null | wc -l)
    local completed=$(grep -l "COMPLETE\|Complete\|Completed" "$log_dir"/*.out 2>/dev/null | wc -l)
    local failed=$(grep -l "Error\|Failed\|Traceback" "$log_dir"/*.out 2>/dev/null | grep -v -l "COMPLETE\|Complete\|Completed" 2>/dev/null | wc -l)

    echo "$total_logs,$completed,$failed"
}

# Function to check if processes are running
check_running() {
    local task=$1
    local count=$(ps aux | grep -c "gen_${task}.py\|gen_conversation.py" | grep -v grep || echo 0)
    echo $count
}

# Continuous monitoring loop
while true; do
    clear
    echo "=================================================="
    echo "Linux Experiment Status"
    echo "Updated: $(date)"
    echo "=================================================="
    echo ""

    # Overall progress
    total_completed=0
    total_failed=0
    total_running=0

    for task in math gsm biography mmlu; do
        stats=$(count_completions "$task")
        IFS=',' read -r total_logs completed failed <<< "$stats"
        running=$(check_running "$task")

        total_completed=$((total_completed + completed))
        total_failed=$((total_failed + failed))
        total_running=$((total_running + running))

        # Calculate progress
        if [ $total_logs -gt 0 ]; then
            pct=$((completed * 100 / 60))  # Out of 60 total
        else
            pct=0
        fi

        # Status display
        status="○"  # Not started
        if [ $running -gt 0 ]; then
            status="●"  # Running
        elif [ $completed -eq 60 ]; then
            status="✓"  # Complete
        elif [ $total_logs -gt 0 ]; then
            status="◐"  # Partial
        fi

        printf "%-12s %s Completed: %2d/60 (%3d%%) | Running: %d | Failed: %d\n" \
            "$task" "$status" "$completed" "$pct" "$running" "$failed"
    done

    echo ""
    echo "Overall Progress:"
    overall_pct=$((total_completed * 100 / 240))
    printf "  Completed: %3d/240 (%3d%%)\n" "$total_completed" "$overall_pct"
    printf "  Running:   %3d\n" "$total_running"
    printf "  Failed:    %3d\n" "$total_failed"

    echo ""
    echo "GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s | VRAM: %s/%s MB | Util: %s%%\n", $1, $2, $3, $4, $5}'
    else
        echo "  nvidia-smi not available"
    fi

    echo ""
    echo "Python Processes:"
    ps aux | grep "python.*gen_" | grep -v grep | awk '{print "  "$11, $12, $13, $14, $15}' | head -n 5

    echo ""
    echo "=================================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Next refresh in ${INTERVAL}s..."

    sleep "$INTERVAL"
done
