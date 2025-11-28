#!/bin/bash

# Resubmit failed Linux persona diversity experiment jobs
# Usage: bash resubmit_failed.sh [task]
#   task: math, gsm, biography, mmlu, or "all" (default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK=${1:-all}

# Function to find failed jobs for a task
find_and_resubmit_failed() {
    local task=$1
    local config_file="$SCRIPT_DIR/configs/persona_${task}_jobs.txt"
    local log_dir="$SCRIPT_DIR/logs/${task}"

    if [ ! -f "$config_file" ]; then
        echo "No config file found for $task"
        return
    fi

    if [ ! -d "$log_dir" ]; then
        echo "No logs found for $task (never run?)"
        return
    fi

    echo "Checking $task jobs..."

    local failed_jobs=()
    local job_num=0

    # Check each job in config
    while IFS= read -r line; do
        # Skip header
        if [ $job_num -eq 0 ]; then
            job_num=1
            continue
        fi

        # Extract job_id (first field)
        job_id=$(echo "$line" | cut -d',' -f1)
        log_file="$log_dir/job_${job_id}.out"

        # Check if log exists and has errors
        if [ ! -f "$log_file" ]; then
            echo "  Job $job_id: No log file (never ran)"
            failed_jobs+=("$line")
        elif grep -q "Error\|Failed\|Traceback" "$log_file" && ! grep -q "COMPLETE\|Complete\|Completed" "$log_file"; then
            echo "  Job $job_id: Failed (found errors)"
            failed_jobs+=("$line")
        elif ! grep -q "COMPLETE\|Complete\|Completed" "$log_file"; then
            echo "  Job $job_id: Incomplete (no completion marker)"
            failed_jobs+=("$line")
        fi

        job_num=$((job_num + 1))

    done < "$config_file"

    if [ ${#failed_jobs[@]} -eq 0 ]; then
        echo "  ✓ No failed jobs found for $task"
        return 0
    fi

    echo ""
    echo "Found ${#failed_jobs[@]} failed job(s) for $task"
    echo ""

    # Create temporary config file with only failed jobs
    local temp_config="$SCRIPT_DIR/configs/persona_${task}_retry.txt"
    local header=$(head -n 1 "$config_file")

    echo "$header" > "$temp_config"
    for job_line in "${failed_jobs[@]}"; do
        echo "$job_line" >> "$temp_config"
    done

    echo "Created retry config: $temp_config"
    echo ""

    read -p "Resubmit these ${#failed_jobs[@]} jobs for $task? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Resubmitting failed $task jobs..."

        # Backup original config
        mv "$config_file" "${config_file}.backup"
        mv "$temp_config" "$config_file"

        # Run the task script
        bash "$SCRIPT_DIR/run_persona_${task}.sh"

        # Restore original config
        mv "$config_file" "$temp_config"
        mv "${config_file}.backup" "$config_file"

        echo "✓ Resubmission complete for $task"
        echo "  Retry config saved as: $temp_config"
    else
        echo "Skipping resubmission for $task"
        rm "$temp_config"
    fi
    echo ""
}

# Main logic
echo "=================================================="
echo "Failed Job Resubmission Tool - Linux"
echo "=================================================="
echo ""

if [ "$TASK" == "all" ]; then
    for t in math gsm biography mmlu; do
        find_and_resubmit_failed "$t"
    done
else
    find_and_resubmit_failed "$TASK"
fi

echo "=================================================="
echo "Resubmission check complete"
echo "=================================================="
