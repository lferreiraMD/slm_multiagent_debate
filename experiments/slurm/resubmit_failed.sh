#!/bin/bash

# Resubmit failed persona diversity experiment jobs
# Usage: bash resubmit_failed.sh [task]
#   task: math, gsm, biography, mmlu, or "all" (default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=${1:-all}

# Function to find failed jobs for a task
find_failed_jobs() {
    local task=$1
    local job_id_file="$SCRIPT_DIR/logs/last_${task}_job_id.txt"

    if [ ! -f "$job_id_file" ]; then
        echo "No job ID file found for $task"
        return
    fi

    local job_id=$(cat "$job_id_file")
    local failed_tasks=()

    echo "Checking $task jobs (Job ID: $job_id)..."

    # Check log files for failures
    for log_file in "$SCRIPT_DIR/logs/persona_${task}_${job_id}_"*.out; do
        if [ -f "$log_file" ]; then
            # Extract task ID from filename
            task_id=$(basename "$log_file" | sed "s/persona_${task}_${job_id}_\([0-9]*\)\.out/\1/")

            # Check for non-zero exit code
            exit_code=$(grep "Exit code:" "$log_file" | tail -n 1 | awk '{print $3}')

            if [ -n "$exit_code" ] && [ "$exit_code" != "0" ]; then
                failed_tasks+=("$task_id")
                echo "  Task $task_id failed with exit code $exit_code"
            fi
        fi
    done

    # Check for missing output files (jobs that never ran or were killed)
    for i in {2..60}; do  # Skip task 1 (header)
        if [ ! -f "$SCRIPT_DIR/logs/persona_${task}_${job_id}_${i}.out" ]; then
            failed_tasks+=("$i")
            echo "  Task $i has no output file (never ran or killed)"
        fi
    done

    if [ ${#failed_tasks[@]} -eq 0 ]; then
        echo "  No failed jobs found for $task"
        return
    fi

    # Create array specification for resubmission
    local array_spec=$(IFS=,; echo "${failed_tasks[*]}")

    echo ""
    echo "Found ${#failed_tasks[@]} failed job(s) for $task"
    echo "Failed task IDs: $array_spec"
    echo ""
    read -p "Resubmit these jobs? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Resubmitting failed $task jobs..."
        sbatch --array="$array_spec" "$SCRIPT_DIR/run_persona_${task}.slurm"
        echo "Resubmission complete!"
    else
        echo "Skipping resubmission for $task"
    fi
    echo ""
}

# Main logic
echo "=================================================="
echo "Failed Job Resubmission Tool"
echo "=================================================="
echo ""

if [ "$TASK" == "all" ]; then
    for t in math gsm biography mmlu; do
        find_failed_jobs "$t"
    done
else
    find_failed_jobs "$TASK"
fi

echo "=================================================="
echo "Resubmission check complete"
echo "=================================================="
