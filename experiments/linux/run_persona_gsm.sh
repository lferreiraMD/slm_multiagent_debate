#!/bin/bash

# Linux parallel execution script for persona diversity GSM experiments
# Runs 60 experiments using GNU parallel or background processes
# Optimized for Ubuntu with vLLM on NVIDIA GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="gsm"

echo "=================================================="
echo "Linux Persona Diversity Experiments - GSM Task"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Configuration
CONFIG_FILE="$SCRIPT_DIR/configs/persona_${TASK}_jobs.txt"
LOG_DIR="$SCRIPT_DIR/logs/${TASK}"
MAX_PARALLEL=${MAX_PARALLEL:-2}  # Default: 2 parallel jobs (for dual GPU setup)

# Create log directory
mkdir -p "$LOG_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Run generate_job_configs.py first:"
    echo "  python3 $SCRIPT_DIR/generate_job_configs.py"
    exit 1
fi

# Count total jobs (excluding header)
TOTAL_JOBS=$(($(wc -l < "$CONFIG_FILE") - 1))

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Total jobs: $TOTAL_JOBS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Log directory: $LOG_DIR"
echo "=================================================="
echo ""

# Function to run a single job
run_job() {
    local job_line="$1"
    local job_num="$2"

    # Parse CSV line
    IFS=',' read -r job_id model_alias n_agents rounds task num_param num_value random_seed personas_tuple <<< "$job_line"

    # Remove quotes from personas_tuple
    personas_tuple=$(echo "$personas_tuple" | sed 's/^"//;s/"$//')

    # Convert persona tuple to space-separated args
    personas_args=$(echo "$personas_tuple" | sed "s/[()']//g" | sed 's/, / /g')

    echo "[Job $job_num/$TOTAL_JOBS] Starting: model=$model_alias agents=$n_agents"

    # Navigate to task directory
    cd "$PROJECT_ROOT/tasks/$task"

    # Run experiment
    python3 gen_${task}.py \
        --model "$model_alias" \
        --agents "$n_agents" \
        --rounds "$rounds" \
        --num-problems "$num_value" \
        --agent-personas $personas_args \
        > "$LOG_DIR/job_${job_id}.out" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[Job $job_num/$TOTAL_JOBS] ✓ Completed: model=$model_alias agents=$n_agents"
    else
        echo "[Job $job_num/$TOTAL_JOBS] ✗ Failed (exit $exit_code): model=$model_alias agents=$n_agents"
    fi

    return $exit_code
}

export -f run_job
export PROJECT_ROOT TASK LOG_DIR TOTAL_JOBS

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job execution"
    echo ""

    # Skip header and run jobs in parallel
    tail -n +2 "$CONFIG_FILE" | nl -v 1 | parallel --colsep '\t' --jobs "$MAX_PARALLEL" run_job {2} {1}

    exit_code=$?

else
    echo "GNU parallel not found, using background processes"
    echo "Install with: sudo apt-get install parallel"
    echo ""

    # Fallback: use background processes with job control
    job_num=0
    active_jobs=0

    while IFS= read -r line; do
        # Skip header
        if [ $job_num -eq 0 ]; then
            job_num=1
            continue
        fi

        # Wait if we've hit max parallel jobs
        while [ $active_jobs -ge $MAX_PARALLEL ]; do
            wait -n  # Wait for any job to finish
            active_jobs=$((active_jobs - 1))
        done

        # Run job in background
        run_job "$line" "$job_num" &
        active_jobs=$((active_jobs + 1))
        job_num=$((job_num + 1))

    done < "$CONFIG_FILE"

    # Wait for all remaining jobs
    wait
    exit_code=$?
fi

echo ""
echo "=================================================="
echo "GSM Task Experiments Complete"
echo "=================================================="
echo "Exit code: $exit_code"
echo "Logs: $LOG_DIR"
echo ""

# Count successes and failures
SUCCESS_COUNT=$(grep -l "Exit code: 0" "$LOG_DIR"/*.out 2>/dev/null | wc -l || echo 0)
FAILURE_COUNT=$(( TOTAL_JOBS - SUCCESS_COUNT ))

echo "Results:"
echo "  Successful: $SUCCESS_COUNT / $TOTAL_JOBS"
echo "  Failed: $FAILURE_COUNT / $TOTAL_JOBS"
echo "=================================================="

exit $exit_code
