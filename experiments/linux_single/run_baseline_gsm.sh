#!/bin/bash

# Linux SINGLE GPU parallel execution script for baseline GSM experiments
# Runs 36 experiments using GNU parallel or background processes
# Optimized for Ubuntu with vLLM on single NVIDIA RTX 3090 (24GB VRAM)

set -e

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="gsm"
RESULTS_DIR="$PROJECT_ROOT/results/baseline/$TASK"

echo "=================================================="
echo "Linux Single GPU Baseline - GSM Task"
echo "=================================================="
echo "Hardware: Single RTX 3090 (24GB VRAM)"
echo "Models: 9 (0.6B-8B, excluding 14B)"
echo "Agent Counts: [1, 3, 5, 7] (single-agent baseline + multiagent debate)"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""

# GPU memory check function
check_gpu_memory() {
    local model_alias="$1"
    local n_agents="$2"

    local mem_info=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from utils.cuda_cleanup import get_cuda_memory_stats

stats = get_cuda_memory_stats(device=0)
if stats:
    print(f\"{stats['free']:.2f},{stats['total']:.2f}\")
else:
    print('0,0')
" 2>/dev/null)

    IFS=',' read -r free_gb total_gb <<< "$mem_info"

    local required_gb=0
    case "$model_alias" in
        *0.6b*|*vibethinker*) required_gb=2 ;;
        *1.7b*|*deepseek*) required_gb=4 ;;
        *3b*|*smallthinker*) required_gb=7 ;;
        *4b*) required_gb=9 ;;
        *7b*) required_gb=15 ;;
        *8b*) required_gb=17 ;;
        *) required_gb=10 ;;
    esac

    local overhead=$(echo "$n_agents * 0.5" | bc -l)
    required_gb=$(echo "$required_gb + $overhead" | bc -l)

    local available=$(echo "$free_gb - 1.0" | bc -l)
    local sufficient=$(echo "$available >= $required_gb" | bc -l)

    if [ "$(echo "$sufficient == 1" | bc -l)" -eq 1 ]; then
        echo "GPU memory OK: ${free_gb}GB free, ~${required_gb}GB required"
        return 0
    else
        echo "WARNING: Low GPU memory! ${free_gb}GB free, ~${required_gb}GB required"
        echo "Consider reducing MAX_PARALLEL or clearing GPU memory"
        return 1
    fi
}

export -f check_gpu_memory

CONFIG_FILE="$SCRIPT_DIR/configs/baseline_${TASK}_jobs.txt"
LOG_DIR="$SCRIPT_DIR/logs/${TASK}_baseline"
MAX_PARALLEL=${MAX_PARALLEL:-2}

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Run generate_baseline_configs.py first:"
    echo "  python3 $SCRIPT_DIR/generate_baseline_configs.py"
    exit 1
fi

TOTAL_JOBS=$(($(wc -l < "$CONFIG_FILE") - 1))

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Total jobs: $TOTAL_JOBS (expected: 36)"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Log directory: $LOG_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "=================================================="
echo ""

run_job() {
    local job_line="$1"
    local job_num="$2"

    eval "$(python3 -c "
import csv
import sys

try:
    row = next(csv.reader(['$job_line']))
    job_id, model_alias, n_agents, rounds, task, num_param, num_value, random_seed = row

    print(f'job_id={job_id}')
    print(f'model_alias={model_alias}')
    print(f'n_agents={n_agents}')
    print(f'rounds={rounds}')
    print(f'task={task}')
    print(f'num_param={num_param}')
    print(f'num_value={num_value}')
    print(f'random_seed={random_seed}')
except Exception as e:
    print(f'echo \"ERROR: CSV parsing failed: {e}\" >&2')
    exit(1)
")"

    echo "[Job $job_num/$TOTAL_JOBS] Starting: model=$model_alias agents=$n_agents"

    if ! check_gpu_memory "$model_alias" "$n_agents"; then
        echo "[Job $job_num/$TOTAL_JOBS] ⚠ Skipping due to insufficient GPU memory"
        echo "[Job $job_num/$TOTAL_JOBS] ⚠ Run with MAX_PARALLEL=1 or clear GPU memory" >> "$LOG_DIR/job_${job_id}.out"
        return 2
    fi

    cd "$PROJECT_ROOT/tasks/$task"

    python3 gen_${task}.py \
        --model "$model_alias" \
        --agents "$n_agents" \
        --rounds "$rounds" \
        --num-problems "$num_value" \
        --output-directory "$RESULTS_DIR" \
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
export PROJECT_ROOT TASK LOG_DIR TOTAL_JOBS RESULTS_DIR

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job execution"
    echo ""

    tail -n +2 "$CONFIG_FILE" | nl -v 1 | parallel --colsep '\t' --jobs "$MAX_PARALLEL" run_job {2} {1}

    exit_code=$?

else
    echo "GNU parallel not found, using background processes"
    echo "Install with: sudo apt-get install parallel"
    echo ""

    job_num=0
    active_jobs=0

    while IFS= read -r line; do
        if [ $job_num -eq 0 ]; then
            job_num=1
            continue
        fi

        while [ $active_jobs -ge $MAX_PARALLEL ]; do
            wait -n
            active_jobs=$((active_jobs - 1))
        done

        run_job "$line" "$job_num" &
        active_jobs=$((active_jobs + 1))
        job_num=$((job_num + 1))

    done < "$CONFIG_FILE"

    wait
    exit_code=$?
fi

echo ""
echo "=================================================="
echo "GSM Task Baseline Experiments Complete"
echo "=================================================="
echo "Exit code: $exit_code"
echo "Logs: $LOG_DIR"
echo "Results: $RESULTS_DIR"
echo ""

SUCCESS_COUNT=$(grep -l "Exit code: 0" "$LOG_DIR"/*.out 2>/dev/null | wc -l || echo 0)
FAILURE_COUNT=$(( TOTAL_JOBS - SUCCESS_COUNT ))

echo "Results:"
echo "  Successful: $SUCCESS_COUNT / $TOTAL_JOBS (expected: 36)"
echo "  Failed: $FAILURE_COUNT / $TOTAL_JOBS"
if [ $FAILURE_COUNT -gt 0 ]; then
    echo ""
    echo "Tip: Check logs for OOM errors. If present, reduce MAX_PARALLEL:"
    echo "  MAX_PARALLEL=1 bash run_baseline_gsm.sh"
fi
echo "=================================================="

exit $exit_code
