#!/bin/bash

# Linux SINGLE GPU parallel execution script for persona diversity math experiments
# Runs 54 experiments using GNU parallel or background processes
# Optimized for Ubuntu with vLLM on single NVIDIA RTX 3090 (24GB VRAM)

set -e

# LAPTOP WITH EXTERNAL GPU
# Force CUDA to use only GPU #1 (RTX 3090)
# GPU 0 = GTX 1650 (4GB) - internal, insufficient VRAM
# GPU 1 = RTX 3090 (24GB) - external, target GPU

# RESEARCH WORKSTATIOS WITH 2X RTX 3090
# Force CUDA to use BOTH GPUs
# GPU 0 = RTX 3090 (24GB) - internal, target GPU
# GPU 1 = RTX 3090 (24GB) - internal, target GPU

export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="math"
RESULTS_DIR="$PROJECT_ROOT/results/persona_experiments/$TASK"

echo "=================================================="
echo "Linux Single GPU Persona Diversity - Math Task"
echo "=================================================="
echo "Hardware: Single RTX 3090 (24GB VRAM)"
echo "Models: 9 (0.6B-8B, excluding 14B)"
echo "Project root: $PROJECT_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""

# GPU memory check function
check_gpu_memory() {
    local model_alias="$1"
    local n_agents="$2"

    # Use Python to check available VRAM
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

    # Estimate required memory (rough heuristics)
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

    # Add overhead for multi-agent (KV cache: 0.5GB per agent)
    local overhead=$(echo "$n_agents * 0.5" | bc -l)
    required_gb=$(echo "$required_gb + $overhead" | bc -l)

    # Check if enough free memory (1GB safety margin)
    local available=$(echo "$free_gb - 1.0" | bc -l)
    local sufficient=$(echo "$available >= $required_gb" | bc -l)

    if [ "$(echo "$sufficient == 1" | bc -l)" -eq 1 ]; then
        echo "GPU memory OK: ${free_gb}GB free, ~${required_gb}GB required"
        return 0
    else
        echo "WARNING: Low GPU memory! ${free_gb}GB free, ~${required_gb}GB required"
        echo "Clear GPU memory or restart the GPU before continuing"
        return 1
    fi
}

export -f check_gpu_memory

# Configuration
CONFIG_FILE="$SCRIPT_DIR/configs/persona_${TASK}_jobs.txt"
LOG_DIR="$SCRIPT_DIR/logs/${TASK}"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Run generate_job_configs.py first:"
    echo "  python3 $SCRIPT_DIR/generate_job_configs.py"
    exit 1
fi

# Count total jobs (excluding header) - should be 54 for single GPU (9 models)
TOTAL_JOBS=$(($(wc -l < "$CONFIG_FILE") - 1))

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Total jobs: $TOTAL_JOBS (expected: 54)"
echo "  Log directory: $LOG_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "=================================================="

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"
echo ""

# Function to run a single job
run_job() {
    local job_line="$1"
    local job_num="$2"

    # Parse CSV line using Python for proper CSV quoting support
    eval "$(python3 -c "
import csv
import ast
import sys

try:
    # Parse CSV line (handles quoted fields with commas correctly)
    row = next(csv.reader(['$job_line']))
    job_id, model_alias, n_agents, rounds, task, num_param, num_value, random_seed, personas_tuple = row

    # Parse personas tuple string representation into list
    try:
        personas = ast.literal_eval(personas_tuple)
        # Build bash array syntax: 'persona1' 'persona2' 'persona3'
        personas_args = ' '.join(repr(p) for p in personas)
    except Exception as e:
        print(f'echo \"ERROR: Failed to parse personas: {e}\" >&2')
        personas_args = ''

    # Output as bash variable assignments
    print(f'job_id={job_id}')
    print(f'model_alias={model_alias}')
    print(f'n_agents={n_agents}')
    print(f'rounds={rounds}')
    print(f'task={task}')
    print(f'num_param={num_param}')
    print(f'num_value={num_value}')
    print(f'random_seed={random_seed}')
    print(f'personas_args=({personas_args})')
except Exception as e:
    print(f'echo \"ERROR: CSV parsing failed: {e}\" >&2')
    exit(1)
")"

    echo "[Job $job_num/$TOTAL_JOBS] Starting: model=$model_alias agents=$n_agents"

    # Pre-flight GPU memory check
    if ! check_gpu_memory "$model_alias" "$n_agents"; then
        echo "[Job $job_num/$TOTAL_JOBS] ⚠ Skipping due to insufficient GPU memory"
        echo "[Job $job_num/$TOTAL_JOBS] ⚠ Run with MAX_PARALLEL=1 or clear GPU memory" >> "$LOG_DIR/job_${job_id}.out"
        return 2  # Return code 2 = skipped due to memory
    fi

    # Navigate to task directory
    cd "$PROJECT_ROOT/tasks/$task"

    # Run experiment
    python3 gen_${task}.py \
        --model "$model_alias" \
        --agents "$n_agents" \
        --rounds "$rounds" \
        --num-problems "$num_value" \
        --agent-personas ${personas_args} \
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

# Run jobs sequentially
echo "Running jobs sequentially..."
echo ""

job_num=0
exit_code=0

while IFS= read -r line; do
    # Skip header
    if [ $job_num -eq 0 ]; then
        job_num=1
        continue
    fi

    run_job "$line" "$job_num"
    job_status=$?

    # Track if any job failed (but continue running others)
    if [ $job_status -ne 0 ]; then
        exit_code=$job_status
    fi

    job_num=$((job_num + 1))

done < "$CONFIG_FILE"

echo ""
echo "=================================================="
echo "Math Task Experiments Complete"
echo "=================================================="
echo "Exit code: $exit_code"
echo "Logs: $LOG_DIR"
echo ""

# Count successes and failures
SUCCESS_COUNT=$(grep -l "Exit code: 0" "$LOG_DIR"/*.out 2>/dev/null | wc -l || echo 0)
FAILURE_COUNT=$(( TOTAL_JOBS - SUCCESS_COUNT ))

echo "Results:"
echo "  Successful: $SUCCESS_COUNT / $TOTAL_JOBS (expected: 54)"
echo "  Failed: $FAILURE_COUNT / $TOTAL_JOBS"
if [ $FAILURE_COUNT -gt 0 ]; then
    echo ""
    echo "Tip: Check logs in $LOG_DIR for errors"
fi
echo "=================================================="

exit $exit_code
