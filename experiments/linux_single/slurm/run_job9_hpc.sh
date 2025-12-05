#!/usr/bin/env bash

# Interactive run for job 9: GSM task, Llama-3.2-3B, 7 agents
# Run from: experiments/linux_single/slurm/

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX_SINGLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$LINUX_SINGLE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_hpc"

mkdir -p "$RESULTS_DIR/gsm"

# Environment
export HF_HOME=/temp_work/ch269957
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
source /etc/bashrc
conda activate slm

# Job 9 parameters
MODEL="vllm-llama32-3b"
N_AGENTS=7
ROUNDS=3
NUM_PROBLEMS=20

echo "=============================================="
echo "Job 9: GSM"
echo "Model: $MODEL"
echo "Agents: $N_AGENTS"
echo "Rounds: $ROUNDS"
echo "Problems: $NUM_PROBLEMS"
echo "=============================================="
echo ""

# Run GSM task
cd "$PROJECT_ROOT/tasks/gsm"

python3 gen_gsm.py \
    --model "$MODEL" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-problems "$NUM_PROBLEMS" \
    --agent-personas \
        'a radical anarchist who views all imposed structures and hierarchies as fundamentally flawed' \
        'a Kantian deontologist who judges all actions strictly by their moral imperative and universal rule application' \
        'a Soviet-era bureaucrat who prioritizes documentation, adherence to arbitrary quotas, and triplicate forms' \
        'a Zen master who communicates only through non-sequiturs, koans, and minimal, cryptic statements' \
        'a deep-sea volcanologist focused on extremes of pressure, heat, and slow geologic change' \
        'a cosmic horror narrator who frames the problem as an ancient, unknowable, and terrifying truth' \
        'a Renaissance painter who values perspective, light, shadow, and visual harmony in the final presentation' \
    --output-directory "$RESULTS_DIR/gsm"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================="
    echo "✓ Job 9 completed successfully"
    echo "Results: $RESULTS_DIR/gsm/"
    echo "=============================================="
else
    echo "✗ Job 9 failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

