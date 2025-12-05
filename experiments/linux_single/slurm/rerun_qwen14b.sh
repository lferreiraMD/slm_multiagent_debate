#!/usr/bin/env bash

# Re-run Qwen3-14B 3 agents across all 4 tasks (incomplete experiments)
# Run from: experiments/linux_single/slurm/

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX_SINGLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$LINUX_SINGLE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_hpc"

# Environment
export HF_HOME=/temp_work/ch269957
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
source /etc/bashrc
conda activate slm

# Common parameters
MODEL="vllm-qwen3-14b"
N_AGENTS=3
ROUNDS=3
PERSONAS=(
    'an enigma machine operator whose primary filter is signal-to-noise ratio and encrypted hidden messages'
    'a Zen master who communicates only through non-sequiturs, koans, and minimal, cryptic statements'
    'a deep-sea volcanologist focused on extremes of pressure, heat, and slow geologic change'
)

echo "=============================================="
echo "Re-running Qwen3-14B 3 agents (all tasks)"
echo "Model: $MODEL"
echo "Agents: $N_AGENTS"
echo "Rounds: $ROUNDS"
echo "=============================================="
echo ""

# Task 1: Math
echo "--- Task 1/4: Math ---"
cd "$PROJECT_ROOT/tasks/math"
mkdir -p "$RESULTS_DIR/math"

#python3 gen_math_clean.py \
#    --model "$MODEL" \
#    --agents "$N_AGENTS" \
#    --rounds "$ROUNDS" \
#    --num-problems 20 \
#    --agent-personas "${PERSONAS[@]}" \
#    --output-directory "$RESULTS_DIR/math"

#MATH_EXIT=$?
#echo "Math task: $([ $MATH_EXIT -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
#echo ""

# Task 2: GSM
echo "--- Task 2/4: GSM ---"
cd "$PROJECT_ROOT/tasks/gsm"
mkdir -p "$RESULTS_DIR/gsm"

#python3 gen_gsm.py \
#    --model "$MODEL" \
#    --agents "$N_AGENTS" \
#    --rounds "$ROUNDS" \
#    --num-problems 20 \
#    --agent-personas "${PERSONAS[@]}" \
#    --output-directory "$RESULTS_DIR/gsm"

#GSM_EXIT=$?
#echo "GSM task: $([ $GSM_EXIT -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
#echo ""

# Task 3: Biography
echo "--- Task 3/4: Biography ---"
cd "$PROJECT_ROOT/tasks/biography"
mkdir -p "$RESULTS_DIR/biography"

python3 gen_conversation.py \
    --model "$MODEL" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-people 20 \
    --agent-personas "${PERSONAS[@]}" \
    --output-directory "$RESULTS_DIR/biography"

BIO_EXIT=$?
echo "Biography task: $([ $BIO_EXIT -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo ""

# Task 4: MMLU
echo "--- Task 4/4: MMLU ---"
cd "$PROJECT_ROOT/tasks/mmlu"
mkdir -p "$RESULTS_DIR/mmlu"

python3 gen_mmlu.py \
    --model "$MODEL" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-questions 20 \
    --agent-personas "${PERSONAS[@]}" \
    --output-directory "$RESULTS_DIR/mmlu"

MMLU_EXIT=$?
echo "MMLU task: $([ $MMLU_EXIT -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo ""

# Summary
echo "=============================================="
echo "Re-run Summary"
echo "=============================================="
echo "Math:      $([ $MATH_EXIT -eq 0 ] && echo '✓' || echo '✗')"
echo "GSM:       $([ $GSM_EXIT -eq 0 ] && echo '✓' || echo '✗')"
echo "Biography: $([ $BIO_EXIT -eq 0 ] && echo '✓' || echo '✗')"
echo "MMLU:      $([ $MMLU_EXIT -eq 0 ] && echo '✓' || echo '✗')"
echo "=============================================="

# Exit with failure if any task failed
if [ $MATH_EXIT -ne 0 ] || [ $GSM_EXIT -ne 0 ] || [ $BIO_EXIT -ne 0 ] || [ $MMLU_EXIT -ne 0 ]; then
    exit 1
fi

exit 0

