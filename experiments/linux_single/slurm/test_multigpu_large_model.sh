#!/usr/bin/env bash

# Test multi-GPU with large model (oss-gpt-120b)
# Goal: Verify tensor_parallel_size=2 works on HPC
# Run from: experiments/linux_single/slurm/

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX_SINGLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$LINUX_SINGLE_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/test_multigpu"

mkdir -p "$RESULTS_DIR/gsm"

# Environment
export HF_HOME=/temp_work/ch269957
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1  # Force both GPUs visible
source /etc/bashrc
conda activate slm

# Test parameters - HARDCODED
MODEL="Qwen/Qwen2.5-32B-Instruct"
# MODEL="meta-llama/Llama-3.1-70B"
# MODEL="openai/gpt-oss-120b" # ==> does not play well with vLLM
N_AGENTS=7
ROUNDS=3
NUM_PROBLEMS=3  # Small test (not full 20)

# Temperature diversity (7 agents)
TEMPERATURES=(0.5 0.6 0.7 0.8 0.9 1.0 1.1)

echo "=============================================="
echo "Multi-GPU Test - Large Model"
echo "=============================================="
echo "Model: $MODEL"
echo "Agents: $N_AGENTS (temperature diversity)"
echo "Rounds: $ROUNDS"
echo "Problems: $NUM_PROBLEMS (quick test)"
echo "Temperatures: ${TEMPERATURES[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=============================================="
echo ""
echo "This will test if tensor_parallel_size=2 works"
echo "when the model requires multiple GPUs."
echo ""

# Run GSM task
cd "$PROJECT_ROOT/tasks/gsm"

python3 gen_gsm.py \
    --model "$MODEL" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-problems "$NUM_PROBLEMS" \
    --agent-temperatures "${TEMPERATURES[@]}" \
    --output-directory "$RESULTS_DIR/gsm"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Multi-GPU test PASSED"
    echo "  Model loaded and ran successfully with TP=2"
    echo "  Results: $RESULTS_DIR/gsm/"
else
    echo "✗ Multi-GPU test FAILED (exit code $EXIT_CODE)"
    echo "  Check logs above for errors"
fi
echo "=============================================="

exit $EXIT_CODE

