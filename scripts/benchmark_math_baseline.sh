#!/bin/bash
# Baseline Math Benchmark Script
# Tests all vLLM models with single agent (no debate) on same 100 problems
#
# Usage: bash scripts/benchmark_math_baseline.sh
# Run from repository root

set -e  # Exit on error

# Configuration
AGENTS=1
ROUNDS=1
NUM_PROBLEMS=100
TASK_DIR="tasks/math"

# vLLM models from config.yaml
MODELS=(
    "vllm-deepseek"
    "vllm-vibethinker-1.5b"
    "vllm-smallthinker-3b"
    "vllm-llama32-3b"
    "vllm-qwen25-7b"
    "vllm-llama31-8b"
    "vllm-qwen25-14b"
)

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -d "$TASK_DIR" ]; then
    echo -e "${RED}Error: Must run from repository root${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create results directory
mkdir -p results/math_baseline

# Header
echo "============================================================"
echo "Math Baseline Benchmark - Single Agent (No Debate)"
echo "============================================================"
echo "Configuration:"
echo "  Agents: $AGENTS"
echo "  Rounds: $ROUNDS"
echo "  Problems: $NUM_PROBLEMS"
echo "  Models: ${#MODELS[@]}"
echo "  Random seed: 0 (from config.yaml - ensures same problems)"
echo "============================================================"
echo ""

# Run experiments
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo -e "${BLUE}[$CURRENT/$TOTAL_MODELS] Testing model: $model${NC}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run experiment
    cd "$TASK_DIR"

    if python3 gen_math.py \
        --model "$model" \
        --agents $AGENTS \
        --rounds $ROUNDS \
        --num-problems $NUM_PROBLEMS; then

        echo -e "${GREEN}✓ Completed: $model${NC}"

        # Move result to baseline directory
        OUTPUT_FILE=$(ls -t math_*_agents${AGENTS}_rounds${ROUNDS}.p 2>/dev/null | head -1)
        if [ -f "$OUTPUT_FILE" ]; then
            cp "$OUTPUT_FILE" "../../results/math_baseline/"
            echo "  Saved: results/math_baseline/$OUTPUT_FILE"
        fi
    else
        echo -e "${RED}✗ Failed: $model${NC}"
        echo "  Continuing with remaining models..."
    fi

    cd ../..
    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

# Summary
echo "============================================================"
echo "Baseline Benchmark Complete"
echo "============================================================"
echo "Results saved to: results/math_baseline/"
echo ""
echo "Result files:"
ls -lh results/math_baseline/math_*_agents${AGENTS}_rounds${ROUNDS}.p 2>/dev/null || echo "No results found"
echo ""
echo "To analyze results:"
echo "  cd results/math_baseline"
echo "  python3 ../../scripts/aggregate_results.py"
echo "============================================================"
