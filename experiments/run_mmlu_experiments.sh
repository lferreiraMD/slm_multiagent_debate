#!/bin/bash

# MMLU Task Experiments
# Tests all vLLM models with varying agents and rounds
# Results saved to: results/baseline/mmlu/

set -e  # Exit on error

# Configuration
AGENTS=(1 3 5 7)
ROUNDS=(2 3 4 5 6)
MODELS=(
    "vllm-qwen3-0.6b"
    "vllm-vibethinker"
    "vllm-deepseek"
    "vllm-qwen3-1.7b"
    "vllm-llama32-3b"
    "vllm-smallthinker"
    "vllm-qwen3-4b"
    "vllm-llama31-8b"
    "vllm-qwen3-8b"
    "vllm-qwen3-14b"
    "vllm-oss-20b"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TASK_SCRIPT="$PROJECT_ROOT/tasks/mmlu/gen_mmlu.py"
OUTPUT_DIR="$PROJECT_ROOT/results/baseline/mmlu"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Counter for progress tracking
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#AGENTS[@]} * ${#ROUNDS[@]}))
CURRENT=0

echo "=========================================="
echo "MMLU Task Baseline Experiments"
echo "=========================================="
echo "Models: ${#MODELS[@]}"
echo "Agent counts: ${AGENTS[@]}"
echo "Round counts: ${ROUNDS[@]}"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Main experiment loop
for model in "${MODELS[@]}"; do
    for agents in "${AGENTS[@]}"; do
        for rounds in "${ROUNDS[@]}"; do
            CURRENT=$((CURRENT + 1))

            echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: model=$model agents=$agents rounds=$rounds"

            # Run experiment and save to baseline directory
            cd "$PROJECT_ROOT/tasks/mmlu"
            python3 gen_mmlu.py \
                --model "$model" \
                --agents "$agents" \
                --rounds "$rounds" \
                --num-questions 100 \
                --random-seed 0

            # Move result file to baseline directory
            # Find the most recently created file matching the pattern
            RESULT_FILE=$(ls -t mmlu_*_agents${agents}_rounds${rounds}.json 2>/dev/null | head -1)
            if [ -n "$RESULT_FILE" ]; then
                mv "$RESULT_FILE" "$OUTPUT_DIR/"
                echo "  ✓ Saved to: $OUTPUT_DIR/$RESULT_FILE"
            else
                echo "  ⚠ Warning: Result file not found"
            fi

            echo ""
        done
    done
done

echo "=========================================="
echo "MMLU experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
