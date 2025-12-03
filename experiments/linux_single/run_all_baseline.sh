#!/bin/bash

# Master control script for Linux SINGLE GPU baseline experiments
# Runs all 96 experiments sequentially (24 per task × 4 tasks)
# Optimized for Ubuntu with vLLM on single NVIDIA RTX 3090 (24GB VRAM)

set -e

# Force CUDA to use only GPU #1 (RTX 3090)
# GPU 0 = GTX 1650 (4GB) - internal, insufficient VRAM
# GPU 1 = RTX 3090 (24GB) - external, target GPU
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Linux Single GPU Baseline - All Tasks"
echo "=================================================="
echo "Hardware: Single RTX 3090 (24GB VRAM)"
echo "Total: 96 experiments (24 per task × 4 tasks)"
echo "Tasks: math, gsm, biography, mmlu"
echo "Models: 6 (0.6B to 14B, filtered by VRAM)"
echo "Agent counts: 1, 3, 5, 7 (single-agent baseline + multiagent)"
echo "=================================================="
echo ""

echo "Configuration:"
echo "  Execution: Sequential (one task at a time)"
echo "  GPU assignment: Single RTX 3090 (auto-detected by vLLM)"
echo ""

# Detect GPU and warn if not single RTX 3090
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

    echo "Detected hardware:"
    echo "  GPUs: $GPU_COUNT"
    echo "  Model: $GPU_NAME"
    echo "  VRAM: ${GPU_VRAM}MB"
    echo ""

    if [ "$GPU_COUNT" -ne 1 ]; then
        echo "⚠ WARNING: Expected 1 GPU, found $GPU_COUNT"
        echo "   This config is optimized for SINGLE RTX 3090"
        echo "   For multi-GPU, use experiments/linux/ instead"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi

    if [ "$GPU_VRAM" -lt 20000 ]; then
        echo "⚠ WARNING: Expected ~24GB VRAM, found ${GPU_VRAM}MB"
        echo "   Larger models (7B-14B) may fail due to OOM"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

read -p "Start all experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Track start time
START_TIME=$(date +%s)

echo ""
echo "=================================================="
echo "Starting Experiments"
echo "=================================================="

# Run each task sequentially
TASKS=("math" "gsm" "biography" "mmlu")
FAILED_TASKS=()

for task in "${TASKS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running $task experiments..."
    echo "=================================================="

    if bash "$SCRIPT_DIR/run_baseline_${task}.sh"; then
        echo "✓ $task experiments completed successfully"
    else
        echo "✗ $task experiments failed"
        FAILED_TASKS+=("$task")
    fi
done

# Track end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================================="
echo "ALL BASELINE EXPERIMENTS COMPLETE"
echo "=================================================="
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Summary
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "Status: ✓ All tasks completed successfully"
else
    echo "Status: ✗ Some tasks failed"
    echo "Failed tasks: ${FAILED_TASKS[*]}"
fi

echo ""
echo "Logs: $SCRIPT_DIR/logs/"
echo ""
echo "Next steps:"
echo "  1. Check logs for any errors:"
echo "     tail $SCRIPT_DIR/logs/*/job_*.out"
echo ""
echo "  2. Aggregate results:"
echo "     cd ../.."
echo "     python3 scripts/aggregate_results.py"
echo ""
echo "  3. Run persona diversity experiments:"
echo "     bash $SCRIPT_DIR/run_all_experiments.sh"
echo "=================================================="

if [ ${#FAILED_TASKS[@]} -ne 0 ]; then
    exit 1
fi
