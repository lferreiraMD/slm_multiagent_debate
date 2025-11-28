#!/bin/bash

# Master control script for Linux persona diversity experiments
# Runs all 240 experiments sequentially (60 per task × 4 tasks)
# Optimized for Ubuntu with vLLM on NVIDIA GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Linux Persona Diversity Experiments - All Tasks"
echo "=================================================="
echo "Total: 240 experiments (60 per task × 4 tasks)"
echo "Tasks: math, gsm, biography, mmlu"
echo "Models: 10 (0.6B to 14B)"
echo "Agent counts: 2-7 (6 configurations per model)"
echo "Persona selection: MaxDet v2 (extreme personas)"
echo "=================================================="
echo ""

# Configuration
export MAX_PARALLEL=${MAX_PARALLEL:-2}  # Override with env var if needed

echo "Configuration:"
echo "  Max parallel jobs: $MAX_PARALLEL"
echo "  GPU assignment: Automatic (vLLM will detect)"
echo ""

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

    if bash "$SCRIPT_DIR/run_persona_${task}.sh"; then
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
echo "ALL EXPERIMENTS COMPLETE"
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
echo "     python3 scripts/aggregate_persona_results.py"
echo ""
echo "  3. Resubmit failed jobs if needed:"
echo "     bash $SCRIPT_DIR/resubmit_failed.sh"
echo "=================================================="

if [ ${#FAILED_TASKS[@]} -ne 0 ]; then
    exit 1
fi
