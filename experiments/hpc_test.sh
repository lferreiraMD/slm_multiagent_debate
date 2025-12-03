#!/bin/bash

# HPC Test Script
# Quick sanity check to verify the repository works on HPC/Linux with vLLM
# Runs all 4 tasks with minimal configuration (2 agents, 2 rounds)
# Uses vllm-vibethinker (1.5B) for fast testing

set -e  # Exit on first error

# Test configuration
MODEL="Qwen/Qwen3-0.6B"
AGENTS=3
ROUNDS=2
NUM_PROBLEMS=2  # Small number for quick testing

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "HPC REPOSITORY TEST"
echo "=========================================="
echo "Model: $MODEL"
echo "Agents: $AGENTS"
echo "Rounds: $ROUNDS"
echo "Problems per task: $NUM_PROBLEMS"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="
echo ""
echo "This will test all 4 tasks to verify:"
echo "  ✓ vLLM backend is working"
echo "  ✓ Models can be loaded"
echo "  ✓ Generation scripts run successfully"
echo "  ✓ Results are saved correctly"
echo ""
echo "=========================================="
echo ""

# Test 1: Math Task
echo "[1/4] Testing Math Task..."
echo "----------------------------------------"
cd "$PROJECT_ROOT/tasks/math"
python3 gen_math_clean.py \
    --model "$MODEL" \
    --agents "$AGENTS" \
    --rounds "$ROUNDS" \
    --num-problems "$NUM_PROBLEMS" \
    
MATH_RESULT=$(ls -t math_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
if [ -n "$MATH_RESULT" ]; then
    echo "✓ Math task completed: $MATH_RESULT"
    rm "$MATH_RESULT"  # Clean up test file
else
    echo "✗ Math task failed: No output file found"
    exit 1
fi
echo ""

# Test 2: GSM Task
echo "[2/4] Testing GSM Task..."
echo "----------------------------------------"
cd "$PROJECT_ROOT/tasks/gsm"
python3 gen_gsm.py \
    --model "$MODEL" \
    --agents "$AGENTS" \
    --rounds "$ROUNDS" \
    --num-problems "$NUM_PROBLEMS" \

GSM_RESULT=$(ls -t gsm_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
if [ -n "$GSM_RESULT" ]; then
    echo "✓ GSM task completed: $GSM_RESULT"
    rm "$GSM_RESULT"  # Clean up test file
else
    echo "✗ GSM task failed: No output file found"
    exit 1
fi
echo ""

# Test 3: Biography Task
echo "[3/4] Testing Biography Task..."
echo "----------------------------------------"
cd "$PROJECT_ROOT/tasks/biography"
python3 gen_conversation.py \
    --model "$MODEL" \
    --agents "$AGENTS" \
    --rounds "$ROUNDS" \
    --num-people "$NUM_PROBLEMS" \

BIOGRAPHY_RESULT=$(ls -t biography_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
if [ -n "$BIOGRAPHY_RESULT" ]; then
    echo "✓ Biography task completed: $BIOGRAPHY_RESULT"
    rm "$BIOGRAPHY_RESULT"  # Clean up test file
else
    echo "✗ Biography task failed: No output file found"
    exit 1
fi
echo ""

# Test 4: MMLU Task
echo "[4/4] Testing MMLU Task..."
echo "----------------------------------------"
cd "$PROJECT_ROOT/tasks/mmlu"
python3 gen_mmlu.py \
    --model "$MODEL" \
    --agents "$AGENTS" \
    --rounds "$ROUNDS" \
    --num-questions "$NUM_PROBLEMS" \

MMLU_RESULT=$(ls -t mmlu_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
if [ -n "$MMLU_RESULT" ]; then
    echo "✓ MMLU task completed: $MMLU_RESULT"
    rm "$MMLU_RESULT"  # Clean up test file
else
    echo "✗ MMLU task failed: No output file found"
    exit 1
fi
echo ""

# All tests passed
echo "=========================================="
echo "✓ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "HPC setup is working correctly. You can now:"
echo "  1. Run full experiments with ./experiments/run_*_experiments.sh"
echo "  2. Optionally pre-cache models with python3 experiments/download_models.py"
echo ""
echo "Test files were automatically cleaned up."
echo "=========================================="