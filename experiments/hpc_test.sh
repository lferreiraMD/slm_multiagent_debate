#!/bin/bash

# HPC Test Script
# Comprehensive sanity check to verify the repository works on HPC/Linux with vLLM
# Tests all uncommented vLLM models from config.yaml
# Runs all 4 tasks with minimal configuration (3 agents, 2 rounds, 2 problems)

set -e

# Test configuration
AGENTS=2
ROUNDS=3
NUM_PROBLEMS=1  # Small number for quick testing

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# All uncommented vLLM models from config.yaml (lines 67-80)
MODELS=(
    "vllm-qwen3-0.6b"
    "vllm-vibethinker"
    "vllm-llama32-3b"
    "vllm-mistral-7b"
    "vllm-qwen3-14b"
)

echo "=========================================="
echo "HPC COMPREHENSIVE REPOSITORY TEST"
echo "=========================================="
echo "Testing ${#MODELS[@]} vLLM models across 4 tasks"
echo "Models: ${MODELS[@]}"
echo "Agents: $AGENTS"
echo "Rounds: $ROUNDS"
echo "Problems per task: $NUM_PROBLEMS"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="
echo ""
echo "This will verify:"
echo "  ✓ vLLM backend is working"
echo "  ✓ All models can be loaded"
echo "  ✓ All generation scripts run successfully"
echo "  ✓ Results are saved correctly"
echo ""
echo "=========================================="
echo ""

TOTAL_TESTS=$((${#MODELS[@]} * 4))
CURRENT_TEST=0
FAILED_TESTS=()

# Test each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║  Testing Model: $MODEL"
    echo "╚════════════════════════════════════════╝"
    echo ""

    # Test 1: Math Task
    ((CURRENT_TEST++))
    echo "[$CURRENT_TEST/$TOTAL_TESTS] Math Task - $MODEL"
    echo "----------------------------------------"
    cd "$PROJECT_ROOT/tasks/math"

    if python3 gen_math_clean.py \
        --model "$MODEL" \
        --agents "$AGENTS" \
        --rounds "$ROUNDS" \
        --num-problems "$NUM_PROBLEMS" 2>&1 | tee /tmp/hpc_test_math.log; then

        # Math saves as .p (pickle) not .json
        MATH_RESULT=$(ls -t math_*_agents${AGENTS}_rounds${ROUNDS}.p 2>/dev/null | head -1)
        if [ -n "$MATH_RESULT" ]; then
            echo "✓ Math task completed: $MATH_RESULT"
            rm "$MATH_RESULT"  # Clean up test file
        else
            echo "✗ Math task failed: No output file found (.p expected)"
            FAILED_TESTS+=("Math - $MODEL (no output file)")
        fi
    else
        echo "✗ Math task failed: Script error"
        FAILED_TESTS+=("Math - $MODEL (script error)")
    fi
    echo ""

    # Test 2: GSM Task
    ((CURRENT_TEST++))
    echo "[$CURRENT_TEST/$TOTAL_TESTS] GSM Task - $MODEL"
    echo "----------------------------------------"
    cd "$PROJECT_ROOT/tasks/gsm"

    if python3 gen_gsm.py \
        --model "$MODEL" \
        --agents "$AGENTS" \
        --rounds "$ROUNDS" \
        --num-problems "$NUM_PROBLEMS" 2>&1 | tee /tmp/hpc_test_gsm.log; then

        GSM_RESULT=$(ls -t gsm_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
        if [ -n "$GSM_RESULT" ]; then
            echo "✓ GSM task completed: $GSM_RESULT"
            rm "$GSM_RESULT"  # Clean up test file
        else
            echo "✗ GSM task failed: No output file found (.json expected)"
            FAILED_TESTS+=("GSM - $MODEL (no output file)")
        fi
    else
        echo "✗ GSM task failed: Script error"
        FAILED_TESTS+=("GSM - $MODEL (script error)")
    fi
    echo ""

    # Test 3: Biography Task
    ((CURRENT_TEST++))
    echo "[$CURRENT_TEST/$TOTAL_TESTS] Biography Task - $MODEL"
    echo "----------------------------------------"
    cd "$PROJECT_ROOT/tasks/biography"

    if python3 gen_conversation.py \
        --model "$MODEL" \
        --agents "$AGENTS" \
        --rounds "$ROUNDS" \
        --num-people "$NUM_PROBLEMS" 2>&1 | tee /tmp/hpc_test_bio.log; then

        BIOGRAPHY_RESULT=$(ls -t biography_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
        if [ -n "$BIOGRAPHY_RESULT" ]; then
            echo "✓ Biography task completed: $BIOGRAPHY_RESULT"
            rm "$BIOGRAPHY_RESULT"  # Clean up test file
        else
            echo "✗ Biography task failed: No output file found (.json expected)"
            FAILED_TESTS+=("Biography - $MODEL (no output file)")
        fi
    else
        echo "✗ Biography task failed: Script error"
        FAILED_TESTS+=("Biography - $MODEL (script error)")
    fi
    echo ""

    # Test 4: MMLU Task
    ((CURRENT_TEST++))
    echo "[$CURRENT_TEST/$TOTAL_TESTS] MMLU Task - $MODEL"
    echo "----------------------------------------"
    cd "$PROJECT_ROOT/tasks/mmlu"

    if python3 gen_mmlu.py \
        --model "$MODEL" \
        --agents "$AGENTS" \
        --rounds "$ROUNDS" \
        --num-questions "$NUM_PROBLEMS" 2>&1 | tee /tmp/hpc_test_mmlu.log; then

        MMLU_RESULT=$(ls -t mmlu_*_agents${AGENTS}_rounds${ROUNDS}.json 2>/dev/null | head -1)
        if [ -n "$MMLU_RESULT" ]; then
            echo "✓ MMLU task completed: $MMLU_RESULT"
            rm "$MMLU_RESULT"  # Clean up test file
        else
            echo "✗ MMLU task failed: No output file found (.json expected)"
            FAILED_TESTS+=("MMLU - $MODEL (no output file)")
        fi
    else
        echo "✗ MMLU task failed: Script error"
        FAILED_TESTS+=("MMLU - $MODEL (script error)")
    fi
    echo ""
done

# Summary
echo "=========================================="
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✓ ALL TESTS PASSED! ($TOTAL_TESTS/$TOTAL_TESTS)"
    echo "=========================================="
    echo ""
    echo "HPC setup is working correctly. You can now:"
    echo "  1. Run baseline experiments: cd experiments/linux_single && bash run_all_baseline.sh"
    echo "  2. Run persona experiments: cd experiments/linux_single/slurm && bash launch.sh"
    echo "  3. Aggregate results: cd experiments/linux_single && python3 aggregate_baseline_results.py"
    echo ""
    echo "All test files were automatically cleaned up."
    echo "=========================================="
    exit 0
else
    echo "✗ SOME TESTS FAILED (${#FAILED_TESTS[@]}/$TOTAL_TESTS failed)"
    echo "=========================================="
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "Check logs in /tmp/hpc_test_*.log for details"
    echo "=========================================="
    exit 1
fi