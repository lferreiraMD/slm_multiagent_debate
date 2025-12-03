#!/usr/bin/env bash
#SBATCH --array=1-4
#SBATCH --partition=bch-gpu-pe
#SBATCH --job-name=persona_test
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# ============================================================
# SLURM Job Array - Quick Test (4 Jobs: 1 per Task)
# ============================================================
# Tests largest model across all 4 tasks (math, gsm, biography, mmlu)
# Usage: sbatch submit_test_4jobs.sbatch
#
# Array mapping:
#   Task 1: Math
#   Task 2: GSM
#   Task 3: Biography
#   Task 4: MMLU
# ============================================================

set -e

# ============================================================
# ADJUSTABLE SETTINGS
# ============================================================
# Model to test (change this to test different models)
MODEL="vllm-qwen3-14b"

# Experiment parameters
NUM_AGENTS=3
NUM_ROUNDS=3
NUM_PROBLEMS=20  # math, gsm, mmlu
NUM_PEOPLE=20    # biography

# Personas for 3-agent experiments
PERSONA_1='an enigma machine operator whose primary filter is signal-to-noise ratio and encrypted hidden messages'
PERSONA_2='a Zen master who communicates only through non-sequiturs, koans, and minimal, cryptic statements'
PERSONA_3='a deep-sea volcanologist focused on extremes of pressure, heat, and slow geologic change'
# ============================================================

# HPC-DEPENDENT SETTINGS
export HF_HOME=/temp_work/ch269957 # Sets location of HF models

# Get array task ID (1-4)
TASK_ID=$SLURM_ARRAY_TASK_ID

# vLLM flag
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Setup paths using SLURM_SUBMIT_DIR (directory where sbatch was called)
SLURM_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="$(cd "$SLURM_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SLURM_DIR/logs"

# Create log and results directories
mkdir -p "$LOG_DIR"/{math,gsm,biography,mmlu}
mkdir -p "$RESULTS_DIR"/{math,gsm,biography,mmlu}

# Initialize conda by sourcing system bashrc (defines conda shell function)
source /etc/bashrc

# Activate conda environment
conda activate slm

echo "============================================================"
echo "SLURM Test Array - Task: $TASK_ID"
echo "Model: $MODEL"
echo "Agents: $NUM_AGENTS, Rounds: $NUM_ROUNDS"
echo "============================================================"

# Run the appropriate task
case $TASK_ID in
    1)
        echo "[Task 1] MATH - $MODEL, $NUM_AGENTS agents"
        cd "$PROJECT_ROOT/tasks/math"
        srun -p bch-gpu-pe -c 1 --mem 16G --gpus 1 \
            python3 gen_math_clean.py \
                --model "$MODEL" \
                --agents $NUM_AGENTS \
                --rounds $NUM_ROUNDS \
                --num-problems $NUM_PROBLEMS \
                --agent-personas \
                    "$PERSONA_1" \
                    "$PERSONA_2" \
                    "$PERSONA_3" \
                --output-directory "$RESULTS_DIR/math" \
            > "$LOG_DIR/math/test_task1.out" 2>&1
        EXIT_CODE=$?
        ;;

    2)
        echo "[Task 2] GSM - $MODEL, $NUM_AGENTS agents"
        cd "$PROJECT_ROOT/tasks/gsm"
        srun -p bch-gpu-pe -c 1 --mem 16G --gpus 1 \
            python3 gen_gsm.py \
                --model "$MODEL" \
                --agents $NUM_AGENTS \
                --rounds $NUM_ROUNDS \
                --num-problems $NUM_PROBLEMS \
                --agent-personas \
                    "$PERSONA_1" \
                    "$PERSONA_2" \
                    "$PERSONA_3" \
                --output-directory "$RESULTS_DIR/gsm" \
            > "$LOG_DIR/gsm/test_task2.out" 2>&1
        EXIT_CODE=$?
        ;;

    3)
        echo "[Task 3] BIOGRAPHY - $MODEL, $NUM_AGENTS agents"
        cd "$PROJECT_ROOT/tasks/biography"
        srun -p bch-gpu-pe -c 1 --mem 16G --gpus 1 \
            python3 gen_conversation.py \
                --model "$MODEL" \
                --agents $NUM_AGENTS \
                --rounds $NUM_ROUNDS \
                --num-people $NUM_PEOPLE \
                --agent-personas \
                    "$PERSONA_1" \
                    "$PERSONA_2" \
                    "$PERSONA_3" \
                --output-directory "$RESULTS_DIR/biography" \
            > "$LOG_DIR/biography/test_task3.out" 2>&1
        EXIT_CODE=$?
        ;;

    4)
        echo "[Task 4] MMLU - $MODEL, $NUM_AGENTS agents"
        cd "$PROJECT_ROOT/tasks/mmlu"
        srun -p bch-gpu-pe -c 1 --mem 16G --gpus 1 \
            python3 gen_mmlu.py \
                --model "$MODEL" \
                --agents $NUM_AGENTS \
                --rounds $NUM_ROUNDS \
                --num-questions $NUM_PROBLEMS \
                --agent-personas \
                    "$PERSONA_1" \
                    "$PERSONA_2" \
                    "$PERSONA_3" \
                --output-directory "$RESULTS_DIR/mmlu" \
            > "$LOG_DIR/mmlu/test_task4.out" 2>&1
        EXIT_CODE=$?
        ;;

    *)
        echo "ERROR: Invalid task ID: $TASK_ID"
        exit 1
        ;;
esac

# CUDA cleanup
python3 "$PROJECT_ROOT/utils/test_cuda_cleanup.py" 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Test task $TASK_ID completed"
else
    echo "✗ FAILED: Test task $TASK_ID with exit code $EXIT_CODE"
fi

echo "============================================================"
exit $EXIT_CODE
