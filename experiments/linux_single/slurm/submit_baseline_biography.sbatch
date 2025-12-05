#!/usr/bin/env bash

#SBATCH --array=1-20
#SBATCH --partition=bch-gpu-pe
#SBATCH --job-name=bio_base
#SBATCH --output=logs/biography_baseline/job_%A_%a.out
#SBATCH --error=logs/biography_baseline/job_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# HARDCODED Biography Baseline Experiments
# All 20 jobs defined as case statements (no CSV parsing, no personas)

set -e

###### Setup environment
# Directories (submitted from: .../slurm/)
SLURM_DIR="${SLURM_SUBMIT_DIR}"                         # .../experiments/linux_single/slurm
LINUX_SINGLE_DIR="$(cd "$SLURM_DIR/.." && pwd)"         # .../experiments/linux_single
PROJECT_ROOT="$(cd "$LINUX_SINGLE_DIR/../.." && pwd)"   # PROJECT_ROOT
RESULTS_DIR="$SLURM_DIR/results_hpc"                    # .../slurm/results_hpc
LOG_DIR="$SLURM_DIR/logs"                               # .../slurm/logs

mkdir -p "$LOG_DIR/biography_baseline"
mkdir -p "$RESULTS_DIR/biography"

# Huggingface cache
export HF_HOME=/temp_work/ch269957

# CUDA + conda settings
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
source /etc/bashrc
conda activate slm

# Hardcoded job parameters (no CSV parsing)
case $SLURM_ARRAY_TASK_ID in
    1)
        JOB_ID=1
        MODEL="vllm-qwen3-0.6b"
        N_AGENTS=1
        ;;
    2)
        JOB_ID=2
        MODEL="vllm-qwen3-0.6b"
        N_AGENTS=3
        ;;
    3)
        JOB_ID=3
        MODEL="vllm-qwen3-0.6b"
        N_AGENTS=5
        ;;
    4)
        JOB_ID=4
        MODEL="vllm-qwen3-0.6b"
        N_AGENTS=7
        ;;
    5)
        JOB_ID=5
        MODEL="vllm-vibethinker"
        N_AGENTS=1
        ;;
    6)
        JOB_ID=6
        MODEL="vllm-vibethinker"
        N_AGENTS=3
        ;;
    7)
        JOB_ID=7
        MODEL="vllm-vibethinker"
        N_AGENTS=5
        ;;
    8)
        JOB_ID=8
        MODEL="vllm-vibethinker"
        N_AGENTS=7
        ;;
    9)
        JOB_ID=9
        MODEL="vllm-llama32-3b"
        N_AGENTS=1
        ;;
    10)
        JOB_ID=10
        MODEL="vllm-llama32-3b"
        N_AGENTS=3
        ;;
    11)
        JOB_ID=11
        MODEL="vllm-llama32-3b"
        N_AGENTS=5
        ;;
    12)
        JOB_ID=12
        MODEL="vllm-llama32-3b"
        N_AGENTS=7
        ;;
    13)
        JOB_ID=13
        MODEL="vllm-mistral-7b"
        N_AGENTS=1
        ;;
    14)
        JOB_ID=14
        MODEL="vllm-mistral-7b"
        N_AGENTS=3
        ;;
    15)
        JOB_ID=15
        MODEL="vllm-mistral-7b"
        N_AGENTS=5
        ;;
    16)
        JOB_ID=16
        MODEL="vllm-mistral-7b"
        N_AGENTS=7
        ;;
    17)
        JOB_ID=17
        MODEL="vllm-qwen3-14b"
        N_AGENTS=1
        ;;
    18)
        JOB_ID=18
        MODEL="vllm-qwen3-14b"
        N_AGENTS=3
        ;;
    19)
        JOB_ID=19
        MODEL="vllm-qwen3-14b"
        N_AGENTS=5
        ;;
    20)
        JOB_ID=20
        MODEL="vllm-qwen3-14b"
        N_AGENTS=7
        ;;
    *)
        echo "Error: Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

# Common parameters for all jobs
ROUNDS=3
TASK="biography"
NUM_PEOPLE=20
RANDOM_SEED=42

echo "=============================================="
echo "Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $JOB_ID"
echo "Model: $MODEL"
echo "Agents: $N_AGENTS"
echo "Rounds: $ROUNDS"
echo "Task: $TASK"
echo "Num People: $NUM_PEOPLE"
echo "=============================================="

# Run biography task
cd "$PROJECT_ROOT/tasks/$TASK"

python3 gen_conversation.py \
    --model "$MODEL" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-people "$NUM_PEOPLE" \
    --output-directory "$RESULTS_DIR/$TASK"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Job $JOB_ID completed successfully"
else
    echo "✗ Job $JOB_ID failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

