#!/bin/bash

#SBATCH --array=1-34
#SBATCH --partition=bch-gpu-pe
#SBATCH --job-name=missing
#SBATCH --output=logs/missing_%A_%a.out
#SBATCH --error=logs/missing_%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Submit Missing Persona Experiments
#
# This script runs only the persona experiments that are missing from results_hpc/.
#
# Usage:
#   1. Generate the missing job config:
#      python3 generate_missing_persona_config.py
#
#   2. Update --array parameter above to match number of missing jobs
#      (shown in generate_missing_persona_config.py output)
#
#   3. Submit the job array:
#      sbatch submit_missing_personas.sbatch

set -e

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_HOME=/temp_work/ch269957

# GPU setup (single GPU)
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source /etc/bashrc
conda activate slm

# Read job config
CONFIG_FILE="$SCRIPT_DIR/persona_jobs_missing.txt"
JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

if [ -z "$JOB_LINE" ]; then
    echo "Error: No job found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse CSV line using Python
eval "$(python3 -c "
import csv
import sys
import ast

try:
    row = next(csv.reader(['$JOB_LINE']))
    job_id, model_alias, n_agents, rounds, task, num_param, num_value, random_seed, personas_tuple = row

    # Parse personas tuple: ('kantian', 'deep-sea', 'renaissance') → kantian deep-sea renaissance
    personas_tuple_parsed = ast.literal_eval(personas_tuple)
    personas_space_sep = ' '.join(personas_tuple_parsed)

    print(f'JOB_ID={job_id}')
    print(f'MODEL_ALIAS={model_alias}')
    print(f'N_AGENTS={n_agents}')
    print(f'ROUNDS={rounds}')
    print(f'TASK={task}')
    print(f'PERSONAS=\"{personas_space_sep}\"')
    print(f'RANDOM_SEED={random_seed}')
except Exception as e:
    print(f'echo \"ERROR: CSV parsing failed: {e}\" >&2', file=sys.stderr)
    sys.exit(1)
")"

echo "=============================================="
echo "Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $JOB_ID"
echo "Model: $MODEL_ALIAS"
echo "Agents: $N_AGENTS"
echo "Rounds: $ROUNDS"
echo "Task: $TASK"
echo "Personas: $PERSONAS"
echo "=============================================="

# Results directory
RESULTS_DIR="$SCRIPT_DIR/results_hpc"

# Run task-specific generation script
cd "$PROJECT_ROOT/tasks/$TASK"

case "$TASK" in
    math)
        python3 gen_math.py \
            --model "$MODEL_ALIAS" \
            --agents "$N_AGENTS" \
            --rounds "$ROUNDS" \
            --num-problems 100 \
            --agent-personas $PERSONAS \
            --output-directory "$RESULTS_DIR/$TASK" \
            --seed "$RANDOM_SEED"
        ;;

    gsm)
        python3 gen_gsm.py \
            --model "$MODEL_ALIAS" \
            --agents "$N_AGENTS" \
            --rounds "$ROUNDS" \
            --num-problems 100 \
            --agent-personas $PERSONAS \
            --output-directory "$RESULTS_DIR/$TASK" \
            --seed "$RANDOM_SEED"
        ;;

    biography)
        python3 gen_conversation.py \
            --model "$MODEL_ALIAS" \
            --agents "$N_AGENTS" \
            --rounds "$ROUNDS" \
            --num-people 20 \
            --agent-personas $PERSONAS \
            --output-directory "$RESULTS_DIR/$TASK" \
            --seed "$RANDOM_SEED"
        ;;

    mmlu)
        python3 gen_mmlu.py \
            --model "$MODEL_ALIAS" \
            --agents "$N_AGENTS" \
            --rounds "$ROUNDS" \
            --num-questions 100 \
            --agent-personas $PERSONAS \
            --output-directory "$RESULTS_DIR/$TASK" \
            --seed "$RANDOM_SEED"
        ;;

    *)
        echo "Error: Unknown task '$TASK'"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Job completed successfully"
else
    echo "✗ Job failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

