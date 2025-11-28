#!/bin/bash
#SBATCH --job-name=persona_gsm
#SBATCH --output=logs/persona_gsm_%A_%a.out
#SBATCH --error=logs/persona_gsm_%A_%a.err
#SBATCH --array=1-60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# SLURM array script for persona diversity GSM experiments
# Runs 60 experiments: 10 models × 6 agent counts (2-7)
# Uses optimal MaxDet v2 persona combinations from personas/summary_personas.csv

# Setup
echo "=================================================="
echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../..

# Activate environment (modify as needed for your HPC setup)
# module load python/3.10
# source venv/bin/activate

# Read job configuration
CONFIG_FILE="experiments/configs/persona_gsm_jobs.txt"
JOB_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE | tail -n 1)

# Parse CSV line (skip header)
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Skipping header line"
    exit 0
fi

# Extract fields using awk with proper CSV parsing
JOB_ID=$(echo "$JOB_LINE" | awk -F',' '{print $1}')
MODEL_ALIAS=$(echo "$JOB_LINE" | awk -F',' '{print $2}')
N_AGENTS=$(echo "$JOB_LINE" | awk -F',' '{print $3}')
ROUNDS=$(echo "$JOB_LINE" | awk -F',' '{print $4}')
TASK=$(echo "$JOB_LINE" | awk -F',' '{print $5}')
NUM_PARAM=$(echo "$JOB_LINE" | awk -F',' '{print $6}')
NUM_VALUE=$(echo "$JOB_LINE" | awk -F',' '{print $7}')
RANDOM_SEED=$(echo "$JOB_LINE" | awk -F',' '{print $8}')

# Extract personas tuple (everything after 8th comma, remove quotes)
PERSONAS_TUPLE=$(echo "$JOB_LINE" | awk -F',' '{for(i=9;i<=NF;i++) printf "%s%s", $i, (i<NF?",":"")}' | sed 's/^"//;s/"$//')

echo "Job Configuration:"
echo "  Job ID: $JOB_ID"
echo "  Model: $MODEL_ALIAS"
echo "  Agents: $N_AGENTS"
echo "  Rounds: $ROUNDS"
echo "  Num problems: $NUM_VALUE"
echo "  Personas: $PERSONAS_TUPLE"
echo "=================================================="

# Convert persona tuple string to space-separated arguments
# Parse Python tuple format: ('persona1', 'persona2', ...) → persona1 persona2 ...
PERSONAS_ARGS=$(echo "$PERSONAS_TUPLE" | sed "s/[()']//g" | sed 's/, / /g')

# Run experiment
cd tasks/gsm

echo "Running experiment..."
python3 gen_gsm.py \
    --model "$MODEL_ALIAS" \
    --agents "$N_AGENTS" \
    --rounds "$ROUNDS" \
    --num-problems "$NUM_VALUE" \
    --agent-personas $PERSONAS_ARGS

EXIT_CODE=$?

echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
