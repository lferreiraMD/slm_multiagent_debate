#!/bin/bash

# Submit all persona diversity experiments to SLURM
# Total: 240 experiments (60 per task × 4 tasks)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Submitting Persona Diversity Experiments"
echo "=================================================="
echo "Total experiments: 240 (60 per task)"
echo "Tasks: math, gsm, biography, mmlu"
echo "Models: 10 (0.6B to 14B)"
echo "Agent counts: 2-7 (6 configurations per model)"
echo "Persona selection: MaxDet v2 (extreme personas)"
echo "=================================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Submit math experiments (60 jobs)
echo "Submitting math experiments..."
MATH_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_persona_math.slurm")
echo "  Math job array submitted: $MATH_JOB_ID (60 tasks)"

# Submit GSM experiments (60 jobs)
echo "Submitting GSM experiments..."
GSM_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_persona_gsm.slurm")
echo "  GSM job array submitted: $GSM_JOB_ID (60 tasks)"

# Submit biography experiments (60 jobs)
echo "Submitting biography experiments..."
BIOGRAPHY_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_persona_biography.slurm")
echo "  Biography job array submitted: $BIOGRAPHY_JOB_ID (60 tasks)"

# Submit MMLU experiments (60 jobs)
echo "Submitting MMLU experiments..."
MMLU_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_persona_mmlu.slurm")
echo "  MMLU job array submitted: $MMLU_JOB_ID (60 tasks)"

echo ""
echo "=================================================="
echo "All experiments submitted successfully!"
echo "=================================================="
echo "Job IDs:"
echo "  Math:      $MATH_JOB_ID"
echo "  GSM:       $GSM_JOB_ID"
echo "  Biography: $BIOGRAPHY_JOB_ID"
echo "  MMLU:      $MMLU_JOB_ID"
echo ""
echo "Monitoring commands:"
echo "  squeue -u \$USER                    # Check job status"
echo "  squeue -j $MATH_JOB_ID             # Check math jobs"
echo "  squeue -j $GSM_JOB_ID              # Check GSM jobs"
echo "  squeue -j $BIOGRAPHY_JOB_ID        # Check biography jobs"
echo "  squeue -j $MMLU_JOB_ID             # Check MMLU jobs"
echo ""
echo "  bash $SCRIPT_DIR/monitor_persona_jobs.sh  # Continuous monitoring"
echo ""
echo "Log files:"
echo "  $SCRIPT_DIR/logs/persona_*_\${JOB_ID}_\${TASK_ID}.{out,err}"
echo "=================================================="

# Save job IDs for later reference
echo "$MATH_JOB_ID" > "$SCRIPT_DIR/logs/last_math_job_id.txt"
echo "$GSM_JOB_ID" > "$SCRIPT_DIR/logs/last_gsm_job_id.txt"
echo "$BIOGRAPHY_JOB_ID" > "$SCRIPT_DIR/logs/last_biography_job_id.txt"
echo "$MMLU_JOB_ID" > "$SCRIPT_DIR/logs/last_mmlu_job_id.txt"

echo "Job IDs saved to $SCRIPT_DIR/logs/last_*_job_id.txt"
echo ""
