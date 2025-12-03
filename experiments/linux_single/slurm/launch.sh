#!/usr/bin/env bash
# Simple launcher script for persona experiments
# Usage: bash launch.sh

cd "$(dirname "$0")"

echo "Submitting 60 persona diversity experiments..."
echo ""
echo "Array mapping:"
echo "  Tasks 1-15:  Math"
echo "  Tasks 16-30: GSM"
echo "  Tasks 31-45: Biography"
echo "  Tasks 46-60: MMLU"
echo ""

sbatch submit_all_personas.sbatch

echo ""
echo "Jobs submitted! Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor logs in real-time:"
echo "  tail -f logs/slurm_*.out"
echo ""
