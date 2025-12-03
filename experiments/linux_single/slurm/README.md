# SLURM Persona Experiments

Submit all 60 persona diversity experiments as independent SLURM jobs with a single command.

## Quick Start

**Single command to submit all 60 jobs:**

```bash
cd experiments/linux_single/slurm
bash launch.sh
```

**What this does:**
- Submits SLURM job array with 60 independent tasks
- Each task runs one persona experiment (5 models × 3 agent counts × 4 tasks)
- SLURM scheduler distributes jobs across available GPUs automatically
- All jobs run in parallel (as GPU resources allow)

**Alternative (direct sbatch):**
```bash
sbatch submit_all_personas.sbatch
```

## What Gets Submitted

- **60 independent SLURM jobs** (via job array 1-60)
- **Task mapping:**
  - Tasks 1-15: Math (5 models × 3 agent counts)
  - Tasks 16-30: GSM (5 models × 3 agent counts)
  - Tasks 31-45: Biography (5 models × 3 agent counts)
  - Tasks 46-60: MMLU (5 models × 3 agent counts)
- **Models tested:** qwen3-0.6b, vibethinker, llama32-3b, mistral-7b, qwen3-14b
- **Agent counts:** 3, 5, 7 agents per experiment
- **Each job:**
  - Uses 1 GPU, 1 CPU, 16GB RAM
  - Runs on `bch-gpu-pe` partition
  - Activates pyenv environment "slm"
  - Logs to `logs/slurm_{jobid}_{taskid}.out`
  - Results saved to `../results/{task}/`

**Total experiments:** 60 (15 per task)

## Monitoring Jobs

**Check job status:**
```bash
squeue -u $USER
```

**Monitor logs in real-time:**
```bash
# All logs
tail -f logs/slurm_*.out

# Specific task
tail -f logs/slurm_*_15.out  # Task 15 (last math job)
```

**Check which jobs completed:**
```bash
grep -l "SUCCESS" logs/slurm_*.out | wc -l
```

**Find failed jobs:**
```bash
grep -l "FAILED" logs/slurm_*.out
```

## Restarting Failed Jobs

If some jobs fail, resubmit only those:

```bash
# Example: Restart tasks 5, 12, and 23
sbatch --array=5,12,23 submit_all_personas.sbatch
```

## Results

Results are saved to:
- `../results/math/`
- `../results/gsm/`
- `../results/biography/`
- `../results/mmlu/`

Same location as local runs, so analysis scripts work identically.

## Cancel All Jobs

```bash
scancel -u $USER
```

Or cancel specific job array:
```bash
scancel {job_id}
```
