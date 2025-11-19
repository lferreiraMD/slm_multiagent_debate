# Baseline Experiments

This directory contains scripts to run comprehensive baseline experiments across all vLLM models.

## Quick Start: HPC Testing

**First time on HPC?** Run the test script to verify everything works:

```bash
# Test all 4 tasks with vllm-vibethinker (1.5B, fast)
./experiments/hpc_test.sh
```

This runs a quick sanity check:
- ✓ Tests all 4 tasks (math, gsm, biography, mmlu)
- ✓ Uses minimal configuration (2 agents, 2 rounds, 10 problems)
- ✓ Verifies vLLM backend and model loading
- ✓ Auto-cleans up test files
- ✓ Takes ~5-10 minutes on dual RTX 3090

**If test passes:** You're ready to run full experiments!

## Pre-Download Models (Optional)

Optionally pre-download model tokenizers/configs to reduce first-run delays:

```bash
# Download tokenizer/config for all vLLM models to ~/.cache/huggingface/hub/
python3 experiments/download_models.py

# Or download a specific model
python3 experiments/download_models.py --model vllm-llama32-3b
```

**Note:** Model weights (10-40GB each) will still download on first use by vLLM.

## Experiment Configuration

**Models Tested:** 11 vLLM models (from config.yaml)
- vllm-qwen3-0.6b, vllm-vibethinker, vllm-deepseek, vllm-qwen3-1.7b
- vllm-llama32-3b, vllm-smallthinker, vllm-qwen3-4b
- vllm-llama31-8b, vllm-qwen3-8b, vllm-qwen3-14b, vllm-oss-20b

**Agent Counts:** 1, 3, 5, 7

**Round Counts:** 2, 3, 4, 5, 6

**Total Experiments:** 880 (11 models × 4 agent counts × 5 round counts × 4 tasks)

## Usage

### Run Individual Tasks

```bash
# Math task (100 problems per experiment)
./experiments/run_math_experiments.sh

# GSM task (100 problems per experiment)
./experiments/run_gsm_experiments.sh

# Biography task (40 people per experiment)
./experiments/run_biography_experiments.sh

# MMLU task (100 questions per experiment)
./experiments/run_mmlu_experiments.sh
```

### Run All Tasks in Parallel (HPC)

```bash
# Example: Run all 4 tasks simultaneously on different GPUs/nodes
./experiments/run_math_experiments.sh &
./experiments/run_gsm_experiments.sh &
./experiments/run_biography_experiments.sh &
./experiments/run_mmlu_experiments.sh &
wait
```

### Run Specific Subset (Manual Editing)

Edit the MODELS, AGENTS, or ROUNDS arrays in the scripts to test specific configurations:

```bash
# Example: Test only small models
MODELS=(
    "vllm-qwen3-0.6b"
    "vllm-deepseek"
    "vllm-llama32-3b"
)

# Example: Test fewer round counts
ROUNDS=(2 3 4)
```

## Output

All results are saved to:
```
results/baseline/
├── math/          # Math task results
├── gsm/           # GSM task results
├── biography/     # Biography task results
└── mmlu/          # MMLU task results
```

**Filename Format:** `{task}_{model}_agents{N}_rounds{N}.json`

Example: `math_Llama-3.2-3B_agents3_rounds4.json`

## Progress Tracking

Each script displays:
- Total number of experiments
- Current progress: `[123/220] Running: model=vllm-llama32-3b agents=3 rounds=4`
- Result file location after each experiment

## Error Handling

- Scripts exit on first error (`set -e`)
- Each experiment runs independently (one failure doesn't affect others)
- Missing result files trigger a warning but don't stop execution

## Estimated Runtime (on dual RTX 3090)

Per experiment (approximate):
- **Math:** ~5-6 minutes (100 problems)
- **GSM:** ~10-12 minutes (100 problems)
- **Biography:** ~15-20 minutes (40 people)
- **MMLU:** ~8-10 minutes (100 questions)

**Total per task:** ~40-60 hours for 220 experiments
**All 4 tasks (parallel):** ~60 hours on quad-GPU setup

Smaller models (0.6B-3B) run much faster than large models (8B-20B).

## Notes

- Default random seeds ensure reproducibility across runs
- Results can be analyzed using `scripts/aggregate_results.py`
- Model diversity experiments (different models per agent) are NOT included in these baseline scripts
