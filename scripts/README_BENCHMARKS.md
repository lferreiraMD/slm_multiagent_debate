# Benchmark Scripts

Automated scripts for running baseline and debate experiments across all models.

## Available Scripts

### 1. `benchmark_gsm_baseline.sh`
Baseline GSM benchmark with single agent (no debate).

**Configuration:**
- **Agents:** 1 (no debate)
- **Rounds:** 1 (single response)
- **Problems:** 100 (same random subset for all models)
- **Models:** All 7 vLLM models

**Usage:**
```bash
# From repository root
bash scripts/benchmark_gsm_baseline.sh
```

**Output:** `results/gsm_baseline/gsm_*_agents1_rounds1.json`

**Duration:** ~10-15 minutes per model (~2 hours total for 7 models)

---

### 2. `benchmark_math_baseline.sh`
Baseline math benchmark with single agent (no debate).

**Configuration:**
- **Agents:** 1 (no debate)
- **Rounds:** 1 (single response)
- **Problems:** 100 (same random subset for all models)
- **Models:** All 7 vLLM models

**Usage:**
```bash
# From repository root
bash scripts/benchmark_math_baseline.sh
```

**Output:** `results/math_baseline/math_*_agents1_rounds1.p`

**Duration:** ~5-8 minutes per model (~1 hour total for 7 models)

---

## Features

### ✅ Same Problems Across Models
All models are tested on the **same 100 problems** thanks to:
- Fixed `random_seed: 0` in `config.yaml`
- Same shuffling algorithm in each task script

This ensures fair comparison across models.

### ✅ Error Handling
- Continues if one model fails
- Reports successes and failures
- Saves partial results

### ✅ Progress Tracking
- Shows current model and progress (e.g., "[3/7]")
- Timestamps for each model
- Color-coded output (green=success, red=failure)

### ✅ Result Organization
- Copies results to dedicated baseline directories
- Preserves original files in task directories
- Easy to find and analyze

---

## Creating Custom Benchmarks

### Template for New Benchmark Script

```bash
#!/bin/bash
# Custom Benchmark Script
# Description: [Your benchmark description]

set -e  # Exit on error

# Configuration
AGENTS=3        # Modify as needed
ROUNDS=2        # Modify as needed
NUM_PROBLEMS=50 # Modify as needed
TASK_DIR="tasks/gsm"  # Change to your task

# Models to test
MODELS=(
    "vllm-llama32-3b"
    "vllm-qwen25-7b"
    # Add/remove models as needed
)

# ... rest of script (copy from existing benchmark)
```

### Common Modifications

1. **Test fewer models:**
   ```bash
   MODELS=("vllm-llama32-3b" "vllm-qwen25-7b")
   ```

2. **Test multiagent debate:**
   ```bash
   AGENTS=3
   ROUNDS=2
   ```

3. **Model diversity experiment:**
   ```bash
   # In gen_*.py call, use --agent-models instead:
   --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek
   ```

4. **Quick test (fewer problems):**
   ```bash
   NUM_PROBLEMS=10
   ```

---

## Analyzing Results

### 1. View Individual Results
```bash
# GSM results
cat results/gsm_baseline/gsm_llama32-3b_agents1_rounds1.json | jq

# Math results (pickle format)
python3 -c "import pickle; print(pickle.load(open('results/math_baseline/math_llama32-3b_agents1_rounds1.p', 'rb')))"
```

### 2. Aggregate All Results
```bash
cd results/gsm_baseline
python3 ../../scripts/aggregate_results.py
# Creates summary.csv and summary.p
```

### 3. Generate Plots
```bash
python3 scripts/plot_by_task.py
# Creates plots/gsm_comparison.png, plots/math_comparison.png, etc.
```

---

## Performance Expectations (on 2x RTX 3090)

| Model Size | Math (100 problems) | GSM (100 problems) |
|------------|---------------------|---------------------|
| 1.5B       | ~5 min              | ~10 min             |
| 3B         | ~6 min              | ~12 min             |
| 7B         | ~8 min              | ~15 min             |
| 8B         | ~9 min              | ~16 min             |
| 14B        | ~12 min             | ~20 min             |

**Total for all 7 models:**
- Math: ~1 hour
- GSM: ~2 hours

---

## Troubleshooting

### Script hangs after completion
- Fixed in latest version with vLLM engine shutdown
- If issue persists: `killall python3`

### CUDA out of memory
- Models use approximately:
  - 1.5-3B: 6-8 GB VRAM
  - 7-8B: 14-16 GB VRAM
  - 14B: 28 GB VRAM
- 14B models fit on single RTX 3090 (24GB) but leave little headroom
- Solution: Close other GPU processes or test 14B models separately

### Model download delays first run
- vLLM downloads models from HuggingFace on first use
- Cached in `~/.cache/huggingface/` for subsequent runs
- First run: +5-10 minutes per model for download

### Wrong directory error
```
Error: Must run from repository root
```
- Solution: `cd /path/to/slm_multiagent_debate` before running script

---

## Advanced Usage

### Run in background with logging
```bash
nohup bash scripts/benchmark_gsm_baseline.sh > gsm_baseline.log 2>&1 &
tail -f gsm_baseline.log
```

### Run specific subset of models
Edit the script and modify the `MODELS` array:
```bash
MODELS=(
    "vllm-llama32-3b"
    "vllm-qwen25-7b"
)
```

### Parallel execution (if you have multiple GPUs)
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_gsm_baseline.sh

# Terminal 2
CUDA_VISIBLE_DEVICES=1 bash scripts/benchmark_math_baseline.sh
```

---

## Next Steps

After running baselines:
1. Compare single-agent vs multiagent debate
2. Test model diversity (different models per agent)
3. Test parameter diversity (different temperatures per agent)
4. Scale to full dataset (1000+ problems)
