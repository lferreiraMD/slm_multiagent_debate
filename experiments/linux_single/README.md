# Linux Single GPU Persona Diversity Experiments

Complete infrastructure for running 216 persona diversity experiments on Ubuntu with vLLM and single RTX 3090 (24GB VRAM).

## Quick Start

```bash
# 1. Setup (one-time)
bash setup.sh

# 2. Test system
bash test_system.sh

# 3. Run all experiments
bash run_all_experiments.sh

# 4. Monitor (in separate terminal)
bash monitor_experiments.sh
```

## System Requirements

- **OS**: Ubuntu 22.04 or later
- **GPU**: Single NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CUDA**: 12.0 or later
- **Python**: 3.10+
- **Dependencies**: vLLM 0.11.0, PyTorch 2.8.0, transformers 4.57.1

**Note:** This configuration excludes the 14B model (vllm-qwen3-14b) which requires 28GB VRAM. For dual GPU setups, use `experiments/linux/` instead.

### Multi-GPU Systems

If your system has multiple GPUs (e.g., internal GPU + external RTX 3090), the scripts automatically use **GPU #1** via `CUDA_VISIBLE_DEVICES=1`.

**Default configuration:**
- GPU 0: Internal GPU (e.g., GTX 1650 with 4GB) - **SKIPPED**
- GPU 1: RTX 3090 (24GB) - **USED**

**To use a different GPU:**
```bash
# Use GPU #0 instead
CUDA_VISIBLE_DEVICES=0 bash run_persona_math.sh

# Use GPU #2
CUDA_VISIBLE_DEVICES=2 bash run_persona_math.sh
```

**Verify GPU selection:**
```bash
CUDA_VISIBLE_DEVICES=1 nvidia-smi
# Should show only the RTX 3090
```

## Installation

```bash
# Install dependencies
pip3 install -r requirements_hpc.txt

# Optional: Install GNU parallel for faster execution
sudo apt-get install parallel
```

## Configuration

### Model Selection (config.yaml)

**IMPORTANT:** All vLLM model specifications are managed in `config.yaml` at the project root. This is the **single source of truth** for model definitions and VRAM requirements.

**How it works:**
1. `config.yaml` defines all available vLLM models with metadata (VRAM, parameter count, description)
2. `generate_job_configs.py` reads config.yaml and filters models based on `--max-vram-gb` limit
3. Models exceeding the VRAM limit are automatically excluded

**Example: Adding a new model**

```yaml
# In config.yaml:
models:
  vllm-new-model: "org/new-model-name"

model_metadata:
  vllm-new-model:
    vram_gb: 12
    params: "5B"
    description: "New 5B parameter model"
```

Then regenerate configs:

```bash
bash setup.sh  # Or: python3 generate_job_configs.py
```

**Customizing VRAM limit:**

```bash
# Generate configs for 16GB GPU instead of 24GB
python3 generate_job_configs.py --max-vram-gb 16

# Or for 48GB dual GPU setup
python3 generate_job_configs.py --max-vram-gb 48
```

### Parallelism

Control how many experiments run simultaneously:

```bash
# Default: 2 parallel jobs (safe for small models <4B)
bash run_all_experiments.sh

# Sequential (recommended for 7B-8B models)
MAX_PARALLEL=1 bash run_all_experiments.sh

# Or set per task
MAX_PARALLEL=1 bash run_persona_math.sh
```

**Recommended settings:**
- Single RTX 3090: `MAX_PARALLEL=2` (default, for small models <4B)
- Conservative (large models 7B-8B): `MAX_PARALLEL=1`

### GPU Memory

vLLM auto-detects and configures GPU memory. For manual control, see `utils/gpu_config.py`.

## Experiments

### Total: 216 Experiments

- **Tasks**: 4 (math, gsm, biography, mmlu)
- **Experiments per task**: 54
- **Models**: 9 (0.6B to 8B parameters, excludes 14B)
- **Agent counts**: 6 (2-7 agents)
- **Persona selection**: MaxDet v2 (extreme personas)

### Excluded Models

Models are automatically excluded based on VRAM requirements defined in `config.yaml`:

**Default (24GB VRAM limit):**
- `vllm-qwen3-14b` - Requires ~28GB VRAM (exceeds 24GB limit)
- `vllm-oss-20b` - Requires ~40GB VRAM (exceeds 24GB limit)

**Note:** To see which models are included/excluded for your configuration, run:
```bash
python3 generate_job_configs.py --max-vram-gb 24
```

### Task Details

| Task | Script | Problems | Rounds | Dataset |
|------|--------|----------|--------|---------|
| Math | `run_persona_math.sh` | 100 | 3 | Generated arithmetic |
| GSM | `run_persona_gsm.sh` | 100 | 2 | GSM8K word problems |
| Biography | `run_persona_biography.sh` | 40 | 2 | Computer scientists |
| MMLU | `run_persona_mmlu.sh` | 100 | 2 | Multiple choice QA |

## Scripts

### Execution

- `setup.sh`: One-time setup (generates configs, makes scripts executable)
- `run_all_experiments.sh`: Run all 216 experiments sequentially
- `run_persona_math.sh`: Run 54 math experiments
- `run_persona_gsm.sh`: Run 54 GSM experiments
- `run_persona_biography.sh`: Run 54 biography experiments
- `run_persona_mmlu.sh`: Run 54 MMLU experiments

### Management

- `test_system.sh`: Validate setup with 4 quick tests
- `monitor_experiments.sh`: Real-time progress monitoring
- `resubmit_failed.sh`: Resubmit failed experiments

### Configuration

- `generate_job_configs.py`: Generate job config files (auto-run by setup.sh)

## Workflow

### 1. Initial Setup

```bash
cd experiments/linux_single
bash setup.sh
```

This will:
- Make all scripts executable
- Generate job configurations (216 jobs across 4 files: 54 per task)
- Create log directories

### 2. Validation

```bash
bash test_system.sh
```

Tests:
- Python and vLLM installation
- NVIDIA GPU availability (warns if not single GPU)
- Job config files (validates 54 jobs per task)
- Generation scripts with persona support
- 4 quick experiments (1 per task, 3 problems each)

### 3. Run Experiments

#### Option A: All Tasks (Sequential)

```bash
bash run_all_experiments.sh
```

**Duration**: ~10-14 hours (single GPU, depends on hardware)

#### Option B: Individual Tasks

```bash
# Run tasks separately
bash run_persona_math.sh        # ~2.5 hours (54 jobs)
bash run_persona_gsm.sh         # ~3 hours
bash run_persona_biography.sh  # ~3 hours
bash run_persona_mmlu.sh        # ~3 hours
```

#### Option C: Custom Parallelism

```bash
# Sequential (safest, for all models)
MAX_PARALLEL=1 bash run_all_experiments.sh

# Parallel (default, safe for small models <4B)
MAX_PARALLEL=2 bash run_all_experiments.sh
```

### 4. Monitor Progress

In a separate terminal:

```bash
bash monitor_experiments.sh
```

Shows:
- Per-task progress (completed/running/failed out of 54)
- Overall progress (out of 216)
- GPU status (VRAM %, utilization, temperature)
- Running Python processes

Press Ctrl+C to stop monitoring (experiments continue in background).

### 5. Handle Failures

```bash
# Check for failed jobs
bash resubmit_failed.sh

# Or resubmit specific task
bash resubmit_failed.sh math
```

### 6. Aggregate Results

```bash
cd ../..
python3 scripts/aggregate_results.py
python3 scripts/aggregate_persona_results.py
```

## Output

### Logs

```
experiments/linux_single/logs/
├── math/
│   ├── job_1.out
│   ├── job_2.out
│   └── ...
├── gsm/
├── biography/
├── mmlu/
└── test/
    └── test_*.out
```

### Results

```
tasks/
├── math/
│   ├── math_Qwen3-0.6B_persona_*_agents2_rounds3.p
│   └── ...
├── gsm/
│   ├── gsm_Qwen3-0.6B_persona_*_agents2_rounds2.json
│   └── ...
├── biography/
└── mmlu/
```

### Aggregated

```
results/
├── summary.p                 # All results (baseline + diversity)
├── summary.csv              # Human-readable
├── persona_comparison.p     # Persona vs baseline analysis
└── persona_comparison.csv   # Human-readable
```

## GPU Memory Management

### Pre-flight Checks

Each task runner performs automatic GPU memory checks before launching experiments:
- Estimates required VRAM based on model size
- Checks available GPU memory
- Skips experiments if insufficient memory detected
- Logs warnings for manual intervention

### Memory Requirements by Model

| Model | VRAM Required | Safe with MAX_PARALLEL=2? | Notes |
|-------|---------------|---------------------------|-------|
| qwen3-0.6b | ~2GB | ✓ Yes | Minimal memory usage |
| vibethinker (1.5B) | ~4GB | ✓ Yes | Safe for parallel runs |
| deepseek (1.5B) | ~4GB | ✓ Yes | Safe for parallel runs |
| llama32-3b | ~7GB | ✓ Yes | Safe for parallel runs |
| smallthinker (3B) | ~7GB | ✓ Yes | Safe for parallel runs |
| qwen3-4b | ~9GB | ⚠ Maybe | Use MAX_PARALLEL=1 for safety |
| qwen25-7b | ~15GB | ✗ No | Requires MAX_PARALLEL=1 |
| llama31-8b | ~17GB | ✗ No | Requires MAX_PARALLEL=1 |

**Tip:** Multi-agent experiments add ~0.5GB per additional agent (KV cache overhead).

## Parallel Execution Details

### With GNU Parallel

If `parallel` is installed:
```bash
sudo apt-get install parallel
```

Execution uses GNU parallel for optimal job scheduling:
- Automatic load balancing
- Efficient resource utilization
- Better error handling

### Without GNU Parallel

Falls back to bash background jobs:
- Uses `wait -n` for job control
- Less efficient but functional
- Install GNU parallel for better performance

## Troubleshooting

### "Config file not found"

```bash
cd experiments/linux_single
python3 generate_job_configs.py
```

### "vLLM not installed"

```bash
pip3 install vllm==0.11.0
```

### "CUDA out of memory"

This is the most common issue with single GPU setups. Try these solutions:

1. Reduce parallelism (most effective):
   ```bash
   MAX_PARALLEL=1 bash run_persona_math.sh
   ```

2. Run smaller models first to validate:
   ```bash
   # Edit configs/persona_*_jobs.txt to keep only 0.6B-3B models
   ```

3. Increase GPU memory utilization safety margin:
   - Edit utils/gpu_config.py
   - Reduce `gpu_memory_utilization` from 0.8 to 0.7 for 'debate' use case

4. Clear GPU memory between tasks:
   ```bash
   # After each task, manually clear:
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

5. Monitor memory during execution:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Jobs hanging/not completing

Check vLLM engine cleanup in generation scripts. All scripts call `model_cache.shutdown()`.

### GNU parallel warnings

Ignore warnings about citation. To silence:
```bash
parallel --citation  # Run once, type "will cite"
```

### Expected completion rates

With single RTX 3090, expect:
- **Small models (0.6B-3B)**: 100% completion with MAX_PARALLEL=2
- **Medium models (4B)**: 95-100% completion with MAX_PARALLEL=1
- **Large models (7B-8B)**: 90-95% completion with MAX_PARALLEL=1

Some 8B + 7 agents jobs may skip due to memory constraints. This is expected and handled gracefully.

## Performance Tips

1. **Use GNU parallel**: 2-3x faster than background jobs
2. **Match parallelism to model size**:
   - Small models (<4B): MAX_PARALLEL=2
   - Large models (7B-8B): MAX_PARALLEL=1
3. **Monitor GPU VRAM**: Ensure models fit in memory
4. **Use tmux/screen**: Keep sessions alive on remote servers
5. **Run overnight**: 10-14 hours for full 216 experiments

## Advanced Usage

### Resume Interrupted Runs

If experiments are interrupted:

```bash
# Check what completed
bash monitor_experiments.sh

# Resubmit failures
bash resubmit_failed.sh
```

### Custom Job Subsets

Edit config files to run specific experiments:

```bash
# Run only small models (first 3 models × 6 agents = 18 jobs)
head -n 19 configs/persona_math_jobs.txt > configs/persona_math_small.txt
# Edit run_persona_math.sh to use persona_math_small.txt
```

### Distributed Execution

Run tasks on different machines:

```bash
# Machine 1 (with RTX 3090)
bash run_persona_math.sh
bash run_persona_gsm.sh

# Machine 2 (with RTX 3090)
bash run_persona_biography.sh
bash run_persona_mmlu.sh
```

Sync results afterward:
```bash
rsync -av tasks/ user@machine2:/path/to/project/tasks/
```

## Comparison with Dual GPU Setup

| Aspect | Single RTX 3090 (linux_single/) | Dual RTX 3090 (linux/) |
|--------|--------------------------------|------------------------|
| Total VRAM | 24GB | 48GB |
| Models | 9 (exclude 14B) | 10 (include 14B) |
| Experiments | 216 (54/task) | 240 (60/task) |
| Duration | 10-14 hours | 8-12 hours |
| MAX_PARALLEL | 1-2 (depends on model) | 2 (default) |
| OOM Risk | Higher for 7B-8B models | Lower |

**When to use linux_single/**:
- You have a single GPU with 24GB VRAM
- You want to avoid dual GPU setup complexity
- You're okay with slightly longer execution times

**When to use linux/**:
- You have 2+ GPUs with 24GB+ VRAM each
- You want to include the 14B model
- You need faster execution times

## Contact

For issues or questions:
- Check logs: `experiments/linux_single/logs/`
- Review test output: `bash test_system.sh`
- See main project README: `../../README.md`
