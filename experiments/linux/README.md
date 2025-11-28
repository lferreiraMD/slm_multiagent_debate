# Linux Persona Diversity Experiments

Complete infrastructure for running 240 persona diversity experiments on Ubuntu with vLLM.

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
- **GPU**: NVIDIA GPU(s) with CUDA support (tested on 2x RTX 3090, 48GB VRAM)
- **CUDA**: 12.0 or later
- **Python**: 3.10+
- **Dependencies**: vLLM 0.11.0, PyTorch 2.8.0, transformers 4.57.1

## Installation

```bash
# Install dependencies
pip3 install -r requirements_hpc.txt

# Optional: Install GNU parallel for faster execution
sudo apt-get install parallel
```

## Configuration

### Parallelism

Control how many experiments run simultaneously:

```bash
# Default: 2 parallel jobs (for dual GPU)
bash run_all_experiments.sh

# Custom parallelism (4 GPUs)
MAX_PARALLEL=4 bash run_all_experiments.sh

# Or set per task
MAX_PARALLEL=3 bash run_persona_math.sh
```

**Recommended settings:**
- 1 GPU: `MAX_PARALLEL=1`
- 2 GPUs: `MAX_PARALLEL=2` (default)
- 4 GPUs: `MAX_PARALLEL=4`

### GPU Memory

vLLM auto-detects and configures GPU memory. For manual control, see `utils/gpu_config.py`.

## Experiments

### Total: 240 Experiments

- **Tasks**: 4 (math, gsm, biography, mmlu)
- **Experiments per task**: 60
- **Models**: 10 (0.6B to 14B parameters)
- **Agent counts**: 6 (2-7 agents)
- **Persona selection**: MaxDet v2 (extreme personas)

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
- `run_all_experiments.sh`: Run all 240 experiments sequentially
- `run_persona_math.sh`: Run 60 math experiments
- `run_persona_gsm.sh`: Run 60 GSM experiments
- `run_persona_biography.sh`: Run 60 biography experiments
- `run_persona_mmlu.sh`: Run 60 MMLU experiments

### Management

- `test_system.sh`: Validate setup with 4 quick tests
- `monitor_experiments.sh`: Real-time progress monitoring
- `resubmit_failed.sh`: Resubmit failed experiments

### Configuration

- `generate_job_configs.py`: Generate job config files (auto-run by setup.sh)

## Workflow

### 1. Initial Setup

```bash
cd experiments/linux
bash setup.sh
```

This will:
- Make all scripts executable
- Generate job configurations (240 jobs across 4 files)
- Create log directories

### 2. Validation

```bash
bash test_system.sh
```

Tests:
- Python and vLLM installation
- NVIDIA GPU availability
- Job config files
- Generation scripts with persona support
- 4 quick experiments (1 per task, 3 problems each)

### 3. Run Experiments

#### Option A: All Tasks (Sequential)

```bash
bash run_all_experiments.sh
```

**Duration**: ~8-12 hours (depends on hardware)

#### Option B: Individual Tasks

```bash
# Run tasks separately (can run in parallel on different machines)
bash run_persona_math.sh        # ~2 hours
bash run_persona_gsm.sh         # ~2-3 hours
bash run_persona_biography.sh  # ~2-3 hours
bash run_persona_mmlu.sh        # ~2-3 hours
```

#### Option C: Custom Parallelism

```bash
# Single GPU (sequential)
MAX_PARALLEL=1 bash run_all_experiments.sh

# Dual GPU (2x parallel, default)
MAX_PARALLEL=2 bash run_all_experiments.sh

# Quad GPU (4x parallel)
MAX_PARALLEL=4 bash run_all_experiments.sh
```

### 4. Monitor Progress

In a separate terminal:

```bash
bash monitor_experiments.sh
```

Shows:
- Per-task progress (completed/running/failed)
- Overall progress (out of 240)
- GPU status (VRAM, utilization)
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
experiments/linux/logs/
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
cd experiments/linux
python3 generate_job_configs.py
```

### "vLLM not installed"

```bash
pip3 install vllm==0.11.0
```

### "CUDA out of memory"

Reduce parallelism:
```bash
MAX_PARALLEL=1 bash run_persona_math.sh
```

Or use smaller models first:
```bash
# Edit configs/persona_*_jobs.txt to filter by model size
```

### Jobs hanging/not completing

Check vLLM engine cleanup in generation scripts. All scripts call `model_cache.shutdown()`.

### GNU parallel warnings

Ignore warnings about citation. To silence:
```bash
parallel --citation  # Run once, type "will cite"
```

## Performance Tips

1. **Use GNU parallel**: 2-3x faster than background jobs
2. **Match parallelism to GPUs**: `MAX_PARALLEL = num_gpus`
3. **Monitor GPU VRAM**: Ensure models fit in memory
4. **Run tasks separately**: Distribute across multiple machines
5. **Use tmux/screen**: Keep sessions alive on remote servers

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
# Run only small models (first 3)
head -n 4 configs/persona_math_jobs.txt > configs/persona_math_small.txt
# Edit run_persona_math.sh to use persona_math_small.txt
```

### Distributed Execution

Run tasks on different machines:

```bash
# Machine 1
bash run_persona_math.sh
bash run_persona_gsm.sh

# Machine 2
bash run_persona_biography.sh
bash run_persona_mmlu.sh
```

Sync results afterward:
```bash
rsync -av tasks/ user@machine2:/path/to/project/tasks/
```

## Contact

For issues or questions:
- Check logs: `experiments/linux/logs/`
- Review test output: `bash test_system.sh`
- See main project README: `../../README.md`
