# Beyond Symmetric Agents: Cognitive Diversity and Multiagent Debate in Small Language Models

**Adapting the multiagent debate methodology to locally-hosted small language models**

**Course Project:** Topics in Machine Learning: Compositional AI Systems
COMPSCI 2821R, Fall 2025, Harvard School of Engineering and Applied Sciences

## About This Project

This repository adapts the research from ["Improving Factuality and Reasoning in Language Models through Multiagent Debate"](https://arxiv.org/abs/2305.14325) (Du et al., 2023) to work with local Small Language Models (SLMs) instead of OpenAI's GPT models.

## Contributors

- **Leonardo Ferreira** - <leonardo.ferreira@childrens.harvard.edu>
- **Vennela Jonnala** - <vjonnala@college.harvard.edu>
- **Gardenia Liu** - <gardenialiu@college.harvard.edu>
- **Kaden Zheng** - <kadenzheng@college.harvard.edu>

## Abstract

Multi-Agent Debate (MAD) has been a promising mechanism to improve reasoning and factual consistency in language models. In multi-agent debate, multiple agents propose answers, critique each other, and converge to an ideally superior solution. Prior work (Du et al., 2023) treats agents as symmetric peers, but doesn't fully answer why multi-agent debate helps. In this research project, we propose that cognitive diversity among agents, such as variation in reasoning style, prompting priors, or heuristics, is a key driver of multi-agent debate gains.

We adopt a small-language-model (SLM) setting (e.g., 1.5B-14B parameter range) to examine this hypothesis in a cost-effective, reproducible environment. We construct multiple debate conditions: (1) homogeneous groups of agents all using the same model and prompt style; (2) heterogeneous groups where the same model is prompted to adopt distinct reasoning styles (such as "intuitive", "slow", "skeptic"); (3) heterogeneous groups composed of different models; and (4) heterogeneous groups varying on the decoding parameters. We hold the number of agents and rounds constant, and evaluate on benchmark reasoning and factuality tasks (such as GSM8K word problems, biography generation, and MMLU multiple-choice questions).

In this paper, we introduce a diversity-gain metric that quantifies improvements in outcome quality (accuracy) as a function of response embedding and argument diversity (measured via cosine distances, disagreement rates), as referenced in a critique of MAD presented by Wynn et. al., 2025. We then test whether higher intra-group stylistic/response variance correlates with higher accuracy gains.

### Original Work
- **Paper:** [ArXiv:2305.14325](https://arxiv.org/abs/2305.14325)
- **Authors:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch
- **Original Repository:** [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate)

### Our Adaptation Goals
1. **Cost Reduction:** Eliminate API costs by using local inference
2. **Model Diversity:** Test whether debate benefits smaller open-source models
3. **Reproducibility:** Enable offline experiments without API dependencies
4. **Accessibility:** Make multiagent debate experiments accessible to researchers with consumer hardware

## What is Multiagent Debate?

Multiple LLM agents independently solve the same problem, then see each other's solutions and refine their answers over several rounds. This iterative debate process has been shown to improve:
- Factual accuracy (biography generation)
- Mathematical reasoning (GSM8K, arithmetic)
- Knowledge recall (MMLU benchmark)

**Key insight:** Even when using the same underlying model, independent agents with different "perspectives" can correct each other's errors through debate.

---

## Installation

### Prerequisites
- **Python:** 3.10+ (tested with 3.12.7)
- **Platform:** macOS (Apple Silicon), Linux (NVIDIA GPUs), or Windows

### 1. Clone Repository
```bash
git clone https://github.com/lferreiraMD/slm_multiagent_debate.git
cd slm_multiagent_debate
```

### 2. Install Dependencies

#### **macOS (Apple Silicon - M1/M2/M3/M4)**
```bash
pip install -r requirements.txt
```
- MLX packages will automatically install for Apple Silicon
- Models are already optimized and ready to use

#### **Linux/HPC (NVIDIA GPUs)**
```bash
pip install -r requirements_hpc.txt
```
- Installs vLLM, PyTorch with CUDA, transformers
- Tested on Ubuntu 22.04 with dual RTX 3090 (48GB VRAM)
- CUDA 12.4 and cuDNN installed automatically via PyTorch

**Verify GPU setup:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
nvidia-smi  # Check GPU status
```

#### **Cross-Platform (Ollama)**
```bash
pip install -r requirements.txt

# Install Ollama
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows:
# Download from https://ollama.com/download/windows

# Pull models
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:1.5b
```

### 3. Verify Installation
```bash
# Check vLLM (Linux only)
python -c "from vllm import LLM; print('vLLM ready')"

# Check MLX (macOS only)
python -c "import mlx_lm; print('MLX ready')"

# Check Ollama (all platforms)
ollama list
```

---

## Quick Start: HPC/Linux Setup

**For HPC users running experiments with vLLM**, follow this workflow:

### Step 1: Download Models (Required First Step)
Pre-download all model tokenizers and configs to avoid runtime delays:

```bash
python experiments/download_models.py
```

This caches tokenizers/configs for all 11 vLLM models. Model weights (10-40GB each) will download automatically on first use.

**Duration:** ~5-10 minutes
**Disk space:** ~250GB for all files (tokenizers/configs/model weights)

### Step 2: Run HPC Test (Verify Setup)
Test that everything works before running full experiments:

```bash
./experiments/hpc_test.sh
```

This runs a quick sanity check:
- ✓ Tests all 4 tasks (math, gsm, biography, mmlu)
- ✓ Minimal configuration (2 agents, 2 rounds, 10 problems)
- ✓ Uses vllm-vibethinker (1.5B, fast model)
- ✓ Verifies vLLM backend and model loading
- ✓ Auto-cleans up test files

**Duration:** ~5-10 minutes on dual RTX 3090
**If test passes:** You're ready for full experiments!

### Step 3: Run Full Experiments
After successful testing, run the baseline experiments:

```bash
# Run individual tasks (220 experiments each)
./experiments/run_math_experiments.sh      # ~40-60 hours
./experiments/run_gsm_experiments.sh       # ~40-60 hours
./experiments/run_biography_experiments.sh # ~40-60 hours
./experiments/run_mmlu_experiments.sh      # ~40-60 hours

# Or run all tasks in parallel (recommended for multi-GPU systems)
./experiments/run_math_experiments.sh &
./experiments/run_gsm_experiments.sh &
./experiments/run_biography_experiments.sh &
./experiments/run_mmlu_experiments.sh &
wait
```

Each script tests:
- **11 models** (vllm-qwen3-0.6b through vllm-oss-20b)
- **4 agent counts** (1, 3, 5, 7)
- **5 round counts** (2, 3, 4, 5, 6)
- **Total:** 220 experiments per task, 880 total

**Results:** Saved to `results/baseline/{task}/`

See `experiments/README.md` for detailed documentation.

---

## Available Tasks

All tasks support inline evaluation for immediate feedback on performance.

### 📊 Math Task - Arithmetic Reasoning
Simple arithmetic expressions testing order of operations.

**Task:** Evaluate expressions like `a+b*c+d-e*f`
**Config:** 2 agents, 3 rounds, 100 problems (default)
**Evaluation:** Automated exact match (inline)

```bash
cd tasks/math

# Default configuration
python3 gen_math.py

# Custom configuration
python3 gen_math.py --model vllm-llama32-3b --agents 3 --rounds 2 --num-problems 50
```

**Output:** `math_{model}_agents{N}_rounds{N}.p` (pickle format with results and accuracy)

---

### 🧮 GSM Task - Grade School Math
Multi-step word problems from GSM8K dataset requiring arithmetic reasoning.

**Task:** Word problems (e.g., "Janet has 3 eggs...calculate profit")
**Config:** 3 agents, 2 rounds, 100 problems (default)
**Dataset:** [OpenAI GSM8K](https://github.com/openai/grade-school-math) (included)
**Evaluation:** Automated answer extraction and comparison (inline)

```bash
cd tasks/gsm

# Default configuration
python3 gen_gsm.py

# Custom configuration
python3 gen_gsm.py --model vllm-qwen25-7b --agents 4 --rounds 3 --num-problems 50

# Outputs: gsm_{model}_agents{N}_rounds{N}.json
#   - Includes accuracy metrics printed during generation
```

---

### 👤 Biography Task - Factual Biography Generation
Generate factual bullet-point biographies of computer scientists.

**Task:** Generate biographies with factual accuracy
**Config:** 3 agents, 2 rounds, 40 people (default)
**Dataset:** Ground truth biographies in `data/biography/article.json`
**Evaluation:** Manual or GPT-4 based fact-checking

```bash
cd tasks/biography

# Generate biographies
python3 gen_conversation.py --model vllm-llama32-3b --agents 3 --rounds 2

# Evaluate factuality (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
python3 eval_conversation.py

# Outputs: biography_{model}_agents{N}_rounds{N}.json
```

---

### 📚 MMLU Task - Multiple-Choice Questions
Multiple-choice questions across academic subjects from the MMLU benchmark.

**Task:** Multi-subject multiple-choice questions (A/B/C/D)
**Config:** 3 agents, 2 rounds, 100 questions (default)
**Dataset:** [MMLU](https://github.com/hendrycks/test) (included in `data/mmlu/`)
**Evaluation:** Automated answer extraction and comparison (inline)

```bash
cd tasks/mmlu

# Generate and evaluate (inline evaluation during generation)
python3 gen_mmlu.py --model vllm-llama32-3b --agents 3 --rounds 2 --num-questions 100

# Optional: Re-evaluate saved results with detailed debug output
python3 eval_mmlu.py --input-file mmlu_{model}_agents{N}_rounds{N}.json --debug

# Outputs: mmlu_{model}_agents{N}_rounds{N}.json
#   - Includes accuracy metrics printed during generation
```

---

## Platform Support

### ✅ Fully Tested Platforms

| Platform | Backend | Status | Hardware |
|----------|---------|--------|----------|
| macOS (Apple Silicon) | MLX | ✅ Tested | M4 Pro, 48GB RAM |
| Linux (NVIDIA GPU) | vLLM | ✅ Tested | 2x RTX 3090 (48GB), 128GB RAM |
| Cross-platform | Ollama | ⚠️ Ready | Any (CPU/GPU) |

### Backend Auto-Detection

The codebase automatically selects the best available backend:
1. **MLX** - macOS with Apple Silicon
2. **vLLM** - Linux with NVIDIA GPUs
3. **Ollama** - Cross-platform fallback

No code changes needed to switch platforms!

---

## Available Models

### Model Aliases

Use short aliases instead of full HuggingFace paths:

| Alias | Platform | Full Path | Size |
|-------|----------|-----------|------|
| `vllm-deepseek` | Linux/vLLM | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B |
| `vllm-vibethinker-1.5b` | Linux/vLLM | `WeiboAI/VibeThinker-1.5B` | 1.5B |
| `vllm-smallthinker-3b` | Linux/vLLM | `PowerInfer/SmallThinker-3B-Preview` | 3B |
| `vllm-llama32-3b` | Linux/vLLM | `meta-llama/Llama-3.2-3B-Instruct` | 3B |
| `vllm-qwen25-7b` | Linux/vLLM | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| `vllm-llama31-8b` | Linux/vLLM | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B |
| `vllm-qwen25-14b` | Linux/vLLM | `Qwen/Qwen2.5-14B-Instruct` | 14B |

**Mac (MLX) aliases:** `deepseek`, `llama32-3b`, `smallthinker`, `qwen25-7b`, `llama31-8b`, `qwen25-14b`
**Ollama aliases:** `ollama-deepseek`, `ollama-llama32`, `ollama-qwen25-7b`, etc.

See `config.yaml` for complete list.

---

## Running Experiments

### Basic Usage

```bash
# Single experiment
cd tasks/gsm
python3 gen_gsm.py --model vllm-llama32-3b --agents 3 --rounds 2 --num-problems 100
```

### Model Diversity (Cognitive Diversity Experiments)

Use different models for each agent:

```bash
python3 gen_gsm.py \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agents 3 \
  --rounds 2
```

This will:
- Agent 1: Llama 3.2 3B
- Agent 2: Qwen 2.5 7B
- Agent 3: DeepSeek 1.5B
- Output filename: `gsm_deepseek+llama32-3b+qwen25-7b_agents3_rounds2.json`

### Temperature Diversity (Parameter Diversity Experiments)

Use different sampling temperatures for each agent to create cognitive diversity through varying creativity/randomness:

```bash
python3 gen_gsm.py \
  --model vllm-llama32-3b \
  --agents 3 \
  --rounds 2 \
  --agent-temperatures 0.7 1.0 1.3
```

This will:
- Agent 1: Temperature 0.7 (more conservative, focused)
- Agent 2: Temperature 1.0 (balanced, default)
- Agent 3: Temperature 1.3 (more creative, exploratory)
- Output filename: `gsm_Llama-3.2-3B_temp0.7+1.0+1.3_agents3_rounds2.json`

**Temperature Guide:**
- **0.0-0.5:** Deterministic, focused on most likely responses
- **0.7:** Good balance of consistency and variety
- **1.0:** Default, balanced exploration
- **1.3-1.5:** More creative, diverse responses
- **>1.5:** Highly random (may reduce coherence)

### Combined Diversity (Model + Temperature)

Combine both model and temperature diversity for maximum cognitive variation:

```bash
python3 gen_gsm.py \
  --agents 3 \
  --rounds 2 \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agent-temperatures 0.7 1.0 1.3
```

This creates the richest diversity:
- Agent 1: Llama 3.2 3B @ temp=0.7
- Agent 2: Qwen 2.5 7B @ temp=1.0
- Agent 3: DeepSeek 1.5B @ temp=1.3
- Output filename: `gsm_deepseek+llama32-3b+qwen25-7b_temp0.7+1.0+1.3_agents3_rounds2.json`

**Note:** All generation scripts (`gen_math.py`, `gen_gsm.py`, `gen_conversation.py`, `gen_mmlu.py`) support both `--agent-models` and `--agent-temperatures` arguments. The number of models/temperatures must match the number of agents.

---

## Automated Benchmarks

### Baseline Benchmarks

Run all models with single agent (no debate) for baseline comparison:

```bash
# GSM baseline (100 problems, all 7 models)
bash scripts/benchmark_gsm_baseline.sh
# Duration: ~2 hours on dual RTX 3090
# Output: results/gsm_baseline/

# Math baseline (100 problems, all 7 models)
bash scripts/benchmark_math_baseline.sh
# Duration: ~1 hour on dual RTX 3090
# Output: results/math_baseline/
```

**Features:**
- ✅ Same 100 problems for all models (fair comparison via `random_seed: 0`)
- ✅ Error handling (continues if one model fails)
- ✅ Progress tracking with timestamps
- ✅ Automatic result organization

See `scripts/README_BENCHMARKS.md` for details on creating custom benchmarks.

---

## Analyzing Results

### View Results

```bash
# View accuracy from GSM results
grep "Final accuracy" tasks/gsm/gsm_*.json

# View math results
python3 -c "import pickle; d=pickle.load(open('tasks/math/math_llama32-3b_agents3_rounds2.p','rb')); print(d['accuracy'])"
```

### Aggregate Results

```bash
# Combine all results into summary
python3 scripts/aggregate_results.py

# Output:
#   results/summary.csv  (human-readable)
#   results/summary.p    (pickle format)
```

### Generate Plots

```bash
# Per-model plots
python3 scripts/plot_by_model.py
# Output: plots/{model}_{task}.png

# Task comparison plots (all models)
python3 scripts/plot_by_task.py
# Output: plots/{task}_comparison.png
```

---

## Configuration

### config.yaml

Central configuration file for all experiments:

```yaml
# Default model
model: "deepseek"  # Or any alias

# Generation parameters (matching GPT-3.5 defaults)
generation:
  temperature: 1.0
  max_tokens: null  # Auto-determined per task
  top_p: 1.0
  n: 1

# Experiment defaults
experiments:
  math:
    agents: 2
    rounds: 3
    num_problems: 100
    random_seed: 0  # Ensures reproducibility
```

Override via command-line:
```bash
python3 gen_gsm.py --model vllm-qwen25-7b --agents 4 --rounds 3
```

---

## Project Structure

```
.
├── data/                   # Datasets (included in repo)
│   ├── biography/          # Ground truth biographies (article.json)
│   ├── gsm8k/              # GSM8K dataset (train.jsonl, test.jsonl)
│   └── mmlu/               # MMLU benchmark (*_test.csv files)
│
├── tasks/                  # Task implementations
│   ├── math/               # Arithmetic reasoning
│   │   └── gen_math.py     # Generation with inline evaluation
│   ├── gsm/                # Grade school math (GSM8K)
│   │   ├── gen_gsm.py      # Generation with inline evaluation
│   │   └── eval_gsm.py     # Standalone evaluation (legacy)
│   ├── biography/          # Computer scientist biographies
│   │   ├── gen_conversation.py   # Generation script
│   │   └── eval_conversation.py  # GPT-4 based evaluation
│   └── mmlu/               # MMLU multiple-choice questions
│       ├── gen_mmlu.py     # Generation with inline evaluation
│       └── eval_mmlu.py    # Standalone evaluation with debug mode
│
├── experiments/            # HPC experiment infrastructure (NEW)
│   ├── download_models.py  # Pre-cache model tokenizers/configs
│   ├── hpc_test.sh         # Quick sanity check (all 4 tasks)
│   ├── run_math_experiments.sh       # 220 math experiments
│   ├── run_gsm_experiments.sh        # 220 GSM experiments
│   ├── run_biography_experiments.sh  # 220 biography experiments
│   ├── run_mmlu_experiments.sh       # 220 MMLU experiments
│   └── README.md           # Experiment documentation
│
├── utils/                  # Shared utilities
│   ├── llm_wrapper.py      # Multi-backend LLM interface (MLX/vLLM/Ollama)
│   ├── config.py           # Configuration management
│   ├── model_cache.py      # Model loading/caching with cleanup
│   ├── helpers.py          # Shared functions (accuracy, diversity metrics)
│   └── ORIGINAL_STUDY_PARAMETERS.md  # Original paper parameters
│
├── scripts/                # Analysis and benchmarking
│   ├── aggregate_results.py          # Combine experiment results
│   ├── plot_by_model.py              # Per-model visualizations
│   ├── plot_by_task.py               # Task comparison plots
│   ├── benchmark_gsm_baseline.sh     # GSM baseline benchmark
│   ├── benchmark_math_baseline.sh    # Math baseline benchmark
│   ├── README_BENCHMARKS.md          # Benchmark documentation
│   └── test_model_output.py          # Model testing utilities
│
├── results/                # Experiment outputs
│   ├── baseline/           # Baseline experiment results (NEW)
│   │   ├── math/           # Math task results
│   │   ├── gsm/            # GSM task results
│   │   ├── biography/      # Biography task results
│   │   └── mmlu/           # MMLU task results
│   ├── summary.p           # Aggregated results (tracked in git)
│   └── summary.csv         # Human-readable summary (tracked in git)
│
├── personas/               # Cognitive diversity research (NEW)
│   ├── diversity_optimization_2821r.ipynb      # Jupyter notebook
│   ├── diversity_optimization_2821r_mlx.py     # Python script version
│   ├── embedding_search.py                     # Embedding analysis
│   ├── persona_v1_data.txt & persona_v1_results.txt
│   └── persona_v2_data.txt & persona_v2_results.txt
│
├── plots/                  # Generated visualizations (gitignored)
│   └── *.png               # t-SNE plots, experiment results
│
├── text/                   # Project documentation (NEW)
│   └── abstract.txt        # Research abstract
│
├── legacy/                 # Deprecated code (not actively maintained)
│   └── eval_gsm.py         # Old standalone GSM evaluation
│
├── config.yaml             # Central configuration
├── requirements.txt        # Cross-platform dependencies
├── requirements_hpc.txt    # Linux/HPC with vLLM
├── CLAUDE.md               # Internal technical documentation
└── README.md               # This file
```

---

## Troubleshooting

### Script Hangs After Completion

**Symptom:** Script completes but doesn't return to shell prompt
**Cause:** vLLM background processes not properly shut down
**Solution:** Already fixed! All scripts now include automatic cleanup.

If you still experience hanging:
```bash
# Force kill
killall python3

# Check for orphaned processes
ps aux | grep vllm
```

### CUDA Out of Memory

**Symptom:** `torch.cuda.OutOfMemoryError`
**Solution:**

```bash
# Check GPU memory
nvidia-smi

# Use smaller model
python3 gen_gsm.py --model vllm-llama32-3b  # 3B instead of 14B

# Or reduce batch size (not yet implemented)
```

**VRAM Requirements:**
- 1.5-3B models: ~6-8 GB
- 7-8B models: ~14-16 GB
- 14B models: ~28 GB (tight fit on single RTX 3090)

### Model Download Delays

**Symptom:** Long delay on first run
**Cause:** vLLM downloads models from HuggingFace
**Solution:** Models are cached in `~/.cache/huggingface/` after first download

```bash
# Pre-download models
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
```

### Wrong Directory Error

**Symptom:** `Error: Must run from repository root`
**Solution:** Run benchmark scripts from repository root:

```bash
cd /path/to/slm_multiagent_debate
bash scripts/benchmark_gsm_baseline.sh
```

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'utils'`
**Solution:** Run scripts from task directories as intended:

```bash
# Correct
cd tasks/gsm
python3 gen_gsm.py

# Incorrect (from repo root)
python3 tasks/gsm/gen_gsm.py
```

---

## Hardware Requirements

### Minimum
- **CPU:** 4+ cores
- **RAM:** 16GB
- **Disk:** 10GB
- **Models:** 1.5B-3B

### Recommended (Consumer Hardware)
- **CPU:** 8+ cores
- **RAM:** 32GB+
- **GPU:** Apple Silicon M1+ OR NVIDIA GPU with 8GB+ VRAM
- **Disk:** 50GB
- **Models:** Up to 8B with multiagent experiments

### High-Performance (HPC)
- **CPU:** 16+ cores
- **RAM:** 64GB+
- **GPU:** NVIDIA A100/RTX 3090/4090 (24GB+ VRAM)
- **Disk:** 100GB+
- **Models:** Up to 14B with full-scale experiments

**Tested Configuration:**
- Ubuntu 22.04, 2x RTX 3090 (48GB total VRAM), 128GB RAM
- Can run 14B models or multiple 7B agents simultaneously

---

## Performance Expectations

### Generation Speed (Linux, dual RTX 3090)

| Model Size | Tokens/sec | Math (100 problems) | GSM (100 problems) |
|------------|------------|---------------------|---------------------|
| 1.5B       | ~100       | ~5 min              | ~10 min             |
| 3B         | ~100       | ~6 min              | ~12 min             |
| 7B         | ~50        | ~8 min              | ~15 min             |
| 8B         | ~50        | ~9 min              | ~16 min             |
| 14B        | ~30        | ~12 min             | ~20 min             |

**Note:** First run per model adds 5-10 minutes for model download and compilation.

---

## Development Status

### ✅ Completed
- Multi-platform support (Mac MLX, Linux vLLM, Ollama)
- All 4 tasks migrated to local inference
- Inline evaluation for math and GSM tasks
- Model diversity support (different models per agent)
- Parameter diversity support (different temperatures per agent)
- Automated benchmarking scripts
- Results aggregation and visualization
- Comprehensive documentation

### 🔄 In Progress
- Baseline benchmark experiments (GSM, Math)
- Full-scale multiagent debate experiments
- Cognitive diversity analysis (model + temperature variation)

### 📋 Planned
- Prompt diversity experiments (different system prompts per agent)
- Large-scale experiments (1000+ problems per task)
- Biography and MMLU full evaluation
- Statistical analysis of diversity impact on debate performance

---

## Citation

If you use this work, please cite both the original paper and this adaptation:

**Original Paper:**
```bibtex
@article{du2023improving,
  title={Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author={Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B and Mordatch, Igor},
  journal={arXiv preprint arXiv:2305.14325},
  year={2023}
}
```

**This Adaptation:**
```bibtex
@software{slm_multiagent_debate,
  title={Multiagent Debate with Small Language Models},
  author={Ferreira, Leonardo and Jonnala, Vennela and Liu, Gardenia and Zheng, Kaden},
  year={2025},
  url={https://github.com/lferreiraMD/slm_multiagent_debate},
  note={Course project for COMPSCI 2821R, Harvard SEAS}
}
```

---

## Contributing

This is a research project. Contributions, suggestions, and experiment results are welcome!

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## License

MIT License - see LICENSE file for details.

---

## Related Work

- **Original Implementation:** [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate)
- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **vLLM:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Ollama:** [ollama/ollama](https://github.com/ollama/ollama)

---

## Acknowledgments

Built upon the excellent work of Du et al. (2023). This project demonstrates that multiagent debate benefits extend to smaller, locally-hosted models, making this research methodology more accessible and cost-effective.

Special thanks to the MLX, vLLM, and Ollama teams for creating excellent local inference frameworks.
