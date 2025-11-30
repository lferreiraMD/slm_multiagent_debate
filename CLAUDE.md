# Beyond Symmetric Agents: Cognitive Diversity and Multiagent Debate in Small Language Models

**Course Project:** Topics in Machine Learning: Compositional AI Systems
COMPSCI 2821R, Fall 2025, Harvard School of Engineering and Applied Sciences

**Contributors:**
- Leonardo Ferreira <leonardo.ferreira@childrens.harvard.edu>
- Vennela Jonnala <vjonnala@college.harvard.edu>
- Gardenia Liu <gardenialiu@college.harvard.edu>
- Kaden Zheng <kadenzheng@college.harvard.edu>

## 🚀 HPC Quick Start

**100% ready for HPC deployment. No code changes needed.**

```bash
# 1. Clone and install
git clone https://github.com/lferreiraMD/slm_multiagent_debate.git
cd slm_multiagent_debate
pip3 install -r requirements_hpc.txt

# 2. Verify setup (~15 min)
bash experiments/hpc_test.sh

# 3. Run experiments
bash experiments/run_math_experiments.sh  # 220 experiments (~2 hours dual RTX 3090)
```

**Verified:** Ubuntu 22.04, vLLM 0.11.0, 2x RTX 3090 (48GB VRAM), CUDA 12.4

---

## Project Overview

Replicates ["Improving Factuality and Reasoning in Language Models through Multiagent Debate"](https://arxiv.org/abs/2305.14325) (Du et al., 2023) using **locally-hosted Small Language Models** instead of GPT-3.5-turbo.

**Goal:** Test multiagent debate with SLMs and explore whether "cognitive diversity" (model/temperature/persona variations) improves results.

**Motivation:** Reduce API costs, enable experimentation with open-source models, support offline workflows.

### Multiagent Debate Mechanism
1. Multiple agents receive same question
2. Round 1: Independent responses
3. Subsequent rounds: Agents see others' responses and refine answers
4. Aggregation: Majority vote (math) or evaluation (biography/MMLU)

### Implemented Tasks

| Task | Dataset | Default Config | Evaluation | Files |
|------|---------|----------------|------------|-------|
| **Math** | Generated arithmetic (`a+b*c+d-e*f`) | 2 agents, 3 rounds | Automated | `tasks/math/gen_math.py` |
| **GSM** | [GSM8K](https://github.com/openai/grade-school-math) word problems | 3 agents, 2 rounds | Inline, `\boxed{}` format | `tasks/gsm/gen_gsm.py` |
| **Biography** | 40 computer scientists | 3 agents, 2 rounds | Manual/automated fact-check | `tasks/biography/gen_conversation.py` |
| **MMLU** | [MMLU benchmark](https://github.com/hendrycks/test) multiple choice | 3 agents, 2 rounds | Inline, A/B/C/D extraction | `tasks/mmlu/gen_mmlu.py` |

### Key Parameters
- `--agents N`: Number of LLM agents
- `--rounds N`: Debate iterations
- `--agent-models`: List of models per agent (model diversity)
- `--agent-temperatures`: List of temperatures per agent (parameter diversity)
- `--agent-personas`: List of persona callsigns per agent (cognitive diversity)

## Technical Architecture

All generation scripts use `utils.ChatCompletion` wrapper providing OpenAI-compatible interface to MLX-LM/vLLM/Ollama:

```python
from utils import ChatCompletion

completion = ChatCompletion.create(
    model=model_name,
    messages=agent_context,
    **generation_params
)
```

**Backend Support:**
- **MLX** (Mac Apple Silicon) - Auto-detected on macOS ARM64
- **vLLM** (Linux/HPC with NVIDIA GPUs) - High-throughput inference
- **Ollama** (Cross-platform) - Server-based GGUF models

All backends support chat templates, reasoning models (VibeThinker, DeepSeek-R1), `<think>` tag extraction, and configurable token limits.

## Cognitive Diversity Experiments

Three types of diversity to improve multiagent debate:

### 1. Model Diversity
Different agents use different models (varied architectures/training):

```bash
python3 gen_gsm.py \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agents 3 --rounds 2
```

Output: `gsm_deepseek+llama32-3b+qwen25-7b_agents3_rounds2.json`

### 2. Parameter Diversity (Temperature)
Same model, different sampling temperatures (0.7=conservative, 1.3=creative):

```bash
python3 gen_gsm.py \
  --model vllm-llama32-3b \
  --agents 3 --rounds 2 \
  --agent-temperatures 0.7 1.0 1.3
```

Output: `gsm_Llama-3.2-3B_temp0.7+1.0+1.3_agents3_rounds2.json`

### 3. Persona Diversity (Cognitive Style)
Different reasoning styles via system prompts. **100 predefined personas** in `config.yaml`:
- **50 v1 (moderate):** analyst, skeptic, innovator, pragmatic, intuitive
- **50 v2 (extreme):** cryptographer, zenmaster, baroque, anarchist, grandmaster

```bash
python3 gen_gsm.py \
  --model vllm-llama32-3b \
  --agents 3 --rounds 2 \
  --agent-personas skeptic analyst intuitive
```

Output: `gsm_Llama-3.2-3B_persona_skeptic+analyst+intuitive_agents3_rounds2.json`

### 4. Combined Diversity
Maximum cognitive diversity (all three types):

```bash
python3 gen_gsm.py \
  --agents 3 --rounds 2 \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agent-temperatures 0.7 1.0 1.3 \
  --agent-personas skeptic analyst intuitive
```

## Available Models

### Mac M4 Pro (MLX-Optimized)

| Model | Size | Path | Location |
|-------|------|------|----------|
| DeepSeek-R1-Distill-Qwen | 1.5B | `valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16` | `/Users/leonardo/.cache/huggingface/hub` |
| Llama 3.2 Instruct | 3B | `mlx-community/Llama-3.2-3B-Instruct` | Ready |
| SmallThinker | 3B | `valuat/SmallThinker-3B-Preview-mlx-fp16` | Ready |
| Qwen2.5 7B Instruct | 7B | `valuat/Qwen2.5-7B-Instruct-1M-mlx-fp16` | Ready |
| Llama 3.1 8B Instruct | 8B | `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | Ready |
| Qwen2.5 14B Instruct | 14B | `valuat/Qwen2.5-14B-Instruct-1M-mlx-fp16` | Ready |

### HPC Models (vLLM)

**Model Equivalents Across Platforms:**

| Size | Mac (MLX) | Linux (vLLM) | Any (Ollama) |
|------|-----------|--------------|--------------|
| 1.5B | `deepseek` | `vllm-deepseek` | `ollama-deepseek` |
| 3B | `llama32-3b` | `vllm-llama32-3b` | `ollama-llama32` |
| 7B | `qwen25-7b` | `vllm-qwen25-7b` | `ollama-qwen25-7b` |
| 8B | `llama31-8b` | `vllm-llama31-8b` | `ollama-llama31-8b` |

**Download HPC models:**
```bash
# vLLM: Downloads automatically on first use to ~/.cache/huggingface/
python3 experiments/download_models.py

# Ollama:
ollama pull llama3.2:3b
ollama pull deepseek-r1:1.5b
```

## Configuration

**config.yaml** manages centralized configuration:
- **model**: Default model alias (e.g., `"deepseek"`)
- **generation**: GPT-3.5 defaults (temp=1.0, max_tokens=null, top_p=1.0)
- **experiments**: Task-specific configs (agents, rounds, num_problems)
- **models**: Alias → HuggingFace path mappings (single source of truth)
- **model_metadata**: VRAM requirements and parameter counts for vLLM models (used by experiments/linux_single/)
- **personas**: 100 predefined persona definitions

**Model Selection Priority:**
1. CLI: `--model qwen25-7b`
2. config.yaml: `model: "deepseek"`
3. Fallback: `"deepseek"`

## Project Status

### ✅ Phase 1 & 2: Complete (52 items)
- MLX-LM wrapper, model caching, config system
- All 4 tasks adapted (math, GSM, biography, MMLU)
- Model/temperature/persona diversity implemented
- vLLM backend tested on Ubuntu 22.04 + dual RTX 3090
- Requirements files, benchmark scripts, documentation
- **NEW (2025-11-30):** Context compression system (95% token reduction)
- **NEW (2025-11-30):** Dynamic context length from config.yaml
- **NEW (2025-11-30):** 14 models with full metadata (context_length, VRAM, params)
- **NEW (2025-11-30):** Auto-enable temperature diversity (0.5-1.5 range)
- **NEW (2025-11-30):** English-only prompt instructions for multilingual models

### 🔄 Phase 3: Current Experiments (In Progress)
- [x] ✅ Add 4 new English-centric models (Gemma-2, Phi-3, Mistral)
- [x] ✅ Calculate optimal personas for all 14 models (v2, MaxDet)
- [x] ✅ Implement context compression (math task proof of concept)
- [ ] Complete baselines across all 14 models
- [ ] Test Ollama backend
- [ ] Run multiagent debate experiments (2-7 agents)
- [ ] Test cognitive diversity combinations (persona experiments)

### 📋 Phase 4: Analysis & Paper
- [ ] Statistical analysis of debate benefits
- [ ] Large-scale experiments (1000+ problems/task)
- [ ] Write paper and visualizations

## File Structure

```
.
├── data/                   # Datasets (GSM8K, MMLU, biography)
├── tasks/                  # Task implementations (gen_*.py, eval_*.py)
│   ├── math/
│   ├── gsm/
│   ├── biography/
│   └── mmlu/
├── experiments/            # HPC infrastructure
│   ├── linux/              # Dual RTX 3090 (240 experiments)
│   │   ├── README.md       # Multi-GPU documentation
│   │   └── ...
│   ├── linux_single/       # Single RTX 3090 (216 experiments)
│   │   ├── README.md       # Single GPU documentation
│   │   └── ...
│   ├── download_models.py
│   ├── hpc_test.sh
│   ├── run_{task}_experiments.sh
│   └── README.md
├── utils/                  # Shared utilities
│   ├── llm_wrapper.py      # Multi-backend interface (MLX/vLLM/Ollama)
│   ├── config.py           # Configuration management
│   ├── model_cache.py      # Model loading/caching
│   ├── helpers.py          # Shared functions (accuracy, diversity metrics)
│   ├── cuda_cleanup.py     # GPU memory management
│   └── gpu_config.py       # vLLM auto-configuration
├── scripts/                # Analysis tools
│   ├── aggregate_results.py
│   ├── plot_by_{model,task}.py
│   └── benchmark_*.sh
├── results/                # Experiment outputs
│   ├── baseline/{math,gsm,biography,mmlu}/
│   ├── summary.p           # Tracked in git
│   └── summary.csv         # Tracked in git
├── personas/               # Cognitive diversity research
├── plots/                  # Visualizations (gitignored)
├── config.yaml             # Centralized configuration
├── requirements.txt        # Cross-platform dependencies
├── requirements_hpc.txt    # Linux/HPC (vLLM, PyTorch, CUDA)
├── CLAUDE.md              # Project documentation (this file)
├── CONTEXT_COMPRESSION.md # Context compression implementation guide (NEW)
└── README.md              # User-facing docs
```

## Results Analysis

```bash
# 1. Run experiments
cd tasks/math && python3 gen_math.py --model MODEL --agents N --rounds N

# 2. Aggregate results
python3 scripts/aggregate_results.py
# Output: results/summary.{p,csv}

# 3. Generate plots
python3 scripts/plot_by_model.py    # plots/{model}_{task}.png
python3 scripts/plot_by_task.py     # plots/{task}_comparison.png
```

## Math Task Results

**DeepSeek 1.5B:** Best = 37% (3 agents, 3 rounds), Solo = 33%
**Llama 8B:** Best = 97% (2-3 agents, 3 rounds), Solo = 85%

**Key Insight:** Multiagent debate shows stronger benefit for more capable models (+12% for Llama 8B).

## Platform-Specific Notes

### Mac M4 Pro
- **Backend:** MLX-LM (Apple Silicon optimized)
- **Performance:** 40-80 tok/s (3B), 20-40 tok/s (7-8B)
- **Commands:** Use `python3` and `pip3` (NOT `python`/`pip`)

### Ubuntu 22.04 + NVIDIA GPUs
- **Backend:** vLLM 0.11.0 (auto-detected)
- **Hardware:** 2x RTX 3090 (48GB VRAM), 128GB RAM, CUDA 12.4
- **Performance:** ~100 tok/s (3B), ~50 tok/s (7-8B)
- **Memory:** 3B=6GB, 7-8B=14-16GB, 14B=28GB VRAM
- **Installation:** `pip3 install -r requirements_hpc.txt`

### Ollama (Cross-platform)
- **Installation:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Models:** `ollama pull llama3.2:3b`
- **Status:** Code ready, untested

## Key Technical Fixes

### 1. vLLM Hanging Issue
**Problem:** Scripts complete but don't return to shell.
**Cause:** vLLM v0.11.0 background `EngineCore_DP0` processes.
**Solution:** Added `shutdown()` method to `ModelCache` (utils/model_cache.py:125-138). All generation scripts call `model_cache.shutdown()` at end.

### 2. Model Alias Resolution
**Problem:** vLLM aliases not resolved.
**Solution:** Added all vLLM/Ollama aliases to hardcoded `MODEL_ALIASES` dict in utils/config.py:54-78.

### 3. Filename Generation for Diversity
**Problem:** Filenames always used default model name.
**Solution:** Created `get_model_descriptor()` in utils/helpers.py:179-234. Handles single model (short name) and multiple models (amalgamated "model1+model2+model3").

**Example outputs:**
- Single: `gsm_Llama-3.2-3B_agents3_rounds2.json`
- Multiple: `gsm_DeepSeek-R1+Llama-3.2-3B+Qwen2.5-7B_agents3_rounds2.json`

### 4. Context Compression (NEW)
**Problem:** Multiagent debate with many agents/rounds can exceed model context limits (even 32K+ models fail with 7 agents × 3 rounds).
**Solution:** Implemented answer extraction for math task - extracts numerical answers instead of full responses (~95% token reduction).

**Implementation (tasks/math/gen_math.py):**
- `construct_message()` now has `compress_context` parameter (default: True)
- Compressed mode: Shows only answers from other agents (e.g., "Agent 1: 42")
- Original mode: Shows full reasoning chains (preserves debate quality, but uses more tokens)
- Real-time compression reporting: Shows character/token savings per round

**Usage:**
```bash
# Default: Compression enabled (recommended)
python3 gen_math.py --agents 7 --rounds 3

# Disable compression (for comparison or debugging)
python3 gen_math.py --agents 7 --rounds 3 --no-compress-context
```

**Performance Impact:**
- 6 agents × 500 tokens/response = 3,000 tokens → **30 tokens** (99% reduction)
- Enables 7 agents × 3 rounds without context overflow
- Minimal accuracy impact (agents still see all answers, just not full reasoning)

**Status:** Implemented for math task (proof of concept). Can be extended to GSM, biography, MMLU if needed.

## vLLM Auto-Configuration (NEW)

### Status: IN PROGRESS

GPU detection and automatic parameter configuration for vLLM backend:

**Completed:**
- `utils/gpu_config.py`: GPU detection, optimal config generation
- `experiments/download_models.py`: Auto-config for model downloads
- Functions: `detect_vllm_gpus()`, `get_vllm_optimal_config()`, `print_gpu_summary()`

**In Progress:**
- Extending auto-config to content generation (`utils/model_cache.py`, `utils/llm_wrapper.py`)

**Use Cases:**
- `'download'`: Conservative (TP=1, gpu_mem=0.7, max_len=2048, enforce_eager)
- `'production'`: Aggressive (TP=all GPUs, gpu_mem=0.9, max_len=8192, chunked_prefill)
- `'debate'`: Balanced (TP=1, gpu_mem=0.8, max_len=4096)

**Example Detection:**
```python
from utils import detect_vllm_gpus, get_vllm_optimal_config

gpu_info = detect_vllm_gpus()
# Returns: {'count': 2, 'models': ['RTX 3090'], 'vram_per_gpu_gb': 24.0,
#           'total_vram_gb': 48.0, 'has_nvlink': True, 'available_vram_gb': 47.2}

config = get_vllm_optimal_config("meta-llama/Llama-3.2-3B-Instruct", use_case='production', gpu_info=gpu_info)
# Returns: {'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.9,
#           'max_model_len': 8192, 'enable_chunked_prefill': True, ...}
```

**Control:**
- **Opt-out:** Set `VLLM_DISABLE_AUTO_CONFIG=1` to disable
- **Override:** Pass `override_params` dict to `get_vllm_optimal_config()`

## Known Issues

- **Debugging statements:** `pdb.set_trace()` in eval_gsm.py:143, eval_mmlu.py:138 (safe, only triggers on parse errors)
- **vLLM memory cleanup:** Use `model_cache.shutdown()` to avoid stuck processes
- **Single GPU OOM:** For RTX 3090 (24GB), use `experiments/linux_single/` which dynamically filters models from config.yaml based on VRAM requirements and includes pre-flight memory checks. Large models (7B-8B) may require `MAX_PARALLEL=1`. See `experiments/linux_single/README.md` for details.
- **Hardcoded model mapping (Technical Debt):** `utils/persona_loader.py` contains hardcoded `VLLM_TO_CSV_MODEL_MAP` dictionary (lines 23-38) that maps vLLM aliases to CSV model names. This violates the "single source of truth" design principle - model mappings should be dynamically derived from config.yaml instead. When adding new models, this dictionary must be manually updated alongside config.yaml, creating maintenance burden and potential for sync errors. **Future improvement:** Generate this mapping automatically from config.yaml model paths.

## Dependencies

**Mac:** mlx==0.29.2, mlx-lm==0.28.1, numpy, pandas, tqdm
**HPC:** vllm==0.11.0, torch==2.8.0, transformers==4.57.1, cuda==12.4
**Both:** openai==0.27.6 (eval only), python-dotenv>=0.19.0

## Key Implementation Files

**Cognitive Diversity:**
- `utils/helpers.py`: `generate_answer()` (per-agent model/params), `get_model_descriptor()`, `get_temperature_descriptor()`, `get_persona_descriptor()`
- `utils/config.py`: `resolve_model_name()`, `resolve_persona()`, `MODEL_ALIASES`
- All generation scripts: CLI args, validation, filename generation

**vLLM Integration:**
- `utils/llm_wrapper.py`: Backend detection, model loading
- `utils/model_cache.py`: Thread-safe caching, cleanup
- `utils/gpu_config.py`: GPU detection, optimal config
- `utils/cuda_cleanup.py`: Memory management utilities
