# Beyond Symmetric Agents: Cognitive Diversity and Multiagent Debate in Small Language Models

**Course Project:** Topics in Machine Learning: Compositional AI Systems
COMPSCI 2821R, Fall 2025, Harvard School of Engineering and Applied Sciences

**Contributors:**
- Leonardo Ferreira <leonardo.ferreira@childrens.harvard.edu>
- Vennela Jonnala <vjonnala@college.harvard.edu>
- Gardenia Liu <gardenialiu@college.harvard.edu>
- Kaden Zheng <kadenzheng@college.harvard.edu>

## Project Overview

### Original Research
This codebase implements the paper ["Improving Factuality and Reasoning in Language Models through Multiagent Debate"](https://arxiv.org/abs/2305.14325) by Du et al. (2023). The core hypothesis is that multiple language model agents debating a problem over several rounds can improve reasoning accuracy and factual correctness compared to single-shot inference.

**Original Paper:** [Project Page](https://composable-models.github.io/llm_debate/) | [ArXiv](https://arxiv.org/abs/2305.14325)

### Our Adaptation
**Goal:** Replicate the multiagent debate experiments using locally-hosted Small Language Models (SLMs) instead of OpenAI's GPT-3.5-turbo-0301.

**Motivation:**
- Reduce API costs and dependencies
- Enable experimentation with smaller, open-source models
- Explore whether debate mechanisms benefit SLMs similarly to larger models
- Support offline/local research workflows

## Experiment Structure

### How Multiagent Debate Works
1. **Initialization:** Multiple independent agents receive the same question/problem
2. **Round 1:** Each agent generates an initial response independently
3. **Subsequent Rounds:** Each agent sees other agents' responses and refines their answer
4. **Aggregation:** Final answers are collected (majority vote for math, evaluation for biography/factuality)

### Implemented Tasks

#### 1. **Math** (`./tasks/math/`)
- **Task:** Simple arithmetic expressions (e.g., `a+b*c+d-e*f`)
- **Configuration:** 2 agents, 3 rounds
- **Evaluation:** Automated comparison with ground truth
- **Files:** `gen_math.py` (generates and evaluates)

#### 2. **Grade School Math (GSM)** (`./tasks/gsm/`)
- **Task:** Multi-step word problems requiring arithmetic reasoning
- **Dataset:** [OpenAI GSM8K](https://github.com/openai/grade-school-math) (included in `data/gsm8k/`)
- **Configuration:** 3 agents, 2 rounds
- **Evaluation:** Extracts numerical answer from `\boxed{answer}` format
- **Files:** `gen_gsm.py` (generation), `eval_gsm.py` (evaluation)

#### 3. **Biography** (`./tasks/biography/`)
- **Task:** Generate bullet-point biographies of computer scientists
- **Configuration:** 3 agents, 2 rounds, 40 people
- **Evaluation:** Manual or automated fact-checking against Wikipedia/sources
- **Files:** `gen_conversation.py` (generation), `eval_conversation.py` (evaluation)
- **Data:** `data/biography/article.json` with ground truth biographies

#### 4. **MMLU** (`./tasks/mmlu/`)
- **Task:** Multiple-choice questions across academic subjects
- **Dataset:** [MMLU benchmark](https://github.com/hendrycks/test) (included in `data/mmlu/`)
- **Configuration:** Likely 3 agents, 2 rounds (inferred from pattern)
- **Files:** `gen_mmlu.py` (generation), `eval_mmlu.py` (evaluation)

### Key Parameters
- **agents:** Number of independent LLM agents participating in debate
- **rounds:** Number of debate iterations (each agent sees others' responses `rounds-1` times)
- **n:** Number of completions per API call (currently 1)

## Technical Implementation

### Completed Architecture
All generation scripts (`gen_*.py`) have been migrated to use MLX-LM via a custom wrapper:

```python
# Current implementation:
from utils import ChatCompletion

completion = ChatCompletion.create(
    model=model_name,  # MLX model path
    messages=agent_context,
    **generation_params
)
```

The `utils.ChatCompletion` wrapper (in `utils/llm_wrapper.py`) provides OpenAI-compatible interface to MLX-LM, handling:
- Model loading and caching
- Chat template formatting
- Token generation
- Response formatting matching OpenAI structure

### Migration Status

#### ✅ Completed Changes
1. ✅ Created OpenAI-compatible wrapper for MLX-LM (`utils/llm_wrapper.py`)
2. ✅ Updated all `gen_*.py` scripts to use mlx-lm wrapper
3. ✅ Implemented model caching system (`utils/model_cache.py`)
4. ✅ Added progress tracking with tqdm
5. ✅ Implemented configurable model selection via CLI and config.yaml

#### Message Format
All scripts continue to use OpenAI's chat format (now handled by wrapper):
```python
[
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "response"},
    {"role": "user", "content": "follow-up with other agents' responses"},
    ...
]
```

**Note:** Evaluation scripts (`eval_*.py`) still use OpenAI API for GPT-4 based fact-checking and evaluation (common pattern for assessing factuality in biography/MMLU tasks).

## Local LLM Framework Strategy

### Platform-Specific Approach

#### Mac M4 Pro (Local Development): MLX-LM
**Why MLX-LM?**
- Models already downloaded in MLX format (`/Users/leonardo/.cache/huggingface/hub`)
- Native Apple Silicon optimization (best performance on M4 Pro)
- Direct Python API, no server needed
- Already installed: `mlx-lm==0.28.1`, `mlx==0.29.2`
- Supports multiple simultaneous model instances
- Zero additional downloads required

**Installation:**
Already installed! Verify with:
```bash
python3 -m pip show mlx-lm
```

**Usage Example:**
```python
from mlx_lm import load, generate

# Load already-downloaded model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")

# Format chat messages
messages = [
    {"role": "user", "content": "What is 2+2?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate response
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
```

#### HPC / Windows / Linux: Ollama or vLLM
**For HPC with NVIDIA GPUs:**
- Use **Ollama** with GGUF models (simple, cross-platform)
- Or use **vLLM** with HuggingFace models (higher throughput)
- Models need to be downloaded again (MLX format won't work on NVIDIA)

**Ollama Installation (HPC/Windows):**
```bash
# Windows
# Download from https://ollama.com/download/windows

# Linux/HPC
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (GGUF format)
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

**vLLM Alternative (HPC only):**
```bash
pip install vllm
# Use standard HuggingFace models (not MLX versions)
```

## Available Models

### Local Mac M4 Pro (MLX-Optimized, Already Downloaded)
These models are ready to use via `mlx-lm` with zero additional downloads:

| Model | Path | Size | Best For | Instances (48GB) |
|-------|------|------|----------|------------------|
| **DeepSeek-R1-Distill-Qwen** | `valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16` | 1.5B | Reasoning, fast iterations | 8-10 |
| **Llama 3.2 Instruct** | `mlx-community/Llama-3.2-3B-Instruct` | 3B | Balanced performance | 5-8 |
| **SmallThinker** | `valuat/SmallThinker-3B-Preview-mlx-fp16` | 3B | Reasoning tasks | 5-8 |
| **Qwen2.5 7B Instruct** | `valuat/Qwen2.5-7B-Instruct-1M-mlx-fp16` | 7B | Strong math (1M context) | 3-5 |
| **Llama 3.1 8B Instruct** | `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | 8B | General purpose (quantized) | 3-5 |
| **Llama 3.1 8B Instruct** | `valuat/Meta-Llama-3.1-8B-Instruct-mlx-fp16` | 8B | General purpose (fp16) | 3-4 |
| **Qwen2.5 14B Instruct** | `valuat/Qwen2.5-14B-Instruct-1M-mlx-fp16` | 14B | Best performance (1M context) | 2-3 |

**Location:** `/Users/leonardo/.cache/huggingface/hub`

### HPC Models (To Download)
For HPC/Windows deployment, you'll need to download models in GGUF format (for Ollama) or standard HuggingFace format (for vLLM/transformers):

**Via Ollama:**
```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull deepseek-r1:1.5b
```

**Via HuggingFace (for vLLM/transformers):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download standard (non-MLX) versions
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

**Note:** MLX models only work on Apple Silicon. HPC with NVIDIA GPUs requires GGUF (Ollama) or standard PyTorch models (vLLM/transformers).

## Development Workflow

### Phase 1: Local Development (Mac M4 Pro with MLX)
1. Create utility wrapper for OpenAI API → mlx-lm translation
2. Test single-agent inference with `mlx-community/Llama-3.2-3B-Instruct`
3. Validate response format parsing and chat templates
4. Update one `gen_*.py` script as proof-of-concept (math task)

### Phase 2: Multiagent Adaptation
1. Update all `gen_*.py` scripts to use mlx-lm wrapper
2. Add configurable model selection (env var or command-line arg)
3. Implement proper error handling and retries
4. Add progress bars and timing metrics
5. Test with multiple simultaneous model instances

### Phase 3: Local Experimentation
1. Run baseline experiments (single agent, no debate) on all tasks
2. Run multiagent debate with various configurations
3. Compare SLM performance to original GPT-3.5 results
4. Test different model sizes (1.5B → 14B) and families

### Phase 4: HPC Deployment
1. Create abstraction layer supporting both mlx-lm (Mac) and Ollama/vLLM (HPC)
2. Set up Ollama on HPC and download GGUF models
3. Create SLURM/PBS job submission scripts
4. Run large-scale experiments with bigger models and more debate rounds
5. Deploy across team members' machines (Windows/Mac/Linux)

## Project Status
- [x] Cloned original codebase
- [x] Identified available MLX-optimized models (7 models ready)
- [x] Verified mlx-lm installation (v0.28.1)
- [x] Created project documentation (CLAUDE.md, updated README.md)
- [x] Set up GitHub repository
- [x] Downloaded and organized datasets (GSM8K, MMLU, biography)
- [x] Reorganized datasets into data/ directory
- [x] Reorganized task implementations into tasks/ directory
- [x] Created mlx-lm wrapper for OpenAI API compatibility (utils module)
- [x] Created config.yaml for centralized configuration
- [x] Created utils/helpers.py with shared functions
- [x] Adapted math task to use mlx-lm (tasks/math/gen_math.py)
- [x] Adapted GSM task to use mlx-lm (tasks/gsm/gen_gsm.py)
- [x] Adapted biography task to use mlx-lm (tasks/biography/gen_conversation.py)
- [x] Adapted MMLU task to use mlx-lm (tasks/mmlu/gen_mmlu.py)
- [x] Fixed MLX generate() API compatibility issues
- [x] All tasks follow consistent modular pattern
- [x] Created results aggregation script (scripts/aggregate_results.py)
- [x] Created plotting scripts (plot_by_model.py, plot_by_task.py)
- [x] Math task experiments completed: 13 configurations across 2 models
  - DeepSeek 1.5B: 9 configs (1-4 agents, 3-7 rounds) → 26-37% accuracy
  - Llama 3.1 8B: 4 configs (1-5 agents, 3 rounds) → 85-97% accuracy
- [x] Demonstrated multiagent debate benefit (Llama 8B: 85% solo → 97% with 2-3 agents)
- [ ] Test GSM, biography, MMLU tasks
- [ ] Run baseline experiments across all tasks
- [ ] Test additional models (Qwen 7B/14B, SmallThinker, Llama 3.2)
- [ ] Compare debate effectiveness across model sizes and families
- [ ] Set up HPC deployment (Ollama/vLLM)

## Configuration

### config.yaml
The project uses `config.yaml` at the repository root for centralized configuration:

**Key Sections:**
- **model**: Default model (alias or full path). Defaults to `"deepseek"` (1.5B, fastest)
- **generation**: Generation parameters matching original GPT-3.5 defaults (temp=1.0, max_tokens=null, top_p=1.0)
- **experiments**: Task-specific configs (agents, rounds, num_problems) from original study
- **models**: Aliases mapping short names → full HuggingFace paths
- **datasets**: Dataset paths relative to repo root

**Usage in Scripts:**
```python
from utils import load_config, resolve_model_name, get_experiment_config

config = load_config()                    # Load config.yaml
model = resolve_model_name(config["model"])  # "deepseek" → full path
gen_params = config["generation"]
exp_config = get_experiment_config("math")  # Get math task config
```

**Model Selection Priority:**
1. Command-line: `--model qwen25-7b`
2. config.yaml: `model: "deepseek"`
3. Fallback: `"deepseek"` (hardcoded default)

## Experimental Results

### Math Task (Arithmetic Reasoning)

**Dataset:** 100 arithmetic problems (order of operations: `a+b*c+d-e*f`)

**DeepSeek-R1-Distill-Qwen-1.5B (1.5B parameters):**
```
Config                  Accuracy
1 agent,  3 rounds  →   33%
1 agent,  5 rounds  →   29%
1 agent,  7 rounds  →   29%
2 agents, 3 rounds  →   30%
2 agents, 5 rounds  →   30%
2 agents, 7 rounds  →   26%
3 agents, 3 rounds  →   37% ⭐ Best
4 agents, 3 rounds  →   32%
4 agents, 5 rounds  →   28%
```
**Observation:** Modest improvement with debate (33% → 37%), but overall low accuracy. More agents/rounds don't consistently help.

**Meta-Llama-3.1-8B-Instruct (8B parameters, 8-bit quantized):**
```
Config                  Accuracy
1 agent,  3 rounds  →   85%
2 agents, 3 rounds  →   97% ⭐ Best
3 agents, 3 rounds  →   97% ⭐ Best
5 agents, 3 rounds  →   94%
```
**Observation:** Strong baseline (85%) with clear multiagent debate benefit (+12% improvement to 97%). Optimal at 2-3 agents. Diminishing returns at 5 agents.

**Key Insight:** Multiagent debate shows stronger benefit for more capable models. Llama 8B demonstrates the core hypothesis: debate helps models correct errors.

---

## Results Analysis Workflow

### 1. Run Experiments
```bash
# Math task
cd tasks/math
python3 gen_math.py [--model MODEL] [--agents N] [--rounds N]

# Other tasks similarly
cd ../gsm && python3 gen_gsm.py
cd ../biography && python3 gen_conversation.py
cd ../mmlu && python3 gen_mmlu.py
```

### 2. Aggregate Results
```bash
# Scans all task directories and creates summary
python3 scripts/aggregate_results.py
```
Output: `results/summary.p` (DataFrame) and `results/summary.csv` (human-readable)

### 3. Generate Plots
```bash
# Per (model, task) plots
python3 scripts/plot_by_model.py
# Output: plots/{model}_{task}.png

# Task comparison plots (all models)
python3 scripts/plot_by_task.py
# Output: plots/{task}_comparison.png
```

## File Structure
```
.
├── data/              # Datasets
│   ├── biography/     # Ground truth biographies
│   │   └── article.json
│   ├── gsm8k/         # GSM8K dataset
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   └── ...
│   └── mmlu/          # MMLU benchmark
│       └── *_test.csv
├── tasks/             # Task implementations
│   ├── math/          # Arithmetic problems
│   │   └── gen_math.py (includes inline evaluation)
│   ├── gsm/           # Grade school math
│   │   └── gen_gsm.py (includes inline evaluation)
│   ├── biography/     # Computer scientist biographies
│   │   ├── gen_conversation.py
│   │   └── eval_conversation.py
│   └── mmlu/          # MMLU benchmark
│       ├── gen_mmlu.py
│       └── eval_mmlu.py
├── utils/             # Shared utilities
│   ├── llm_wrapper.py      # OpenAI-compatible ChatCompletion
│   ├── config.py           # Configuration management
│   ├── model_cache.py      # Model caching
│   └── helpers.py          # Shared functions
├── scripts/           # Analysis scripts
│   ├── aggregate_results.py  # Aggregate experiment results
│   ├── plot_by_model.py      # Generate per-model plots
│   └── plot_by_task.py       # Generate comparison plots
├── results/           # Experiment results
│   ├── summary.p      # Aggregated results DataFrame
│   └── summary.csv    # Human-readable summary
├── plots/             # Generated visualizations
│   └── *.png          # Result plots (gitignored)
├── legacy/            # Deprecated/unused code (not actively maintained)
│   └── eval_gsm.py    # Standalone GSM eval (superseded by inline eval in gen_gsm.py)
├── config.yaml        # Centralized configuration
├── requirements.txt
├── README.md
└── CLAUDE.md          # This file
```

### Legacy Directory

The `legacy/` directory contains deprecated or superseded code that is no longer actively used but preserved for reference or potential future use. Files moved to legacy include:

- **eval_gsm.py** - Standalone evaluation script for GSM task. Superseded by inline evaluation in `gen_gsm.py` (which now evaluates during generation, similar to the math task). The standalone version remains available for:
  - Re-evaluating old JSON files without re-running generation
  - Testing alternative evaluation strategies
  - Debugging specific failure cases

**Note:** Code in the legacy directory is not actively maintained and may not be compatible with current configurations.

## Dependencies

### Mac M4 Pro (Current Setup)
**Already Installed:**
- `mlx==0.29.2` (Apple Silicon optimization)
- `mlx-lm==0.28.1` (LLM inference)
- `mlx-metal==0.29.2` (Metal acceleration)
- `numpy==1.22.4`
- `pandas==1.5.3`
- `tqdm==4.64.1`

**To Update:**
- `openai==0.27.6` → Will create compatibility wrapper instead of removing

### HPC/Windows Deployment (Future)
**To Install:**
- `ollama` (GGUF model serving) OR `vllm` (high-throughput inference)
- `transformers` (for non-MLX HuggingFace models)
- `torch` (PyTorch with CUDA support)

## Notes for Future Sessions

### Current Implementation Status
- ✅ **All generation scripts migrated to MLX-LM** - Uses `utils.ChatCompletion` wrapper
- ✅ **Model caching implemented** - Automatically caches loaded models between runs
- ✅ **Configuration centralized** - All settings in `config.yaml`, supports CLI overrides
- ✅ **Results tracking configured** - Summary files tracked in git, individual runs ignored

### Repository Management
- **Results tracking:** `results/summary.p` and `results/summary.csv` are tracked in git
- **Individual experiment files:** Stored in `tasks/{task}/math_*.p` format (gitignored for size)
- **Plots:** Generated visualizations in `plots/` (gitignored, regenerated from summary data)
- **Datasets:** All datasets (GSM8K, MMLU, biography) committed to repo for reproducibility

### Known Issues
- **Eval scripts contain debugging statements:** `pdb.set_trace()` present in:
  - `tasks/gsm/eval_gsm.py:143`
  - `tasks/mmlu/eval_mmlu.py:138`
  - Note: These are in evaluation scripts only (not generation scripts)
  - Trigger only on parsing errors; safe to leave for debugging

### Data Requirements
- **GSM:** `data/gsm8k/test.jsonl` (from https://github.com/openai/grade-school-math)
- **Biography:** `data/biography/article.json` (40 computer scientist biographies)
- **MMLU:** `data/mmlu/*_test.csv` (from https://github.com/hendrycks/test)
- **Math:** Generated on-the-fly (no external data needed)

### Platform-Specific Notes
- **Mac M4 Pro (Current):**
  - Uses MLX-LM with models at `/Users/leonardo/.cache/huggingface/hub`
  - 7 models ready (1.5B-14B parameters)
  - Model wrapper handles chat templates automatically
  - **IMPORTANT:** Use `python3` and `pip3` commands (NOT `python` or `pip`)
    - Example: `python3 gen_math.py` or `pip3 install -r requirements.txt`
    - The system does not have `python` symlinked to `python3`
- **HPC/Windows (Future):**
  - MLX models incompatible with NVIDIA GPUs
  - Will need GGUF (Ollama) or PyTorch (vLLM/transformers) versions
  - Cross-platform abstraction layer needed

### Performance Characteristics
- **MLX on M4 Pro:** 40-80 tokens/sec (3B), 20-40 tokens/sec (7-8B)
- **Memory usage:** Model caching reduces load time but increases RAM usage
- **Experiment duration:** Math task (100 problems, 3 agents, 3 rounds) ≈ 20-40 minutes with DeepSeek 1.5B
- **Multiagent overhead:** Slower than GPT-3.5 API but zero cost after setup

### Testing Recommendations
- Test each model family (Llama, Qwen, DeepSeek) at least once per task
- Chat template formatting varies by model - verify output quality
- Watch for truncated responses with long contexts (especially biography task)
- Compare single-agent vs multiagent to quantify debate benefit
