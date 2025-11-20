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
- **Evaluation:** Inline evaluation during generation, extracts from `\boxed{answer}` format
- **Files:** `gen_gsm.py` (generation with inline eval), `eval_gsm.py` (legacy, for re-evaluating old files)

#### 3. **Biography** (`./tasks/biography/`)
- **Task:** Generate bullet-point biographies of computer scientists
- **Configuration:** 3 agents, 2 rounds, 40 people
- **Evaluation:** Manual or automated fact-checking against Wikipedia/sources
- **Files:** `gen_conversation.py` (generation), `eval_conversation.py` (evaluation)
- **Data:** `data/biography/article.json` with ground truth biographies

#### 4. **MMLU** (`./tasks/mmlu/`)
- **Task:** Multiple-choice questions across academic subjects (A/B/C/D)
- **Dataset:** [MMLU benchmark](https://github.com/hendrycks/test) (included in `data/mmlu/`)
- **Configuration:** 3 agents, 2 rounds
- **Evaluation:** Inline evaluation during generation, extracts letter answers
- **Files:** `gen_mmlu.py` (generation with inline eval), `eval_mmlu.py` (standalone eval with detailed parsing)

### Key Parameters
- **agents:** Number of independent LLM agents participating in debate
- **rounds:** Number of debate iterations (each agent sees others' responses `rounds-1` times)
- **n:** Number of completions per API call (currently 1)
- **agent-models:** Optional list of models (one per agent) for model diversity
- **agent-temperatures:** Optional list of temperatures (one per agent) for parameter diversity
- **agent-personas:** Optional list of persona callsigns (one per agent) for cognitive diversity

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

## Cognitive Diversity Experiments

A key research focus of this project is exploring how cognitive diversity among agents affects multiagent debate performance. We implement three types of diversity:

### 1. Model Diversity
Different agents use different language models, providing diversity through varied architectures, training data, and capabilities.

**Implementation:** Use `--agent-models` to specify different models per agent.

```bash
python3 gen_gsm.py \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agents 3 \
  --rounds 2
```

**Technical Details:**
- The `generate_answer()` function in `utils/helpers.py` handles per-agent model selection
- Each agent gets assigned `agent_models[agent_id]` instead of the default model
- Output filename includes all models: `gsm_deepseek+llama32-3b+qwen25-7b_agents3_rounds2.json`

### 2. Parameter Diversity (Temperature)
Agents use the same model but with different sampling temperatures, creating diversity through varied generation randomness.

**Implementation:** Use `--agent-temperatures` to specify different temperatures per agent.

```bash
python3 gen_gsm.py \
  --model vllm-llama32-3b \
  --agents 3 \
  --rounds 2 \
  --agent-temperatures 0.7 1.0 1.3
```

**Technical Details:**
- Temperature controls randomness in token sampling (0.0 = deterministic, >1.0 = more random)
- Agent 1 (temp=0.7): Conservative, focused on high-probability responses
- Agent 2 (temp=1.0): Balanced, default behavior
- Agent 3 (temp=1.3): Creative, explores lower-probability options
- The `generate_answer()` function uses `agent_gen_params[agent_id]` for per-agent parameters
- Output filename includes temperatures: `gsm_Llama-3.2-3B_temp0.7+1.0+1.3_agents3_rounds2.json`
- Helper function `get_temperature_descriptor()` in `utils/helpers.py` creates filename component

**Temperature Guidelines:**
- **0.0-0.5:** Highly deterministic, minimal diversity
- **0.7:** Good for precision tasks, slight exploration
- **1.0:** Default, balanced exploration/exploitation
- **1.3-1.5:** Increased diversity, good for creative tasks
- **>1.5:** High randomness, may reduce coherence

### 3. Persona Diversity (Cognitive Style)
Agents use different personas (reasoning styles/perspectives) defined via system prompts, creating diversity through varied cognitive approaches.

**Implementation:** Use `--agent-personas` to specify different persona callsigns per agent.

```bash
python3 gen_gsm.py \
  --model vllm-llama32-3b \
  --agents 3 \
  --rounds 2 \
  --agent-personas skeptic analyst intuitive
```

**Technical Details:**
- **100 predefined personas** available in `config.yaml` with callsign aliases
  - **50 v1 personas:** Moderate, professional styles (analyst, skeptic, innovator, etc.)
  - **50 v2 personas:** Extreme, creative styles (cryptographer, zenmaster, baroque, etc.)
- Each persona is injected as a system message: `{"role": "system", "content": "You are {persona}."}`
- The `generate_answer()` function prepends the system message before the first user message
- Persona resolution via `resolve_persona()` supports both callsigns and full descriptions
- Output filename includes persona descriptor: `gsm_Llama-3.2-3B_persona_skeptic+analyst+intuitive_agents3_rounds2.json`
- Helper function `get_persona_descriptor()` extracts short names for filenames

**Example Personas (v1 - Moderate):**
- `skeptic`: "a skeptical questioner who challenges assumptions rigorously"
- `analyst`: "a meticulous analyst who examines every detail carefully"
- `intuitive`: "an intuitive thinker who relies on pattern recognition and gut feelings"
- `pragmatic`: "a pragmatic problem-solver focused on practical solutions"
- `innovator`: "a creative innovator who generates novel solutions and perspectives"

**Example Personas (v2 - Extreme):**
- `cryptographer`: "a nihilistic cryptographer who only trusts solutions verifiable by zero-knowledge proofs"
- `zenmaster`: "a Zen master who communicates only through non-sequiturs, koans, and minimal, cryptic statements"
- `baroque`: "a Baroque music theorist fixated on harmonic counterpoint and structural symmetry"
- `anarchist`: "a radical anarchist who views all imposed structures and hierarchies as fundamentally flawed"
- `grandmaster`: "an expert chess grandmaster who analyzes all moves based on look-ahead, board state, and counterplay"

**Persona Guidelines:**
- **v1 personas:** Best for standard reasoning tasks, professional diversity
- **v2 personas:** Best for creative exploration, maximum cognitive diversity
- **Mix v1+v2:** Combine moderate and extreme personas for hybrid diversity
- **Full descriptions:** Can also pass full persona descriptions instead of callsigns

### 4. Combined Diversity (Model + Temperature + Persona)
Maximum cognitive diversity by combining model, parameter, and persona variation.

```bash
python3 gen_gsm.py \
  --agents 3 \
  --rounds 2 \
  --agent-models vllm-llama32-3b vllm-qwen25-7b vllm-deepseek \
  --agent-temperatures 0.7 1.0 1.3 \
  --agent-personas skeptic analyst intuitive
```

**Output filename:** `gsm_deepseek+llama32-3b+qwen25-7b_temp0.7+1.0+1.3_persona_skeptic+analyst+intuitive_agents3_rounds2.json`

This creates the richest possible cognitive diversity:
- **Model diversity:** Different architectures and training (DeepSeek 1.5B, Llama 3B, Qwen 7B)
- **Temperature diversity:** Different sampling strategies (conservative, balanced, creative)
- **Persona diversity:** Different reasoning styles (skeptical, analytical, intuitive)

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

#### HPC / Windows / Linux: vLLM or Ollama ✅ **FULLY IMPLEMENTED**

**The codebase now supports three backends with automatic detection:**
- **MLX** (Mac Apple Silicon) - Auto-detected on macOS ARM64
- **vLLM** (Linux/HPC with NVIDIA GPUs) - Optimized for high-throughput inference
- **Ollama** (Cross-platform) - Simple server-based approach

All backends support:
- ✅ Chat template formatting
- ✅ Reasoning model handling (VibeThinker, DeepSeek-R1)
- ✅ `<think>` tag extraction
- ✅ Configurable token limits (40960 for reasoning models)

**vLLM Setup (Linux/HPC with NVIDIA GPUs):**
```bash
# 1. Uncomment vLLM dependencies in requirements.txt
# 2. Install
pip install vllm torch transformers

# 3. Backend auto-detects vLLM (MLX import fails on Linux)
# 4. Run experiments
python3 gen_math.py --model vllm-llama32-3b --agents 2 --rounds 3
```

**Ollama Setup (Cross-platform):**
```bash
# Windows: Download from https://ollama.com/download/windows
# Linux/HPC:
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (GGUF format)
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:1.5b

# Run experiments
python3 gen_math.py --model ollama-llama32 --agents 2 --rounds 3
```

**Model Equivalents Across Platforms:**

| Size | Mac (MLX) | Linux (vLLM) | Any (Ollama) |
|------|-----------|--------------|--------------|
| 1.5B | `deepseek` | `vllm-deepseek` | `ollama-deepseek` |
| 3B | `llama32-3b` | `vllm-llama32-3b` | `ollama-llama32` |
| 7B | `qwen25-7b` | `vllm-qwen25-7b` | `ollama-qwen25-7b` |
| 8B | `llama31-8b` | `vllm-llama31-8b` | `ollama-llama31-8b` |
| 14B | `qwen25-14b` | `vllm-qwen25-14b` | `ollama-qwen25-14b` |

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

### ✅ Phase 1: Local Development (Mac M4 Pro) - COMPLETED
- [x] Cloned original codebase
- [x] Identified available MLX-optimized models (8 models ready including VibeThinker)
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
- [x] Implemented per-agent model diversity support (Condition 3: Model Diversity)
- [x] Merged GSM evaluation into generation for inline feedback
- [x] Added VibeThinker reasoning model with special handling
- [x] Implemented reasoning model support (<think> tag extraction, 40960 token limit)
- [x] Created results aggregation script (scripts/aggregate_results.py)
- [x] Created plotting scripts (plot_by_model.py, plot_by_task.py)
- [x] Math task experiments completed: 13 configurations across 2 models
  - DeepSeek 1.5B: 9 configs (1-4 agents, 3-7 rounds) → 26-37% accuracy
  - Llama 3.1 8B: 4 configs (1-5 agents, 3 rounds) → 85-97% accuracy
- [x] Demonstrated multiagent debate benefit (Llama 8B: 85% solo → 97% with 2-3 agents)
- [x] Implemented parameter diversity (temperature variation per agent)
  - Added `--agent-temperatures` CLI argument to all generation scripts
  - Created `get_temperature_descriptor()` helper function
  - Updated filename generation to include temperature descriptor
  - Supports combined model + temperature diversity
- [x] Refactored evaluation architecture for consistency and reusability
  - Created generic `compute_accuracy()` in utils/helpers.py (takes parse_fn as parameter)
  - All tasks now use shared accuracy computation with task-specific parsing
  - Added inline evaluation to gen_mmlu.py (consistent with math/GSM)
  - Completely rewrote eval_mmlu.py with proper MMLU answer parsing (A/B/C/D)
  - Fixed critical bugs in original eval_mmlu.py (was using wrong parsers for letters)

### ✅ Phase 2: Linux/HPC GPU Support - COMPLETED (Nov 12, 2025)
- [x] **vLLM backend tested and validated on Ubuntu 22.04 with dual RTX 3090**
  - Test environment: 2x RTX 3090 (48GB VRAM), 128GB RAM, CUDA 12.4
  - Successfully loaded and ran Llama-3.2-3B-Instruct
  - Performance: ~100 tokens/sec output, 6.02 GiB VRAM usage
  - Math task test: 5 problems, 2 agents, 2 rounds → 60% accuracy
  - Model download and torch.compile work correctly
- [x] **Fixed MODEL_ALIASES in utils/config.py**
  - Issue: vLLM model aliases not in hardcoded MODEL_ALIASES dict
  - Solution: Added all 7 vLLM models (deepseek, vibethinker, smallthinker, llama32-3b, qwen25-7b, llama31-8b, qwen25-14b)
  - Also added 5 Ollama model aliases for cross-platform support
- [x] **Fixed vLLM hanging issue**
  - Issue: Scripts complete but don't return to shell (vLLM engine processes don't shut down)
  - Root cause: vLLM v0.11.0 creates background `EngineCore_DP0` processes
  - Solution: Added `shutdown()` method to ModelCache (utils/model_cache.py:101-114)
  - Updated all 4 generation scripts to call `model_cache.shutdown()` at end
  - Files updated: gen_math.py, gen_gsm.py, gen_conversation.py, gen_mmlu.py
- [x] **Fixed filename generation for model diversity experiments**
  - Issue: Output filenames always used default model name, even with `--agent-models`
  - Solution: Created `get_model_descriptor()` function (utils/helpers.py:179-234)
  - Logic: Single model → short name, Multiple models → "model1+model2+model3"
  - General approach: Strips prefixes/suffixes without hardcoding model names
  - Updated all 4 generation scripts to use `get_model_descriptor(model_name, agent_models)`
- [x] **Created comprehensive requirements files**
  - `requirements.txt`: Cross-platform with conditional MLX install for macOS ARM64
  - `requirements_hpc.txt`: Linux/HPC specific with vLLM, PyTorch, CUDA
  - Documented verified versions (vLLM 0.11.0, PyTorch 2.8.0, transformers 4.57.1)
- [x] **Created automated benchmark scripts**
  - `scripts/benchmark_gsm_baseline.sh`: Test all 7 models on 100 GSM problems
  - `scripts/benchmark_math_baseline.sh`: Test all 7 models on 100 math problems
  - Features: Same problems via random_seed=0, error handling, progress tracking
  - `scripts/README_BENCHMARKS.md`: Complete benchmark documentation
- [x] **Updated documentation**
  - README.md: Comprehensive team-facing guide with platform-specific instructions
  - CLAUDE.md: This document with technical implementation details

### 🔄 Phase 3: Current Experiments - IN PROGRESS
- [x] Created baseline benchmark infrastructure
- [ ] Complete GSM baseline across all 7 models (~2 hours on dual RTX 3090)
- [ ] Complete math baseline across all 7 models (~1 hour on dual RTX 3090)
- [ ] Test Ollama backend as alternative
- [ ] Run multiagent debate experiments (2-5 agents)
- [ ] Test model diversity (different models per agent)
- [ ] Test GSM, biography, MMLU tasks with reasoning models
- [ ] Compare debate effectiveness across model sizes and families

### 📋 Phase 4: Analysis & Paper - PLANNED
- [ ] Analyze cognitive diversity metrics
- [ ] Run parameter diversity experiments (temperature variation across agents)
- [ ] Test prompt diversity (different system prompts)
- [ ] Large-scale experiments (1000+ problems per task)
- [ ] Statistical analysis of debate benefits
- [ ] Write paper and prepare visualizations

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
├── data/                   # Datasets (committed to repo)
│   ├── biography/          # Ground truth biographies
│   │   └── article.json
│   ├── gsm8k/              # GSM8K dataset
│   │   ├── train.jsonl
│   │   ├── test.jsonl
│   │   └── ...
│   └── mmlu/               # MMLU benchmark
│       └── *_test.csv
│
├── tasks/                  # Task implementations
│   ├── math/               # Arithmetic problems
│   │   └── gen_math.py     # Generation with inline evaluation
│   ├── gsm/                # Grade school math
│   │   ├── gen_gsm.py      # Generation with inline evaluation
│   │   └── eval_gsm.py     # Standalone evaluation (legacy)
│   ├── biography/          # Computer scientist biographies
│   │   ├── gen_conversation.py
│   │   └── eval_conversation.py
│   └── mmlu/               # MMLU benchmark
│       ├── gen_mmlu.py     # Generation with inline evaluation
│       └── eval_mmlu.py    # Standalone eval with debug mode
│
├── experiments/            # HPC experiment infrastructure
│   ├── download_models.py  # Pre-cache model tokenizers/configs for vLLM
│   ├── hpc_test.sh         # Quick sanity check (all 4 tasks, 2 agents, 2 rounds)
│   ├── run_math_experiments.sh       # 220 math baseline experiments
│   ├── run_gsm_experiments.sh        # 220 GSM baseline experiments
│   ├── run_biography_experiments.sh  # 220 biography baseline experiments
│   ├── run_mmlu_experiments.sh       # 220 MMLU baseline experiments
│   └── README.md           # Experiment workflow documentation
│
├── utils/                  # Shared utilities
│   ├── llm_wrapper.py      # Multi-backend LLM interface (MLX/vLLM/Ollama)
│   ├── config.py           # Configuration management
│   ├── model_cache.py      # Model loading/caching with cleanup
│   ├── helpers.py          # Shared functions (compute_accuracy, diversity metrics)
│   └── ORIGINAL_STUDY_PARAMETERS.md  # Original paper parameters
│
├── scripts/                # Analysis and benchmarking
│   ├── aggregate_results.py          # Aggregate experiment results
│   ├── plot_by_model.py              # Generate per-model plots
│   ├── plot_by_task.py               # Generate comparison plots
│   ├── benchmark_gsm_baseline.sh     # GSM baseline benchmark
│   ├── benchmark_math_baseline.sh    # Math baseline benchmark
│   ├── README_BENCHMARKS.md          # Benchmark documentation
│   └── test_model_output.py          # Model testing utilities
│
├── results/                # Experiment outputs
│   ├── baseline/           # Baseline experiment results (experiments/)
│   │   ├── math/           # Math task baseline results
│   │   ├── gsm/            # GSM task baseline results
│   │   ├── biography/      # Biography task baseline results
│   │   └── mmlu/           # MMLU task baseline results
│   ├── summary.p           # Aggregated results DataFrame (tracked in git)
│   └── summary.csv         # Human-readable summary (tracked in git)
│
├── personas/               # Cognitive diversity research
│   ├── diversity_optimization_2821r.ipynb      # Jupyter notebook
│   ├── diversity_optimization_2821r_mlx.py     # Python script version
│   ├── embedding_search.py                     # Embedding analysis
│   ├── persona_v1_data.txt & persona_v1_results.txt
│   └── persona_v2_data.txt & persona_v2_results.txt
│
├── plots/                  # Generated visualizations (gitignored)
│   └── *.png               # Result plots, t-SNE visualizations
│
├── text/                   # Project documentation
│   └── abstract.txt        # Research abstract
│
├── legacy/                 # Deprecated/unused code (not actively maintained)
│   └── eval_gsm.py         # Standalone GSM eval (superseded by inline eval)
│
├── config.yaml             # Centralized configuration
├── requirements.txt        # Cross-platform dependencies
├── requirements_hpc.txt    # Linux/HPC dependencies (vLLM, PyTorch, CUDA)
├── README.md               # User-facing documentation
└── CLAUDE.md               # This file (internal technical documentation)
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
- ✅ **Cognitive diversity support** - Three types of diversity fully implemented:
  - **Model diversity:** `--agent-models` allows different models per agent
  - **Temperature diversity:** `--agent-temperatures` allows different sampling temperatures per agent
  - **Persona diversity:** `--agent-personas` allows different reasoning styles/perspectives per agent (NEW!)
  - All three features work together for maximum cognitive diversity
  - Filenames automatically include diversity descriptors (e.g., `gsm_model1+model2_temp0.7+1.0_persona_skeptic+analyst_agents2_rounds2.json`)
- ✅ **100 predefined personas** - 50 moderate (v1) + 50 extreme (v2) personas with unique callsigns

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

#### **Mac M4 Pro (Development)**
- **Backend:** MLX-LM (Apple Silicon optimized)
- **Models location:** `/Users/leonardo/.cache/huggingface/hub`
- **Models ready:** 7 MLX-optimized models (1.5B-14B parameters)
- **Model wrapper:** Handles chat templates automatically
- **Performance:** 40-80 tokens/sec (3B), 20-40 tokens/sec (7-8B)
- **IMPORTANT:** Use `python3` and `pip3` commands (NOT `python` or `pip`)
  - Example: `python3 gen_math.py` or `pip3 install -r requirements.txt`
  - The system does not have `python` symlinked to `python3`

#### **Ubuntu 22.04 with NVIDIA GPUs (HPC - Now Tested!)**
- **Backend:** vLLM 0.11.0 (auto-detected)
- **Hardware:** 2x RTX 3090 (24GB each = 48GB VRAM), 128GB RAM
- **CUDA:** 12.4 (auto-installed via PyTorch 2.8.0)
- **Models:** Downloads from HuggingFace on first use, cached in `~/.cache/huggingface/`
- **Performance:** ~100 tokens/sec (3B models)
- **Memory usage:**
  - 3B models: ~6GB VRAM
  - 7-8B models: ~14-16GB VRAM
  - 14B models: ~28GB VRAM (tight fit on single RTX 3090)
- **First run overhead:** +5-10 minutes for model download and torch.compile
- **Installation:** `pip3 install -r requirements_hpc.txt`
- **Known working models:**
  - `vllm-llama32-3b` (meta-llama/Llama-3.2-3B-Instruct) ✅ Tested
  - `vllm-deepseek` (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) - Ready
  - `vllm-vibethinker-1.5b` (WeiboAI/VibeThinker-1.5B) - Ready
  - `vllm-smallthinker-3b` (PowerInfer/SmallThinker-3B-Preview) - Ready
  - `vllm-qwen25-7b`, `vllm-llama31-8b`, `vllm-qwen25-14b` - Ready

#### **Cross-Platform (Ollama - Ready but Untested)**
- **Backend:** Ollama (GGUF models)
- **Installation:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Models:** Download with `ollama pull <model>`
- **Use case:** Teams without dedicated GPUs or on Windows
- **Status:** Code ready, not yet tested on this project

### Performance Characteristics
- **MLX on M4 Pro:** 40-80 tokens/sec (3B), 20-40 tokens/sec (7-8B)
- **vLLM on dual RTX 3090:** ~100 tokens/sec (3B), ~50 tokens/sec (7-8B)
- **Memory usage:** Model caching reduces load time but increases RAM usage
- **Experiment duration:**
  - Math task (100 problems, 1 agent, vLLM 3B): ~5-6 minutes
  - GSM task (100 problems, 1 agent, vLLM 3B): ~10-12 minutes
  - Mac M4 Pro (DeepSeek 1.5B, 3 agents, 3 rounds): ~20-40 minutes
- **Multiagent overhead:** Slower than GPT-3.5 API but zero cost after setup

## Key Technical Fixes (Nov 12, 2025)

### 1. vLLM Hanging Issue
**Problem:** Scripts complete successfully but never return to shell prompt.

**Root Cause:** vLLM v0.11.0 creates background engine process `EngineCore_DP0` that doesn't automatically terminate when Python script ends.

**Solution:**
```python
# utils/model_cache.py:101-114
def shutdown(self):
    """Shutdown all cached models properly (especially vLLM engines)."""
    with self._cache_lock:
        for cache_key, (model, tokenizer) in self._cache.items():
            if "vllm:" in cache_key:
                try:
                    # vLLM LLM objects need explicit destruction
                    if hasattr(model, '__del__'):
                        del model
                except Exception as e:
                    print(f"[ModelCache] Warning: Failed to shutdown {cache_key}: {e}")
        self._cache.clear()
```

All generation scripts now call `model_cache.shutdown()` at the end.

**Files Modified:**
- `utils/model_cache.py` - Added shutdown() method
- `tasks/math/gen_math.py` - Added cleanup call
- `tasks/gsm/gen_gsm.py` - Added cleanup call
- `tasks/biography/gen_conversation.py` - Added cleanup call
- `tasks/mmlu/gen_mmlu.py` - Added cleanup call

### 2. Model Alias Resolution
**Problem:** vLLM model aliases like `vllm-llama32-3b` were not being resolved to actual HuggingFace paths.

**Root Cause:** Hardcoded `MODEL_ALIASES` dict in `utils/config.py` didn't include vLLM models that were only in `config.yaml`.

**Solution:** Added all vLLM and Ollama model aliases to the hardcoded dict in `utils/config.py:54-78`:
```python
MODEL_ALIASES = {
    # MLX models (Mac)
    "deepseek": "valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16",
    ...
    # vLLM models (Linux/HPC)
    "vllm-deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "vllm-llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
    ...
    # Ollama models (Cross-platform)
    "ollama-deepseek": "deepseek-r1:1.5b",
    ...
}
```

### 3. Filename Generation for Model Diversity
**Problem:** Output filenames always used default model name, even when using `--agent-models` with different models.

**Root Cause:** Filename generation used `model_name.split('/')[-1]` which always referred to the fallback model, not the actual models being used.

**Solution:** Created general `get_model_descriptor()` function in `utils/helpers.py:179-234`:

```python
def get_model_descriptor(model_name: str, agent_models: Optional[List[str]] = None) -> str:
    """
    Generate descriptive model name for output filenames.

    Two cases:
    1. Single model (all agents use same) → extract short name
    2. Multiple distinct models → create amalgamated name like "model1+model2+model3"
    """
    def _extract_short_name(full_path: str) -> str:
        """General approach: strips prefixes/suffixes without hardcoding."""
        name = full_path.split('/')[-1]
        # Remove backend prefixes
        for prefix in ["vllm-", "ollama-"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        # Remove common suffixes
        for suffix in ["-Instruct", "-mlx-fp16", etc.]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    if agent_models:
        unique_models = list(set(agent_models))
        if len(unique_models) == 1:
            return _extract_short_name(unique_models[0])
        else:
            short_names = sorted([_extract_short_name(m) for m in unique_models])
            return "+".join(short_names)
    else:
        return _extract_short_name(model_name)
```

**Example outputs:**
- Single model: `gsm_Llama-3.2-3B_agents3_rounds2.json`
- Multiple models: `gsm_DeepSeek-R1+Llama-3.2-3B+Qwen2.5-7B_agents3_rounds2.json`

**Files Modified:**
- `utils/helpers.py` - Added `get_model_descriptor()` function
- `utils/__init__.py` - Exported new function
- All 4 generation scripts - Now use `get_model_descriptor(model_name, agent_models)`

### Testing Recommendations
- Test each model family (Llama, Qwen, DeepSeek) at least once per task
- Chat template formatting varies by model - verify output quality
- Watch for truncated responses with long contexts (especially biography task)
- Compare single-agent vs multiagent to quantify debate benefit

### Key Implementation Files for Cognitive Diversity

**Parameter Diversity (Temperature) Implementation:**
1. **`utils/helpers.py`** - Core functions
   - `generate_answer()` (lines 12-50): Handles per-agent model and parameter selection
   - `get_temperature_descriptor()` (lines 237-265): Creates filename component from temperature list

2. **All generation scripts** - CLI and usage
   - Argument parsing: `parser.add_argument("--agent-temperatures", type=float, nargs="+")`
   - Parameter validation: Ensures number of temperatures matches number of agents
   - Per-agent param creation: Builds list of generation param dicts with different temperatures
   - Filename generation: Uses `get_temperature_descriptor()` to create descriptive filename

3. **Files modified:**
   - `tasks/math/gen_math.py`
   - `tasks/gsm/gen_gsm.py`
   - `tasks/biography/gen_conversation.py`
   - `tasks/mmlu/gen_mmlu.py`
   - `utils/helpers.py`
   - `utils/__init__.py` (exports `get_temperature_descriptor`)

**Model Diversity Implementation:**
1. **`utils/helpers.py`**
   - `generate_answer()`: Handles per-agent model selection via `agent_models[agent_id]`
   - `get_model_descriptor()` (lines 179-234): Creates filename component from model list

2. **`utils/config.py`**
   - `resolve_model_name()`: Maps aliases to full HuggingFace paths
   - `MODEL_ALIASES` dict: Platform-specific model aliases (MLX, vLLM, Ollama)
