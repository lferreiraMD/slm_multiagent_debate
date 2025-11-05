# LLM Multiagent Debate - Local SLM Adaptation

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

### Current Architecture
```python
# All scripts use this pattern:
import openai

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=agent_context,
    n=1
)
```

### Migration to Local SLMs

#### Required Changes
1. Replace `openai.ChatCompletion.create()` calls with local inference
2. Update `requirements.txt` to include local LLM framework
3. Handle potential differences in response formatting
4. Adjust for slower inference times (add progress tracking, consider batching)
5. Update model name/path configuration

#### Message Format
All scripts use OpenAI's chat format:
```python
[
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "response"},
    {"role": "user", "content": "follow-up with other agents' responses"},
    ...
]
```

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
- [ ] Create mlx-lm wrapper for OpenAI API compatibility
- [ ] Test single-agent inference with Llama 3.2 3B
- [ ] Adapt math task to use mlx-lm
- [ ] Adapt GSM task to use mlx-lm
- [ ] Adapt biography task to use mlx-lm
- [ ] Adapt MMLU task to use mlx-lm
- [ ] Run baseline experiments (no debate)
- [ ] Run multiagent debate experiments
- [ ] Compare results across model sizes
- [ ] Set up HPC deployment (Ollama/vLLM)

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
│   │   └── gen_math.py
│   ├── gsm/           # Grade school math
│   │   ├── gen_gsm.py
│   │   └── eval_gsm.py
│   ├── biography/     # Computer scientist biographies
│   │   ├── gen_conversation.py
│   │   └── eval_conversation.py
│   └── mmlu/          # MMLU benchmark
│       ├── gen_mmlu.py
│       └── eval_mmlu.py
├── scripts/           # Utility scripts
│   └── download_datasets.sh
├── requirements.txt
├── README.md
└── CLAUDE.md          # This file
```

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

### Repository Management
- **TODO: Revisit results tracking** - Currently results/ is gitignored. Once we have baseline results from multiple models, we should commit them to the repo for comparability and reproducibility.
  - Format: results/{task}/{model}_a{agents}_r{rounds}.json
  - Include metadata: model, params, timestamp, performance metrics
  - Consider adding results/ to repo after initial experiments complete
- All datasets (GSM8K, MMLU, biography) are kept in repo for reproducibility

### Code Adaptation
- All `gen_*.py` files have hardcoded dataset paths that need updating
- Original code includes `pdb.set_trace()` debugging statements (remove these: gsm line 67-68, math line 149-150)
- Need to create `openai`-compatible wrapper for `mlx-lm` to minimize code changes
- Chat template handling varies by model - test with each model family

### Data Requirements
- GSM: Already included in `data/gsm8k/` (originally from https://github.com/openai/grade-school-math)
- Biography: Already included in `data/biography/article.json`
- MMLU: Already included in `data/mmlu/` (originally from https://github.com/hendrycks/test)
- Math: Generated on-the-fly (no external data needed)

### Platform-Specific Notes
- **Mac M4 Pro:** Use mlx-lm with models at `/Users/leonardo/.cache/huggingface/hub` (ready to use)
- **HPC/Windows:** MLX models won't work - need GGUF (Ollama) or standard PyTorch models
- **Cross-platform wrapper:** Create abstraction that detects platform and uses appropriate backend

### Performance Expectations
- MLX on M4 Pro: ~40-80 tokens/sec for 3B models, ~20-40 tokens/sec for 7-8B models
- Multiagent debate will be slower than original (GPT-3.5 API) but more cost-effective
- Consider caching model loads between agents to save memory
