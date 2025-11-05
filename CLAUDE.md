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

#### 1. **Math** (`./math/`)
- **Task:** Simple arithmetic expressions (e.g., `a+b*c+d-e*f`)
- **Configuration:** 2 agents, 3 rounds
- **Evaluation:** Automated comparison with ground truth
- **Files:** `gen_math.py` (generates and evaluates)

#### 2. **Grade School Math (GSM)** (`./gsm/`)
- **Task:** Multi-step word problems requiring arithmetic reasoning
- **Dataset:** [OpenAI GSM8K](https://github.com/openai/grade-school-math)
- **Configuration:** 3 agents, 2 rounds
- **Evaluation:** Extracts numerical answer from `\boxed{answer}` format
- **Files:** `gen_gsm.py` (generation), `eval_gsm.py` (evaluation)

#### 3. **Biography** (`./biography/`)
- **Task:** Generate bullet-point biographies of computer scientists
- **Configuration:** 3 agents, 2 rounds, 40 people
- **Evaluation:** Manual or automated fact-checking against Wikipedia/sources
- **Files:** `gen_conversation.py` (generation), `eval_conversation.py` (evaluation)
- **Data:** Requires `article.json` with ground truth biographies

#### 4. **MMLU** (`./mmlu/`)
- **Task:** Multiple-choice questions across academic subjects
- **Dataset:** [MMLU benchmark](https://github.com/hendrycks/test)
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

## Local LLM Framework: Ollama

### Why Ollama?
- Cross-platform (macOS M4 Pro, Windows PCs, Linux HPC)
- Simple model management (`ollama pull <model>`)
- Automatic hardware optimization (MLX on Apple Silicon, CUDA on NVIDIA)
- OpenAI-compatible API out of the box
- Supports running multiple instances simultaneously
- Easy to deploy on HPC with Docker/Singularity
- Simple and minimal dependencies

### Installation
```bash
# macOS (M4 Pro - uses MLX optimization automatically)
brew install ollama

# Windows
# Download from https://ollama.com/download/windows

# Linux/HPC
curl -fsSL https://ollama.com/install.sh | sh

# Python client
pip install ollama
```

### Usage Example
```python
import openai

# Point to local Ollama server
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # dummy key

completion = openai.ChatCompletion.create(
    model="llama3.2:3b",
    messages=agent_context,
    n=1
)
```

## Available Models

### Already Downloaded (MLX-Optimized)
These models are already in the local HuggingFace cache and optimized for M4 Pro:

1. **Llama 3.2 3B Instruct** (`mlx-community/Llama-3.2-3B-Instruct`)
   - Size: 3B parameters
   - Fast inference, good for initial testing
   - Balanced reasoning capabilities

2. **Llama 3.1 8B Instruct** (two versions available)
   - `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` (quantized)
   - `valuat/Meta-Llama-3.1-8B-Instruct-mlx-fp16` (full precision)
   - Strong general-purpose performance

3. **Qwen2.5 7B Instruct** (`valuat/Qwen2.5-7B-Instruct-1M-mlx-fp16`)
   - Size: 7B parameters
   - 1M token context window
   - Excellent math and reasoning performance

4. **Qwen2.5 14B Instruct** (`valuat/Qwen2.5-14B-Instruct-1M-mlx-fp16`)
   - Size: 14B parameters
   - 1M token context window
   - Best performance, may fit 2-3 simultaneous instances on 48GB

5. **DeepSeek-R1-Distill-Qwen 1.5B** (`valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16`)
   - Size: 1.5B parameters
   - Distilled from reasoning model
   - Very fast, good for high-throughput experiments

6. **SmallThinker 3B** (`valuat/SmallThinker-3B-Preview-mlx-fp16`)
   - Size: 3B parameters
   - Designed for reasoning tasks
   - Good candidate for debate experiments

### Recommended via Ollama (if needed)
If the HuggingFace models require additional setup, these are readily available through Ollama:
- `ollama pull llama3.2:3b`
- `ollama pull qwen2.5:3b`
- `ollama pull qwen2.5:7b`
- `ollama pull phi3:3.8b`

### Capacity on 48GB Mac M4 Pro
- **3B models:** 5-8 simultaneous instances
- **7-8B models:** 3-5 simultaneous instances
- **14B models:** 2-3 simultaneous instances
- Mix and match based on experiment needs

## Development Workflow

### Phase 1: Single-Model Testing
1. Install Ollama and test with one local model
2. Create utility wrapper for OpenAI → Ollama API translation
3. Test single-agent inference on each task type
4. Validate response format parsing

### Phase 2: Multiagent Adaptation
1. Update `gen_*.py` scripts to use local inference
2. Add configurable model selection
3. Implement proper error handling and retries
4. Add progress bars and timing metrics

### Phase 3: Experimentation
1. Run baseline experiments (single agent, no debate)
2. Run multiagent debate with various configurations
3. Compare SLM performance to original GPT-3.5 results
4. Test different model sizes and families

### Phase 4: Scaling to HPC
1. Containerize with Ollama + dependencies
2. Create job submission scripts for batch processing
3. Run larger-scale experiments with bigger models
4. Deploy across team members' machines (Windows/Mac/Linux)

## Project Status
- [x] Cloned original codebase
- [x] Identified available MLX-optimized models
- [ ] Set up Ollama infrastructure
- [ ] Create API wrapper/abstraction layer
- [ ] Test single-agent inference
- [ ] Adapt math task to local SLMs
- [ ] Adapt GSM task to local SLMs
- [ ] Adapt biography task to local SLMs
- [ ] Adapt MMLU task to local SLMs
- [ ] Run baseline comparisons
- [ ] Document results and model comparisons

## File Structure
```
.
├── math/          # Arithmetic problems
│   └── gen_math.py
├── gsm/           # Grade school math
│   ├── gen_gsm.py
│   └── eval_gsm.py
├── biography/     # Computer scientist biographies
│   ├── gen_conversation.py
│   └── eval_conversation.py
├── mmlu/          # MMLU benchmark
│   ├── gen_mmlu.py
│   └── eval_mmlu.py
├── requirements.txt
├── README.md
└── CLAUDE.md      # This file
```

## Dependencies
**Current:**
- `openai==0.27.6` (to be replaced/adapted)
- `numpy==1.22.4`
- `pandas==1.5.3`
- `tqdm==4.64.1`

**To Add:**
- `ollama` (for local LLM inference)

## Notes for Future Sessions
- All `gen_*.py` files have hardcoded dataset paths that need updating
- Original code includes `pdb.set_trace()` debugging statements (line 67-68 in gsm, 149-150 in math)
- GSM requires downloading the dataset separately
- Biography task requires `article.json` with ground truth data
- Current evaluation is mostly automated for math tasks, semi-manual for biography/factuality
- HuggingFace models are already downloaded in MLX format at `/Users/leonardo/.cache/huggingface/hub`
- Need to verify compatibility between HuggingFace MLX models and Ollama (may need mlx-lm for direct usage)
