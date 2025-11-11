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

We adopt a small-language-model (SLM) setting (e.g., 1.5B-4B parameter range) to examine this hypothesis in a cost-effective, reproducible environment. We construct multiple debate conditions: (1) homogeneous groups of agents all using the same model and prompt style; (2) heterogeneous groups where the same model is prompted to adopt distinct reasoning styles (such as "intuitive", "slow", "skeptic"); (3) heterogeneous groups composed of different models; and (4) heterogeneous groups varying on the decoding parameters. We hold the number of agents and rounds constant, and evaluate on benchmark reasoning and factuality tasks (such as GSM8K word problems, biography generation, and MMLU multiple-choice questions).

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

## Research Hypothesis: Cognitive Diversity

Multi-Agent Debate (MAD) has been a promising mechanism to improve reasoning and factual consistency in language models. In multi-agent debate, multiple agents propose answers, critique each other, and converge to an ideally superior solution. Prior work (Du et al., 2023) treats agents as symmetric peers, but doesn't fully answer why multi-agent debate helps. In this research project, we propose that **cognitive diversity among agents**, such as variation in reasoning style, prompting priors, or heuristics, is a key driver of multi-agent debate gains.

We adopt a small-language-model (SLM) setting (e.g., 1.5B-4B parameter range) to examine this hypothesis in a cost-effective, reproducible environment. We construct multiple debate conditions:

1. **Homogeneous groups** of agents all using the same model and prompt style
2. **Heterogeneous groups** where the same model is prompted to adopt distinct reasoning styles (such as "intuitive", "slow", "skeptic")
3. **Heterogeneous groups** composed of different models
4. **Heterogeneous groups** varying on the decoding parameters

We hold the number of agents and rounds constant, and evaluate on benchmark reasoning and factuality tasks (such as GSM8K word problems, biography generation, and MMLU multiple-choice questions).

In this paper, we introduce a **diversity-gain metric** that quantifies improvements in outcome quality (accuracy) as a function of response embedding and argument diversity (measured via cosine distances, disagreement rates), as referenced in a critique of MAD presented by Wynn et. al., 2025. We then test whether higher intra-group stylistic/response variance correlates with higher accuracy gains.

## What is Multiagent Debate?

Multiple LLM agents independently solve the same problem, then see each other's solutions and refine their answers over several rounds. This iterative debate process has been shown to improve:
- Factual accuracy (biography generation)
- Mathematical reasoning (GSM8K, arithmetic)
- Knowledge recall (MMLU benchmark)

**Key insight:** Even when using the same underlying model, independent agents with different "perspectives" can correct each other's errors through debate.

## Available Experiments

### 📊 Math (`./tasks/math/`)
Simple arithmetic expressions testing order of operations
- **Task:** Evaluate `a+b*c+d-e*f`
- **Config:** 2 agents, 3 rounds, 100 problems
- **Evaluation:** Automated exact match
```bash
cd tasks/math && python gen_math.py
```

### 🧮 Grade School Math (`./tasks/gsm/`)
Multi-step word problems from GSM8K dataset
- **Task:** Word problems requiring multi-step reasoning
- **Config:** 3 agents, 2 rounds, 100 problems
- **Dataset:** [OpenAI GSM8K](https://github.com/openai/grade-school-math)
```bash
cd tasks/gsm
python gen_gsm.py      # Generate answers
python eval_gsm.py     # Evaluate results
```

### 👤 Biography (`./tasks/biography/`)
Factual biography generation for computer scientists
- **Task:** Generate bullet-point biographies
- **Config:** 3 agents, 2 rounds, 40 people
- **Evaluation:** Fact-checking against ground truth
```bash
cd tasks/biography
python gen_conversation.py    # Generate biographies
python eval_conversation.py   # Evaluate factuality
```

### 📚 MMLU (`./tasks/mmlu/`)
Multiple-choice questions across academic subjects
- **Task:** MMLU benchmark questions
- **Config:** 3 agents, 2 rounds
- **Dataset:** [MMLU](https://github.com/hendrycks/test)
```bash
cd tasks/mmlu
python gen_mmlu.py     # Generate answers
python eval_mmlu.py    # Evaluate accuracy
```

## Setup

### Prerequisites
- Python 3.8+
- **Mac M4 Pro:** MLX-LM (already installed)
- **HPC/Windows:** [Ollama](https://ollama.com) or vLLM for local inference

### Installation

**Mac M4 Pro (Local Development):**
```bash
# Clone repository
git clone https://github.com/lferreiraMD/slm_multiagent_debate.git
cd slm_multiagent_debate

# Install dependencies (mlx-lm already installed)
pip install -r requirements.txt

# Verify MLX installation
python3 -m pip show mlx-lm
```

**HPC/Windows (Future Deployment):**
```bash
# Option 1: Ollama (simpler, cross-platform)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Download from https://ollama.com/download/windows

ollama pull llama3.2:3b
ollama pull qwen2.5:7b

# Option 2: vLLM (higher throughput, HPC only)
pip install vllm transformers
```

### Available Models

**Mac M4 Pro (MLX-optimized, ready to use):**

| Model | Size | Best For | Simultaneous Instances |
|-------|------|----------|------------------------|
| **DeepSeek-R1-Distill-Qwen** | 1.5B | Reasoning tasks, fast iterations | 8-10 |
| **Llama 3.2 Instruct** | 3B | Balanced performance | 5-8 |
| **SmallThinker** | 3B | Reasoning, debate experiments | 5-8 |
| **Qwen2.5 Instruct** | 7B | Strong math (1M context) | 3-5 |
| **Llama 3.1 Instruct** | 8B | General purpose | 3-5 |
| **Qwen2.5 Instruct** | 14B | Best overall performance | 2-3 |

These models are already downloaded and optimized for M4 Pro via MLX. No additional downloads needed!

**HPC/Windows (requires GGUF or PyTorch models):**
- Models need to be downloaded separately (MLX format only works on Apple Silicon)
- Use Ollama (GGUF) or vLLM/transformers (PyTorch) formats

## Configuration

### Model Selection
All generation scripts have been migrated to use local MLX models via a custom wrapper:

```python
# Current implementation (OpenAI-compatible):
from utils import ChatCompletion

completion = ChatCompletion.create(
    model=model_name,  # Can be alias like "deepseek" or full path
    messages=agent_context,
    temperature=1.0,
    max_tokens=2048
)
```

The `utils.ChatCompletion` wrapper provides:
- Automatic model loading and caching
- OpenAI-compatible API interface
- Chat template formatting
- MLX-LM inference backend

**Model Selection (Priority Order):**
1. Command-line argument: `--model llama32-3b`
2. config.yaml default: `model: "deepseek"`
3. Available aliases: `deepseek`, `llama32-3b`, `smallthinker`, `qwen25-7b`, `llama31-8b`, `qwen25-14b`

## Running Experiments

### Quick Start - Math Task
```bash
cd tasks/math

# Use default model (DeepSeek 1.5B)
python gen_math.py

# Or specify a model
python gen_math.py --model llama32-3b --agents 3 --rounds 2

# Or use a different configuration
python gen_math.py --model qwen25-7b --agents 4 --rounds 5
```

### Full Workflow Example (GSM)
```bash
# 1. Datasets are already included in data/gsm8k/
cd tasks/gsm

# 2. Run generation with default config (3 agents, 2 rounds)
python gen_gsm.py

# Or customize the experiment
python gen_gsm.py --model smallthinker --agents 4 --rounds 3

# 3. Evaluate results (uses GPT-4 for verification)
python eval_gsm.py
```

### Analysis Workflow
```bash
# 1. Run experiments across tasks
cd tasks/math && python gen_math.py --model deepseek
cd ../gsm && python gen_gsm.py --model deepseek
cd ../biography && python gen_conversation.py --model deepseek

# 2. Aggregate all results
python scripts/aggregate_results.py
# Creates: results/summary.p and results/summary.csv

# 3. Generate visualizations
python scripts/plot_by_model.py   # Per-model plots
python scripts/plot_by_task.py    # Task comparison plots
# Outputs to: plots/*.png
```

## Project Structure

```
.
├── data/              # Datasets (included in repo)
│   ├── biography/     # Ground truth biographies (article.json)
│   ├── gsm8k/         # GSM8K dataset files (train.jsonl, test.jsonl)
│   └── mmlu/          # MMLU test files (*_test.csv)
├── tasks/             # Task implementations
│   ├── math/          # Arithmetic problems (gen_math.py)
│   ├── gsm/           # Grade School Math (gen_gsm.py, eval_gsm.py)
│   ├── biography/     # Biography generation (gen_conversation.py, eval_conversation.py)
│   └── mmlu/          # MMLU benchmark (gen_mmlu.py, eval_mmlu.py)
├── utils/             # Shared utilities
│   ├── llm_wrapper.py      # OpenAI-compatible ChatCompletion wrapper
│   ├── config.py           # Configuration management
│   ├── model_cache.py      # Model loading/caching
│   └── helpers.py          # Shared functions
├── scripts/           # Analysis scripts
│   ├── aggregate_results.py  # Combine results from all experiments
│   ├── plot_by_model.py      # Generate per-model performance plots
│   └── plot_by_task.py       # Generate task comparison plots
├── results/           # Experiment outputs
│   ├── summary.p      # Aggregated results (pickle)
│   └── summary.csv    # Human-readable summary
├── plots/             # Generated visualizations (*.png)
├── config.yaml        # Centralized configuration
├── requirements.txt
├── CLAUDE.md          # Detailed technical documentation
└── README.md          # This file
```

## Results & Analysis

### Current Experimental Results

**Math Task (Arithmetic Reasoning)**
- **Models tested:** DeepSeek-R1-Distill-Qwen-1.5B, Meta-Llama-3.1-8B-Instruct-8bit
- **Configurations:** 13 experiments (1-5 agents, 3-7 rounds)
- **Dataset:** 100 arithmetic problems per configuration

**Initial Findings (Math Task):**
```
DeepSeek 1.5B Performance:
- Single agent (1 agent, 3 rounds): 33% accuracy
- Best configuration (3 agents, 3 rounds): 37% accuracy
- Performance ranges: 28-37% across configurations
```

**Analysis Tools:**
- `scripts/aggregate_results.py`: Combines results from all experiments
- `scripts/plot_by_model.py`: Generates performance plots per (model, task)
- `scripts/plot_by_task.py`: Generates comparison plots across models
- Output: `results/summary.csv` and visualizations in `plots/`

**Ongoing Experiments:**
- GSM (grade school math word problems)
- Biography (factual biography generation)
- MMLU (multi-subject multiple choice)

### Performance Metrics

Results are automatically aggregated and tracked across:
- **Model sizes:** 1.5B → 14B parameters
- **Agent counts:** Single-agent vs multiagent debate (2-5 agents)
- **Debate rounds:** 2-7 rounds of refinement
- **Model families:** Llama, Qwen, DeepSeek, SmallThinker

See `results/summary.csv` for detailed performance data.

## Development Roadmap

**Phase 1: Local Development (Mac M4 Pro)** ✅ **COMPLETED**
- [x] Repository setup and documentation
- [x] Verify MLX-LM installation and available models
- [x] Download and organize datasets (GSM8K, MMLU, biography)
- [x] Reorganize codebase structure (data/ and tasks/ directories)
- [x] Create OpenAI-compatible wrapper for mlx-lm
- [x] Adapt all tasks to use local MLX models (math, GSM, biography, MMLU)
- [x] Test single-agent and multiagent debate locally
- [x] Create results aggregation and visualization scripts

**Phase 2: Experimentation** 🔄 **IN PROGRESS**
- [x] Initial math task experiments (13 configurations, 2 models)
- [x] Results aggregation and visualization pipeline
- [ ] Run baseline experiments (no debate) across all tasks
- [ ] Complete GSM, biography, and MMLU task experiments
- [ ] Compare SLM performance to GPT-3.5 baseline (from original paper)
- [ ] Test multiple model sizes (1.5B → 14B) and families
- [ ] Document optimal configurations per task

**Phase 3: HPC Scaling** 📋 **PLANNED**
- [ ] Create platform abstraction layer (MLX vs Ollama/vLLM)
- [ ] Set up Ollama/vLLM on HPC with GGUF models
- [ ] Create SLURM job submission scripts
- [ ] Run large-scale experiments with bigger models
- [ ] Cross-platform testing and deployment

## Hardware Requirements

**Minimum:**
- 16GB RAM
- 10GB disk space
- Works with 1.5B-3B models

**Recommended:**
- 32GB+ RAM
- 50GB disk space
- Apple Silicon (M1/M2/M3/M4) or NVIDIA GPU
- Enables 7B-14B models with multiple simultaneous agents

## Contributing

This is a research project. Contributions, suggestions, and experiment results are welcome!

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

## License

MIT License - see original repository for details.

## Related Work

- **Original Implementation:** [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate)
- **Open-source LLM Debate:** [gauss5930/LLM-Agora](https://github.com/gauss5930/LLM-Agora)
- **Debate Logs:** [Additional examples](https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0)

## Acknowledgments

Built upon the excellent work of Du et al. (2023). This project demonstrates that multiagent debate benefits extend to smaller, locally-hosted models.
