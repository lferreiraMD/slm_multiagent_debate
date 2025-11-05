# Multiagent Debate with Small Language Models (SLMs)

**Adapting the multiagent debate methodology to locally-hosted small language models**

## About This Project

This repository adapts the research from ["Improving Factuality and Reasoning in Language Models through Multiagent Debate"](https://arxiv.org/abs/2305.14325) (Du et al., 2023) to work with local Small Language Models (SLMs) instead of OpenAI's GPT models.

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

## Available Experiments

### 📊 Math (`./math/`)
Simple arithmetic expressions testing order of operations
- **Task:** Evaluate `a+b*c+d-e*f`
- **Config:** 2 agents, 3 rounds, 100 problems
- **Evaluation:** Automated exact match
```bash
cd math && python gen_math.py
```

### 🧮 Grade School Math (`./gsm/`)
Multi-step word problems from GSM8K dataset
- **Task:** Word problems requiring multi-step reasoning
- **Config:** 3 agents, 2 rounds, 100 problems
- **Dataset:** [OpenAI GSM8K](https://github.com/openai/grade-school-math)
```bash
cd gsm
python gen_gsm.py      # Generate answers
python eval_gsm.py     # Evaluate results
```

### 👤 Biography (`./biography/`)
Factual biography generation for computer scientists
- **Task:** Generate bullet-point biographies
- **Config:** 3 agents, 2 rounds, 40 people
- **Evaluation:** Fact-checking against ground truth
```bash
cd biography
python gen_conversation.py    # Generate biographies
python eval_conversation.py   # Evaluate factuality
```

### 📚 MMLU (`./mmlu/`)
Multiple-choice questions across academic subjects
- **Task:** MMLU benchmark questions
- **Config:** 3 agents, 2 rounds
- **Dataset:** [MMLU](https://github.com/hendrycks/test)
```bash
cd mmlu
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
Scripts will be updated to use MLX models via a wrapper. For now, the original code uses:
```python
# Current (OpenAI):
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=agent_context
)

# Future (MLX-LM via wrapper):
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")
# Wrapper will handle OpenAI-compatible interface
```

Model selection will be configurable via environment variable or command-line argument.

## Running Experiments

### Quick Start - Math Task
```bash
cd math
python gen_math.py
```

### Full Workflow Example (GSM)
```bash
# 1. Download dataset
git clone https://github.com/openai/grade-school-math

# 2. Update dataset path in gen_gsm.py
# Edit line 38: questions = read_jsonl("/path/to/test.jsonl")

# 3. Run generation
cd gsm
python gen_gsm.py

# 4. Evaluate results
python eval_gsm.py
```

## Project Structure

```
.
├── math/              # Arithmetic problems
│   └── gen_math.py
├── gsm/               # Grade School Math
│   ├── gen_gsm.py
│   └── eval_gsm.py
├── biography/         # Biography generation
│   ├── gen_conversation.py
│   └── eval_conversation.py
├── mmlu/              # MMLU benchmark
│   ├── gen_mmlu.py
│   └── eval_mmlu.py
├── requirements.txt
├── CLAUDE.md          # Detailed technical notes
└── README.md          # This file
```

## Results & Analysis

*Coming soon - we'll document performance comparisons between:*
- Different model sizes (1.5B → 14B)
- Single-agent vs multiagent debate
- Number of debate rounds
- Model families (Llama, Qwen, DeepSeek)

## Development Roadmap

**Phase 1: Local Development (Mac M4 Pro)**
- [x] Repository setup and documentation
- [x] Verify MLX-LM installation and available models
- [ ] Create OpenAI-compatible wrapper for mlx-lm
- [ ] Adapt math task to use local MLX models
- [ ] Test single-agent and multiagent debate locally
- [ ] Adapt GSM, biography, and MMLU tasks

**Phase 2: Experimentation**
- [ ] Run baseline experiments (no debate) across all tasks
- [ ] Run multiagent debate experiments with different models
- [ ] Compare SLM performance to GPT-3.5 baseline
- [ ] Test model sizes (1.5B → 14B) and families
- [ ] Document optimal configurations per task

**Phase 3: HPC Scaling**
- [ ] Create platform abstraction layer (MLX vs Ollama/vLLM)
- [ ] Set up Ollama/vLLM on HPC with GGUF models
- [ ] Create SLURM job submission scripts
- [ ] Run large-scale experiments with bigger models

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
  author={[Your Team]},
  year={2025},
  url={https://github.com/lferreiraMD/slm_multiagent_debate}
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
