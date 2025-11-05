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
- [Ollama](https://ollama.com) for local LLM inference

### Installation

1. **Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download/windows
```

2. **Clone and install dependencies:**
```bash
git clone https://github.com/lferreiraMD/slm_multiagent_debate.git
cd slm_multiagent_debate
pip install -r requirements.txt
```

3. **Pull a model (optional - if not using HuggingFace models):**
```bash
ollama pull llama3.2:3b
# or
ollama pull qwen2.5:7b
```

### Available Models

We're testing with MLX-optimized models (optimized for Apple Silicon):

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| **DeepSeek-R1-Distill-Qwen** | 1.5B | Reasoning tasks, fast iterations | ⚡⚡⚡ |
| **Llama 3.2 Instruct** | 3B | Balanced performance | ⚡⚡ |
| **SmallThinker** | 3B | Reasoning, debate experiments | ⚡⚡ |
| **Qwen2.5 Instruct** | 7B | Strong math performance | ⚡ |
| **Llama 3.1 Instruct** | 8B | General purpose | ⚡ |
| **Qwen2.5 Instruct** | 14B | Best overall performance | ⚡ |

All models support 3-8 simultaneous instances on 48GB M4 Pro.

## Configuration

### Environment Setup
Copy `.env.example` to `.env` and configure:
```bash
cp .env .env.example
```

### Model Selection
Edit the `model` parameter in each `gen_*.py` script:
```python
# Change from:
model="gpt-3.5-turbo-0301"

# To:
model="llama3.2:3b"  # or your preferred model
```

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

- [x] Repository setup and documentation
- [ ] Adapt scripts to use Ollama/local inference
- [ ] Run baseline experiments (no debate)
- [ ] Run multiagent debate experiments
- [ ] Compare SLM performance to GPT-3.5 baseline
- [ ] Document optimal configurations per task
- [ ] Scale to HPC for larger models

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
