# Original Study Parameters & Datasets

## Generation Parameters (GPT-3.5-turbo-0301)

All experiments used **default OpenAI parameters**:

```python
openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=agent_context,
    n=1,
    # All other parameters at default:
    temperature=1.0,          # Default (not specified in code)
    max_tokens=None,          # Unlimited (not specified in code)
    top_p=1.0,                # Default (not specified in code)
    frequency_penalty=0.0,    # Default
    presence_penalty=0.0      # Default
)
```

**Important:** We must use the same parameters for fair comparison:
- `temperature=1.0` (high randomness, encourages diverse agent responses)
- `max_tokens=None` (let model decide when to stop)
- `top_p=1.0` (nucleus sampling with full distribution)

## Experiment Configurations

| Task | Agents | Rounds | Problems | Random Seed | Dataset Required |
|------|--------|--------|----------|-------------|------------------|
| **Math** | 2 | 3 | 100 | 0 | No (generated) |
| **GSM** | 3 | 2 | 100 | 0 | **Yes** |
| **Biography** | 3 | 2 | 40 | 1 | **Yes** |
| **MMLU** | 3 | 2 | 100 | 0 | **Yes** |

## Required Datasets

### 1. Math (No External Dataset)
Generated on-the-fly:
```python
a, b, c, d, e, f = np.random.randint(0, 30, size=6)
answer = a + b * c + d - e * f
question = "What is the result of {}+{}*{}+{}-{}*{}?"
```
- Uses `numpy.random.seed(0)` for reproducibility

### 2. GSM8K (Grade School Math)
**Source:** https://github.com/openai/grade-school-math

**File needed:** `grade_school_math/data/test.jsonl`

**Download:**
```bash
git clone https://github.com/openai/grade-school-math.git data/gsm8k
```

**Format:**
```json
{
  "question": "Natalia sold clips to 48 of her friends in April...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n..."
}
```

**Usage in code:**
- Line 38: Reads from hardcoded path
- Line 39: `random.shuffle(questions)` with seed 0
- Line 41: Uses first 100 questions

### 3. Biography (Computer Scientists)
**Source:** Custom dataset (already included in this repo)

**File location:** `biography/article.json` ✓

**Format (inferred):**
```json
{
  "Alan Turing (1912-1954)": { ... },
  "Grace Hopper (1906-1992)": { ... },
  ...
}
```

**Usage in code:**
- Line 56: Reads `article.json` from biography directory
- Line 59-60: Sorts keys, filters to remove dates
- Line 61-62: `random.shuffle(people)` with seed 1
- Line 70: Uses first 40 people

**Action needed:** None - dataset already present

### 4. MMLU (Massive Multitask Language Understanding)
**Source:** https://github.com/hendrycks/test

**Files needed:** `data/test/*.csv` (57 subject CSV files)

**Download:**
```bash
git clone https://github.com/hendrycks/test.git data/mmlu
```

**Format (CSV):**
```csv
Question,A,B,C,D,Answer
"What is the capital of France?","London","Berlin","Paris","Madrid","C"
```

**Usage in code:**
- Line 60: Globs all CSV files in test directory
- Line 64: `random.seed(0)`
- Line 67-70: Randomly selects 100 questions across all subjects

## Dataset Setup Script

Run `scripts/download_datasets.sh` to download GSM8K and MMLU:
```bash
cd scripts
./download_datasets.sh
```

This will:
- Download GSM8K to `./data/gsm8k/`
- Download MMLU to `./data/mmlu/`
- Note: Biography dataset already exists at `./biography/article.json`

## Path Updates Needed

All `gen_*.py` files have hardcoded paths that need updating:

| File | Current Path | New Path |
|------|-------------|----------|
| `gsm/gen_gsm.py:38` | `/data/vision/billf/scratch/.../test.jsonl` | `../data/gsm8k/grade_school_math/data/test.jsonl` |
| `biography/gen_conversation.py:56` | `article.json` | `../data/biography/article.json` or `./article.json` |
| `mmlu/gen_mmlu.py:60` | `/data/vision/billf/scratch/.../test/*.csv` | `../data/mmlu/data/test/*.csv` |

## MLX-LM Equivalent Parameters

When implementing with `mlx-lm`, use:
```python
from mlx_lm import generate

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    temp=1.0,              # Matches temperature=1.0
    top_p=1.0,             # Matches top_p=1.0
    max_tokens=2048,       # Reasonable limit (OpenAI had none, but practical limit)
    repetition_penalty=1.0 # No penalty (match OpenAI default)
)
```

## Notes for Replication

1. **Exact reproducibility:** Using same random seeds (0 for most, 1 for biography)
2. **Model differences:** SLMs will behave differently than GPT-3.5, even with same parameters
3. **Temperature 1.0:** High temperature = more randomness = more diverse agent responses (critical for debate)
4. **Sampling:** Each agent samples independently (introduces natural variation even with same model)
5. **Evaluation:** Keep original evaluation metrics for fair comparison

## References

- Original paper: https://arxiv.org/abs/2305.14325
- Additional logs: https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0
