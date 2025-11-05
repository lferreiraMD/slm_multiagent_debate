#!/bin/bash
set -e

# Script to download datasets for multiagent debate experiments
# Based on: https://github.com/composable-models/llm_multiagent_debate

echo "=== Downloading datasets for LLM Multiagent Debate ==="
echo ""

# Create data directory
mkdir -p data
cd data

# 1. GSM8K (Grade School Math)
if [ ! -d "gsm8k" ]; then
    echo "[1/3] Downloading GSM8K (Grade School Math)..."
    git clone https://github.com/openai/grade-school-math.git gsm8k
    echo "✓ GSM8K downloaded to ./data/gsm8k/"
else
    echo "[1/3] GSM8K already exists, skipping..."
fi

# 2. MMLU (Massive Multitask Language Understanding)
if [ ! -d "mmlu" ]; then
    echo "[2/3] Downloading MMLU..."
    git clone https://github.com/hendrycks/test.git mmlu
    echo "✓ MMLU downloaded to ./data/mmlu/"
else
    echo "[2/3] MMLU already exists, skipping..."
fi

# 3. Biography dataset (already exists in repo)
echo "[3/3] Biography dataset (article.json already in repo)..."
echo "✓ Biography dataset: ../biography/article.json"

cd ..

echo ""
echo "=== Dataset Summary ==="
echo "✓ GSM8K: $(find data/gsm8k/grade_school_math/data/test.jsonl -type f 2>/dev/null | wc -l | tr -d ' ') file"
echo "✓ MMLU:  $(find data/mmlu/data/test/*.csv -type f 2>/dev/null | wc -l | tr -d ' ') subjects"
echo "✓ Biography: article.json already in repo"
echo ""
echo "Next steps:"
echo "  1. Verify datasets: ls -R data/"
echo "  2. Update paths in gen_*.py files (see ORIGINAL_STUDY_PARAMETERS.md)"
