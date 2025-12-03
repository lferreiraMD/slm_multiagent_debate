#!/usr/bin/env bash
# Context overflow stress test: 7 agents, 3 rounds, all tasks
# Tests all vLLM models (excluding qwen3-0.6b, llama, oss-gpt-20b)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=========================================="
echo "Context Overflow Stress Test"
echo "7 agents, 3 rounds, 2 problems per task"
echo "=========================================="

# ==========================================
# MATH TASK
# ==========================================
echo -e "\n[1/4] MATH TASK"
cd "$PROJECT_ROOT/tasks/math"

echo "  Testing vllm-qwen3-0.6b..."
python3 gen_math_clean.py --model vllm-qwen3-0.6b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-vibethinker..."
python3 gen_math_clean.py --model vllm-vibethinker --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-smallthinker..."
#python3 gen_math_clean.py --model vllm-smallthinker --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-llama32-3b..."
python3 gen_math_clean.py --model vllm-llama32-3b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-mistral-7b..."
python3 gen_math_clean.py --model vllm-mistral-7b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-9b..."
#python3 gen_math_clean.py --model vllm-gemma2-9b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-14b..."
python3 gen_math_clean.py --model vllm-qwen3-14b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-deepseek..."
#python3 gen_math_clean.py --model vllm-deepseek --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-1.7b..."
#python3 gen_math_clean.py --model vllm-qwen3-1.7b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-2b..."
#python3 gen_math_clean.py --model vllm-gemma2-2b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-phi3-mini..."
#python3 gen_math_clean.py --model vllm-phi3-mini --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-4b..."
#python3 gen_math_clean.py --model vllm-qwen3-4b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-8b..."
#python3 gen_math_clean.py --model vllm-qwen3-8b --agents 7 --rounds 3 --num-problems 2
#ython3 ../../utils/test_cuda_cleanup.py

# ==========================================
# GSM TASK
# ==========================================
echo -e "\n[2/4] GSM TASK"
cd "$PROJECT_ROOT/tasks/gsm"

echo "  Testing vllm-qwen3-0.6b..."
python3 gen_gsm.py --model vllm-qwen3-0.6b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-vibethinker..."
python3 gen_gsm.py --model vllm-vibethinker --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-llama32-3b..."
python3 gen_gsm.py --model vllm-llama32-3b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-smallthinker..."
#python3 gen_gsm.py --model vllm-smallthinker --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-mistral-7b..."
python3 gen_gsm.py --model vllm-mistral-7b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-9b..."
#python3 gen_gsm.py --model vllm-gemma2-9b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-14b..."
python3 gen_gsm.py --model vllm-qwen3-14b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-deepseek..."
#python3 gen_gsm.py --model vllm-deepseek --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-1.7b..."
#python3 gen_gsm.py --model vllm-qwen3-1.7b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-2b..."
#python3 gen_gsm.py --model vllm-gemma2-2b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-phi3-mini..."
#python3 gen_gsm.py --model vllm-phi3-mini --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-4b..."
#python3 gen_gsm.py --model vllm-qwen3-4b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-8b..."
#python3 gen_gsm.py --model vllm-qwen3-8b --agents 7 --rounds 3 --num-problems 2
#python3 ../../utils/test_cuda_cleanup.py

# ==========================================
# BIOGRAPHY TASK
# ==========================================
echo -e "\n[3/4] BIOGRAPHY TASK"
cd "$PROJECT_ROOT/tasks/biography"

echo "  Testing vllm-qwen3-0.6b..."
python3 gen_gsm.py --model vllm-qwen3-0.6b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-vibethinker..."
python3 gen_conversation.py --model vllm-vibethinker --agents 7 --rounds 3 --num-people 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-llama32-3b..."
python3 gen_gsm.py --model vllm-llama32-3b --agents 7 --rounds 3 --num-problems 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-smallthinker..."
#python3 gen_conversation.py --model vllm-smallthinker --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-mistral-7b..."
python3 gen_conversation.py --model vllm-mistral-7b --agents 7 --rounds 3 --num-people 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-9b..."
#python3 gen_conversation.py --model vllm-gemma2-9b --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-14b..."
python3 gen_conversation.py --model vllm-qwen3-14b --agents 7 --rounds 3 --num-people 2
python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-deepseek..."
#python3 gen_conversation.py --model vllm-deepseek --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-1.7b..."
#python3 gen_conversation.py --model vllm-qwen3-1.7b --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-gemma2-2b..."
#python3 gen_conversation.py --model vllm-gemma2-2b --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-phi3-mini..."
#python3 gen_conversation.py --model vllm-phi3-mini --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-4b..."
#python3 gen_conversation.py --model vllm-qwen3-4b --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

#echo "  Testing vllm-qwen3-8b..."
#python3 gen_conversation.py --model vllm-qwen3-8b --agents 7 --rounds 3 --num-people 2
#python3 ../../utils/test_cuda_cleanup.py

# ==========================================
# MMLU TASK
# ==========================================
echo -e "\n[4/4] MMLU TASK"
cd "$PROJECT_ROOT/tasks/mmlu"

echo "  Testing vllm-vibethinker..."
python3 gen_mmlu.py --model vllm-vibethinker --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-smallthinker..."
python3 gen_mmlu.py --model vllm-smallthinker --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-mistral-7b..."
python3 gen_mmlu.py --model vllm-mistral-7b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-gemma2-9b..."
python3 gen_mmlu.py --model vllm-gemma2-9b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-14b..."
python3 gen_mmlu.py --model vllm-qwen3-14b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-deepseek..."
python3 gen_mmlu.py --model vllm-deepseek --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-1.7b..."
python3 gen_mmlu.py --model vllm-qwen3-1.7b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-gemma2-2b..."
python3 gen_mmlu.py --model vllm-gemma2-2b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-phi3-mini..."
python3 gen_mmlu.py --model vllm-phi3-mini --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-4b..."
python3 gen_mmlu.py --model vllm-qwen3-4b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo "  Testing vllm-qwen3-8b..."
python3 gen_mmlu.py --model vllm-qwen3-8b --agents 7 --rounds 3 --num-questions 2
python3 ../../utils/test_cuda_cleanup.py

echo -e "\n=========================================="
echo "Stress test complete"
echo "=========================================="
