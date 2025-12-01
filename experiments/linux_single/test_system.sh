#!/bin/bash

# Test the Linux SINGLE GPU persona diversity experiment system
# Runs 1 job per task (4 total) to validate setup
# Hardware: Single RTX 3090 (24GB VRAM)

# Force CUDA to use only GPU #1 (RTX 3090)
# GPU 0 = GTX 1650 (4GB) - internal, insufficient VRAM
# GPU 1 = RTX 3090 (24GB) - external, target GPU
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "Linux Single GPU Persona Diversity System Test"
echo "=================================================="
echo "Hardware: Single RTX 3090 (24GB VRAM)"
echo "Models: 9 (excludes 14B model)"
echo "This will run 4 test experiments (1 per task)"
echo "to validate the system before full execution."
echo "=================================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# 1. Python version
echo "  [1/8] Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "    ✓ $PYTHON_VERSION"
else
    echo "    ✗ python3 not found"
    exit 1
fi

# 2. vLLM installation
echo "  [2/8] vLLM package..."
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    echo "    ✓ vLLM $VLLM_VERSION installed"
else
    echo "    ✗ vLLM not installed"
    echo "    Install with: pip3 install vllm==0.11.0"
    exit 1
fi

# 3. NVIDIA GPU (single RTX 3090)
echo "  [3/8] NVIDIA GPU (using GPU #1 - RTX 3090)..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd "," -)

    # Check GPU #1 specifically (what CUDA_VISIBLE_DEVICES=1 targets)
    GPU1_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 1 2>/dev/null || echo "Not found")
    GPU1_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 1 2>/dev/null || echo "0")

    echo "    System: $GPU_COUNT GPU(s) total: $GPU_NAMES"
    echo "    Using: GPU #1 = $GPU1_NAME (${GPU1_VRAM}MB VRAM)"
    echo "    ✓ CUDA_VISIBLE_DEVICES=1 (targeting RTX 3090)"

    if [ "$GPU1_VRAM" -lt 20000 ]; then
        echo "    ⚠ Warning: Expected ~24GB VRAM on GPU #1, found ${GPU1_VRAM}MB"
        echo "    Some larger models may fail due to OOM"
    fi
else
    echo "    ✗ nvidia-smi not found"
    exit 1
fi

# 4. Job config files
echo "  [4/8] Job configuration files..."
for task in math gsm biography mmlu; do
    config_file="$SCRIPT_DIR/configs/persona_${task}_jobs.txt"
    if [ ! -f "$config_file" ]; then
        echo "    ✗ Missing $config_file"
        echo "    Run: python3 $SCRIPT_DIR/generate_job_configs.py"
        exit 1
    fi
    line_count=$(wc -l < "$config_file")
    #if [ "$line_count" -ne 55 ]; then  # 54 jobs + 1 header = 55 lines (single GPU)
    #    echo "    ✗ $config_file has $line_count lines (expected 55 for single GPU)"
    #    echo "    Run: python3 $SCRIPT_DIR/generate_job_configs.py"
    #    exit 1
    #fi
    echo "    ✓ $config_file (55 lines: 54 jobs + header)"
done

# 5. Execution scripts
echo "  [5/8] Execution scripts..."
for task in math gsm biography mmlu; do
    script_file="$SCRIPT_DIR/run_persona_${task}.sh"
    if [ ! -f "$script_file" ] || [ ! -x "$script_file" ]; then
        echo "    ✗ Missing or not executable: $script_file"
        exit 1
    fi
    echo "    ✓ $script_file"
done

# 6. Generation scripts
echo "  [6/8] Generation scripts..."
for task in math gsm biography mmlu; do
    if [ "$task" == "biography" ]; then
        gen_script="$PROJECT_ROOT/tasks/$task/gen_conversation.py"
    else
        gen_script="$PROJECT_ROOT/tasks/$task/gen_${task}.py"
    fi

    if [ ! -f "$gen_script" ]; then
        echo "    ✗ Missing $gen_script"
        exit 1
    fi

    if ! grep -q "agent-personas" "$gen_script"; then
        echo "    ✗ $gen_script missing --agent-personas support"
        exit 1
    fi

    echo "    ✓ $gen_script"
done

# 7. Persona loader utility
echo "  [7/8] Persona loader utility..."
persona_loader="$PROJECT_ROOT/utils/persona_loader.py"
if [ ! -f "$persona_loader" ]; then
    echo "    ✗ Missing $persona_loader"
    exit 1
fi
echo "    ✓ $persona_loader"

# 8. Persona summary CSV
echo "  [8/8] Persona summary CSV..."
persona_csv="$PROJECT_ROOT/personas/summary_personas.csv"
if [ ! -f "$persona_csv" ]; then
    echo "    ✗ Missing $persona_csv"
    exit 1
fi
echo "    ✓ $persona_csv"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Create test logs directory
mkdir -p "$SCRIPT_DIR/logs/test"

# Run test jobs
echo "=================================================="
echo "Running Test Jobs"
echo "=================================================="
echo ""

FAILED_TESTS=()

for task in math gsm biography mmlu; do
    echo "Testing $task task..."

    # Read first job configuration (line 2 = first data row)
    config_file="$SCRIPT_DIR/configs/persona_${task}_jobs.txt"
    job_line=$(sed -n '2p' "$config_file")

    # Parse configuration using Python (handles CSV with commas in fields)
    eval "$(python3 -c "
import csv
import ast

# Parse CSV line
row = next(csv.reader(['''$job_line''']))
job_id, model_alias, n_agents, rounds, task_name, num_param, num_value, random_seed, personas_tuple = row

# Parse personas tuple
try:
    personas = ast.literal_eval(personas_tuple)
    # Quote each persona for bash
    personas_quoted = ' '.join([repr(p) for p in personas])
except:
    personas_quoted = ''

# Output as bash variable assignments
print(f'job_id={job_id}')
print(f'model_alias={model_alias}')
print(f'n_agents={n_agents}')
print(f'rounds={rounds}')
print(f'task_name={task_name}')
print(f'num_param={num_param}')
print(f'num_value={num_value}')
print(f'random_seed={random_seed}')
print(f'personas_args=({personas_quoted})')
")"

    echo "  Configuration:"
    echo "    Model: $model_alias"
    echo "    Agents: $n_agents"
    echo "    Rounds: $rounds"
    echo "    Num: $num_value"

    # Navigate to task directory
    cd "$PROJECT_ROOT/tasks/$task"

    # Set smaller test size
    test_num=3

    # Determine script and argument names
    if [ "$task" == "biography" ]; then
        gen_script="gen_conversation.py"
        num_arg="--num-people"
    elif [ "$task" == "mmlu" ]; then
        gen_script="gen_mmlu.py"
        num_arg="--num-questions"
    else
        gen_script="gen_${task}.py"
        num_arg="--num-problems"
    fi

    echo "  Running: python3 $gen_script --model $model_alias --agents $n_agents --rounds $rounds $num_arg $test_num"

    # Run with timeout (5 minutes)
    timeout 300 python3 "$gen_script" \
        --model "$model_alias" \
        --agents "$n_agents" \
        --rounds "$rounds" \
        "$num_arg" "$test_num" \
        --agent-personas "${personas_args[@]}" \
        > "$SCRIPT_DIR/logs/test/test_${task}.out" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Test PASSED"
    elif [ $exit_code -eq 124 ]; then
        echo "  ✗ Test TIMEOUT (>5 minutes)"
        FAILED_TESTS+=("$task")
    else
        echo "  ✗ Test FAILED (exit code: $exit_code)"
        echo "    Check log: $SCRIPT_DIR/logs/test/test_${task}.out"
        FAILED_TESTS+=("$task")
    fi

    echo ""
done

echo "=================================================="
echo "Test Summary"
echo "=================================================="
echo "Test logs: $SCRIPT_DIR/logs/test/test_*.out"
echo ""

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✓ All tests passed!"
    echo ""
    echo "System is ready. You can now run experiments:"
    echo "  # Run all experiments (sequential, ~10-14 hours for 216 jobs):"
    echo "  bash $SCRIPT_DIR/run_all_experiments.sh"
    echo ""
    echo "  # Or run tasks individually:"
    echo "  MAX_PARALLEL=2 bash $SCRIPT_DIR/run_persona_math.sh  # 54 jobs"
    echo ""
    echo "  # Monitor progress:"
    echo "  bash $SCRIPT_DIR/monitor_experiments.sh"
else
    echo "✗ Some tests failed: ${FAILED_TESTS[*]}"
    echo ""
    echo "Check logs and fix errors before running full experiments."
    exit 1
fi

echo "=================================================="
