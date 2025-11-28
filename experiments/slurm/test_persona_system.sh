#!/bin/bash

# Test the persona diversity experiment system with a small subset
# Runs 1 job per task (4 total) to validate setup before full submission

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "Persona Diversity System Test"
echo "=================================================="
echo "This will run 4 test experiments (1 per task)"
echo "to validate the system before full submission."
echo "=================================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# 1. Check job config files
echo "  [1/5] Job configuration files..."
for task in math gsm biography mmlu; do
    config_file="$PROJECT_ROOT/experiments/configs/persona_${task}_jobs.txt"
    if [ ! -f "$config_file" ]; then
        echo "    ERROR: Missing $config_file"
        exit 1
    fi
    line_count=$(wc -l < "$config_file")
    if [ "$line_count" -ne 61 ]; then  # 1 header + 60 jobs
        echo "    ERROR: $config_file has $line_count lines (expected 61)"
        exit 1
    fi
    echo "    ✓ $config_file (61 lines)"
done

# 2. Check SLURM scripts
echo "  [2/5] SLURM array scripts..."
for task in math gsm biography mmlu; do
    slurm_script="$SCRIPT_DIR/run_persona_${task}.slurm"
    if [ ! -f "$slurm_script" ]; then
        echo "    ERROR: Missing $slurm_script"
        exit 1
    fi
    echo "    ✓ $slurm_script"
done

# 3. Check generation scripts
echo "  [3/5] Generation scripts..."
for task in math gsm biography mmlu; do
    if [ "$task" == "biography" ]; then
        gen_script="$PROJECT_ROOT/tasks/$task/gen_conversation.py"
    else
        gen_script="$PROJECT_ROOT/tasks/$task/gen_${task}.py"
    fi

    if [ ! -f "$gen_script" ]; then
        echo "    ERROR: Missing $gen_script"
        exit 1
    fi

    # Check for --agent-personas argument
    if ! grep -q "agent-personas" "$gen_script"; then
        echo "    ERROR: $gen_script missing --agent-personas support"
        exit 1
    fi

    echo "    ✓ $gen_script"
done

# 4. Check persona loader utility
echo "  [4/5] Persona loader utility..."
persona_loader="$PROJECT_ROOT/utils/persona_loader.py"
if [ ! -f "$persona_loader" ]; then
    echo "    ERROR: Missing $persona_loader"
    exit 1
fi
echo "    ✓ $persona_loader"

# 5. Check persona summary CSV
echo "  [5/5] Persona summary CSV..."
persona_csv="$PROJECT_ROOT/personas/summary_personas.csv"
if [ ! -f "$persona_csv" ]; then
    echo "    ERROR: Missing $persona_csv"
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

# Test each task with job ID 2 (first non-header job)
for task in math gsm biography mmlu; do
    echo "Testing $task task..."

    # Read job configuration (line 2 = first data row)
    config_file="$PROJECT_ROOT/experiments/configs/persona_${task}_jobs.txt"
    job_line=$(sed -n '2p' "$config_file")

    # Parse configuration
    model_alias=$(echo "$job_line" | awk -F',' '{print $2}')
    n_agents=$(echo "$job_line" | awk -F',' '{print $3}')
    rounds=$(echo "$job_line" | awk -F',' '{print $4}')
    num_value=$(echo "$job_line" | awk -F',' '{print $7}')
    personas_tuple=$(echo "$job_line" | awk -F',' '{for(i=9;i<=NF;i++) printf "%s%s", $i, (i<NF?",":"")}' | sed 's/^"//;s/"$//')

    # Convert personas to space-separated args
    personas_args=$(echo "$personas_tuple" | sed "s/[()']//g" | sed 's/, / /g')

    echo "  Configuration:"
    echo "    Model: $model_alias"
    echo "    Agents: $n_agents"
    echo "    Rounds: $rounds"
    echo "    Num: $num_value"
    echo "    Personas: $personas_tuple"

    # Navigate to task directory
    cd "$PROJECT_ROOT/tasks/$task"

    # Set smaller test size
    test_num=5  # Just 5 problems/questions/people for testing

    # Run generation script
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

    echo "  Running: python3 $gen_script --model $model_alias --agents $n_agents --rounds $rounds $num_arg $test_num --agent-personas $personas_args"

    # Run with timeout
    timeout 300 python3 "$gen_script" \
        --model "$model_alias" \
        --agents "$n_agents" \
        --rounds "$rounds" \
        "$num_arg" "$test_num" \
        --agent-personas $personas_args \
        > "$SCRIPT_DIR/logs/test/test_${task}.out" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Test PASSED"
    elif [ $exit_code -eq 124 ]; then
        echo "  ✗ Test TIMEOUT (>5 minutes)"
    else
        echo "  ✗ Test FAILED (exit code: $exit_code)"
        echo "    Check log: $SCRIPT_DIR/logs/test/test_${task}.out"
    fi

    echo ""
done

echo "=================================================="
echo "Test Summary"
echo "=================================================="
echo "Test logs: $SCRIPT_DIR/logs/test/test_*.out"
echo ""
echo "If all tests passed, you can submit the full"
echo "experiment batch with:"
echo "  bash $SCRIPT_DIR/submit_all_persona_experiments.sh"
echo "=================================================="
