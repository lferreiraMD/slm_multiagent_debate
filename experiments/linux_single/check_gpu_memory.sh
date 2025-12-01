#!/usr/bin/env bash
#
# check_gpu_memory.sh - Multi-GPU memory availability checker
#
# Usage: check_gpu_memory.sh MODEL_ALIAS N_AGENTS [PROJECT_ROOT]
#
# Checks available VRAM across multiple GPUs (respects CUDA_VISIBLE_DEVICES)
# Returns 0 if ANY GPU has enough memory, 1 otherwise
#

check_gpu_memory() {
    local model_alias="$1"
    local n_agents="$2"
    local project_root="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

    # Parse CUDA_VISIBLE_DEVICES to get available GPU indices
    local visible_gpus=()
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # If not set, detect all available GPUs
        mapfile -t visible_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    else
        # Parse comma-separated GPU indices (handle spaces)
        IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
        for gpu in "${gpu_array[@]}"; do
            # Trim whitespace
            gpu=$(echo "$gpu" | xargs)
            visible_gpus+=("$gpu")
        done
    fi

    if [ ${#visible_gpus[@]} -eq 0 ]; then
        echo "ERROR: No GPUs detected or CUDA_VISIBLE_DEVICES is invalid"
        return 1
    fi

    echo "Detected ${#visible_gpus[@]} visible GPU(s): ${visible_gpus[*]}"

    # If no model specified, just report GPU status and exit
    if [ -z "$model_alias" ]; then
        echo "No model specified - reporting GPU memory status only"
        echo ""

        local total_free_gb=0
        local total_capacity_gb=0

        for gpu_idx in "${visible_gpus[@]}"; do
            local gpu_mem=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null)

            if [ -z "$gpu_mem" ]; then
                echo "  GPU $gpu_idx: Unable to query (may not exist)"
                continue
            fi

            IFS=',' read -r free_mb total_mb <<< "$gpu_mem"
            free_mb=$(echo "$free_mb" | xargs)
            total_mb=$(echo "$total_mb" | xargs)

            local free_gb=$(echo "scale=2; $free_mb / 1024" | bc -l)
            local total_gb=$(echo "scale=2; $total_mb / 1024" | bc -l)
            local used_gb=$(echo "scale=2; $total_gb - $free_gb" | bc -l)
            local pct_used=$(echo "scale=1; $used_gb * 100 / $total_gb" | bc -l)

            echo "  GPU $gpu_idx: ${free_gb}GB free / ${total_gb}GB total (${used_gb}GB used, ${pct_used}%)"

            # Accumulate totals
            total_free_gb=$(echo "$total_free_gb + $free_gb" | bc -l)
            total_capacity_gb=$(echo "$total_capacity_gb + $total_gb" | bc -l)
        done

        # Display totals
        echo ""
        echo "Total across all GPUs:"
        echo "  Free VRAM:      ${total_free_gb}GB (currently available)"
        echo "  Potential VRAM: ${total_capacity_gb}GB (total capacity when clear)"

        return 0
    fi

    # Estimate required memory (rough heuristics)
    # IMPORTANT: Order matters! Check larger sizes first (14b before 4b, 8b before 18b, etc.)
    local required_gb=0
    case "$model_alias" in
        *14b*) required_gb=28 ;;
        *0.6b*|*vibethinker*) required_gb=2 ;;
        *1.7b*|*deepseek*) required_gb=4 ;;
        *3b*|*smallthinker*) required_gb=7 ;;
        *4b*) required_gb=9 ;;
        *7b*) required_gb=15 ;;
        *8b*) required_gb=17 ;;
        *9b*) required_gb=19 ;;
        *) required_gb=10 ;;
    esac

    # Add overhead for multi-agent (KV cache: 0.5GB per agent)
    local overhead=0
    if [ -n "$n_agents" ]; then
        overhead=$(echo "$n_agents * 0.5" | bc -l)
        required_gb=$(echo "$required_gb + $overhead" | bc -l)
        echo "Required VRAM: ~${required_gb}GB (model + ${n_agents} agents × 0.5GB KV cache)"
    else
        echo "Required VRAM: ~${required_gb}GB (model only, no multi-agent overhead)"
    fi

    # Check each visible GPU
    local has_sufficient_gpu=false
    local suitable_gpus=()

    for gpu_idx in "${visible_gpus[@]}"; do
        # Query this specific GPU's memory via nvidia-smi
        local gpu_mem=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null)

        if [ -z "$gpu_mem" ]; then
            echo "  GPU $gpu_idx: Unable to query (may not exist)"
            continue
        fi

        # Parse memory (format: "free_mb, total_mb")
        IFS=',' read -r free_mb total_mb <<< "$gpu_mem"
        # Trim whitespace
        free_mb=$(echo "$free_mb" | xargs)
        total_mb=$(echo "$total_mb" | xargs)

        # Convert MB to GB
        local free_gb=$(echo "scale=2; $free_mb / 1024" | bc -l)
        local total_gb=$(echo "scale=2; $total_mb / 1024" | bc -l)

        # Check if enough free memory (1GB safety margin)
        local available=$(echo "$free_gb - 1.0" | bc -l)
        local sufficient=$(echo "$available >= $required_gb" | bc -l)

        if [ "$(echo "$sufficient == 1" | bc -l)" -eq 1 ]; then
            echo "  GPU $gpu_idx: ✓ ${free_gb}GB free / ${total_gb}GB total (SUFFICIENT)"
            suitable_gpus+=("$gpu_idx")
            has_sufficient_gpu=true
        else
            echo "  GPU $gpu_idx: ✗ ${free_gb}GB free / ${total_gb}GB total (insufficient)"
        fi
    done

    echo ""

    if [ "$has_sufficient_gpu" = true ]; then
        echo "✓ GPU memory check PASSED"
        echo "  Suitable GPU(s): ${suitable_gpus[*]}"
        echo "  You can run the job on any of these GPUs"
        return 0
    else
        echo "✗ GPU memory check FAILED"
        echo "  None of the visible GPUs have enough free memory"
        echo "  Required: ~${required_gb}GB, Available: ${free_gb}GB (last checked GPU)"
        echo "  Clear GPU memory or use a smaller model"
        return 1
    fi
}

# If script is executed directly (not sourced), run the function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Show help if requested
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        echo "Usage: $0 [MODEL_ALIAS [N_AGENTS [PROJECT_ROOT]]]"
        echo ""
        echo "Parameters (all optional):"
        echo "  MODEL_ALIAS   - Model name to check (e.g., vllm-qwen3-0.6b, vllm-mistral-7b)"
        echo "  N_AGENTS      - Number of agents for multi-agent overhead calculation"
        echo "  PROJECT_ROOT  - Project root directory (auto-detected if not specified)"
        echo ""
        echo "Examples:"
        echo "  $0                          # Just report GPU memory status"
        echo "  $0 vllm-qwen3-0.6b          # Check if 0.6B model fits (no multi-agent)"
        echo "  $0 vllm-qwen3-0.6b 3        # Check if 0.6B model + 3 agents fits"
        echo "  $0 vllm-qwen3-14b 7         # Check if 14B model + 7 agents fits"
        echo ""
        echo "Environment:"
        echo "  CUDA_VISIBLE_DEVICES: Controls which GPUs to check (e.g., '0,1')"
        echo ""
        echo "Model size detection (case-insensitive pattern matching):"
        echo "  *14b* → 28GB   *9b* → 19GB   *8b* → 17GB   *7b* → 15GB"
        echo "  *4b*  → 9GB    *3b* → 7GB    *1.7b* → 4GB  *0.6b* → 2GB"
        exit 0
    fi

    check_gpu_memory "$1" "$2" "$3"
fi
