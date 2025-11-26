"""
GPU detection and automatic configuration for vLLM backend.

This module provides runtime GPU detection and optimal vLLM parameter configuration
based on hardware capabilities. Only applicable to vLLM backend (Linux/HPC with NVIDIA GPUs).

For MLX (Mac) and Ollama backends, this module has no effect.
"""

import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def is_vllm_backend() -> bool:
    """
    Check if vLLM backend is available and should be used.

    Returns:
        True if running on Linux with torch+CUDA available
    """
    if platform.system() != "Linux":
        return False

    if not TORCH_AVAILABLE:
        return False

    if not torch.cuda.is_available():
        return False

    # Check if vLLM is importable
    try:
        import vllm
        return True
    except ImportError:
        return False


def detect_nvlink() -> bool:
    """
    Detect if NVLink is available between GPUs.

    Returns:
        True if NVLink detected, False otherwise
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', 'topo', '-m'],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Look for "NV" in topology matrix (indicates NVLink)
        output = result.stdout.lower()
        return 'nvlink' in output or ' nv' in output

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False


def detect_vllm_gpus() -> Dict:
    """
    Detect GPU configuration for vLLM backend.

    Returns:
        Dictionary with GPU information:
        {
            'count': int,                    # Number of GPUs
            'models': List[str],             # GPU model names
            'vram_per_gpu_gb': float,        # VRAM per GPU in GB
            'total_vram_gb': float,          # Total VRAM across all GPUs
            'has_nvlink': bool,              # NVLink available
            'available_vram_gb': float,      # Available (unused) VRAM in GB
        }

        Returns empty dict if vLLM backend not available.
    """
    if not is_vllm_backend():
        return {}

    try:
        gpu_count = torch.cuda.device_count()

        if gpu_count == 0:
            return {}

        # Get GPU models and VRAM
        gpu_models = []
        vram_per_gpu = []
        available_vram = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_models.append(props.name)
            vram_gb = props.total_memory / (1024**3)
            vram_per_gpu.append(vram_gb)

            # Get currently allocated memory
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            available_vram.append(vram_gb - allocated_gb)

        # Detect NVLink
        has_nvlink = detect_nvlink() if gpu_count > 1 else False

        return {
            'count': gpu_count,
            'models': gpu_models,
            'vram_per_gpu_gb': vram_per_gpu[0] if vram_per_gpu else 0.0,
            'total_vram_gb': sum(vram_per_gpu),
            'has_nvlink': has_nvlink,
            'available_vram_gb': sum(available_vram),
        }

    except Exception as e:
        print(f"Warning: GPU detection failed: {e}", file=sys.stderr)
        return {}


def estimate_model_size_gb(model_name: str) -> float:
    """
    Estimate model size in GB based on model name.

    Args:
        model_name: HuggingFace model path or alias

    Returns:
        Estimated size in GB (fp16)
    """
    model_lower = model_name.lower()

    # Extract parameter count from common naming patterns
    if '0.5b' in model_lower or '0.6b' in model_lower:
        return 1.0  # ~0.5-0.6B params
    elif '1b' in model_lower or '1.5b' in model_lower or '1.7b' in model_lower:
        return 3.0  # ~1-1.7B params
    elif '3b' in model_lower:
        return 6.0  # ~3B params
    elif '4b' in model_lower:
        return 8.0  # ~4B params
    elif '7b' in model_lower:
        return 14.0  # ~7B params
    elif '8b' in model_lower:
        return 16.0  # ~8B params
    elif '13b' in model_lower or '14b' in model_lower:
        return 28.0  # ~13-14B params
    elif '20b' in model_lower:
        return 40.0  # ~20B params
    elif '70b' in model_lower:
        return 140.0  # ~70B params
    else:
        # Default assumption: medium model
        return 8.0


def get_vllm_optimal_config(
    model_name: str,
    use_case: str = 'production',
    gpu_info: Optional[Dict] = None,
    override_params: Optional[Dict] = None
) -> Dict:
    """
    Get optimal vLLM LLM() parameters based on GPU detection.

    Args:
        model_name: HuggingFace model path
        use_case: One of ['download', 'production', 'debate']
            - 'download': Conservative, sequential model loading
            - 'production': Aggressive, maximum throughput
            - 'debate': Balanced, multiple models
        gpu_info: Optional GPU info dict (auto-detected if None)
        override_params: Optional dict of parameters to override auto-config

    Returns:
        Dictionary of vLLM LLM() parameters, or empty dict if not vLLM backend
    """
    # Only return config if vLLM backend
    if not is_vllm_backend():
        return {}

    # Auto-detect GPUs if not provided
    if gpu_info is None:
        gpu_info = detect_vllm_gpus()

    if not gpu_info or gpu_info['count'] == 0:
        print("Warning: No GPUs detected, using default vLLM config", file=sys.stderr)
        return override_params or {}

    # Extract GPU info
    gpu_count = gpu_info['count']
    vram_per_gpu = gpu_info['vram_per_gpu_gb']
    has_nvlink = gpu_info['has_nvlink']

    # Estimate model size
    model_size_gb = estimate_model_size_gb(model_name)

    # Base configuration
    config = {
        'enable_sleep_mode': True,  # Always enable for proper cleanup
        'disable_custom_all_reduce': True,  # Avoid custom kernel issues
    }

    # Use case specific configurations
    if use_case == 'download':
        # Conservative: sequential model loading
        config.update({
            'tensor_parallel_size': 1,  # Load one model at a time
            'gpu_memory_utilization': 0.7,  # Leave room for cleanup
            'max_model_len': 2048,  # Minimal KV cache
            'enforce_eager': True,  # Save memory, fix flash-attn issues
        })

    elif use_case == 'production':
        # Aggressive: maximize throughput
        # Determine tensor_parallel_size
        if model_size_gb > vram_per_gpu * 0.8:
            # Model doesn't fit on single GPU, use all GPUs
            tensor_parallel_size = gpu_count
        elif gpu_count > 1:
            # Model fits on single GPU, but multi-GPU available
            # Use TP for better latency (per vLLM best practices)
            tensor_parallel_size = gpu_count
        else:
            tensor_parallel_size = 1

        config.update({
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': 0.9,  # Maximize KV cache
            'max_model_len': 8192,  # Full context
            'enable_chunked_prefill': True,  # Better throughput
            'max_num_batched_tokens': 8192,
        })

        # Warn if multi-GPU without NVLink
        if tensor_parallel_size > 1 and not has_nvlink:
            print(
                f"Warning: Using tensor_parallel_size={tensor_parallel_size} "
                f"without NVLink. Performance may be suboptimal.",
                file=sys.stderr
            )

    elif use_case == 'debate':
        # Balanced: multiple models simultaneously
        config.update({
            'tensor_parallel_size': 1,  # Each agent on separate GPU
            'gpu_memory_utilization': 0.8,  # Balanced
            'max_model_len': 4096,  # Moderate context
        })

    else:
        raise ValueError(f"Unknown use_case: {use_case}. Must be 'download', 'production', or 'debate'")

    # Apply overrides if provided
    if override_params:
        config.update(override_params)

    return config


def print_gpu_summary(gpu_info: Optional[Dict] = None, use_case: Optional[str] = None, config: Optional[Dict] = None) -> None:
    """
    Print formatted GPU configuration summary.

    Args:
        gpu_info: Optional GPU info dict (auto-detected if None)
        use_case: Optional use case string to display
        config: Optional vLLM config dict to display
    """
    if not is_vllm_backend():
        print("vLLM backend not available (not Linux or CUDA not available)")
        return

    # Auto-detect if not provided
    if gpu_info is None:
        gpu_info = detect_vllm_gpus()

    if not gpu_info or gpu_info['count'] == 0:
        print("No GPUs detected")
        return

    print("=" * 70)
    print("vLLM Backend - GPU Configuration")
    print("=" * 70)

    # GPU Hardware Info
    gpu_count = gpu_info['count']
    if gpu_count == 1:
        print(f"  GPU: {gpu_info['models'][0]}")
    else:
        # Check if all GPUs are same model
        if len(set(gpu_info['models'])) == 1:
            print(f"  GPUs: {gpu_count}x {gpu_info['models'][0]}")
        else:
            print(f"  GPUs: {gpu_count} GPUs")
            for i, model in enumerate(gpu_info['models']):
                print(f"    GPU {i}: {model}")

    print(f"  VRAM per GPU: {gpu_info['vram_per_gpu_gb']:.1f} GB")
    print(f"  Total VRAM: {gpu_info['total_vram_gb']:.1f} GB")
    print(f"  Available VRAM: {gpu_info['available_vram_gb']:.1f} GB")

    if gpu_count > 1:
        nvlink_status = "✓" if gpu_info['has_nvlink'] else "✗"
        print(f"  NVLink: {nvlink_status}")

    # vLLM Configuration
    if use_case or config:
        print()
        if use_case:
            print(f"Auto-Configuration (use_case='{use_case}'):")
        else:
            print("vLLM Configuration:")

        if config:
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")

    print("=" * 70)


def get_gpu_info_string(gpu_info: Optional[Dict] = None) -> str:
    """
    Get a concise GPU info string for logging.

    Args:
        gpu_info: Optional GPU info dict (auto-detected if None)

    Returns:
        String like "2x RTX 3090 (48GB, NVLink)"
    """
    if not is_vllm_backend():
        return "vLLM not available"

    if gpu_info is None:
        gpu_info = detect_vllm_gpus()

    if not gpu_info or gpu_info['count'] == 0:
        return "No GPUs"

    gpu_count = gpu_info['count']

    # GPU model string
    if len(set(gpu_info['models'])) == 1:
        # All same model
        model_str = gpu_info['models'][0]
        # Simplify model name (remove "NVIDIA GeForce" etc.)
        model_str = model_str.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        if gpu_count > 1:
            gpu_str = f"{gpu_count}x {model_str}"
        else:
            gpu_str = model_str
    else:
        # Mixed models
        gpu_str = f"{gpu_count} GPUs"

    # VRAM
    vram_str = f"{gpu_info['total_vram_gb']:.0f}GB"

    # NVLink
    features = []
    if gpu_count > 1 and gpu_info['has_nvlink']:
        features.append("NVLink")

    # Combine
    if features:
        return f"{gpu_str} ({vram_str}, {', '.join(features)})"
    else:
        return f"{gpu_str} ({vram_str})"
