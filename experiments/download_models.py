#!/usr/bin/env python3
"""
Download all vLLM models to local HuggingFace cache.

This script pre-downloads all models used in the baseline experiments
to avoid downloads during runtime. Downloads tokenizer and model config,
which triggers HuggingFace to cache all necessary files.

IMPORTANT: This script is intended for HPC/Linux with vLLM backend.
           Do NOT run on Mac (MLX models are stored differently).

Usage:
    python3 download_models.py [--model MODEL_NAME]

    # If you encounter CUDA memory fragmentation issues, set:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Examples:
    # Download all vLLM models
    python3 download_models.py

    # Download specific model
    python3 download_models.py --model vllm-llama32-3b
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# Suppress NCCL process group warning from vLLM worker processes
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
os.environ['NCCL_DEBUG'] = 'ERROR'  # Only show NCCL errors
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'  # Only show C++ errors
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')

from vllm import LLM
from transformers import AutoTokenizer

from huggingface_hub import login
import os
from dotenv import load_dotenv
import logging

# Suppress verbose logging from vLLM and related libraries
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace token from environment
hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    login(token=hf_token)
    print("✓ HuggingFace login successful!")
else:
    print("⚠ HF_TOKEN not found in .env file or environment variables")
    print("  Some models may require authentication to download.")

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    dist = None


# Add project root to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoConfig
from utils import resolve_model_name

# vLLM model aliases from config.yaml
VLLM_MODELS = {
    'vllm-qwen3-0.6b':      "Qwen/Qwen3-0.6B",
    'vllm-vibethinker':	    "WeiboAI/VibeThinker-1.5B",
    'vllm-deepseek':	    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    'vllm-qwen3-1.7b':	    "Qwen/Qwen3-1.7B",
    'vllm-llama32-3b':	    "meta-llama/Llama-3.2-3B-Instruct",
    'vllm-smallthinker':	"PowerInfer/SmallThinker-3B-Preview",
    'vllm-qwen3-4b':	    "Qwen/Qwen3-4B-Instruct-2507",
    'vllm-llama31-8b':	    "meta-llama/Llama-3.1-8B-Instruct",
    'vllm-qwen3-8b':	    "Qwen/Qwen3-8B",
    'vllm-qwen3-14b':	    "Qwen/Qwen3-14B",
    'vllm-oss-20b':	        "openai/gpt-oss-20b"
}


def download_model(model_alias: str) -> bool:
    """
    Download a single model to HuggingFace cache.

    This downloads tokenizer and config files, which is sufficient for vLLM
    to load the model. vLLM will download weights on first use if needed.

    Args:
        model_alias: Model alias (e.g., "vllm-llama32-3b")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve alias to full HuggingFace path
        model_path = resolve_model_name(model_alias)

        print(f"\n{'='*60}")
        print(f"Downloading: {model_alias}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        # Download tokenizer (includes vocab files)
        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        print(f"  ✓ Tokenizer cached ({len(tokenizer)} tokens)")

        # Download model config (includes architecture info)
        print("  → Downloading model config...")
        _ = AutoConfig.from_pretrained(
            model_path,
        )
        print(f"  ✓ Config cached")

        # Download model weights
        print("  → Downloading model weights...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Adjust for multi-GPU
            max_model_len=2048,  # Limit context length to reduce KV cache memory
            gpu_memory_utilization=0.7,  # Use 70% of GPU memory (leaves room for cleanup)
            enforce_eager=True,  # Disable flash-attn and use eager mode (fixes compatibility issues)
            disable_custom_all_reduce=True  # Disable custom kernels
        )

        # Explicitly delete model to free GPU memory
        print("  → Cleaning up GPU memory...")
        del llm

        # Aggressive cleanup to ensure GPU memory is fully released
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Destroy distributed process group if initialized
            if dist is not None and dist.is_initialized():
                dist.destroy_process_group()

            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.ipc_collect()  # Collect CUDA IPC memory

        gc.collect()  # Force garbage collection

        # Give GPU a moment to fully release memory
        import time
        time.sleep(2)

        print(f"  ✓ Successfully cached {model_alias}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {model_alias}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download vLLM models to local HuggingFace cache"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to download (default: all vLLM models)"
    )
    args = parser.parse_args()

    # Determine which models to download
    if args.model:
        models_to_download = [args.model]
    else:
        models_to_download = VLLM_MODELS

    print("="*60)
    print("MODEL DOWNLOAD SCRIPT (vLLM)")
    print("="*60)
    print(f"Models to download: {len(models_to_download)}")
    print(f"Cache location: ~/.cache/huggingface/hub/")
    print(f"Note: Downloads tokenizer, config, and full model weights")
    print("="*60)

    # Download each model
    results = {}
    for i, model_alias in enumerate(models_to_download, 1):
        # Handle both cases: dict (VLLM_MODELS) and list ([args.model])
        if isinstance(models_to_download, dict):
            model_path = models_to_download[model_alias]
        else:
            # models_to_download is a list, model_alias is already the model name
            model_path = model_alias

        print(f"\n[{i}/{len(models_to_download)}] Processing {model_alias}...")
        success = download_model(model_path)
        results[model_alias] = success

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    successful = [m for m, success in results.items() if success]
    failed = [m for m, success in results.items() if not success]

    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for model in successful:
        print(f"  - {model}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for model in failed:
            print(f"  - {model}")
        print("\nSome models failed to download. Check errors above.")
        sys.exit(1)
    else:
        print("\n✓ All models successfully cached!")
        print("\nYou can now run experiments without download delays.")
        sys.exit(0)


if __name__ == "__main__":
    main()
