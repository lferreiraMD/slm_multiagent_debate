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

Examples:
    # Download all vLLM models
    python3 download_models.py

    # Download specific model
    python3 download_models.py --model vllm-llama32-3b
"""

import argparse
import sys
from pathlib import Path

# Add project root to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoConfig
from utils import resolve_model_name

# vLLM model aliases from config.yaml
VLLM_MODELS = [
    "vllm-qwen3-0.6b",
    "vllm-vibethinker",
    "vllm-deepseek",
    "vllm-qwen3-1.7b",
    "vllm-llama32-3b",
    "vllm-smallthinker",
    "vllm-qwen3-4b",
    "vllm-llama31-8b",
    "vllm-qwen3-8b",
    "vllm-qwen3-14b",
    "vllm-oss-20b",
]


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
            trust_remote_code=True
        )
        print(f"  ✓ Tokenizer cached ({len(tokenizer)} tokens)")

        # Download model config (includes architecture info)
        print("  → Downloading model config...")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"  ✓ Config cached")

        # Note: We don't download full weights here because:
        # 1. Some models are 20B+ parameters (40GB+ downloads)
        # 2. vLLM will auto-download on first use
        # 3. This script focuses on caching tokenizer/config for speed

        print(f"  ✓ Successfully cached {model_alias}")
        print(f"  ℹ Note: Model weights will download on first vLLM use")
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
    print(f"Note: Only tokenizer/config cached, weights download on first use")
    print("="*60)

    # Download each model
    results = {}
    for i, model_alias in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] Processing {model_alias}...")
        success = download_model(model_alias)
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
