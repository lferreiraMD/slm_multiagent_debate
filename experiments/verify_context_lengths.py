#!/usr/bin/env python3
"""
Verify model context lengths by loading models with vLLM and checking auto-detected max_model_len.

This script loads each vLLM model WITHOUT specifying max_model_len and reports what vLLM
auto-detects from the model's config.json. Use this to verify/update config.yaml values.

Usage:
    python3 experiments/verify_context_lengths.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config
from vllm import LLM


def verify_model_context_length(model_alias: str, model_path: str) -> dict:
    """
    Load model with vLLM and check auto-detected max_model_len.

    Args:
        model_alias: Model alias from config.yaml
        model_path: HuggingFace model path

    Returns:
        Dict with: {alias, path, detected_length, status, error}
    """
    print(f"\n{'='*70}")
    print(f"Model: {model_alias}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")

    result = {
        'alias': model_alias,
        'path': model_path,
        'detected_length': None,
        'status': 'unknown',
        'error': None
    }

    try:
        # Load model WITHOUT specifying max_model_len
        # vLLM will auto-detect from model config
        print("Loading model (auto-detecting context length)...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=2,  # Single GPU for simplicity
            gpu_memory_utilization=0.95,  # Conservative for testing
            disable_custom_all_reduce=True,
            enforce_eager=True,  # Avoid flash-attn issues
        )

        # Get the detected max_model_len
        detected_len = llm.llm_engine.model_config.max_model_len
        result['detected_length'] = detected_len
        result['status'] = 'success'

        print(f"✓ Detected context length: {detected_len}")

        # Cleanup
        del llm
        import gc
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"✗ Error: {e}")

    return result


def main():
    print("="*70)
    print("vLLM Model Context Length Verification")
    print("="*70)
    print("\nThis script loads each vLLM model and checks the auto-detected")
    print("max_model_len value. Results will be compared to config.yaml.")
    print()

    # Load config
    config = load_config()
    models = config.get('models', {})
    model_metadata = config.get('model_metadata', {})

    # Get all vLLM models
    vllm_models = {
        alias: path for alias, path in models.items()
        if alias.startswith('vllm-')
    }

    print(f"Found {len(vllm_models)} vLLM models to verify\n")

    results = []

    for alias, path in vllm_models.items():
        result = verify_model_context_length(alias, path)
        results.append(result)

    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    print()

    # Compare with config.yaml
    discrepancies = []

    for result in results:
        alias = result['alias']
        detected = result['detected_length']
        status = result['status']

        if status == 'success':
            # Check config.yaml value
            config_value = model_metadata.get(alias, {}).get('context_length', None)

            if config_value is None:
                status_str = "⚠️  NOT IN CONFIG"
                discrepancies.append((alias, detected, config_value, "missing"))
            elif config_value != detected:
                status_str = f"✗ MISMATCH (config={config_value}, detected={detected})"
                discrepancies.append((alias, detected, config_value, "mismatch"))
            else:
                status_str = "✓ MATCH"

            print(f"{alias:25s} {detected:>8d}  {status_str}")
        else:
            print(f"{alias:25s} {'ERROR':>8s}  {result['error'][:40]}")

    # Report discrepancies
    if discrepancies:
        print("\n" + "="*70)
        print("DISCREPANCIES FOUND")
        print("="*70)
        print()
        print("Update config.yaml with these values:")
        print()

        for alias, detected, config_val, issue_type in discrepancies:
            if issue_type == "mismatch":
                print(f"  {alias}:")
                print(f"    context_length: {detected}  # was {config_val}")
            else:
                print(f"  {alias}:")
                print(f"    context_length: {detected}  # NEW")
        print()
    else:
        print("\n✓ All values match config.yaml")

    print("="*70)


if __name__ == "__main__":
    main()
