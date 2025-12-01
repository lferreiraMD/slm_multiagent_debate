#!/usr/bin/env python3
"""
Generate job configuration files for Linux SINGLE GPU baseline experiments.

Dynamically loads vLLM models from config.yaml and filters based on VRAM constraints.

Baseline experiments test agent performance WITHOUT persona diversity:
- Single agent (n=1) + multiagent debate (n=3,5,7)
- No persona system prompt overrides
- Serves as comparison baseline for persona experiments

Usage:
    python3 generate_baseline_configs.py [--max-vram-gb LIMIT]

Arguments:
    --max-vram-gb: Maximum VRAM in GB (default: 24 for single RTX 3090)
"""

import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import load_config

# Load config to get agent counts
config = load_config()

# Hardware constraints
MAX_VRAM_GB = 48
SEED = 42

# Load agent counts from config.yaml and add 1 for single-agent baseline
AGENT_COUNTS_CONFIG = config.get('agent_counts', [3, 5, 7])
AGENT_COUNTS = [1] + AGENT_COUNTS_CONFIG  # [1, 3, 5, 7]


def load_vllm_models(max_vram_gb=MAX_VRAM_GB):
    """
    Load vLLM models from config.yaml and filter by VRAM constraints.

    Args:
        max_vram_gb: Maximum VRAM available (default: 24 GB)

    Returns:
        List of model aliases that fit within VRAM limit
    """
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get models and metadata
    models_section = config.get('models', {})
    model_metadata = config.get('model_metadata', {})

    # Filter vLLM models by VRAM requirement
    valid_models = []
    excluded_models = []

    for model_alias, metadata in model_metadata.items():
        if not model_alias.startswith('vllm-'):
            continue

        # Skip if model is commented out (not in models section)
        if model_alias not in models_section:
            continue

        vram_required = metadata.get('vram_gb', 999)

        if vram_required <= max_vram_gb:
            valid_models.append(model_alias)
        else:
            excluded_models.append((model_alias, vram_required, metadata.get('params', 'unknown')))

    # Sort by VRAM requirement (smallest first)
    valid_models.sort(key=lambda m: model_metadata[m]['vram_gb'])

    print(f"\n{'='*70}")
    print(f"Model Selection: MAX_VRAM_GB = {max_vram_gb} GB")
    print(f"{'='*70}")
    print(f"Valid models: {len(valid_models)}")
    for model in valid_models:
        vram = model_metadata[model]['vram_gb']
        params = model_metadata[model]['params']
        print(f"  ✓ {model:25s} ({params:>4s}, ~{vram}GB VRAM)")

    if excluded_models:
        print(f"\nExcluded models: {len(excluded_models)}")
        for model, vram, params in excluded_models:
            print(f"  ✗ {model:25s} ({params:>4s}, ~{vram}GB VRAM - exceeds {max_vram_gb}GB limit)")

    print(f"{'='*70}\n")

    return valid_models


# Task configurations
TASKS = {
    'math': {
        'rounds': 3,
        'num_param': 'num_problems',
        'num_value': 100,
        'random_seed': SEED
    },
    'gsm': {
        'rounds': 3,
        'num_param': 'num_problems',
        'num_value': 100,
        'random_seed': SEED
    },
    'biography': {
        'rounds': 3,
        'num_param': 'num_people',
        'num_value': 40,
        'random_seed': SEED
    },
    'mmlu': {
        'rounds': 3,
        'num_param': 'num_questions',
        'num_value': 100,
        'random_seed': SEED
    }
}


def generate_task_jobs(task_name, task_config, models):
    """
    Generate job configurations for a single task (baseline, no personas).

    Args:
        task_name: Task name (math, gsm, biography, mmlu)
        task_config: Task configuration dict
        models: List of model aliases to use

    Returns:
        List of job configuration dicts
    """
    jobs = []
    job_id = 1

    for model in models:
        for n_agents in AGENT_COUNTS:
            # Create job configuration (NO personas for baseline)
            job = {
                'job_id': job_id,
                'model_alias': model,
                'n_agents': n_agents,
                'rounds': task_config['rounds'],
                'task': task_name,
                'num_param': task_config['num_param'],
                'num_value': task_config['num_value'],
                'random_seed': task_config['random_seed']
            }

            jobs.append(job)
            job_id += 1

    return jobs


def save_jobs_to_csv(jobs, output_path):
    """Save job configurations to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'job_id', 'model_alias', 'n_agents', 'rounds', 'task',
            'num_param', 'num_value', 'random_seed'
        ])

        writer.writeheader()
        writer.writerows(jobs)

    # Force Unix line endings (handles cross-platform file sync issues)
    with open(output_path, 'rb') as f:
        content = f.read()
    content = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    with open(output_path, 'wb') as f:
        f.write(content)

    print(f"  Saved {len(jobs)} jobs to {output_path}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate job configurations for Linux baseline experiments'
    )
    parser.add_argument(
        '--max-vram-gb',
        type=int,
        default=MAX_VRAM_GB,
        help=f'Maximum VRAM in GB (default: {MAX_VRAM_GB} for dual RTX 3090)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Linux BASELINE Experiment Job Configuration Generator")
    print("=" * 70)
    print(f"Hardware: Single RTX 3090 ({args.max_vram_gb}GB VRAM limit)")
    print(f"Agent Counts: {AGENT_COUNTS} (1 single-agent baseline + multiagent debate)")
    print("=" * 70)

    # Load models dynamically from config.yaml
    models = load_vllm_models(max_vram_gb=args.max_vram_gb)

    if not models:
        print("ERROR: No models found that fit within VRAM constraints!")
        print(f"Try increasing --max-vram-gb (current: {args.max_vram_gb}GB)")
        sys.exit(1)

    # Create configs directory
    configs_dir = Path(__file__).parent / 'configs'
    configs_dir.mkdir(exist_ok=True)

    total_jobs = 0

    # Generate configs for each task
    for task_name, task_config in TASKS.items():
        print(f"\nGenerating {task_name} configurations...")

        jobs = generate_task_jobs(task_name, task_config, models)

        output_path = configs_dir / f'baseline_{task_name}_jobs.txt'
        save_jobs_to_csv(jobs, output_path)

        total_jobs += len(jobs)

    # Calculate expected totals
    expected_per_task = len(models) * len(AGENT_COUNTS)
    expected_total = expected_per_task * len(TASKS)

    print("\n" + "=" * 70)
    print("Configuration Generation Complete")
    print("=" * 70)
    print(f"Total jobs: {total_jobs} ({expected_total} expected)")
    print(f"Jobs per task: {expected_per_task} ({len(models)} models × {len(AGENT_COUNTS)} agent counts)")
    print(f"Output directory: {configs_dir}")
    print("\nGenerated files:")
    for task_name in TASKS.keys():
        print(f"  - baseline_{task_name}_jobs.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
