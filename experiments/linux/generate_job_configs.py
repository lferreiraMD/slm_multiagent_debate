#!/usr/bin/env python3
"""
Generate job configuration files for Linux persona diversity experiments.

Creates CSV files mapping job IDs to experiment parameters for all 4 tasks.
Each task gets 60 experiments (10 models × 6 agent counts).

Usage:
    python3 generate_job_configs.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.persona_loader import get_optimal_personas

# Configuration
MODELS = [
    'vllm-qwen3-0.6b',
    'vllm-vibethinker',
    'vllm-qwen3-1.7b',
    'vllm-deepseek',
    'vllm-llama32-3b',
    'vllm-smallthinker',
    'vllm-qwen3-4b',
    'vllm-qwen25-7b',
    'vllm-llama31-8b',
    'vllm-qwen3-14b'
]

AGENT_COUNTS = [2, 3, 4, 5, 6, 7]

# Task configurations
TASKS = {
    'math': {
        'rounds': 3,
        'num_param': 'num_problems',
        'num_value': 100,
        'random_seed': 0
    },
    'gsm': {
        'rounds': 2,
        'num_param': 'num_problems',
        'num_value': 100,
        'random_seed': 0
    },
    'biography': {
        'rounds': 2,
        'num_param': 'num_people',
        'num_value': 40,
        'random_seed': 0
    },
    'mmlu': {
        'rounds': 2,
        'num_param': 'num_questions',
        'num_value': 100,
        'random_seed': 0
    }
}


def generate_task_jobs(task_name, task_config):
    """
    Generate job configurations for a single task.

    Args:
        task_name: Task name (math, gsm, biography, mmlu)
        task_config: Task configuration dict

    Returns:
        List of job configuration dicts
    """
    jobs = []
    job_id = 1

    for model in MODELS:
        for n_agents in AGENT_COUNTS:
            # Get optimal personas for this model/agent count
            try:
                personas = get_optimal_personas(
                    model_name=model,
                    n_agents=n_agents,
                    version='v2',
                    selection_method='maxdet'
                )
            except Exception as e:
                print(f"Warning: Could not get personas for {model} with {n_agents} agents: {e}")
                continue

            # Create job configuration
            job = {
                'job_id': job_id,
                'model_alias': model,
                'n_agents': n_agents,
                'rounds': task_config['rounds'],
                'task': task_name,
                'num_param': task_config['num_param'],
                'num_value': task_config['num_value'],
                'random_seed': task_config['random_seed'],
                'personas_tuple': str(tuple(personas))
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
            'num_param', 'num_value', 'random_seed', 'personas_tuple'
        ])

        writer.writeheader()
        writer.writerows(jobs)

    print(f"  Saved {len(jobs)} jobs to {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Linux Persona Diversity Job Configuration Generator")
    print("=" * 60)

    # Create configs directory
    configs_dir = Path(__file__).parent / 'configs'
    configs_dir.mkdir(exist_ok=True)

    total_jobs = 0

    # Generate configs for each task
    for task_name, task_config in TASKS.items():
        print(f"\nGenerating {task_name} configurations...")

        jobs = generate_task_jobs(task_name, task_config)

        output_path = configs_dir / f'persona_{task_name}_jobs.txt'
        save_jobs_to_csv(jobs, output_path)

        total_jobs += len(jobs)

    print("\n" + "=" * 60)
    print("Configuration Generation Complete")
    print("=" * 60)
    print(f"Total jobs: {total_jobs}")
    print(f"Jobs per task: {len(MODELS) * len(AGENT_COUNTS)}")
    print(f"Output directory: {configs_dir}")
    print("\nGenerated files:")
    for task_name in TASKS.keys():
        print(f"  - persona_{task_name}_jobs.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
