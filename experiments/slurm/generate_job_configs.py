#!/usr/bin/env python3
"""
Generate SLURM job configuration files for persona experiments.

Creates one config file per task with all model/agent/persona combinations.
Each row represents one SLURM array job with optimal MaxDet personas.

Usage:
    python3 generate_job_configs.py

Output:
    experiments/configs/persona_math_jobs.txt
    experiments/configs/persona_gsm_jobs.txt
    experiments/configs/persona_biography_jobs.txt
    experiments/configs/persona_mmlu_jobs.txt
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.persona_loader import get_optimal_personas, load_persona_database
import pandas as pd


# Model configuration
MODELS = [
    'vllm-qwen3-0.6b',
    'vllm-vibethinker',
    'vllm-deepseek',
    'vllm-qwen3-1.7b',
    'vllm-llama32-3b',
    'vllm-smallthinker',
    'vllm-qwen3-4b',
    'vllm-llama31-8b',
    'vllm-qwen3-8b',
    'vllm-qwen3-14b',
]

AGENT_COUNTS = [2, 3, 4, 5, 6, 7]

# Task-specific configuration
TASK_CONFIGS = {
    'math': {
        'rounds': 3,
        'num_problems': 100,
        'random_seed': 0
    },
    'gsm': {
        'rounds': 2,
        'num_problems': 100,
        'random_seed': 0
    },
    'biography': {
        'rounds': 2,
        'num_people': 40,
        'random_seed': 1
    },
    'mmlu': {
        'rounds': 2,
        'num_questions': 100,
        'random_seed': 0
    }
}


def generate_job_configs(output_dir: Path = None):
    """
    Generate all job configuration files.

    Args:
        output_dir: Directory to write config files (default: experiments/configs)
    """
    # Default output directory
    if output_dir is None:
        output_dir = project_root / 'experiments' / 'configs'

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load persona database once (will be cached)
    csv_path = project_root / 'personas' / 'summary_personas.csv'
    db = load_persona_database(str(csv_path))

    print("=" * 80)
    print("Generating SLURM Job Configuration Files")
    print("=" * 80)
    print(f"Persona database: {len(db)} rows")
    print(f"Models: {len(MODELS)}")
    print(f"Agent counts: {AGENT_COUNTS}")
    print(f"Output directory: {output_dir}")
    print()

    # Generate config for each task
    for task, task_cfg in TASK_CONFIGS.items():
        print(f"Task: {task}")
        print("-" * 80)

        job_id = 1
        rows = []
        skipped = []

        for model in MODELS:
            for n_agents in AGENT_COUNTS:
                try:
                    # Get optimal MaxDet personas for this model/agent count
                    personas = get_optimal_personas(
                        model,
                        n_agents,
                        version='v2',
                        selection_method='maxdet'
                    )

                    # Format personas as tuple string for CSV storage
                    personas_str = str(tuple(personas))

                    # Determine parameter name based on task
                    if task == 'biography':
                        num_param = 'num_people'
                        num_value = task_cfg['num_people']
                    elif task == 'mmlu':
                        num_param = 'num_questions'
                        num_value = task_cfg['num_questions']
                    else:
                        num_param = 'num_problems'
                        num_value = task_cfg['num_problems']

                    # Create job row
                    row = {
                        'job_id': job_id,
                        'model_alias': model,
                        'n_agents': n_agents,
                        'rounds': task_cfg['rounds'],
                        'task': task,
                        'num_param': num_param,
                        'num_value': num_value,
                        'random_seed': task_cfg['random_seed'],
                        'personas_tuple': personas_str
                    }
                    rows.append(row)
                    job_id += 1

                except Exception as e:
                    skipped.append(f"{model} with {n_agents} agents: {str(e)}")
                    continue

        # Write config file
        df = pd.DataFrame(rows)
        output_file = output_dir / f'persona_{task}_jobs.txt'
        df.to_csv(output_file, index=False)

        print(f"  Generated: {len(rows)} jobs")
        if skipped:
            print(f"  Skipped: {len(skipped)} configurations")
            for skip_msg in skipped[:3]:  # Show first 3
                print(f"    - {skip_msg}")
            if len(skipped) > 3:
                print(f"    ... and {len(skipped) - 3} more")
        print(f"  Output: {output_file}")
        print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Count total jobs
    total_jobs = 0
    for task in TASK_CONFIGS.keys():
        config_file = output_dir / f'persona_{task}_jobs.txt'
        if config_file.exists():
            df = pd.read_csv(config_file)
            job_count = len(df)
            total_jobs += job_count
            print(f"  {task:12s}: {job_count:3d} jobs")

    print(f"  {'Total':12s}: {total_jobs:3d} jobs")
    print()
    print("✓ All configuration files generated successfully!")
    print()
    print("Next steps:")
    print("  1. Review config files in experiments/configs/")
    print("  2. Run: bash experiments/slurm/submit_all_persona_experiments.sh")
    print("=" * 80)


if __name__ == '__main__':
    try:
        generate_job_configs()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
