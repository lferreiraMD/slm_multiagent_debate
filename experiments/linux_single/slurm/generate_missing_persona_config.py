#!/usr/bin/env python3
"""
Generate job config for missing persona experiments.

Compares planned persona jobs against existing result files and creates
a new config containing only the missing jobs.

Usage (on HPC):
    cd experiments/linux_single/slurm
    python3 generate_missing_persona_config.py
"""

import sys
from pathlib import Path
import csv

SCRIPT_DIR = Path(__file__).parent
LINUX_SINGLE_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = LINUX_SINGLE_DIR / "configs"
RESULTS_DIR = SCRIPT_DIR / "results_hpc"

# Task-specific result file patterns
TASK_EXTENSIONS = {
    'math': '.p',
    'gsm': '.json',
    'biography': '.json',
    'mmlu': '.json',
}


def parse_job_config(config_file: Path):
    """Parse persona job config CSV."""
    jobs = []
    with open(config_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(row)
    return jobs


def find_matching_result(job: dict, results_dir: Path) -> bool:
    """
    Check if a result file exists for this job.

    Uses glob pattern matching on (task, model, n_agents, rounds) since
    personas are fixed for each configuration.

    Returns:
        True if result exists, False otherwise
    """
    task = job['task']
    model = job['model_alias'].replace('vllm-', '')  # Remove vllm- prefix
    n_agents = job['n_agents']
    rounds = job['rounds']

    # Build filename pattern (personas can be anything)
    ext = TASK_EXTENSIONS[task]

    # Get short model name for filename
    model_short_map = {
        'qwen3-0.6b': 'Qwen3-0.6B',
        'vibethinker': 'VibeThinker-1.5B',
        'llama32-3b': 'Llama-3.2-3B',
        'mistral-7b': 'Mistral-7B-Instruct-v0.3',
        'qwen3-14b': 'Qwen3-14B',
    }
    model_short = model_short_map.get(model.lower(), model)

    # Use glob to find any file matching this pattern
    pattern = f"{task}_{model_short}_persona_*_agents{n_agents}_rounds{rounds}{ext}"
    task_dir = results_dir / task

    if not task_dir.exists():
        return False

    matches = list(task_dir.glob(pattern))
    return len(matches) > 0


def check_missing_jobs(original_config: Path, results_dir: Path):
    """
    Identify which jobs are missing results.

    Returns:
        List of job dicts that need to be run
    """
    jobs = parse_job_config(original_config)
    missing_jobs = []

    print(f"Checking {len(jobs)} jobs against existing results...")
    print()

    for job in jobs:
        task = job['task']
        model = job['model_alias']
        n_agents = job['n_agents']
        rounds = job['rounds']

        result_exists = find_matching_result(job, results_dir)

        if not result_exists:
            missing_jobs.append(job)
            print(f"  ✗ Missing: {task} - {model} (agents={n_agents}, rounds={rounds})")
        else:
            print(f"  ✓ Exists: {task} - {model} (agents={n_agents}, rounds={rounds})")

    return missing_jobs


def write_filtered_config(missing_jobs: list, output_file: Path):
    """Write new job config with only missing jobs."""
    if not missing_jobs:
        print("\n✓ All jobs completed! No missing results.")
        return

    # Get fieldnames from first job
    fieldnames = list(missing_jobs[0].keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_jobs)

    print(f"\n✓ Wrote {len(missing_jobs)} missing jobs to: {output_file}")


def main():
    """Main script."""
    print("=" * 70)
    print("Generate Missing Persona Job Config")
    print("=" * 70)
    print()

    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return 1

    # Process each task
    tasks = ['math', 'gsm', 'biography', 'mmlu']
    all_missing_jobs = []

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print('='*70)

        # Input file
        original_config = CONFIGS_DIR / f"persona_{task}_jobs.txt"

        if not original_config.exists():
            print(f"  Warning: Config not found: {original_config}")
            continue

        # Find missing jobs for this task
        missing_jobs = check_missing_jobs(original_config, RESULTS_DIR)
        all_missing_jobs.extend(missing_jobs)

        print(f"  → {len(missing_jobs)} missing jobs for {task}")

    # Write combined filtered config
    print()
    print("=" * 70)
    print(f"Summary: {len(all_missing_jobs)} total jobs need to run")
    print("=" * 70)

    output_config = SCRIPT_DIR / "persona_jobs_missing.txt"
    write_filtered_config(all_missing_jobs, output_config)

    return 0


if __name__ == "__main__":
    sys.exit(main())

