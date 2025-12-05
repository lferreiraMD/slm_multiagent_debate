#!/usr/bin/env python3
"""
Aggregate and analyze persona experiment results from HPC runs.

Scans results_hpc/ directory for completed persona experiments and computes
accuracy metrics for each task.

Usage:
    cd experiments/linux_single/slurm
    python3 aggregate_persona_results.py
"""

import os
import sys
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import most_frequent
from utils.config import load_config


def get_full_model_name(short_name: str) -> str:
    """
    Get full model name from config.yaml by matching against model paths.

    Args:
        short_name: Model name from filename (e.g., "Llama-3.2-3B")

    Returns:
        Full model name from HuggingFace path (e.g., "Llama-3.2-3B-Instruct")
    """
    try:
        config = load_config()
        models = config.get('models', {})

        # Search for model alias whose path contains the short name
        for alias, path in models.items():
            # Extract model name from path (e.g., "meta-llama/Llama-3.2-3B-Instruct" → "Llama-3.2-3B-Instruct")
            if '/' in path:
                full_name = path.split('/')[-1]
            else:
                full_name = path

            # Check if short name matches (case-insensitive, partial match)
            if short_name.lower() in full_name.lower():
                return full_name

        # If no match, return original
        return short_name

    except Exception as e:
        print(f"Warning: Failed to resolve model name '{short_name}': {e}")
        return short_name


def parse_filename(filename: str, task: str) -> Optional[Dict]:
    """
    Parse result filename to extract metadata.

    Format examples:
        math_Qwen3-0.6B_persona_kantian+deep-sea+renaissance_agents3_rounds3.p
        gsm_Mistral-7B_persona_quantum+zen_agents5_rounds3.json

    Returns:
        Dict with: model_name, personas, num_agents, num_rounds
    """
    try:
        # Remove task prefix and extension
        name = filename.replace(f"{task}_", "").replace(".p", "").replace(".json", "")

        # Extract pattern: ModelName_persona_persona1+persona2+..._agentsN_roundsM
        pattern = r"^(.+?)_persona_(.+?)_agents(\d+)_rounds(\d+)$"
        match = re.match(pattern, name)

        if not match:
            return None

        model_name = match.group(1)
        personas_str = match.group(2)
        num_agents = int(match.group(3))
        num_rounds = int(match.group(4))

        # Parse personas (abbreviated names separated by +)
        personas = personas_str.split('+')

        return {
            'model_name': model_name,
            'personas': personas,
            'num_personas': len(personas),
            'num_agents': num_agents,
            'num_rounds': num_rounds,
            'filename': filename,
        }
    except Exception as e:
        print(f"Warning: Failed to parse filename {filename}: {e}")
        return None


def parse_math_answer(text: str) -> Optional[int]:
    """Extract numerical answer from math response."""
    import re

    # Try to find the last number in the response
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate_math(data: Dict) -> Tuple[float, float, int]:
    """
    Evaluate math task results.

    Args:
        data: Dict mapping problem -> (agent_contexts, ground_truth)

    Returns:
        (mean_accuracy, std_accuracy, num_problems)
    """
    scores = []

    for problem, (agent_contexts, ground_truth) in data.items():
        # Extract answers from all agents' final responses
        text_answers = []
        for agent_context in agent_contexts:
            # Get last message content
            if agent_context and len(agent_context) > 0:
                last_msg = agent_context[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    text_answer = parse_math_answer(last_msg['content'])
                    if text_answer is not None:
                        text_answers.append(text_answer)

        # Majority vote
        if text_answers:
            final_answer = most_frequent(text_answers)
            scores.append(1.0 if final_answer == ground_truth else 0.0)

    if not scores:
        return (0.0, 0.0, 0)

    return (np.mean(scores), np.std(scores), len(scores))


def parse_gsm_answer(text: str) -> Optional[float]:
    """Extract numerical answer from GSM response (looks for \\boxed{} format)."""
    import re

    # Look for \boxed{number} format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        try:
            # Extract number, handle commas
            num_str = boxed_match.group(1).replace(',', '')
            return float(num_str)
        except:
            pass

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None


def evaluate_gsm(data: Dict) -> Tuple[float, float, int]:
    """
    Evaluate GSM task results.

    Args:
        data: Dict mapping question -> ([agent_conversations], ground_truth_string)

    Returns:
        (mean_accuracy, std_accuracy, num_problems)
    """
    scores = []

    for question, problem_data in data.items():
        if not isinstance(problem_data, list) or len(problem_data) < 2:
            continue

        agent_conversations = problem_data[0]  # List of agent conversation lists
        ground_truth_str = problem_data[1]     # String with answer

        # Extract ground truth number from string (e.g., "#### 52")
        import re
        gt_match = re.search(r'####\s*(\d+)', ground_truth_str)
        if not gt_match:
            continue
        ground_truth = float(gt_match.group(1))

        # Extract answers from all agents
        text_answers = []
        for agent_conv in agent_conversations:
            if isinstance(agent_conv, list) and len(agent_conv) > 0:
                # Get last message in conversation
                last_msg = agent_conv[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    answer = parse_gsm_answer(last_msg['content'])
                    if answer is not None:
                        text_answers.append(answer)

        # Majority vote
        if text_answers:
            final_answer = most_frequent(text_answers)
            scores.append(1.0 if abs(final_answer - ground_truth) < 0.01 else 0.0)

    if not scores:
        return (0.0, 0.0, 0)

    return (np.mean(scores), np.std(scores), len(scores))


def evaluate_mmlu(data: Dict) -> Tuple[float, float, int]:
    """
    Evaluate MMLU task results.

    Args:
        data: Dict mapping question -> ([agent_conversations], correct_answer)

    Returns:
        (mean_accuracy, std_accuracy, num_problems)
    """
    scores = []

    for question, problem_data in data.items():
        if not isinstance(problem_data, list) or len(problem_data) < 2:
            continue

        agent_conversations = problem_data[0]  # List of agent conversation lists
        correct_answer = problem_data[1]       # Single letter: 'A', 'B', 'C', or 'D'

        # Extract answers from all agents
        text_answers = []
        for agent_conv in agent_conversations:
            if isinstance(agent_conv, list) and len(agent_conv) > 0:
                # Get last message in conversation
                last_msg = agent_conv[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    content = last_msg['content']
                    # Look for answer letter
                    import re
                    match = re.search(r'\b([A-D])\b', content)
                    if match:
                        text_answers.append(match.group(1))

        # Majority vote
        if text_answers:
            final_answer = most_frequent(text_answers)
            scores.append(1.0 if final_answer == correct_answer else 0.0)

    if not scores:
        return (0.0, 0.0, 0)

    return (np.mean(scores), np.std(scores), len(scores))


def load_persona_metrics() -> pd.DataFrame:
    """
    Load persona diversity metrics from summary_personas.csv.

    Returns:
        DataFrame with model, n_agents, persona_version, maxmin_score, maxdet_volume
    """
    personas_csv = PROJECT_ROOT / "personas" / "summary_personas.csv"
    if not personas_csv.exists():
        print(f"Warning: {personas_csv} not found, skipping diversity metrics")
        return pd.DataFrame()

    df = pd.read_csv(personas_csv)
    # Keep only columns we need
    return df[['model', 'n_agents', 'persona_version', 'maxmin_score', 'maxdet_volume']]


def process_results_directory(results_dir: Path) -> pd.DataFrame:
    """
    Process all result files in results_hpc directory.

    Returns:
        DataFrame with columns: task, model_name, num_agents, num_rounds,
                               num_personas, personas_str, avg_accuracy,
                               std_accuracy, num_problems, filename,
                               maxmin_score, maxdet_volume
    """
    records = []

    tasks = ['math', 'gsm', 'biography', 'mmlu']

    for task in tasks:
        task_dir = results_dir / task
        if not task_dir.exists():
            print(f"Warning: {task_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing {task} results...")

        # Get all result files
        if task == 'math':
            files = list(task_dir.glob("*.p"))
        else:
            files = list(task_dir.glob("*.json"))

        print(f"  Found {len(files)} result files")

        for filepath in files:
            filename = filepath.name

            # Parse filename
            metadata = parse_filename(filename, task)
            if not metadata:
                print(f"  ⚠ Skipping {filename} (parse error)")
                continue

            # Load and evaluate results
            try:
                if task == 'math':
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    mean_acc, std_acc, num_problems = evaluate_math(data)

                elif task == 'gsm':
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    mean_acc, std_acc, num_problems = evaluate_gsm(data)

                elif task == 'mmlu':
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    mean_acc, std_acc, num_problems = evaluate_mmlu(data)

                elif task == 'biography':
                    # Biography requires LLM evaluation - skip for now
                    print(f"  ℹ {filename}: Biography requires manual evaluation")
                    mean_acc, std_acc, num_problems = (None, None, None)

                # Create record with normalized model name for persona matching
                record = {
                    'task': task,
                    'model_name': get_full_model_name(metadata['model_name']),
                    'num_agents': metadata['num_agents'],
                    'num_rounds': metadata['num_rounds'],
                    'num_personas': metadata['num_personas'],
                    'personas_str': '+'.join(metadata['personas']),
                    'avg_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'num_problems': num_problems,
                    'filename': filename,
                }

                records.append(record)

                if mean_acc is not None:
                    print(f"  ✓ {filename}: {mean_acc:.1%} accuracy")

            except Exception as e:
                print(f"  ✗ {filename}: Error - {e}")
                continue

    # Create DataFrame
    df = pd.DataFrame(records)

    # Load and join persona diversity metrics
    persona_metrics = load_persona_metrics()
    if not persona_metrics.empty:
        # All HPC experiments use v2 personas
        persona_metrics_v2 = persona_metrics[persona_metrics['persona_version'] == 'v2'].copy()

        # Join on model_name and num_agents
        df = df.merge(
            persona_metrics_v2[['model', 'n_agents', 'maxmin_score', 'maxdet_volume']],
            left_on=['model_name', 'num_agents'],
            right_on=['model', 'n_agents'],
            how='left'
        )

        # Drop duplicate 'model' column from join
        if 'model' in df.columns:
            df = df.drop(columns=['model'])

        # Report any missing matches
        missing = df['maxmin_score'].isna().sum()
        if missing > 0:
            print(f"\nWarning: {missing}/{len(df)} experiments missing diversity metrics")
            print("Models without matches:")
            for model in df[df['maxmin_score'].isna()]['model_name'].unique():
                print(f"  - {model}")

    return df


def main():
    """Main aggregation script."""
    print("=" * 70)
    print("Persona Experiment Results Aggregator")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results_hpc"
    output_dir = script_dir

    if not results_dir.exists():
        print(f"\nError: {results_dir} does not exist!")
        print("Expected structure:")
        print("  slurm/results_hpc/")
        print("    ├── math/")
        print("    ├── gsm/")
        print("    ├── biography/")
        print("    └── mmlu/")
        return 1

    print(f"\nScanning: {results_dir}")

    # Process all results
    df = process_results_directory(results_dir)

    if df.empty:
        print("\n⚠ No results found to aggregate")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print(f"\nTotal experiments: {len(df)}")
    print(f"  Tasks: {df['task'].value_counts().to_dict()}")
    print(f"  Models: {df['model_name'].nunique()} unique")
    print(f"  Agent counts: {sorted(df['num_agents'].unique())}")

    # Average accuracy by task
    print("\nAccuracy by Task:")
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        valid = task_df.dropna(subset=['avg_accuracy'])
        if not valid.empty:
            mean_acc = valid['avg_accuracy'].mean()
            print(f"  {task:12s}: {mean_acc:.1%} (n={len(valid)})")

    # Save results
    output_csv = output_dir / "persona_results_summary.csv"
    output_pkl = output_dir / "persona_results_summary.p"

    df.to_csv(output_csv, index=False)
    df.to_pickle(output_pkl)

    print(f"\n✓ Results saved:")
    print(f"  CSV: {output_csv}")
    print(f"  Pickle: {output_pkl}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

