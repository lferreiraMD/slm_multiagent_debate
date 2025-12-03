#!/usr/bin/env python3
"""
Aggregate baseline results from linux_single experiments.

Scans experiments/linux_single/results/ directory for all result files,
computes accuracy for math, GSM, and MMLU tasks, and creates a single
comprehensive DataFrame with all results.

Usage:
    python experiments/linux_single/aggregate_baseline_results.py
"""

import sys
from pathlib import Path
import pickle
import json
import re
import pandas as pd
import numpy as np
from glob import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.helpers import most_frequent


def parse_filename(filepath):
    """
    Extract task, model, agents, rounds, diversity type from result filename.

    Returns:
        dict with keys: task, model_name, num_agents, num_rounds, diversity_type, filepath
        or None if parsing fails
    """
    filename = Path(filepath).name

    if '_' not in filename:
        print(f"Warning: Could not parse filename: {filename}")
        return None

    task = filename.split('_')[0]

    # Extract agents and rounds (always at the end)
    agents_rounds_pattern = r'agents(\d+)_rounds(\d+)\.(p|json)$'
    agents_rounds_match = re.search(agents_rounds_pattern, filename)

    if not agents_rounds_match:
        print(f"Warning: Could not parse agents/rounds in filename: {filename}")
        return None

    num_agents = int(agents_rounds_match.group(1))
    num_rounds = int(agents_rounds_match.group(2))

    # Extract middle part (between task and agents/rounds)
    middle_part = filename[len(task)+1:agents_rounds_match.start()-1]

    # Determine diversity type and model name
    diversity_type = 'baseline'
    model_name = middle_part

    # Check for temperature diversity
    if 'temp' in middle_part:
        diversity_type = 'temperature'
        model_name = middle_part.split('_temp')[0]

    # Check for persona diversity
    elif 'persona' in middle_part:
        diversity_type = 'persona'
        model_name = middle_part.split('_persona')[0]

    # Check for model diversity
    elif '+' in middle_part and 'temp' not in middle_part:
        diversity_type = 'model'

    return {
        'task': task,
        'model_name': model_name,
        'num_agents': num_agents,
        'num_rounds': num_rounds,
        'diversity_type': diversity_type,
        'filepath': filepath
    }


def parse_math_answer(sentence):
    """Extract numerical answer from text."""
    if sentence is None:
        return None

    parts = str(sentence).split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue

    return None


def evaluate_math(data):
    """Evaluate math task results."""
    scores = []

    for problem, (agent_contexts, ground_truth) in data.items():
        text_answers = []

        for agent_context in agent_contexts:
            if len(agent_context) == 0:
                continue

            text_answer = agent_context[-1]['content']
            text_answer = str(text_answer).replace(",", ".")
            text_answer = parse_math_answer(text_answer)

            if text_answer is not None:
                text_answers.append(text_answer)

        if len(text_answers) == 0:
            continue

        try:
            final_answer = most_frequent(text_answers)
            if final_answer == ground_truth:
                scores.append(1.0)
            else:
                scores.append(0.0)
        except:
            continue

    if len(scores) == 0:
        return None

    return (np.mean(scores), np.std(scores))


def parse_gsm_answer(input_str):
    """Extract answer from \\boxed{answer} format."""
    pattern = r"\\boxed\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def solve_gsm_answer(input_str):
    """Fallback: extract last number from text."""
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None


def evaluate_gsm(data):
    """Evaluate GSM task results."""
    scores = []

    for question, (agent_contexts, ground_truth) in data.items():
        pred_answers = []

        for agent_context in agent_contexts:
            if len(agent_context) == 0:
                continue

            pred_solution = agent_context[-1]['content']

            # Try boxed format first
            pred_answer = parse_gsm_answer(pred_solution)
            if pred_answer is None:
                # Fallback to last number
                pred_answer = solve_gsm_answer(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if len(pred_answers) == 0:
            continue

        try:
            final_answer = most_frequent(pred_answers)
            gt_answer = solve_gsm_answer(ground_truth)

            if gt_answer is not None and final_answer is not None:
                if float(gt_answer) == float(final_answer):
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        except:
            continue

    if len(scores) == 0:
        return None

    return (np.mean(scores), np.std(scores))


def parse_mmlu_answer(text):
    """Extract answer from (X) format at end of response."""
    pattern = r'\(([A-D])\)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None


def evaluate_mmlu(data):
    """Evaluate MMLU task results."""
    scores = []

    for question, (agent_contexts, ground_truth) in data.items():
        pred_answers = []

        for agent_context in agent_contexts:
            if len(agent_context) == 0:
                continue

            pred_solution = agent_context[-1]['content']
            pred_answer = parse_mmlu_answer(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if len(pred_answers) == 0:
            continue

        try:
            final_answer = most_frequent(pred_answers)

            if final_answer == ground_truth:
                scores.append(1.0)
            else:
                scores.append(0.0)
        except:
            continue

    if len(scores) == 0:
        return None

    return (np.mean(scores), np.std(scores))


def evaluate_result_file(filepath, task):
    """
    Load and evaluate a result file.

    Returns:
        (mean_accuracy, std_accuracy, num_problems) or None if evaluation fails
    """
    try:
        # Load data
        if filepath.endswith('.p'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            return None

        num_problems = len(data)

        # Evaluate based on task
        if task == 'math':
            result = evaluate_math(data)
        elif task == 'gsm':
            result = evaluate_gsm(data)
        elif task == 'mmlu':
            result = evaluate_mmlu(data)
        elif task == 'biography':
            # Biography requires LLM-based evaluation - skip for now
            return None
        else:
            return None

        if result is None:
            return None

        avg_acc, std_acc = result
        return (avg_acc, std_acc, num_problems)

    except Exception as e:
        print(f"  Error evaluating {filepath}: {e}")
        return None


def aggregate_results(results_dir):
    """
    Scan results directory and aggregate all results into DataFrame.

    Args:
        results_dir: Path to experiments/linux_single/results/

    Returns:
        pandas DataFrame
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    # Find all result files in task subdirectories
    result_files = []
    task_counts = {}

    for task_subdir in ['math', 'gsm', 'biography', 'mmlu']:
        task_path = results_dir / task_subdir
        if task_path.exists():
            p_files = glob(str(task_path / "*.p"))
            json_files = glob(str(task_path / "*.json"))
            count = len(p_files) + len(json_files)
            task_counts[task_subdir] = count
            print(f"  {task_subdir}: {len(p_files)} .p files, {len(json_files)} .json files")
            result_files.extend(p_files)
            result_files.extend(json_files)
        else:
            task_counts[task_subdir] = 0
            print(f"  {task_subdir}: directory not found")

    total_files = sum(task_counts.values())
    print(f"\nTotal: {total_files} result files")

    # Process each file
    rows = []
    processed = 0
    skipped = 0

    for filepath in result_files:
        # Parse filename
        metadata = parse_filename(filepath)
        if metadata is None:
            skipped += 1
            continue

        # Evaluate results
        result = evaluate_result_file(filepath, metadata['task'])
        if result is None:
            if metadata['task'] == 'biography':
                # Biography skipped (requires LLM evaluation)
                skipped += 1
            else:
                print(f"  Warning: Evaluation failed for {Path(filepath).name}")
                skipped += 1
            continue

        avg_acc, std_acc, num_problems = result
        processed += 1

        # Add to results
        row = {
            'task': metadata['task'],
            'model_name': metadata['model_name'],
            'num_agents': metadata['num_agents'],
            'num_rounds': metadata['num_rounds'],
            'diversity_type': metadata['diversity_type'],
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'num_problems': num_problems,
            'filename': Path(filepath).name
        }
        rows.append(row)

    print(f"\nProcessed: {processed} files")
    print(f"Skipped: {skipped} files (including {task_counts.get('biography', 0)} biography files)")

    # Create DataFrame
    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(['task', 'diversity_type', 'model_name', 'num_agents', 'num_rounds'])

    return df


def main():
    """Main entry point."""
    print("=" * 70)
    print("Linux Single GPU - Baseline Results Aggregation")
    print("=" * 70)

    # Get directories
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'

    print(f"Results directory: {results_dir}")
    print(f"Scanning for result files...\n")

    # Aggregate results
    df = aggregate_results(results_dir)

    if len(df) == 0:
        print("\nNo results found to aggregate.")
        return

    # Display summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY BY TASK")
    print("=" * 70)

    for task in ['math', 'gsm', 'mmlu']:
        task_df = df[df['task'] == task]
        if len(task_df) > 0:
            print(f"\n{task.upper()}:")
            print(f"  Total experiments: {len(task_df)}")
            print(f"  Avg accuracy: {task_df['avg_accuracy'].mean():.3f} ± {task_df['avg_accuracy'].std():.3f}")
            print(f"  Best accuracy: {task_df['avg_accuracy'].max():.3f}")
            print(f"  Worst accuracy: {task_df['avg_accuracy'].min():.3f}")

    # Display full results table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    display_df = df[['task', 'model_name', 'num_agents', 'num_rounds',
                     'diversity_type', 'avg_accuracy', 'std_accuracy', 'num_problems']]
    print(display_df.to_string(index=False))

    # Save results
    output_csv = script_dir / 'aggregated_results.csv'
    output_pkl = script_dir / 'aggregated_results.pkl'

    df.to_csv(output_csv, index=False)
    df.to_pickle(output_pkl)

    print("\n" + "=" * 70)
    print("Results saved:")
    print(f"  CSV: {output_csv}")
    print(f"  Pickle: {output_pkl}")
    print(f"  Total entries: {len(df)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
