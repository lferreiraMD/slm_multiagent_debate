#!/usr/bin/env python3
"""
Aggregate results from all multiagent debate experiments.

Scans results/ directory for .p and .json result files, computes accuracy
for each run, and maintains a summary DataFrame.

Usage:
    python scripts/aggregate_results.py
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import most_frequent


def parse_filename(filepath):
    """
    Extract task, model, agents, rounds, diversity type from result filename.

    Expected formats:
    - math_{model}_agents{N}_rounds{N}.p
    - gsm_{model}_agents{N}_rounds{N}.json
    - biography_{model}_agents{N}_rounds{N}.json
    - mmlu_{model}_agents{N}_rounds{N}.json
    - gsm_{model}_temp0.7+1.0+1.3_agents{N}_rounds{N}.json  (temperature diversity)
    - gsm_{model}_persona_{personas}_agents{N}_rounds{N}.json  (persona diversity)
    - gsm_{model1+model2+model3}_agents{N}_rounds{N}.json  (model diversity)

    Returns:
        dict with keys: task, model_name, num_agents, num_rounds, diversity_type, filepath
        or None if parsing fails
    """
    filename = Path(filepath).name

    # Extract task (first component before _)
    if not '_' in filename:
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
    diversity_type = 'baseline'  # Default
    model_name = middle_part

    # Check for temperature diversity (pattern: temp{float}+{float}+...)
    if 'temp' in middle_part:
        diversity_type = 'temperature'
        # Extract model name (everything before _temp)
        model_name = middle_part.split('_temp')[0]

    # Check for persona diversity (pattern: persona_{personas})
    elif 'persona' in middle_part:
        diversity_type = 'persona'
        # Extract model name (everything before _persona)
        model_name = middle_part.split('_persona')[0]

    # Check for model diversity (pattern: {model1+model2+model3})
    elif '+' in middle_part:
        diversity_type = 'model'
        # model_name stays as is (amalgamated name)

    return {
        'task': task,
        'model_name': model_name,
        'num_agents': num_agents,
        'num_rounds': num_rounds,
        'diversity_type': diversity_type,
        'filepath': filepath
    }


def parse_math_answer(sentence):
    """Extract numerical answer from text (for math task)."""
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
    """
    Evaluate math task results.

    Args:
        data: Dict with problem tuples as keys, (agent_contexts, answer) as values

    Returns:
        (mean_accuracy, std_accuracy) or None if failed
    """
    scores = []

    for problem, (agent_contexts, ground_truth) in data.items():
        text_answers = []

        for agent_context in agent_contexts:
            # Get final response from agent
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
            # Majority vote
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
    """Extract answer from \\boxed{answer} format (for GSM)."""
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
    """
    Evaluate GSM task results.

    Args:
        data: Dict with question as key, (agent_contexts, answer) as values

    Returns:
        (mean_accuracy, std_accuracy) or None if failed
    """
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
            # Majority vote
            final_answer = most_frequent(pred_answers)

            # Extract ground truth number
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


def evaluate_biography(data):
    """
    Evaluate biography task results.

    For now, returns None (requires LLM-based fact-checking).
    TODO: Implement simplified evaluation or skip.

    Args:
        data: Dict with person as key, agent_contexts as values

    Returns:
        None (not implemented)
    """
    # Biography evaluation requires LLM-based fact-checking
    # Skip for now - would need to implement or import eval_conversation.py logic
    print("Warning: Biography evaluation not implemented (requires LLM fact-checking)")
    return None


def parse_mmlu_answer(text):
    """Extract answer from (X) format at end of response."""
    # Look for (A), (B), (C), or (D) at the end
    pattern = r'\(([A-D])\)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # Take last match
    return None


def evaluate_mmlu(data):
    """
    Evaluate MMLU task results.

    Args:
        data: Dict with question as key, (agent_contexts, answer) as values

    Returns:
        (mean_accuracy, std_accuracy) or None if failed
    """
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
            # Majority vote
            final_answer = most_frequent(pred_answers)

            # Ground truth is already a letter (A, B, C, D)
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

    Args:
        filepath: Path to .p or .json result file
        task: Task name (math, gsm, biography, mmlu)

    Returns:
        (mean_accuracy, std_accuracy) or None if evaluation fails
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
            print(f"Warning: Unknown file format: {filepath}")
            return None

        # Evaluate based on task
        if task == 'math':
            return evaluate_math(data)
        elif task == 'gsm':
            return evaluate_gsm(data)
        elif task == 'biography':
            return evaluate_biography(data)
        elif task == 'mmlu':
            return evaluate_mmlu(data)
        else:
            print(f"Warning: Unknown task: {task}")
            return None

    except Exception as e:
        print(f"Error evaluating {filepath}: {e}")
        return None


def aggregate_results(tasks_dir='tasks'):
    """
    Scan task directories and aggregate all results into DataFrame.

    Args:
        tasks_dir: Directory containing task subdirectories (math, gsm, biography, mmlu)

    Returns:
        pandas DataFrame with columns: task, model_name, number_agents,
        number_rounds, diversity_type, average_accuracy, stdev_accuracy
    """
    tasks_dir = Path(tasks_dir)

    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return pd.DataFrame(columns=['task', 'model_name', 'num_agents', 'num_rounds',
                                     'diversity_type', 'avg_accuracy', 'std_accuracy'])

    # Find all result files in task subdirectories
    result_files = []

    # Scan each task directory
    for task_subdir in ['math', 'gsm', 'biography', 'mmlu']:
        task_path = tasks_dir / task_subdir
        if task_path.exists():
            p_files = glob(str(task_path / "*.p"))
            json_files = glob(str(task_path / "*.json"))
            print(f"  {task_subdir}: {len(p_files)} .p files, {len(json_files)} .json files")
            result_files.extend(p_files)
            result_files.extend(json_files)
        else:
            print(f"  {task_subdir}: directory not found")

    print(f"\nTotal: {len(result_files)} result files across task directories")

    # Process each file
    rows = []

    for filepath in result_files:
        print(f"\nProcessing: {filepath}")

        # Parse filename
        metadata = parse_filename(filepath)
        if metadata is None:
            continue

        # Evaluate results
        result = evaluate_result_file(filepath, metadata['task'])
        if result is None:
            print(f"  Skipping (evaluation failed or not implemented)")
            continue

        avg_acc, std_acc = result

        # Add to results
        row = {
            'task': metadata['task'],
            'model_name': metadata['model_name'],
            'num_agents': metadata['num_agents'],
            'num_rounds': metadata['num_rounds'],
            'diversity_type': metadata['diversity_type'],
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc
        }
        rows.append(row)

        print(f"  Diversity: {metadata['diversity_type']}, Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df


def main():
    """Main entry point."""
    print("=" * 60)
    print("Multiagent Debate Results Aggregation")
    print("=" * 60)

    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Project root: {project_root}")
    print(f"Scanning for results...\n")

    # Aggregate results from task directories
    tasks_dir = project_root / 'tasks'
    df = aggregate_results(tasks_dir)

    if len(df) == 0:
        print("\nNo results found to aggregate.")
        return

    # Sort by task, model, diversity_type, agents, rounds
    df = df.sort_values(['task', 'model_name', 'diversity_type', 'num_agents', 'num_rounds'])

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save results to project root
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / 'summary.p'

    # Load existing if it exists and append
    if output_path.exists():
        print(f"\nLoading existing results from {output_path}")
        try:
            existing_df = pd.read_pickle(output_path)

            # Merge with new results (avoid duplicates)
            # Create unique key (include diversity_type)
            df['_key'] = df.apply(lambda row: f"{row['task']}_{row['model_name']}_{row['diversity_type']}_a{row['num_agents']}_r{row['num_rounds']}", axis=1)
            existing_df['_key'] = existing_df.apply(lambda row: f"{row['task']}_{row['model_name']}_{row.get('diversity_type', 'baseline')}_a{row['num_agents']}_r{row['num_rounds']}", axis=1)

            # Keep new results, add existing ones that aren't duplicates
            new_keys = set(df['_key'])
            existing_unique = existing_df[~existing_df['_key'].isin(new_keys)]

            # Combine
            df = pd.concat([df, existing_unique], ignore_index=True)
            df = df.drop('_key', axis=1)
            df = df.sort_values(['task', 'model_name', 'diversity_type', 'num_agents', 'num_rounds'])

            print(f"Merged with {len(existing_unique)} existing results")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            print("Saving new results only")

    # Save
    df.to_pickle(output_path)
    print(f"\nResults saved to: {output_path}")
    print(f"Total entries: {len(df)}")

    # Also save as CSV for easy viewing
    csv_path = results_dir / 'summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
