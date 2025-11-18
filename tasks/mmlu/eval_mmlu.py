"""
Evaluate MMLU (Massive Multitask Language Understanding) multiagent debate results.

This script evaluates multiple-choice question answering performance by:
1. Extracting predicted answers (A/B/C/D) from agent responses
2. Using majority voting across multiple agents
3. Comparing with ground truth answers
4. Computing accuracy metrics

Usage:
    python3 eval_mmlu.py --input-file mmlu_model_agents3_rounds2.json
    python3 eval_mmlu.py --input-file results.json --debug
"""

import sys
from pathlib import Path
import json
import numpy as np
import re
import argparse
from collections import Counter
from typing import List, Optional


def parse_answer(input_str: str) -> Optional[str]:
    """
    Extract multiple-choice answer (A/B/C/D) from model response.

    Tries multiple patterns in order of preference:
    1. (X) format - e.g., "The answer is (B)"
    2. Answer: X format - e.g., "Answer: B"
    3. X) format at line start - e.g., "B) is correct"
    4. Standalone letter - e.g., "B"

    Args:
        input_str: Model's response text

    Returns:
        Single letter (A/B/C/D) in uppercase, or None if no answer found

    Examples:
        >>> parse_answer("I think the answer is (B)")
        'B'
        >>> parse_answer("Answer: C is correct")
        'C'
        >>> parse_answer("The correct choice is D)")
        'D'
    """
    if not input_str:
        return None

    # Pattern 1: (X) format - most common in prompts
    pattern1 = r'\(([A-Da-d])\)'
    matches = re.findall(pattern1, input_str)
    if matches:
        # Take the last match (final answer after reasoning)
        return matches[-1].upper()

    # Pattern 2: "Answer: X" or "answer is X" format
    pattern2 = r'[Aa]nswer[:\s]+([A-Da-d])\b'
    matches = re.findall(pattern2, input_str)
    if matches:
        return matches[-1].upper()

    # Pattern 3: X) format at line boundaries
    pattern3 = r'(?:^|\n)\s*([A-Da-d])\)'
    matches = re.findall(pattern3, input_str)
    if matches:
        return matches[-1].upper()

    # Pattern 4: Standalone letter (A/B/C/D) - last resort
    # Only match if it appears near end of response (last 200 chars)
    tail = input_str[-200:] if len(input_str) > 200 else input_str
    pattern4 = r'\b([A-Da-d])\b'
    matches = re.findall(pattern4, tail)
    if matches:
        return matches[-1].upper()

    return None


def most_frequent(items: List[str]) -> Optional[str]:
    """
    Find most common item in list using majority voting.

    Args:
        items: List of items to vote on

    Returns:
        Most frequent item, or None if list is empty or all None

    Examples:
        >>> most_frequent(['A', 'B', 'A', 'A'])
        'A'
        >>> most_frequent(['A', 'B'])  # Tie goes to first occurrence
        'A'
        >>> most_frequent([])
        None
    """
    if not items:
        return None

    # Filter out None values
    valid_items = [item for item in items if item is not None]
    if not valid_items:
        return None

    # Use Counter for efficient counting
    counts = Counter(valid_items)
    most_common = counts.most_common(1)

    return most_common[0][0] if most_common else None


def compute_accuracy(gt: str, pred_solutions: List[str], debug: bool = False) -> int:
    """
    Compute accuracy for a single question with multiple agent predictions.

    Uses majority voting: extracts answer from each agent's response,
    then selects the most common answer for comparison with ground truth.

    Args:
        gt: Ground truth answer (single letter A/B/C/D)
        pred_solutions: List of predicted solution strings from multiple agents
        debug: If True, print detailed parsing information

    Returns:
        1 if correct, 0 if incorrect

    Examples:
        >>> compute_accuracy("B", ["I choose (B)", "Answer is (B)", "(A) is wrong"])
        1
        >>> compute_accuracy("A", ["(B)", "(C)", "(B)"])
        0
    """
    # Normalize ground truth
    gt = gt.strip().upper()

    # Extract answers from each agent
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)
        elif debug:
            print(f"  [WARNING] Could not parse answer from: {pred_solution[:100]}...")

    # Check if we got any valid predictions
    if not pred_answers:
        if debug:
            print(f"  [ERROR] No valid answers parsed from {len(pred_solutions)} agent responses")
        return 0

    # Use majority voting
    pred_answer = most_frequent(pred_answers)

    if pred_answer is None:
        if debug:
            print(f"  [ERROR] Majority voting returned None")
        return 0

    # Compare with ground truth
    correct = (gt == pred_answer)

    if debug:
        print(f"  GT: {gt}, Predictions: {pred_answers}, Majority: {pred_answer}, Correct: {correct}")

    return 1 if correct else 0


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU task results from multiagent debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 eval_mmlu.py --input-file mmlu_llama32-3b_agents3_rounds2.json
    python3 eval_mmlu.py --input-file results.json --debug
        """
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to JSON file with debate results (e.g., mmlu_model_agents3_rounds2.json)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing answer parsing details"
    )
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)

    # Load results
    print("=" * 60)
    print("MMLU EVALUATION")
    print("=" * 60)
    print(f"Input file: {args.input_file}")

    try:
        with open(input_path, 'r') as f:
            response_dict = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    questions = list(response_dict.keys())
    print(f"Total questions: {len(questions)}")
    print("=" * 60)

    accuracies = []
    correct_count = 0

    for idx, question in enumerate(questions, 1):
        responses, gt = response_dict[question]

        # Extract final response from each agent
        pred_solutions = []
        for response in responses:
            if response and isinstance(response, list) and len(response) > 0:
                # Get last message content (final answer after all rounds)
                pred_solution = response[-1]['content']
                pred_solutions.append(pred_solution)
            elif args.debug:
                print(f"[WARNING] Invalid response structure for question {idx}")

        # Compute accuracy for this question
        accurate = compute_accuracy(gt, pred_solutions, debug=args.debug)

        if accurate is not None:
            accuracies.append(float(accurate))
            if accurate == 1:
                correct_count += 1

        # Print running statistics every 10 questions
        if idx % 10 == 0 or idx == len(questions):
            mean_acc = np.mean(accuracies)
            std_err = np.std(accuracies) / (len(accuracies) ** 0.5)
            print(f"Progress: {idx}/{len(questions)} questions | "
                  f"Accuracy: {mean_acc:.3f} ± {std_err:.3f} | "
                  f"Correct: {correct_count}/{idx}")

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Questions evaluated: {len(accuracies)}")
    print(f"Correct answers: {correct_count}/{len(accuracies)}")

    if len(accuracies) > 0:
        mean_acc = np.mean(accuracies)
        std_err = np.std(accuracies) / (len(accuracies) ** 0.5)
        print(f"Final accuracy: {mean_acc:.3f} ± {std_err:.3f}")
        print(f"Percentage: {mean_acc * 100:.1f}%")
    else:
        print("No questions evaluated successfully")

    print("=" * 60)

    # Save summary to file
    summary_path = input_path.parent / f"{input_path.stem}_eval_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MMLU EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Questions evaluated: {len(accuracies)}\n")
        f.write(f"Correct answers: {correct_count}/{len(accuracies)}\n")
        if len(accuracies) > 0:
            f.write(f"Accuracy: {mean_acc:.3f} ± {std_err:.3f}\n")
            f.write(f"Percentage: {mean_acc * 100:.1f}%\n")

    print(f"\nSummary saved to: {summary_path}")
