import json
import numpy as np
import time
import re
import argparse
import sys
from pathlib import Path

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None

    if type(pred_solution) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            pred_answers.append(pred_answer)

        # print("pred_answers: ", pred_answers)
        pred_answer = most_frequent(pred_answers)
        # print("pred answer: ", pred_answer)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

    if pred_answer is None:
        return 1

    # try:
    if float(answers) == float(pred_answer):
        return 1
    else:
        return 0
    # except:
    #     import pdb
    #     pdb.set_trace()
    #     print(pred_solution)


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate GSM task results")
    parser.add_argument("--input-file", type=str, required=True,
                       help="Path to JSON file with debate results (e.g., gsm_model_agents2_rounds2.json)")
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Load results
    print(f"Evaluating: {args.input_file}")
    response_dict = json.load(open(args.input_file, "r"))

    questions = list(response_dict.keys())
    print(f"Total questions: {len(questions)}")

    accuracies = []

    for question in questions:
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']

            pred_solutions.append(pred_solution)

        accurate = compute_accuracy(gt, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            import pdb
            pdb.set_trace()
            print(gt)

        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Questions evaluated: {len(accuracies)}")
    print(f"Final accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies) / (len(accuracies) ** 0.5):.3f}")
    print("=" * 60)

