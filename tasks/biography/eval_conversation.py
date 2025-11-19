import sys
from pathlib import Path

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import time
import argparse
from tqdm import tqdm
from utils import (
    ChatCompletion,
    load_config,
    resolve_model_name,
    ModelCache
)

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

    if "uncertain" in string.lower():
        return None
    elif "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def filter_people(person):
    people = person.split("(")[0].strip()
    return people

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate biography generation using a local judge model")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to generated biography JSON file")
    parser.add_argument("--judge-model", type=str, default="qwen25-7b",
                        help="Model to use for fact-checking (default: qwen25-7b)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation results (optional)")
    parser.add_argument("--ground-truth", type=str, default="../../data/biography/article.json",
                        help="Path to ground truth biographies")

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    judge_model = resolve_model_name(args.judge_model)
    model_cache = ModelCache()

    print(f"Loading biography results from: {args.input_file}")
    print(f"Using judge model: {judge_model}")
    print(f"Ground truth file: {args.ground_truth}")

    # Load generated biographies
    response = json.load(open(args.input_file, "r"))

    # Load ground truth
    with open(args.ground_truth, "r") as f:
        gt_data = json.load(f)

    # Filter people names (remove parenthetical suffixes)
    gt_data_filter = {}
    for k, v in gt_data.items():
        k = filter_people(k)
        gt_data_filter[k] = v

    gt_data = gt_data_filter

    people = list(response.keys())

    print(f"\nEvaluating {len(people)} people...")

    accuracies = []
    per_person_results = {}

    # Iterate through people with progress bar
    for person in tqdm(people, desc="Evaluating biographies"):

        if person not in gt_data:
            print(f"\nWarning: {person} not found in ground truth, skipping")
            continue

        gt_description = gt_data[person]
        gt_bullets = parse_bullets(gt_description)
        bio_descriptions = response[person]

        person_accuracies = []

        for agent_idx, description in enumerate(bio_descriptions):

            bio_description = description[-1]['content']

            bio_bullets = parse_bullets(bio_description)
            if len(bio_bullets) == 1:
                if len(bio_bullets[0]) < 400:
                    print(f"\nSkipping {person} (agent {agent_idx}): biography too short")
                    continue

            bio_bullets_text = " ".join(bio_bullets)

            # Evaluate each ground truth fact
            for bullet_idx, bullet in enumerate(gt_bullets):
                # Strengthened prompt for single-word response
                message = [{
                    "role": "user",
                    "content": f"""You must respond with ONLY one word: yes, no, or uncertain. Do not explain your reasoning.

Consider the following biography of {person}:
{bio_bullets_text}

Is the above biography consistent with the fact below?
{bullet}

Carefully check the precise dates, locations, and details between the fact and the biography above.

Answer (one word only):"""
                }]

                retry_count = 0
                max_retries = 3

                while retry_count < max_retries:
                    try:
                        completion = ChatCompletion.create(
                            model=judge_model,
                            messages=message,
                            temperature=0.3,  # Low temperature for consistency
                            max_tokens=10,    # Short response expected
                            n=1
                        )

                        content = completion["choices"][0]["message"]["content"]
                        accurate = parse_yes_no(content)

                        if accurate is not None:
                            accuracies.append(float(accurate))
                            person_accuracies.append(float(accurate))

                        # Success, break retry loop
                        break

                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"\nError evaluating {person}, bullet {bullet_idx}: {e}")
                            print("Max retries reached, skipping this fact")
                        else:
                            print(f"\nRetrying ({retry_count}/{max_retries})...")
                            time.sleep(5)

        # Store per-person results
        if person_accuracies:
            per_person_results[person] = {
                "accuracy": np.mean(person_accuracies),
                "num_facts": len(person_accuracies)
            }

    # Calculate final statistics
    if accuracies:
        mean_accuracy = np.mean(accuracies)
        std_error = np.std(accuracies) / (len(accuracies) ** 0.5)

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Judge Model: {judge_model}")
        print(f"Input File: {args.input_file}")
        print(f"Total Facts Evaluated: {len(accuracies)}")
        print(f"Overall Accuracy: {mean_accuracy:.3f} ± {std_error:.3f}")
        print("="*60)

        # Save results if output file specified
        if args.output_file:
            results = {
                "judge_model": judge_model,
                "input_file": args.input_file,
                "ground_truth_file": args.ground_truth,
                "overall_accuracy": mean_accuracy,
                "std_error": std_error,
                "total_facts": len(accuracies),
                "per_person_results": per_person_results
            }

            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to: {args.output_file}")
    else:
        print("\nNo facts were successfully evaluated!")

    # Cleanup model cache
    print("\nCleaning up models...")
    model_cache.shutdown()
    print("Done!")
