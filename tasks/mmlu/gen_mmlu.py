import sys
from pathlib import Path

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    ChatCompletion,
    load_config,
    resolve_model_name,
    get_experiment_config,
    get_dataset_path,
    construct_assistant_message,
    generate_answer,
    ModelCache,
    get_model_descriptor,
    get_temperature_descriptor,
    compute_accuracy
)
from glob import glob
import pandas as pd
import json
import time
import random
import argparse
import re
import numpy as np
from typing import Optional


def parse_mmlu_answer(input_str: str) -> Optional[str]:
    """Extract multiple-choice answer (A/B/C/D) from model response."""
    if not input_str:
        return None

    # Pattern 1: (X) format
    pattern1 = r'\(([A-Da-d])\)'
    matches = re.findall(pattern1, input_str)
    if matches:
        return matches[-1].upper()

    # Pattern 2: "Answer: X" format
    pattern2 = r'[Aa]nswer[:\s]+([A-Da-d])\b'
    matches = re.findall(pattern2, input_str)
    if matches:
        return matches[-1].upper()

    # Pattern 3: X) format
    pattern3 = r'(?:^|\n)\s*([A-Da-d])\)'
    matches = re.findall(pattern3, input_str)
    if matches:
        return matches[-1].upper()

    return None


def construct_message(other_agents, question, idx):
    if len(other_agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in other_agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MMLU task with multiagent debate")
    parser.add_argument("--model", type=str, default=None, help="Model to use (alias or full path)")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--agent-models", type=str, nargs="+", default=None,
                       help="Per-agent models for model diversity (space-separated aliases or paths)")
    parser.add_argument("--agent-temperatures", type=float, nargs="+", default=None,
                       help="Per-agent temperatures for parameter diversity (space-separated floats)")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Model configuration
    model_name = args.model or config.get("model", "deepseek")
    model_name = resolve_model_name(model_name)
    generation_params = config["generation"]

    # Experiment configuration
    exp_config = get_experiment_config("mmlu")
    agents = args.agents or exp_config["agents"]
    rounds = args.rounds or exp_config["rounds"]
    num_questions = args.num_questions or exp_config["num_questions"]
    random_seed = exp_config["random_seed"]

    # Per-agent model configuration (for model diversity experiments)
    agent_models = None
    if args.agent_models:
        agent_models = [resolve_model_name(m) for m in args.agent_models]
        if len(agent_models) != agents:
            raise ValueError(f"Number of agent models ({len(agent_models)}) must match number of agents ({agents})")
        print(f"Using model diversity with {len(agent_models)} different models")

    # Per-agent temperature configuration (for parameter diversity experiments)
    agent_gen_params = None
    if args.agent_temperatures:
        if len(args.agent_temperatures) != agents:
            raise ValueError(f"Number of temperatures ({len(args.agent_temperatures)}) must match number of agents ({agents})")

        # Create per-agent generation param dicts with different temperatures
        agent_gen_params = []
        for temp in args.agent_temperatures:
            params = generation_params.copy()
            params['temperature'] = temp
            agent_gen_params.append(params)

        print(f"Using temperature diversity with {len(agent_gen_params)} different temperatures")

    # Dataset path
    dataset_path = get_dataset_path("mmlu")

    # Print configuration
    print("=" * 60)
    print("MMLU Task - Multiagent Debate")
    print("=" * 60)
    if agent_models:
        print(f"Model diversity mode:")
        for i, model in enumerate(agent_models):
            print(f"  Agent {i+1}: {model}")
    else:
        print(f"Model: {model_name}")
    if agent_gen_params:
        print(f"Temperature diversity mode:")
        for i, params in enumerate(agent_gen_params):
            print(f"  Agent {i+1}: temp={params['temperature']}")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print(f"Questions: {num_questions}")
    print(f"Dataset: {dataset_path}")
    print(f"Generation params: {generation_params}")
    print("=" * 60)

    tasks = glob(f"{dataset_path}/*.csv")
    print(f"Found {len(tasks)} MMLU test files")

    dfs = [pd.read_csv(task, header=None) for task in tasks]

    random.seed(random_seed)
    response_dict = {}
    accuracies = []

    for question_idx in range(num_questions):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Round {round + 1}, Agent {i + 1}/{agents}, Question {question_idx+1}/{num_questions} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' responses...")

                completion = generate_answer(agent_context, model_name, generation_params,
                                            agent_id=i, agent_models=agent_models,
                                            agent_gen_params=agent_gen_params)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"Agent {i + 1} response: {assistant_message['content'][:100]}...")

        # Inline evaluation
        pred_solutions = [ctx[-1]['content'] for ctx in agent_contexts]
        accurate = compute_accuracy(answer, pred_solutions, parse_fn=parse_mmlu_answer)

        if accurate is not None:
            accuracies.append(float(accurate))
            print(f"\nRunning accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies) / (len(accuracies) ** 0.5):.3f}")

        response_dict[question] = (agent_contexts, answer)

    # Save results
    model_descriptor = get_model_descriptor(model_name, agent_models)
    temp_descriptor = get_temperature_descriptor(agent_gen_params)

    # Build filename with optional temperature descriptor
    if temp_descriptor:
        output_filename = f"mmlu_{model_descriptor}_{temp_descriptor}_agents{agents}_rounds{rounds}.json"
    else:
        output_filename = f"mmlu_{model_descriptor}_agents{agents}_rounds{rounds}.json"

    json.dump(response_dict, open(output_filename, "w"))

    print("\n" + "=" * 60)
    print("GENERATION & EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_filename}")
    print(f"Questions processed: {len(response_dict)}")
    if len(accuracies) > 0:
        print(f"Final accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies) / (len(accuracies) ** 0.5):.3f}")
    print("=" * 60)

    # Cleanup: Shutdown vLLM engines to prevent hanging
    model_cache = ModelCache()
    model_cache.shutdown()
