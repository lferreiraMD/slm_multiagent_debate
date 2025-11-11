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
    read_jsonl,
    generate_answer
)
import json
import numpy as np
import random
import argparse

def construct_message(other_agents, question, idx):
    if len(other_agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in other_agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GSM task with multiagent debate")
    parser.add_argument("--model", type=str, default=None, help="Model to use (alias or full path)")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--num-problems", type=int, default=None, help="Number of problems to evaluate")
    parser.add_argument("--agent-models", type=str, nargs="+", default=None,
                       help="Per-agent models for model diversity (space-separated aliases or paths)")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Model configuration
    model_name = args.model or config.get("model", "deepseek")
    model_name = resolve_model_name(model_name)
    generation_params = config["generation"]

    # Experiment configuration
    exp_config = get_experiment_config("gsm")
    agents = args.agents or exp_config["agents"]
    rounds = args.rounds or exp_config["rounds"]
    num_problems = args.num_problems or exp_config["num_problems"]
    random_seed = exp_config["random_seed"]

    # Per-agent model configuration (for model diversity experiments)
    agent_models = None
    if args.agent_models:
        agent_models = [resolve_model_name(m) for m in args.agent_models]
        if len(agent_models) != agents:
            raise ValueError(f"Number of agent models ({len(agent_models)}) must match number of agents ({agents})")
        print(f"Using model diversity with {len(agent_models)} different models")

    # Dataset path
    dataset_path = get_dataset_path("gsm")

    # Print configuration
    print("=" * 60)
    print("GSM Task - Multiagent Debate")
    print("=" * 60)
    if agent_models:
        print(f"Model diversity mode:")
        for i, model in enumerate(agent_models):
            print(f"  Agent {i+1}: {model}")
    else:
        print(f"Model: {model_name}")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print(f"Problems: {num_problems}")
    print(f"Dataset: {dataset_path}")
    print(f"Generation params: {generation_params}")
    print("=" * 60)

    random.seed(random_seed)

    generated_description = {}

    questions = read_jsonl(dataset_path)
    random.shuffle(questions)

    for data in questions[:num_problems]:
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Round {round + 1}, Agent {i + 1}/{agents} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' responses...")

                completion = generate_answer(agent_context, model_name, generation_params,
                                            agent_id=i, agent_models=agent_models)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"Agent {i + 1} response: {assistant_message['content'][:100]}...")

        generated_description[question] = (agent_contexts, answer)

    # Save results
    output_filename = f"gsm_{model_name.split('/')[-1]}_agents{agents}_rounds{rounds}.json"
    json.dump(generated_description, open(output_filename, "w"))

    print("=" * 60)
    print(f"Results saved to: {output_filename}")
    print(f"Total problems processed: {len(generated_description)}")
    print("=" * 60)
