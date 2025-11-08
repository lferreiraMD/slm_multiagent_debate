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
    construct_assistant_message
)
from glob import glob
import pandas as pd
import json
import time
import random
import argparse

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


def generate_answer(answer_context, model_name, generation_params):
    try:
        completion = ChatCompletion.create(
                  model=model_name,
                  messages=answer_context,
                  **generation_params)
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context, model_name, generation_params)

    return completion


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

    # Dataset path
    dataset_path = get_dataset_path("mmlu")

    # Print configuration
    print("=" * 60)
    print("MMLU Task - Multiagent Debate")
    print("=" * 60)
    print(f"Model: {model_name}")
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

    for i in range(num_questions):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Round {round + 1}, Agent {i + 1}/{agents}, Question {i+1}/{num_questions} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' responses...")

                completion = generate_answer(agent_context, model_name, generation_params)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"Agent {i + 1} response: {assistant_message['content'][:100]}...")

        response_dict[question] = (agent_contexts, answer)

    # Save results
    output_filename = f"mmlu_{model_name.split('/')[-1]}_agents{agents}_rounds{rounds}.json"
    json.dump(response_dict, open(output_filename, "w"))

    print("=" * 60)
    print(f"Results saved to: {output_filename}")
    print(f"Total questions processed: {len(response_dict)}")
    print("=" * 60)
