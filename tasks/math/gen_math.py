import sys
from pathlib import Path

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    ChatCompletion,
    load_config,
    resolve_model_name,
    get_experiment_config,
    construct_assistant_message,
    most_frequent
)
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import argparse


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


def construct_message(other_agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(other_agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in other_agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Math task with multiagent debate")
    parser.add_argument("--model", type=str, default=None, help="Model to use (alias or full path)")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--num-problems", type=int, default=None, help="Number of problems to evaluate")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Model configuration
    model_name = args.model or config.get("model", "deepseek")
    model_name = resolve_model_name(model_name)
    generation_params = config["generation"]

    # Experiment configuration
    exp_config = get_experiment_config("math")
    agents = args.agents or exp_config["agents"]
    rounds = args.rounds or exp_config["rounds"]
    evaluation_round = args.num_problems or exp_config["num_problems"]
    random_seed = exp_config["random_seed"]

    # Print configuration
    print("=" * 60)
    print("Math Task - Multiagent Debate")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print(f"Problems: {evaluation_round}")
    print(f"Generation params: {generation_params}")
    print("=" * 60)

    np.random.seed(random_seed)

    scores = []
    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Round {round + 1}, Agent {i + 1}/{agents} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' responses...")

                completion = generate_answer(agent_context, model_name, generation_params)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"Agent {i + 1} response: {assistant_message['content'][:100]}...")

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    # Save results
    output_filename = f"math_{model_name.split('/')[-1]}_agents{agents}_rounds{rounds}.p"
    pickle.dump(generated_description, open(output_filename, "wb"))

    print("=" * 60)
    print(f"Results saved to: {output_filename}")
    print(f"Final performance: {np.mean(scores):.3f} ± {np.std(scores) / (len(scores) ** 0.5):.3f}")
    print("=" * 60)
