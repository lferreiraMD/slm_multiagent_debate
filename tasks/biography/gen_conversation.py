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
    parse_bullets,
    generate_answer
)
import json
import random
from tqdm import tqdm
import argparse
import time


def filter_people(person):
    """Extract person name before parentheses (e.g., 'John Doe (scientist)' -> 'John Doe')."""
    people = person.split("(")[0].strip()
    return people


def construct_message(other_agents, idx, person, final=False):
    prefix_string = "Here are some bullet point biographies of {} given by other agents: ".format(person)

    if len(other_agents) == 0:
        return {"role": "user", "content": "Closely examine your biography and provide an updated bullet point biography."}

    for i, agent in enumerate(other_agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if final:
        prefix_string = prefix_string + "\n\n Closely examine your biography and the biography of other agents and provide an updated bullet point biography.".format(person, person)
    else:
        prefix_string = prefix_string + "\n\n Using these other biographies of {} as additional advice, what is your updated bullet point biography of the computer scientist {}?".format(person, person)

    return {"role": "user", "content": prefix_string}


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Biography task with multiagent debate")
    parser.add_argument("--model", type=str, default=None, help="Model to use (alias or full path)")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--num-people", type=int, default=None, help="Number of people to process")
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
    exp_config = get_experiment_config("biography")
    agents = args.agents or exp_config["agents"]
    rounds = args.rounds or exp_config["rounds"]
    num_people = args.num_people or exp_config["num_people"]
    random_seed = exp_config["random_seed"]

    # Per-agent model configuration (for model diversity experiments)
    agent_models = None
    if args.agent_models:
        agent_models = [resolve_model_name(m) for m in args.agent_models]
        if len(agent_models) != agents:
            raise ValueError(f"Number of agent models ({len(agent_models)}) must match number of agents ({agents})")
        print(f"Using model diversity with {len(agent_models)} different models")

    # Dataset path
    dataset_path = get_dataset_path("biography")

    # Print configuration
    print("=" * 60)
    print("Biography Task - Multiagent Debate")
    print("=" * 60)
    if agent_models:
        print(f"Model diversity mode:")
        for i, model in enumerate(agent_models):
            print(f"  Agent {i+1}: {model}")
    else:
        print(f"Model: {model_name}")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print(f"People: {num_people}")
    print(f"Dataset: {dataset_path}")
    print(f"Generation params: {generation_params}")
    print("=" * 60)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    people = sorted(data.keys())
    people = [filter_people(person) for person in people]
    random.seed(random_seed)
    random.shuffle(people)

    generated_description = {}

    for person in tqdm(people[:num_people], desc="Processing people"):
        agent_contexts = [[{"role": "user", "content": "Give a bullet point biography of {} highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character. ".format(person)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Round {round + 1}, Agent {i + 1}/{agents}, Person: {person} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                    if round == (rounds - 1):
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=True)
                    else:
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=False)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' biographies...")

                completion = generate_answer(agent_context, model_name, generation_params,
                                            agent_id=i, agent_models=agent_models)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(f"Agent {i + 1} response: {assistant_message['content'][:100]}...")

            bullets = parse_bullets(completion["choices"][0]['message']['content'])

            # The LM just doesn't know this person so no need to create debates
            if len(bullets) == 1:
                break

        generated_description[person] = agent_contexts

    # Save results
    output_filename = f"biography_{model_name.split('/')[-1]}_agents{agents}_rounds{rounds}.json"
    json.dump(generated_description, open(output_filename, "w"))

    print("=" * 60)
    print(f"Results saved to: {output_filename}")
    print(f"Total people processed: {len(generated_description)}")
    print("=" * 60)

