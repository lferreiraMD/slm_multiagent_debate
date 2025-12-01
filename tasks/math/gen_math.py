import sys
from pathlib import Path

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    ChatCompletion,
    load_config,
    resolve_model_name,
    resolve_persona,
    get_experiment_config,
    construct_assistant_message,
    most_frequent,
    generate_answer,
    ModelCache,
    get_model_descriptor,
    get_temperature_descriptor,
    get_persona_descriptor
)
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import argparse
import os


def construct_message(other_agents, question, idx, compress_context=True):
    """
    Construct message for agent with optional context compression.

    Args:
        other_agents: List of other agents' conversation histories
        question: The math problem being solved
        idx: Index of the message to extract from other agents
        compress_context: If True, extract answers only (saves ~95% tokens)
                         If False, include full responses (original behavior)

    Returns:
        Message dict for the agent
    """
    # Use introspection in the case in which there are no other agents.
    if len(other_agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    if compress_context:
        # OPTIMIZED: Extract answers only (reduces context by ~95%)
        # Example: 6 agents × 500 tokens = 3000 tokens → 6 answers × 5 tokens = 30 tokens

        # Calculate original size for comparison
        original_size = sum(len(agent[idx]["content"]) for agent in other_agents)

        prefix_string = "These are the answers from other agents:\n"

        answers = []
        for i, agent in enumerate(other_agents):
            agent_response = agent[idx]["content"]
            extracted_answer = parse_answer(agent_response)

            if extracted_answer is not None:
                answers.append(f"Agent {i+1}: {extracted_answer}")
            else:
                # Fallback: couldn't extract answer, show truncated response
                truncated = agent_response[:100] + "..." if len(agent_response) > 100 else agent_response
                answers.append(f"Agent {i+1}: {truncated}")

        prefix_string += "\n".join(answers)
        prefix_string += "\n\nUse these answers as additional advice. Can you verify your own answer and provide an updated response? Make sure to state your answer at the end of the response."

        # Report compression ratio
        compressed_size = len(prefix_string)
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        print(f"[Context Compression] Original: {original_size} chars → Compressed: {compressed_size} chars (saved {compression_ratio:.1f}%)")

    else:
        # ORIGINAL: Include full responses (may cause context overflow)
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
    parser.add_argument("--agent-models", type=str, nargs="+", default=None,
                       help="Per-agent models for model diversity (space-separated aliases or paths)")
    parser.add_argument("--agent-temperatures", type=float, nargs="+", default=None,
                       help="Per-agent temperatures for parameter diversity (space-separated floats)")
    parser.add_argument("--agent-personas", type=str, nargs="+", default=None,
                       help="Per-agent personas for cognitive diversity (space-separated callsigns or descriptions)")
    parser.add_argument("--compress-context", action="store_true", default=True,
                       help="Compress context by extracting answers only (default: True, saves ~95%% tokens)")
    parser.add_argument("--no-compress-context", dest="compress_context", action="store_false",
                       help="Disable context compression (use full agent responses)")
    parser.add_argument("--output-directory", type=str, default=".",
                       help="Directory to save output files (default: current directory)")
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

    # Per-agent persona configuration (for cognitive diversity experiments)
    agent_personas = None
    if args.agent_personas:
        if len(args.agent_personas) != agents:
            raise ValueError(f"Number of personas ({len(args.agent_personas)}) must match number of agents ({agents})")

        # Resolve persona callsigns to full descriptions
        agent_personas = [resolve_persona(p) for p in args.agent_personas]
        print(f"Using persona diversity with {len(agent_personas)} different personas")

    # Auto-enable temperature diversity if no other diversity exists
    # If multiple agents with same model and no persona diversity, use different temperatures
    temp_diversity_config = config.get("temperature_diversity", {})
    temp_diversity_enabled = temp_diversity_config.get("enabled", True)

    if agents > 1 and agent_models is None and agent_personas is None and agent_gen_params is None and temp_diversity_enabled:
        # Read temperature range from config
        min_temp = temp_diversity_config.get("min_temp", 0.7)
        max_temp = temp_diversity_config.get("max_temp", 1.3)

        # Create temperature range distributed across agents
        temps = np.linspace(min_temp, max_temp, agents)
        agent_gen_params = []
        for temp in temps:
            params = generation_params.copy()
            params['temperature'] = float(temp)
            agent_gen_params.append(params)
        temp_list = [f"{p['temperature']:.2f}" for p in agent_gen_params]
        print(f"Auto-enabled temperature diversity (no model/persona diversity detected)")
        print(f"Using {len(agent_gen_params)} different temperatures: {temp_list}")

    # Print configuration
    print("=" * 60)
    print("Math Task - Multiagent Debate")
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
    if agent_personas:
        print(f"Persona diversity mode:")
        for i, persona in enumerate(agent_personas):
            print(f"  Agent {i+1}: {persona[:60]}...")
    print(f"Agents: {agents}")
    print(f"Rounds: {rounds}")
    print(f"Problems: {evaluation_round}")
    print(f"Context compression: {'ENABLED (answer extraction)' if args.compress_context else 'DISABLED (full responses)'}")
    print(f"Generation params: {generation_params}")
    print("=" * 60)

    np.random.seed(random_seed)

    scores = []
    generated_description = {}

    for problem_idx in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Answer in English only. Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                print(f"\n--- Problem {problem_idx + 1}/{evaluation_round}, Round {round + 1}, Agent {i + 1}/{agents} ---")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1,
                                               compress_context=args.compress_context)
                    agent_context.append(message)

                    print(f"Agent {i + 1} receiving other agents' responses...")

                completion = generate_answer(agent_context, model_name, generation_params,
                                            agent_id=i, agent_models=agent_models,
                                            agent_gen_params=agent_gen_params,
                                            agent_personas=agent_personas)

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
    model_descriptor = get_model_descriptor(model_name, agent_models)
    temp_descriptor = get_temperature_descriptor(agent_gen_params)
    persona_descriptor = get_persona_descriptor(agent_personas)

    # Build filename with optional diversity descriptors
    filename_parts = ["math", model_descriptor]
    if temp_descriptor:
        filename_parts.append(temp_descriptor)
    if persona_descriptor:
        filename_parts.append(persona_descriptor)
    filename_parts.extend([f"agents{agents}", f"rounds{rounds}"])
    output_filename = "_".join(filename_parts) + ".p"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_directory, exist_ok=True)

    # Save to output directory
    output_path = os.path.join(args.output_directory, output_filename)
    pickle.dump(generated_description, open(output_path, "wb"))

    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"Final performance: {np.mean(scores):.3f} ± {np.std(scores) / (len(scores) ** 0.5):.3f}")
    print("=" * 60)

    # Cleanup: Shutdown vLLM engines to prevent hanging
    model_cache = ModelCache()
    model_cache.shutdown()
