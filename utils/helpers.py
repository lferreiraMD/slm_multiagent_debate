"""
Shared helper functions used across multiple tasks.

These utilities are task-agnostic and can be reused by math, GSM, biography, and MMLU tasks.
"""

import json
import time
from typing import List, Any, Dict, Optional, Callable


def generate_answer(
    answer_context: List[Dict[str, str]],
    model_name: str,
    generation_params: Dict[str, Any],
    agent_id: int = 0,
    agent_models: Optional[List[str]] = None,
    agent_gen_params: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate LLM response with optional per-agent model and parameter selection.

    Supports cognitive diversity experiments by allowing different models and
    generation parameters for each agent in multiagent debate.

    Args:
        answer_context: Chat message history for this agent
        model_name: Default model name (fallback if agent_models not provided)
        generation_params: Default generation parameters (fallback if agent_gen_params not provided)
        agent_id: Agent index (0-based) for per-agent selection
        agent_models: Optional list of model names (one per agent). If provided,
                     uses agent_models[agent_id] instead of model_name
        agent_gen_params: Optional list of generation param dicts (one per agent).
                         If provided, uses agent_gen_params[agent_id] instead of generation_params

    Returns:
        OpenAI-compatible completion response dict

    Examples:
        # Homogeneous agents (existing behavior)
        >>> generate_answer(context, "deepseek", {"temperature": 1.0})

        # Model diversity (Condition 3)
        >>> models = ["deepseek", "llama32-3b", "qwen25-7b"]
        >>> generate_answer(context, "deepseek", params, agent_id=1, agent_models=models)
        # Uses llama32-3b for agent 1

        # Decoding diversity (Condition 4)
        >>> params_list = [{"temperature": 0.7}, {"temperature": 1.0}, {"temperature": 1.3}]
        >>> generate_answer(context, "deepseek", params, agent_id=2, agent_gen_params=params_list)
        # Uses temperature=1.3 for agent 2
    """
    from utils import ChatCompletion

    # Select model: per-agent if available, else default
    selected_model = (agent_models[agent_id]
                     if agent_models and len(agent_models) > agent_id
                     else model_name)

    # Select generation params: per-agent if available, else default
    selected_params = (agent_gen_params[agent_id]
                      if agent_gen_params and len(agent_gen_params) > agent_id
                      else generation_params)

    try:
        completion = ChatCompletion.create(
            model=selected_model,
            messages=answer_context,
            **selected_params
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context, model_name, generation_params,
                              agent_id, agent_models, agent_gen_params)

    return completion


def construct_assistant_message(completion: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract assistant message from OpenAI-compatible completion response.

    Args:
        completion: Response dict from ChatCompletion.create()

    Returns:
        Message dict with role="assistant" and content
    """
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def most_frequent(items: List[Any]) -> Any:
    """
    Find the most frequent item in a list (for majority voting).

    Args:
        items: List of items (can be numbers, strings, etc.)

    Returns:
        The most frequently occurring item

    Raises:
        IndexError: If list is empty
    """
    if not items:
        raise ValueError("Cannot find most frequent item in empty list")

    counter = 0
    most_common = items[0]

    for item in items:
        current_frequency = items.count(item)
        if current_frequency > counter:
            counter = current_frequency
            most_common = item

    return most_common


def compute_accuracy(
    gt: str,
    pred_solutions: List[str],
    parse_fn: Callable[[str], Optional[str]],
    normalize_gt: bool = True
) -> int:
    """
    Generic accuracy computation with majority voting across agents.

    Args:
        gt: Ground truth answer string
        pred_solutions: List of predicted solution strings from multiple agents
        parse_fn: Function to extract answer from solution text
                  Should return parsed answer or None if parsing fails
        normalize_gt: If True, normalize ground truth (strip + uppercase) instead of parsing

    Returns:
        1 if correct, 0 if incorrect

    Examples:
        >>> def parse_letter(s): return s.strip().upper()
        >>> compute_accuracy("B", ["b", "B", "b"], parse_fn=parse_letter)
        1
    """
    # Parse ground truth (or just normalize if it's already in final form)
    if normalize_gt:
        gt_parsed = gt.strip().upper()
    else:
        gt_parsed = parse_fn(gt)
        if gt_parsed is None:
            return 0

    # Extract answers from each agent's response
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_fn(pred_solution)
        if pred_answer is not None:
            pred_answers.append(pred_answer)

    # Need at least one valid prediction
    if not pred_answers:
        return 0

    # Use majority voting
    try:
        majority_answer = most_frequent(pred_answers)
    except ValueError:
        return 0

    # Compare with ground truth
    return 1 if majority_answer == gt_parsed else 0


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) file into list of dicts.

    Args:
        path: Path to .jsonl file

    Returns:
        List of parsed JSON objects
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line.strip()]


def parse_bullets(sentence: str) -> List[str]:
    """
    Parse bullet points from text (used in biography task).

    Extracts non-empty lines starting with alphabetic characters.

    Args:
        sentence: Text containing bullet points

    Returns:
        List of cleaned bullet point strings
    """
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except StopIteration:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def write_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """
    Write list of dicts to JSONL file.

    Args:
        data: List of JSON-serializable objects
        path: Output path for .jsonl file
    """
    with open(path, 'w') as fh:
        for item in data:
            fh.write(json.dumps(item) + '\n')


def get_model_descriptor(
    model_name: str,
    agent_models: Optional[List[str]] = None
) -> str:
    """
    Generate descriptive model name for output filenames.

    Two cases:
    1. Single model (all agents use same) → extract short name
    2. Multiple distinct models → create amalgamated name like "model1+model2+model3"

    Args:
        model_name: Single model name (from --model or config default)
        agent_models: Optional list of per-agent models (from --agent-models)

    Returns:
        Short descriptive model name for filename
    """
    def _extract_short_name(full_path: str) -> str:
        """Extract short descriptive name from model path (general approach)."""
        # Get last part: "meta-llama/Llama-3.2-3B-Instruct" → "Llama-3.2-3B-Instruct"
        name = full_path.split('/')[-1]

        # Remove backend prefixes: "vllm-llama32-3b" → "llama32-3b"
        for prefix in ["vllm-", "ollama-"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Remove common suffixes to shorten
        for suffix in ["-Instruct", "-instruct", "-Preview", "-preview",
                      "-mlx-fp16", "-mlx-8bit", "-mlx-4bit", "-8bit", "-4bit"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # If still too long (> 25 chars), truncate intelligently
        if len(name) > 25:
            # Try to keep meaningful parts
            name = name[:25]

        return name

    # Determine which models are actually being used
    if agent_models is not None:
        unique_models = list(set(agent_models))

        if len(unique_models) == 1:
            # Case 1: All agents use the same model
            return _extract_short_name(unique_models[0])
        else:
            # Case 2: Multiple distinct models - create amalgamated name
            short_names = sorted([_extract_short_name(m) for m in unique_models])
            return "+".join(short_names)
    else:
        # Case 1: Single model via --model or config
        return _extract_short_name(model_name)


def get_temperature_descriptor(
    agent_gen_params: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Generate temperature descriptor for output filenames.

    Args:
        agent_gen_params: Optional list of per-agent generation parameter dicts

    Returns:
        Temperature descriptor like "temp0.7+1.0+1.3" or None if homogeneous

    Examples:
        >>> get_temperature_descriptor([{'temperature': 0.7}, {'temperature': 1.0}])
        'temp0.7+1.0'
        >>> get_temperature_descriptor([{'temperature': 1.0}, {'temperature': 1.0}])
        None  # All same, no descriptor needed
        >>> get_temperature_descriptor(None)
        None  # No diversity
    """
    if agent_gen_params is None:
        return None

    # Extract temperatures from param dicts
    temps = [params.get('temperature', 1.0) for params in agent_gen_params]
    unique_temps = list(set(temps))

    # If all agents have the same temperature, no need for descriptor
    if len(unique_temps) == 1:
        return None

    # Create descriptor with sorted temperatures
    temp_strs = [f"{t:.1f}" for t in temps]  # Keep original order, not sorted
    return "temp" + "+".join(temp_strs)
