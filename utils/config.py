"""
Configuration management for multiagent debate experiments.

Handles:
- Model selection
- Generation parameters (matching original study)
- Experiment configurations
- Dataset paths
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


# Default generation parameters (matching GPT-3.5-turbo-0301 defaults)
DEFAULT_GENERATION_PARAMS = {
    "temperature": 1.0,      # High temperature for diverse agent responses
    "max_tokens": None,      # Let model decide (we'll use 2048 as practical limit)
    "top_p": 1.0,           # Full nucleus sampling
    "n": 1,                 # One completion per call
}

# Default experiment configurations (from original study)
DEFAULT_EXPERIMENT_CONFIGS = {
    "math": {
        "agents": 2,
        "rounds": 3,
        "num_problems": 100,
        "random_seed": 0,
    },
    "gsm": {
        "agents": 3,
        "rounds": 2,
        "num_problems": 100,
        "random_seed": 0,
    },
    "biography": {
        "agents": 3,
        "rounds": 2,
        "num_people": 40,
        "random_seed": 1,
    },
    "mmlu": {
        "agents": 3,
        "rounds": 2,
        "num_questions": 100,
        "random_seed": 0,
    },
}

# Model aliases for convenience
MODEL_ALIASES = {
    # MLX models (already downloaded)
    "deepseek": "valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16",
    "llama32-3b": "mlx-community/Llama-3.2-3B-Instruct",
    "smallthinker": "valuat/SmallThinker-3B-Preview-mlx-fp16",
    "qwen25-7b": "valuat/Qwen2.5-7B-Instruct-1M-mlx-fp16",
    "llama31-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "llama31-8b-fp16": "valuat/Meta-Llama-3.1-8B-Instruct-mlx-fp16",
    "qwen25-14b": "valuat/Qwen2.5-14B-Instruct-1M-mlx-fp16",

    # Ollama models (for HPC/Windows)
    "ollama-llama32": "llama3.2:3b",
    "ollama-qwen25": "qwen2.5:7b",
}


def get_generation_params(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get generation parameters, matching original study defaults.

    Args:
        override: Optional dict to override specific parameters

    Returns:
        Dict of generation parameters
    """
    params = DEFAULT_GENERATION_PARAMS.copy()

    if override:
        params.update(override)

    return params


def get_experiment_config(
    task: str,
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get experiment configuration for a task.

    Args:
        task: Task name ('math', 'gsm', 'biography', 'mmlu')
        override: Optional dict to override specific parameters

    Returns:
        Dict of experiment configuration

    Raises:
        ValueError: If task is unknown
    """
    if task not in DEFAULT_EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Must be one of {list(DEFAULT_EXPERIMENT_CONFIGS.keys())}"
        )

    config = DEFAULT_EXPERIMENT_CONFIGS[task].copy()

    if override:
        config.update(override)

    return config


def resolve_model_name(model: str) -> str:
    """
    Resolve model alias to full path/name.

    Args:
        model: Model name or alias

    Returns:
        Full model path/name
    """
    return MODEL_ALIASES.get(model, model)


def get_dataset_path(task: str, base_dir: Optional[str] = None) -> str:
    """
    Get dataset path for a task.

    Args:
        task: Task name
        base_dir: Base directory (defaults to repo root)

    Returns:
        Path to dataset

    Raises:
        ValueError: If task is unknown or dataset not found
    """
    if base_dir is None:
        # Assume we're in a subdirectory (math/, gsm/, etc.)
        base_dir = Path(__file__).parent.parent

    base_dir = Path(base_dir)

    dataset_paths = {
        "gsm": base_dir / "data" / "gsm8k" / "grade_school_math" / "data" / "test.jsonl",
        "biography": base_dir / "biography" / "article.json",
        "mmlu": base_dir / "data" / "mmlu" / "data" / "test",
        "math": None,  # Generated on-the-fly
    }

    if task not in dataset_paths:
        raise ValueError(f"Unknown task: {task}")

    path = dataset_paths[task]

    if path is not None and not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run scripts/download_datasets.sh to download datasets."
        )

    return str(path) if path else None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config.yaml in repo root)

    Returns:
        Configuration dict
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}, using defaults")
        return {
            "generation": DEFAULT_GENERATION_PARAMS,
            "experiments": DEFAULT_EXPERIMENT_CONFIGS,
            "models": MODEL_ALIASES,
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_env_model() -> Optional[str]:
    """
    Get model from environment variable.

    Returns:
        Model name from LLM_MODEL env var, or None
    """
    return os.environ.get("LLM_MODEL")


def get_output_path(
    task: str,
    model_alias: str,
    agents: int,
    rounds: int,
    base_dir: Optional[str] = None
) -> Path:
    """
    Generate output path for experiment results.

    Args:
        task: Task name
        model_alias: Short model identifier
        agents: Number of agents
        rounds: Number of rounds
        base_dir: Base directory (defaults to repo root)

    Returns:
        Path object for output file
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    base_dir = Path(base_dir)

    # Create results directory structure
    results_dir = base_dir / "results" / task
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"{model_alias}_a{agents}_r{rounds}.json"

    return results_dir / filename
