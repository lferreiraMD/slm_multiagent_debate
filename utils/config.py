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
        "random_seed": 42,
    },
    "gsm": {
        "agents": 3,
        "rounds": 2,
        "num_problems": 100,
        "random_seed": 42,
    },
    "biography": {
        "agents": 3,
        "rounds": 2,
        "num_people": 40,
        "random_seed": 42,
    },
    "mmlu": {
        "agents": 3,
        "rounds": 2,
        "num_questions": 100,
        "random_seed": 42,
    },
}

# Model and persona aliases loaded from config.yaml at runtime
# These are populated by load_config() to ensure single source of truth
MODEL_ALIASES = {}
PERSONA_ALIASES = {}


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


def resolve_persona(persona: str) -> str:
    """
    Resolve persona alias to full description.

    Args:
        persona: Persona callsign/alias or full description

    Returns:
        Full persona description

    Raises:
        ValueError: If persona alias not found and doesn't look like a full description
    """
    # Check if it's an alias in PERSONA_ALIASES
    if persona in PERSONA_ALIASES:
        return PERSONA_ALIASES[persona]

    # If not an alias, check if it looks like a full description
    # (contains spaces and is reasonably long)
    if " " in persona and len(persona) > 20:
        return persona

    # Otherwise, it's an unknown alias
    raise ValueError(
        f"Unknown persona alias: '{persona}'. "
        f"Check config.yaml for available personas or provide a full description."
    )


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
        "gsm": base_dir / "data" / "gsm8k" / "test.jsonl",
        "biography": base_dir / "data" / "biography" / "article.json",
        "mmlu": base_dir / "data" / "mmlu",  # Directory containing *_test.csv files
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
    Load configuration from YAML file and populate global aliases.

    This function populates MODEL_ALIASES and PERSONA_ALIASES from config.yaml
    to ensure a single source of truth for all model and persona definitions.

    Args:
        config_path: Path to config file (defaults to config.yaml in repo root)

    Returns:
        Configuration dict
    """
    global MODEL_ALIASES, PERSONA_ALIASES

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}, using defaults")
        return {
            "generation": DEFAULT_GENERATION_PARAMS,
            "experiments": DEFAULT_EXPERIMENT_CONFIGS,
            "models": {},
            "personas": {},
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Populate MODEL_ALIASES from config for resolve_model_name() to use
    if "models" in config:
        MODEL_ALIASES.clear()
        MODEL_ALIASES.update(config["models"])

    # Populate PERSONA_ALIASES from config for resolve_persona() to use
    if "personas" in config:
        PERSONA_ALIASES.clear()
        PERSONA_ALIASES.update(config["personas"])

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
