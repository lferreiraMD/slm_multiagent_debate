#!/usr/bin/env python3
"""
Persona CSV loader and parser for SLURM experiment generation.

Extracts optimal MaxDet persona combinations from personas/summary_personas.csv
for use in multiagent debate experiments.

Usage:
    from utils.persona_loader import get_optimal_personas

    personas = get_optimal_personas('vllm-qwen3-0.6b', n_agents=3, version='v2')
    # Returns: ['persona 1 description', 'persona 2 description', 'persona 3 description']
"""

import ast
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import re


# Mapping from vLLM model aliases to CSV model names
VLLM_TO_CSV_MODEL_MAP = {
    'vllm-qwen3-0.6b': 'Qwen3-0.6B',
    'vllm-vibethinker': 'VibeThinker-1.5B',
    'vllm-deepseek': 'DeepSeek-R1-Distill-Qwen-1.5B',
    'vllm-qwen3-1.7b': 'Qwen3-1.7B',
    'vllm-gemma2-2b': 'gemma-2-2b-it',
    'vllm-llama32-3b': 'Llama-3.2-3B-Instruct',
    'vllm-phi3-mini': 'Phi-3-mini-4k-instruct',
    'vllm-smallthinker': 'SmallThinker-3B-Preview',
    'vllm-qwen3-4b': 'Qwen3-4B-Instruct-2507',
    'vllm-mistral-7b': 'Mistral-7B-Instruct-v0.3',
    'vllm-llama31-8b': 'Llama-3.1-8B-Instruct',
    'vllm-qwen3-8b': 'Qwen3-8B',
    'vllm-gemma2-9b': 'gemma-2-9b-it',
    'vllm-qwen3-14b': 'Qwen3-14B',
}


# Cached database to avoid re-loading CSV on every call
_PERSONA_DATABASE = None


def load_persona_database(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and cache persona CSV database.

    Args:
        csv_path: Path to summary_personas.csv (default: auto-detect from repo root)

    Returns:
        DataFrame with persona optimization results
    """
    global _PERSONA_DATABASE

    if _PERSONA_DATABASE is not None:
        return _PERSONA_DATABASE

    # Auto-detect CSV path if not provided
    if csv_path is None:
        # Try to find from current file location
        current_file = Path(__file__).resolve()
        repo_root = current_file.parent.parent
        csv_path = repo_root / 'personas' / 'summary_personas.csv'

        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find summary_personas.csv at: {csv_path}")

    # Load CSV
    _PERSONA_DATABASE = pd.read_csv(csv_path)

    return _PERSONA_DATABASE


def parse_persona_tuple(tuple_string: str) -> List[str]:
    """
    Parse persona tuple from CSV string format.

    Args:
        tuple_string: String like "('persona 1 description', 'persona 2 description')"

    Returns:
        List of persona description strings

    Example:
        >>> parse_persona_tuple("('a skeptic', 'an analyst')")
        ['a skeptic', 'an analyst']
    """
    try:
        # Safe evaluation of Python tuple literal
        personas = ast.literal_eval(tuple_string)

        if isinstance(personas, tuple):
            return list(personas)
        else:
            raise ValueError(f"Expected tuple, got {type(personas)}")

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse persona tuple: {tuple_string}") from e


def map_vllm_alias_to_csv_model(vllm_alias: str) -> str:
    """
    Map vLLM model alias to CSV model name.

    Args:
        vllm_alias: vLLM alias like "vllm-qwen3-0.6b"

    Returns:
        CSV model name like "Qwen3-0.6B"

    Raises:
        ValueError: If alias not found in mapping

    Example:
        >>> map_vllm_alias_to_csv_model('vllm-deepseek')
        'DeepSeek-R1-Distill-Qwen-1.5B'
    """
    if vllm_alias in VLLM_TO_CSV_MODEL_MAP:
        return VLLM_TO_CSV_MODEL_MAP[vllm_alias]

    # Try to extract model name directly if not in map
    # This handles cases where someone passes the CSV name directly
    for csv_model in VLLM_TO_CSV_MODEL_MAP.values():
        if csv_model.lower() in vllm_alias.lower():
            return csv_model

    raise ValueError(
        f"Unknown vLLM model alias: {vllm_alias}\n"
        f"Known aliases: {list(VLLM_TO_CSV_MODEL_MAP.keys())}"
    )


def get_optimal_personas(
    model_name: str,
    n_agents: int,
    version: str = 'v2',
    selection_method: str = 'maxdet',
    csv_path: Optional[str] = None
) -> List[str]:
    """
    Get optimal persona combinations for a model and agent count.

    Args:
        model_name: vLLM alias (e.g., 'vllm-qwen3-0.6b') or CSV model name (e.g., 'Qwen3-0.6B')
        n_agents: Number of agents (2-7)
        version: Persona version ('v1' or 'v2', default 'v2')
        selection_method: 'maxdet' or 'maxmin' (default 'maxdet')
        csv_path: Optional path to summary_personas.csv (auto-detected if None)

    Returns:
        List of n_agents persona description strings

    Raises:
        ValueError: If no matching entry found in database

    Example:
        >>> personas = get_optimal_personas('vllm-qwen3-0.6b', n_agents=3, version='v2')
        >>> len(personas)
        3
    """
    # Load database
    db = load_persona_database(csv_path)

    # Map vLLM alias to CSV model name
    if model_name.startswith('vllm-'):
        csv_model = map_vllm_alias_to_csv_model(model_name)
    else:
        # Assume it's already a CSV model name
        csv_model = model_name

    # Query database
    query_result = db[
        (db['model'] == csv_model) &
        (db['n_agents'] == n_agents) &
        (db['persona_version'] == version)
    ]

    if len(query_result) == 0:
        raise ValueError(
            f"No personas found for model={csv_model}, n_agents={n_agents}, version={version}\n"
            f"Available models: {db['model'].unique()}\n"
            f"Available agent counts: {sorted(db['n_agents'].unique())}\n"
            f"Available versions: {db['persona_version'].unique()}"
        )

    # Get first matching row
    row = query_result.iloc[0]

    # Extract persona tuple based on selection method
    if selection_method == 'maxdet':
        personas_tuple_str = row['maxdet_personas']
    elif selection_method == 'maxmin':
        personas_tuple_str = row['maxmin_personas']
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}. Must be 'maxdet' or 'maxmin'")

    # Parse tuple string to list
    personas = parse_persona_tuple(personas_tuple_str)

    # Validate count matches
    if len(personas) != n_agents:
        raise ValueError(
            f"Persona count mismatch: expected {n_agents}, got {len(personas)} "
            f"for model={csv_model}, n_agents={n_agents}"
        )

    return personas


def get_all_persona_configs(
    version: str = 'v2',
    selection_method: str = 'maxdet',
    csv_path: Optional[str] = None
) -> List[Dict]:
    """
    Get all available persona configurations from the database.

    Useful for generating complete experiment matrices.

    Args:
        version: Persona version ('v1' or 'v2', default 'v2')
        selection_method: 'maxdet' or 'maxmin' (default 'maxdet')
        csv_path: Optional path to summary_personas.csv

    Returns:
        List of dicts with keys: model, n_agents, personas

    Example:
        >>> configs = get_all_persona_configs(version='v2')
        >>> len(configs)
        60  # 10 models × 6 agent counts
    """
    # Load database
    db = load_persona_database(csv_path)

    # Filter by version
    filtered = db[db['persona_version'] == version]

    configs = []
    for _, row in filtered.iterrows():
        # Parse personas based on selection method
        if selection_method == 'maxdet':
            personas_str = row['maxdet_personas']
        else:
            personas_str = row['maxmin_personas']

        personas = parse_persona_tuple(personas_str)

        configs.append({
            'model': row['model'],
            'n_agents': row['n_agents'],
            'personas': personas
        })

    return configs


# Convenience function for testing
def print_persona_info(model_name: str, n_agents: int, version: str = 'v2'):
    """
    Print persona information for debugging.

    Args:
        model_name: vLLM alias or CSV model name
        n_agents: Number of agents
        version: Persona version
    """
    try:
        personas = get_optimal_personas(model_name, n_agents, version)

        print(f"Model: {model_name}")
        print(f"Agents: {n_agents}")
        print(f"Version: {version}")
        print(f"Personas:")
        for i, p in enumerate(personas, 1):
            print(f"  {i}. {p[:80]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # Test script
    print("Testing persona_loader.py")
    print("=" * 80)

    # Test 1: Load database
    print("\nTest 1: Loading database...")
    db = load_persona_database()
    print(f"✓ Loaded {len(db)} rows")
    print(f"  Models: {db['model'].nunique()}")
    print(f"  Agent counts: {sorted(db['n_agents'].unique())}")
    print(f"  Versions: {db['persona_version'].unique()}")

    # Test 2: Model name mapping
    print("\nTest 2: Model name mapping...")
    for vllm_alias in ['vllm-qwen3-0.6b', 'vllm-deepseek', 'vllm-llama32-3b']:
        csv_name = map_vllm_alias_to_csv_model(vllm_alias)
        print(f"  {vllm_alias} → {csv_name}")

    # Test 3: Get optimal personas
    print("\nTest 3: Getting optimal personas...")
    print_persona_info('vllm-qwen3-0.6b', n_agents=3, version='v2')

    print("\n" + "=" * 80)
    print("All tests passed!")
