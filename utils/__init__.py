"""
Shared utilities for LLM multiagent debate experiments.

This module provides platform-agnostic LLM inference with support for:
- MLX (Apple Silicon)
- Ollama (cross-platform GGUF)
- vLLM (HPC/server with NVIDIA GPUs)
"""

from .llm_wrapper import ChatCompletion
from .config import load_config, get_generation_params, resolve_model_name, resolve_persona, get_experiment_config, get_dataset_path
from .model_cache import ModelCache
from .helpers import construct_assistant_message, most_frequent, compute_accuracy, read_jsonl, parse_bullets, write_jsonl, generate_answer, get_model_descriptor, get_temperature_descriptor, get_persona_descriptor

__all__ = [
    'ChatCompletion',
    'load_config',
    'get_generation_params',
    'resolve_model_name',
    'resolve_persona',
    'get_experiment_config',
    'get_dataset_path',
    'ModelCache',
    'construct_assistant_message',
    'most_frequent',
    'compute_accuracy',
    'read_jsonl',
    'parse_bullets',
    'write_jsonl',
    'generate_answer',
    'get_model_descriptor',
    'get_temperature_descriptor',
    'get_persona_descriptor',
]
