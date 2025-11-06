"""
Shared utilities for LLM multiagent debate experiments.

This module provides platform-agnostic LLM inference with support for:
- MLX (Apple Silicon)
- Ollama (cross-platform GGUF)
- vLLM (HPC/server with NVIDIA GPUs)
"""

from .llm_wrapper import ChatCompletion
from .config import load_config, get_generation_params, resolve_model_name, get_experiment_config, get_dataset_path
from .model_cache import ModelCache

__all__ = [
    'ChatCompletion',
    'load_config',
    'get_generation_params',
    'resolve_model_name',
    'get_experiment_config',
    'get_dataset_path',
    'ModelCache',
]
