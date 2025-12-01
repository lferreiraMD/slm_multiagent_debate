"""
Model caching to avoid reloading the same model multiple times.

Critical for memory efficiency when running multiple agents with the same model.
"""

from typing import Tuple, Any, Optional, Dict
import threading
import os

# Import GPU auto-configuration (only used for vLLM backend)
try:
    from .gpu_config import (
        get_vllm_optimal_config,
        detect_vllm_gpus,
        get_gpu_info_string,
        is_vllm_backend
    )
    GPU_CONFIG_AVAILABLE = True
except ImportError:
    GPU_CONFIG_AVAILABLE = False

# Environment variable to disable auto-config
VLLM_DISABLE_AUTO_CONFIG = os.environ.get('VLLM_DISABLE_AUTO_CONFIG', '').lower() in ('1', 'true', 'yes')


class ModelCache:
    """
    Thread-safe singleton cache for loaded models.

    Ensures we only load each model once, even when running multiple agents.
    This is critical for memory efficiency (e.g., don't load 3 copies of 7B model).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._cache_lock = threading.Lock()
        return cls._instance

    def get_or_load(
        self,
        model_path: str,
        backend: str = "mlx",
        use_case: str = 'production',
        override_params: Optional[Dict] = None
    ) -> Tuple[Any, Any]:
        """
        Get model from cache or load it.

        Args:
            model_path: Path or name of model to load
            backend: Backend to use ('mlx', 'ollama', 'vllm')
            use_case: vLLM use case ('production', 'debate', 'download')
            override_params: Optional dict to override vLLM auto-config

        Returns:
            Tuple of (model, tokenizer) for MLX/vLLM, or model for Ollama
        """
        # Include use_case in cache key for vLLM (different configs = different cache entries)
        if backend == "vllm":
            cache_key = f"{backend}:{use_case}:{model_path}"
        else:
            cache_key = f"{backend}:{model_path}"

        # Check cache first
        with self._cache_lock:
            if cache_key in self._cache:
                print(f"[ModelCache] Using cached model: {model_path}")
                return self._cache[cache_key]

        # Load model
        print(f"[ModelCache] Loading model: {model_path} (backend={backend})")

        if backend == "mlx":
            model, tokenizer = self._load_mlx(model_path)
            result = (model, tokenizer)
        elif backend == "vllm":
            model, tokenizer = self._load_vllm(model_path)
            result = (model, tokenizer)
        elif backend == "ollama":
            # Ollama doesn't need to load models in Python
            result = (None, None)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Cache it
        with self._cache_lock:
            self._cache[cache_key] = result

        return result

    def _load_mlx(self, model_path: str) -> Tuple[Any, Any]:
        """Load MLX model and tokenizer."""
        from mlx_lm import load

        model, tokenizer = load(model_path)
        return model, tokenizer

    def _load_vllm(self, model_path: str) -> Tuple[Any, Any]:
        """Load vLLM model and tokenizer."""
        from vllm import LLM
        from transformers import AutoTokenizer

        # Get context length from config.yaml model_metadata
        max_model_len = self._get_model_context_length(model_path)

        print(f"[ModelCache] Using max_model_len={max_model_len} for {model_path}")

        llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.90,
            tensor_parallel_size=2,     # Adjust for multi-GPU
        #    cpu_offload_gb=24           # CPU offload in GB (for 14B model)
        )

        # Load tokenizer separately for chat template support
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        return llm, tokenizer

    def _get_model_context_length(self, model_path: str) -> int:
        """
        Get model context length from config.yaml model_metadata.

        Args:
            model_path: HuggingFace model path

        Returns:
            Context length in tokens (default: 32768 if not found)
        """
        try:
            from .config import load_config
            config = load_config()

            # Find model alias that matches this model_path
            models = config.get('models', {})
            model_alias = None
            for alias, path in models.items():
                if path == model_path:
                    model_alias = alias
                    break

            if model_alias:
                metadata = config.get('model_metadata', {}).get(model_alias, {})
                context_length = metadata.get('context_length', 32768)
                return context_length
            else:
                print(f"[ModelCache] Warning: Model {model_path} not found in config.yaml, using default context_length=32768")
                return 32768

        except Exception as e:
            print(f"[ModelCache] Warning: Failed to read context_length from config: {e}, using default 32768")
            return 32768

    def shutdown(self):
        """Shutdown all cached models properly (especially vLLM engines)."""
        with self._cache_lock:
            for cache_key, (model, tokenizer) in self._cache.items():
                if "vllm:" in cache_key:
                    try:
                        # vLLM LLM objects need explicit destruction
                        if hasattr(model, '__del__'):
                            del model
                        print(f"[ModelCache] Shut down vLLM model: {cache_key}")
                    except Exception as e:
                        print(f"[ModelCache] Warning: Failed to shutdown {cache_key}: {e}")
            self._cache.clear()
        print("[ModelCache] All models shut down")

    def clear(self):
        """Clear all cached models (alias for shutdown)."""
        self.shutdown()

    def get_cached_models(self):
        """Get list of currently cached models."""
        with self._cache_lock:
            return list(self._cache.keys())

    def estimate_memory_usage_mb(self) -> int:
        """
        Rough estimate of cached model memory usage.

        This is approximate and backend-specific.
        """
        # Simple heuristic: count model sizes from cache keys
        # In practice, this would need backend-specific memory introspection
        total_mb = 0

        with self._cache_lock:
            for key in self._cache.keys():
                # Extract model name and estimate size
                if "1.5b" in key.lower() or "1b" in key.lower():
                    total_mb += 3000  # ~3GB
                elif "3b" in key.lower():
                    total_mb += 6000  # ~6GB
                elif "7b" in key.lower():
                    total_mb += 14000  # ~14GB
                elif "8b" in key.lower():
                    total_mb += 16000  # ~16GB
                elif "14b" in key.lower():
                    total_mb += 28000  # ~28GB

        return total_mb
