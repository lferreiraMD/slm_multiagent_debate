"""
Model caching to avoid reloading the same model multiple times.

Critical for memory efficiency when running multiple agents with the same model.
"""

from typing import Tuple, Any, Optional
import threading


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
        backend: str = "mlx"
    ) -> Tuple[Any, Any]:
        """
        Get model from cache or load it.

        Args:
            model_path: Path or name of model to load
            backend: Backend to use ('mlx', 'ollama', 'vllm')

        Returns:
            Tuple of (model, tokenizer) for MLX/vLLM, or model for Ollama
        """
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
            model = self._load_vllm(model_path)
            result = (model, None)  # vLLM doesn't separate tokenizer
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

    def _load_vllm(self, model_path: str) -> Any:
        """Load vLLM model."""
        from vllm import LLM

        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Adjust for multi-GPU
            trust_remote_code=True
        )
        return llm

    def clear(self):
        """Clear all cached models."""
        with self._cache_lock:
            self._cache.clear()
        print("[ModelCache] Cache cleared")

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
