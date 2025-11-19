"""
OpenAI-compatible wrapper for local LLM inference.

Supports multiple backends:
- MLX (Apple Silicon via mlx-lm)
- Ollama (GGUF models)
- vLLM (HPC with NVIDIA GPUs)

Usage:
    from utils.llm_wrapper import ChatCompletion

    response = ChatCompletion.create(
        model="mlx-community/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=1.0,
        max_tokens=512
    )
"""

import os
import platform
import time
from typing import List, Dict, Optional, Any
from .model_cache import ModelCache


class ChatCompletion:
    """OpenAI ChatCompletion-compatible interface for local LLMs."""

    _backend = None  # Auto-detected: 'mlx', 'ollama', or 'vllm'
    _model_cache = ModelCache()

    @classmethod
    def _detect_backend(cls) -> str:
        """Auto-detect best available backend."""
        if cls._backend is not None:
            return cls._backend

        # Check for MLX on Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx
                import mlx_lm
                cls._backend = "mlx"
                return "mlx"
            except ImportError:
                pass

        # Check for Ollama (look for ollama process or API)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code == 200:
                cls._backend = "ollama"
                return "ollama"
        except:
            pass

        # Check for vLLM
        try:
            import vllm
            cls._backend = "vllm"
            return "vllm"
        except ImportError:
            pass

        raise RuntimeError(
            "No LLM backend detected. Please install one of:\n"
            "  - mlx-lm (Apple Silicon)\n"
            "  - ollama (cross-platform)\n"
            "  - vllm (NVIDIA GPUs)"
        )

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion (OpenAI-compatible interface).

        Args:
            model: Model name/path
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (default 1.0 to match GPT-3.5)
            max_tokens: Maximum tokens to generate (None = model decides)
            top_p: Nucleus sampling parameter
            n: Number of completions (only n=1 supported currently)
            **kwargs: Additional arguments, including an optional 'backend' to override auto-detection.

        Returns:
            OpenAI-compatible response dict with choices
        """
        if n != 1:
            raise NotImplementedError("Only n=1 is currently supported")

        # Allow backend override via kwargs, otherwise auto-detect
        backend = kwargs.pop("backend", None) or cls._detect_backend()

        if backend == "mlx":
            return cls._create_mlx(model, messages, temperature, max_tokens, top_p, **kwargs)
        elif backend == "ollama":
            return cls._create_ollama(model, messages, temperature, max_tokens, top_p, **kwargs)
        elif backend == "vllm":
            return cls._create_vllm(model, messages, temperature, max_tokens, top_p, **kwargs)
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

    @classmethod
    def _create_mlx(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """MLX-LM backend implementation."""
        from mlx_lm import load, generate

        # Load model (cached)
        model_obj, tokenizer = cls._model_cache.get_or_load(model, backend="mlx")

        # Format messages using chat template
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback for models without chat template
            prompt = cls._format_messages_fallback(messages)

        # Generate response
        # Note: MLX max_tokens default is model-dependent, use 2048 as reasonable limit
        # Reasoning models (VibeThinker, DeepSeek-R1) need more tokens for chain-of-thought
        if max_tokens is None:
            # Check if this is a reasoning model that uses <think> tokens
            model_lower = model.lower()
            is_reasoning_model = any(keyword in model_lower for keyword in ['thinker', 'deepseek'])
            # VibeThinker creators recommend 40960 max_token_length
            max_tokens = 40960 if is_reasoning_model else 4096

        # Create sampler with temperature and top_p for diversity
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p if top_p < 1.0 else 0.0  # top_p=1.0 means disabled in MLX
        )

        start_time = time.time()
        # Generate with custom sampler for temperature control
        response_text = generate(
            model_obj,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        latency = time.time() - start_time

        # Post-process reasoning models: extract answer after chain-of-thought
        # Models like VibeThinker and DeepSeek-R1 use <think>...</think> for reasoning
        # We want to preserve the reasoning but prioritize the final answer
        if '<think>' in response_text and '</think>' in response_text:
            # Split by </think> to get the final answer part
            parts = response_text.split('</think>')
            if len(parts) > 1:
                # Keep everything after the last </think> tag
                final_answer = parts[-1].strip()
                # Only use the extracted answer if it's non-empty
                if final_answer:
                    response_text = final_answer

        # Format as OpenAI-compatible response
        return {
            "id": f"mlx-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
            },
            "_metadata": {
                "backend": "mlx",
                "latency_seconds": latency
            }
        }

    @classmethod
    def _create_ollama(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Ollama backend implementation."""
        import requests

        # Ollama API endpoint
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        latency = time.time() - start_time

        result = response.json()
        content = result["message"]["content"]

        # Post-process reasoning models: extract answer after chain-of-thought
        # Models like VibeThinker and DeepSeek-R1 use <think>...</think> for reasoning
        if '<think>' in content and '</think>' in content:
            # Split by </think> to get the final answer part
            parts = content.split('</think>')
            if len(parts) > 1:
                # Keep everything after the last </think> tag
                final_answer = parts[-1].strip()
                # Only use the extracted answer if it's non-empty
                if final_answer:
                    content = final_answer

        # Format as OpenAI-compatible response
        return {
            "id": f"ollama-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            },
            "_metadata": {
                "backend": "ollama",
                "latency_seconds": latency
            }
        }

    @classmethod
    def _create_vllm(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """vLLM backend implementation (for HPC with NVIDIA GPUs)."""
        from vllm import LLM, SamplingParams

        # Load model and tokenizer (cached)
        llm, tokenizer = cls._model_cache.get_or_load(model, backend="vllm")

        # Format messages using chat template if available
        try:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = cls._format_messages_fallback(messages)
        except Exception:
            # Fallback if chat template fails
            prompt = cls._format_messages_fallback(messages)

        # Reasoning models (VibeThinker, DeepSeek-R1) need more tokens for chain-of-thought
        if max_tokens is None:
            # Check if this is a reasoning model that uses <think> tokens
            model_lower = model.lower()
            is_reasoning_model = any(keyword in model_lower for keyword in ['thinker', 'deepseek'])
            # VibeThinker creators recommend 40960 max_token_length
            max_tokens = 40960 if is_reasoning_model else 4096

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # Generate
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        latency = time.time() - start_time

        output = outputs[0]
        response_text = output.outputs[0].text

        # Post-process reasoning models: extract answer after chain-of-thought
        # Models like VibeThinker and DeepSeek-R1 use <think>...</think> for reasoning
        # We want to preserve the reasoning but prioritize the final answer
        if '<think>' in response_text and '</think>' in response_text:
            # Split by </think> to get the final answer part
            parts = response_text.split('</think>')
            if len(parts) > 1:
                # Keep everything after the last </think> tag
                final_answer = parts[-1].strip()
                # Only use the extracted answer if it's non-empty
                if final_answer:
                    response_text = final_answer

        # Format as OpenAI-compatible response
        return {
            "id": f"vllm-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            },
            "_metadata": {
                "backend": "vllm",
                "latency_seconds": latency
            }
        }

    @staticmethod
    def _format_messages_fallback(messages: List[Dict[str, str]]) -> str:
        """
        Fallback message formatting for models without chat templates.

        Formats messages in a simple conversational format.
        """
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        # Add final assistant prompt
        formatted.append("Assistant:")

        return "\n\n".join(formatted)
