#!/usr/bin/env python3
"""
Model Output Diagnostic Script

Tests a model's raw output to diagnose formatting issues, special tokens,
chat template handling, and generation parameters.

Usage:
    python3 scripts/test_model_output.py --model vibethinker
    python3 scripts/test_model_output.py --model deepseek --prompt "What is 2+2?"
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import resolve_model_name
import argparse
import json


def test_model_raw_output(model_name: str, test_prompts: list = None):
    """
    Test model output without any wrapper processing.

    Shows:
    - Raw MLX generation output
    - Chat template application
    - Special tokens
    - Generation parameters effect
    """
    from mlx_lm import load, generate

    print("=" * 80)
    print("MODEL OUTPUT DIAGNOSTIC")
    print("=" * 80)
    print(f"Model: {model_name}")
    print()

    # Load model
    print("Loading model...")
    try:
        model, tokenizer = load(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    print()

    # Check chat template
    print("-" * 80)
    print("CHAT TEMPLATE CHECK")
    print("-" * 80)

    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print(f"Has chat template: {has_chat_template}")

    if has_chat_template:
        print(f"Chat template preview: {tokenizer.chat_template[:200]}...")

    print()

    # Default test prompts if none provided
    if test_prompts is None:
        test_prompts = [
            "What is 2+2?",
            "Explain photosynthesis in one sentence.",
            "Write a haiku about computers."
        ]

    # Test each prompt
    for idx, user_prompt in enumerate(test_prompts, 1):
        print("=" * 80)
        print(f"TEST {idx}: {user_prompt}")
        print("=" * 80)
        print()

        # Test 1: With chat template (if available)
        if has_chat_template:
            print("-" * 80)
            print("A) WITH CHAT TEMPLATE")
            print("-" * 80)

            messages = [{"role": "user", "content": user_prompt}]

            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                print("Formatted prompt:")
                print(f"```\n{formatted_prompt}\n```")
                print()

                print("Generating response (max_tokens=512)...")
                response = generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=512,
                    verbose=False
                )

                print()
                print("RAW OUTPUT:")
                print("=" * 40)
                print(response)
                print("=" * 40)
                print()

                # Analyze output
                analyze_output(response)

            except Exception as e:
                print(f"✗ Error with chat template: {e}")

            print()

        # Test 2: Without chat template (simple format)
        print("-" * 80)
        print("B) WITHOUT CHAT TEMPLATE (Simple Format)")
        print("-" * 80)

        simple_prompt = f"User: {user_prompt}\n\nAssistant:"

        print("Formatted prompt:")
        print(f"```\n{simple_prompt}\n```")
        print()

        try:
            print("Generating response (max_tokens=512)...")
            response = generate(
                model,
                tokenizer,
                prompt=simple_prompt,
                max_tokens=512,
                verbose=False
            )

            print()
            print("RAW OUTPUT:")
            print("=" * 40)
            print(response)
            print("=" * 40)
            print()

            # Analyze output
            analyze_output(response)

        except Exception as e:
            print(f"✗ Error with simple format: {e}")

        print()

    # Check special tokens
    print("=" * 80)
    print("TOKENIZER SPECIAL TOKENS")
    print("=" * 80)

    special_attrs = [
        'bos_token', 'eos_token', 'pad_token', 'unk_token',
        'sep_token', 'cls_token', 'mask_token'
    ]

    for attr in special_attrs:
        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)
            if value:
                print(f"  {attr}: {repr(value)}")

    print()

    # Check added tokens
    if hasattr(tokenizer, 'added_tokens_decoder'):
        added = tokenizer.added_tokens_decoder
        if added:
            print("Added tokens:")
            for token_id, token in list(added.items())[:20]:  # Show first 20
                print(f"  {token_id}: {repr(str(token))}")
            if len(added) > 20:
                print(f"  ... and {len(added) - 20} more")

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


def analyze_output(response: str):
    """Analyze response for common issues."""
    print("OUTPUT ANALYSIS:")
    print()

    # Check for special tokens
    special_patterns = {
        '<think>': 'Reasoning token (chain-of-thought)',
        '</think>': 'End reasoning token',
        '<|im_start|>': 'ChatML start token',
        '<|im_end|>': 'ChatML end token',
        '<|endoftext|>': 'End of text token',
        '<s>': 'BOS token',
        '</s>': 'EOS token',
        '[INST]': 'Llama instruction token',
        '[/INST]': 'Llama end instruction token',
    }

    found_tokens = []
    for token, description in special_patterns.items():
        if token in response:
            found_tokens.append(f"  ✓ Found '{token}' - {description}")

    if found_tokens:
        print("Special tokens detected:")
        for finding in found_tokens:
            print(finding)
    else:
        print("  No special tokens detected")

    print()

    # Check for repetition
    lines = response.split('\n')
    if len(lines) > 5:
        unique_lines = set(lines)
        if len(unique_lines) < len(lines) * 0.5:
            print("  ⚠ High repetition detected (many duplicate lines)")

    # Check length
    print(f"  Length: {len(response)} characters, {len(response.split())} words")

    # Check if output continues original prompt
    print(f"  Starts with prompt echo: {response.startswith('User:') or response.startswith('Assistant:')}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test and diagnose model output behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test VibeThinker
  python3 scripts/test_model_output.py --model vibethinker

  # Test with custom prompt
  python3 scripts/test_model_output.py --model deepseek --prompt "Solve: 2+2*3"

  # Test multiple prompts
  python3 scripts/test_model_output.py --model llama32-3b \\
      --prompt "What is AI?" \\
      --prompt "Explain quantum computing."
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model alias (e.g., 'vibethinker', 'deepseek') or full path"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        action='append',
        help="Test prompt (can be specified multiple times)"
    )

    args = parser.parse_args()

    # Resolve model name
    model_name = resolve_model_name(args.model)

    print()
    print(f"Resolved model: {args.model} -> {model_name}")
    print()

    # Run tests
    test_model_raw_output(model_name, args.prompt)


if __name__ == "__main__":
    main()
