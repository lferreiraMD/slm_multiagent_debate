"""
Shared helper functions used across multiple tasks.

These utilities are task-agnostic and can be reused by math, GSM, biography, and MMLU tasks.
"""

import json
from typing import List, Any, Dict


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
