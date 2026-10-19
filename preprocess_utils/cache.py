"""Caching utilities for preprocessing.

This module provides functions to cache preprocessed data to avoid
reprocessing the same dataset multiple times.
"""

import os
import json
from typing import List, Dict, Any


def check_cache(cache_path: str) -> bool:
    """Check if cache file exists.

    Args:
        cache_path: Path to the cache file

    Returns:
        True if cache file exists and is non-empty, False otherwise
    """
    if not os.path.exists(cache_path):
        return False

    # Check if file is non-empty
    return os.path.getsize(cache_path) > 0


def load_cache(cache_path: str) -> List[Dict[str, Any]]:
    """Load cached data.

    Args:
        cache_path: Path to the cache file

    Returns:
        List of sample dictionaries loaded from cache

    Raises:
        FileNotFoundError: If cache file doesn't exist
        json.JSONDecodeError: If cache file is not valid JSON
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    samples = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                samples.append(json.loads(line))

    return samples


def save_cache(samples: List[Dict[str, Any]], cache_path: str) -> None:
    """Save data to cache.

    Args:
        samples: List of sample dictionaries to cache
        cache_path: Path to the cache file (will be created if doesn't exist)

    Note:
        This function writes in JSONL format (one JSON object per line)
        for compatibility with the rest of the Table-Critic pipeline.
    """
    # Ensure directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
