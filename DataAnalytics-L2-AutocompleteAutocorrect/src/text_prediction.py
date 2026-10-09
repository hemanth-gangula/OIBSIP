"""
text_prediction.py — Shared utilities for corpus loading, tokenization,
vocabulary building, and n-gram construction.
"""

import re
import os
from collections import Counter
from typing import List, Dict, Tuple


def load_corpus(path: str) -> str:
    """
    Load a text file from disk and return its contents as a string.

    Args:
        path: Absolute or relative path to the corpus text file.

    Returns:
        The file contents as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Please ensure the data file is present before running."
        )
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize a text string into lowercase words.

    Steps:
        1. Convert to lowercase.
        2. Extract sequences of alphabetic characters (strips punctuation and digits).
        3. Remove empty tokens.

    Args:
        text: Input text string.

    Returns:
        List of lowercase alphabetic tokens.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t]  # guard against empty strings


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """
    Build a word-frequency dictionary from a list of tokens.

    Args:
        tokens: List of string tokens.

    Returns:
        Dictionary mapping word -> frequency, sorted by frequency descending.
    """
    freq = Counter(tokens)
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def build_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Build an n-gram counter from a list of tokens.

    Args:
        tokens: List of string tokens.
        n:      Size of each n-gram (e.g. 2 for bigrams, 3 for trigrams).

    Returns:
        Counter mapping n-gram tuples -> count.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)
