"""
autocorrect.py — Edit-distance and probabilistic spelling correction.

Two correction approaches:
  1. EditDistanceCorrector  — ranks candidates by edit distance, then frequency.
  2. ProbabilisticCorrector — Norvig-style P(candidate) = freq / total_words,
                              preferring edit-distance-1 candidates over edit-distance-2.
"""

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Dataset parser
# ---------------------------------------------------------------------------

def parse_spell_errors(path: str) -> List[Tuple[str, str]]:
    """
    Parse a spell-errors file in Norvig's format.

    File format (one entry per line)::

        correct_word: misspelling1, misspelling2, misspelling3*5, ...

    The ``*N`` suffix on a misspelling is a frequency count; it is stripped
    because we need only the unique (misspelled, correct) pair.

    Lines that do not match the expected format are skipped with a warning.

    Args:
        path: Path to the spell-errors file.

    Returns:
        List of (misspelled_word, correct_word) tuples.
    """
    pairs: List[Tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                correct = parts[0].strip().lower()
                if not correct:
                    continue
                for raw in parts[1].split(","):
                    misp = re.sub(r"\*\d+$", "", raw.strip()).strip().lower()
                    if misp and misp != correct:
                        pairs.append((misp, correct))
    except FileNotFoundError:
        print(f"Warning: spell-errors file not found at {path}")
    return pairs


# ---------------------------------------------------------------------------
# Edit-distance primitives
# ---------------------------------------------------------------------------

def edit_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses full dynamic-programming table (O(m*n) time, O(m*n) space).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum number of single-character edits (insertions, deletions,
        substitutions) to transform s1 into s2.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def edits1(word: str) -> set:
    """
    Generate all strings that are exactly one edit away from ``word``.

    Edits include:
        - Deletion:      remove one character.
        - Insertion:     add one character (a-z).
        - Substitution:  replace one character (a-z).
        - Transposition: swap two adjacent characters.

    Args:
        word: Input word (lowercase recommended).

    Returns:
        Set of strings within edit distance 1.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes      = [L + R[1:]           for L, R in splits if R]
    transposes   = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces     = [L + c + R[1:]       for L, R in splits if R for c in letters]
    inserts      = [L + c + R           for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word: str) -> set:
    """
    Generate all strings that are exactly two edits away from ``word``.

    Applies ``edits1`` twice. The result can be large (tens of thousands
    of strings for typical words) but provides broader coverage for
    severely misspelled words.

    Args:
        word: Input word.

    Returns:
        Set of strings within edit distance 2.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


# ---------------------------------------------------------------------------
# Corrector classes
# ---------------------------------------------------------------------------

class EditDistanceCorrector:
    """
    Spelling corrector based on edit distance and corpus frequency.

    Strategy:
        1. If the word is already in the vocabulary, return it unchanged.
        2. Generate candidate corrections within `max_edit` edit distance.
        3. Keep only candidates that are in the vocabulary.
        4. Sort by edit distance ascending, then by frequency descending.

    Attributes:
        vocab (Dict[str, int]): Word-to-frequency mapping.
    """

    def __init__(self, vocab: Dict[str, int]):
        """
        Args:
            vocab: Dictionary mapping word -> frequency.
        """
        self.vocab = vocab

    def correct(
        self, word: str, max_edit: int = 2
    ) -> List[Tuple[str, int, int]]:
        """
        Suggest corrections for a potentially misspelled word.

        Args:
            word:     The (possibly misspelled) input word.
            max_edit: Maximum edit distance to consider (1 or 2).

        Returns:
            If the word is already in the vocabulary:
                [(word, 0, freq)] — the word itself, edit-distance 0.
            Otherwise:
                List of (candidate, edit_distance, frequency) tuples,
                sorted by edit_distance ascending, then frequency descending.
                Returns [] if no vocabulary candidates are found.
        """
        word = word.lower()
        if word in self.vocab:
            return [(word, 0, self.vocab[word])]
        # Gather candidates at edit distance 1
        cands1 = {w: self.vocab[w] for w in edits1(word) if w in self.vocab}
        results: List[Tuple[str, int, int]] = [
            (w, 1, freq) for w, freq in cands1.items()
        ]
        if max_edit >= 2 and not results:
            cands2 = {w: self.vocab[w] for w in edits2(word) if w in self.vocab}
            results += [(w, 2, freq) for w, freq in cands2.items()]
        results.sort(key=lambda x: (x[1], -x[2]))
        return results


class ProbabilisticCorrector:
    """
    Norvig-style probabilistic spelling corrector.

    Strategy:
        1. If the word is in the vocabulary, return [(word, 1.0)].
        2. Otherwise find known words at edit distance 1; if any, return them
           ranked by P(w) = freq(w) / total_words.
        3. If no edit-1 candidates exist in the vocabulary, try edit distance 2.
        4. If still nothing, return [].

    This is a distinct approach from EditDistanceCorrector because it:
        - Uses probability mass (normalised by corpus size) rather than raw
          frequency as the ranking score.
        - Strictly prefers edit-distance-1 candidates over edit-distance-2,
          rather than mixing them.
        - Produces probabilities that are directly interpretable as likelihoods.

    Attributes:
        word_freq (Dict[str, int]): Word-to-frequency mapping.
        total_words (int): Sum of all word frequencies.
    """

    def __init__(self, word_freq: Dict[str, int]):
        """
        Args:
            word_freq: Dictionary mapping word -> frequency.
        """
        self.word_freq = word_freq
        self.total_words = sum(word_freq.values())

    def _prob(self, word: str) -> float:
        """Return P(word) = freq / total_words."""
        return self.word_freq.get(word, 0) / max(self.total_words, 1)

    def _known(self, words: set) -> set:
        """Return the subset of `words` that appear in the vocabulary."""
        return {w for w in words if w in self.word_freq}

    def correct(self, word: str) -> List[Tuple[str, float]]:
        """
        Return the most probable spelling corrections for `word`.

        Args:
            word: The (possibly misspelled) input word.

        Returns:
            If the word is in the vocabulary:
                [(word, 1.0)]
            Otherwise:
                List of (candidate, probability) sorted by probability desc.
                Returns [] if no known candidates are found.
        """
        word = word.lower()
        if word in self.word_freq:
            return [(word, 1.0)]
        candidates = (
            self._known(edits1(word))
            or self._known(edits2(word))
        )
        if not candidates:
            return []
        results = [(w, self._prob(w)) for w in candidates]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
