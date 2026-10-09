"""
autocomplete.py — Bigram and Trigram autocomplete models.

BigramModel:  P(w2 | w1) = count(w1, w2) / count(w1)
TrigramModel: P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
              Falls back to BigramModel when the trigram context is unseen.
"""

import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from text_prediction import build_ngrams


class BigramModel:
    """
    A bigram language model for next-word prediction.

    Attributes:
        bigram_counts (Counter): Counts of (w1, w2) bigrams.
        unigram_counts (Counter): Counts of individual words.
        fitted (bool): True after fit() has been called.
    """

    def __init__(self):
        self.bigram_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.fitted: bool = False

    def fit(self, tokens: List[str]) -> "BigramModel":
        """
        Train the model on a list of tokens.

        Args:
            tokens: List of string tokens from the corpus.

        Returns:
            self (for chaining).
        """
        self.unigram_counts = Counter(tokens)
        self.bigram_counts = build_ngrams(tokens, 2)
        self.fitted = True
        return self

    def predict(self, word: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the most likely next words after `word`.

        Args:
            word:  The context word (previous word).
            top_k: Number of top predictions to return.

        Returns:
            List of (predicted_word, probability) tuples sorted by probability
            descending. Returns an empty list if the context word has never
            been seen (avoids crashing on unknown contexts).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        word = word.lower()
        context_count = self.unigram_counts.get(word, 0)
        if context_count == 0:
            return []
        # Collect all bigrams starting with `word`
        candidates = {
            w2: cnt
            for (w1, w2), cnt in self.bigram_counts.items()
            if w1 == word
        }
        if not candidates:
            return []
        # Convert to conditional probability
        results = [
            (w2, cnt / context_count)
            for w2, cnt in candidates.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class TrigramModel:
    """
    A trigram language model for next-word prediction with bigram fallback.

    Attributes:
        trigram_counts (Counter): Counts of (w1, w2, w3) trigrams.
        bigram_counts (Counter): Counts of (w1, w2) bigrams (context).
        bigram_model (BigramModel): Fallback for unseen trigram contexts.
        fitted (bool): True after fit() has been called.
    """

    def __init__(self):
        self.trigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.bigram_model: Optional[BigramModel] = None
        self.fitted: bool = False

    def fit(self, tokens: List[str]) -> "TrigramModel":
        """
        Train the model on a list of tokens.

        Args:
            tokens: List of string tokens.

        Returns:
            self (for chaining).
        """
        self.trigram_counts = build_ngrams(tokens, 3)
        self.bigram_counts = build_ngrams(tokens, 2)
        self.bigram_model = BigramModel().fit(tokens)
        self.fitted = True
        return self

    def predict(
        self, context_tuple: Tuple[str, str], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely next word given a two-word context.

        Falls back to BigramModel predictions if the trigram context has
        never been seen in the training data.

        Args:
            context_tuple: A (w1, w2) tuple representing the two preceding words.
            top_k: Number of top predictions to return.

        Returns:
            List of (predicted_word, probability) tuples sorted descending.
            Returns empty list if neither trigram nor bigram context is known.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        if len(context_tuple) != 2:
            raise ValueError("context_tuple must be a (w1, w2) pair.")
        w1, w2 = context_tuple[0].lower(), context_tuple[1].lower()
        context_bigram = (w1, w2)
        context_count = self.bigram_counts.get(context_bigram, 0)
        if context_count > 0:
            candidates = {
                w3: cnt
                for (c1, c2, w3), cnt in self.trigram_counts.items()
                if c1 == w1 and c2 == w2
            }
            if candidates:
                results = [
                    (w3, cnt / context_count)
                    for w3, cnt in candidates.items()
                ]
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
        # Fallback to bigram on w2
        return self.bigram_model.predict(w2, top_k=top_k)


def get_autocomplete_suggestions(
    text: str,
    bigram_model: BigramModel,
    trigram_model: TrigramModel,
    top_k: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Generate autocomplete suggestions from both bigram and trigram models.

    Args:
        text:          Input text string (one or more words).
        bigram_model:  A fitted BigramModel instance.
        trigram_model: A fitted TrigramModel instance.
        top_k:         Number of predictions per model.

    Returns:
        Dictionary with keys 'bigram' and 'trigram', each mapping to a list
        of (predicted_word, score) tuples sorted by score descending.

    Example:
        >>> suggestions = get_autocomplete_suggestions(
        ...     "machine learning", bigram_model, trigram_model
        ... )
        >>> suggestions['bigram']   # top-3 bigram predictions after 'learning'
        >>> suggestions['trigram']  # top-3 trigram predictions after ('machine', 'learning')
    """
    words = re.findall(r"[a-z]+", text.lower())
    bigram_preds: List[Tuple[str, float]] = []
    trigram_preds: List[Tuple[str, float]] = []
    if len(words) >= 1:
        bigram_preds = bigram_model.predict(words[-1], top_k=top_k)
    if len(words) >= 2:
        trigram_preds = trigram_model.predict(
            (words[-2], words[-1]), top_k=top_k
        )
    elif len(words) == 1:
        # Single word: use bigram prediction as trigram fallback too
        trigram_preds = bigram_preds
    return {"bigram": bigram_preds, "trigram": trigram_preds}
