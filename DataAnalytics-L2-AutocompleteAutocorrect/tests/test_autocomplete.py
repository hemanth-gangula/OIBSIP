"""
Tests for autocomplete models and shared text utilities.
"""
import sys
import os

# Add src/ to path so imports work from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_prediction import tokenize, build_vocab, build_ngrams
from autocomplete import BigramModel, TrigramModel, get_autocomplete_suggestions


SAMPLE_TOKENS = "the cat sat on the mat the cat ate the rat".split()


def test_tokenize_lowercase():
    tokens = tokenize("Hello World, TEST!")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_no_empty_strings():
    tokens = tokenize("  hello   world  ")
    assert "" not in tokens
    assert len(tokens) > 0


def test_tokenize_strips_punctuation():
    tokens = tokenize("don't stop, can't stop.")
    assert all(t.isalpha() for t in tokens)


def test_build_vocab_counts():
    tokens = ["cat", "dog", "cat", "bird", "cat"]
    vocab = build_vocab(tokens)
    assert vocab["cat"] == 3
    assert vocab["dog"] == 1
    assert vocab["bird"] == 1


def test_build_vocab_sorted_by_frequency():
    tokens = ["a", "b", "a", "a", "b"]
    vocab = build_vocab(tokens)
    freqs = list(vocab.values())
    assert freqs == sorted(freqs, reverse=True)


def test_bigram_predict_returns_at_most_top_k():
    model = BigramModel().fit(SAMPLE_TOKENS)
    results = model.predict("the", top_k=3)
    assert len(results) <= 3


def test_bigram_predict_sorted_descending():
    model = BigramModel().fit(SAMPLE_TOKENS)
    results = model.predict("the", top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_bigram_predict_unseen_word_returns_empty():
    model = BigramModel().fit(SAMPLE_TOKENS)
    results = model.predict("zzzzunknownzzzz", top_k=3)
    assert results == []


def test_trigram_predict_returns_at_most_top_k():
    model = TrigramModel().fit(SAMPLE_TOKENS)
    results = model.predict(("the", "cat"), top_k=3)
    assert len(results) <= 3


def test_trigram_predict_unseen_context_fallback():
    """Trigram should fall back gracefully on unseen context (no crash)."""
    model = TrigramModel().fit(SAMPLE_TOKENS)
    # If this context is unseen as trigram, it should fall back to bigram
    results = model.predict(("zzz", "yyy"), top_k=3)
    assert isinstance(results, list)


def test_get_autocomplete_suggestions_keys():
    bigram = BigramModel().fit(SAMPLE_TOKENS)
    trigram = TrigramModel().fit(SAMPLE_TOKENS)
    suggestions = get_autocomplete_suggestions("the cat", bigram, trigram)
    assert "bigram" in suggestions
    assert "trigram" in suggestions


def test_get_autocomplete_suggestions_types():
    bigram = BigramModel().fit(SAMPLE_TOKENS)
    trigram = TrigramModel().fit(SAMPLE_TOKENS)
    suggestions = get_autocomplete_suggestions("the", bigram, trigram)
    for model_key in ["bigram", "trigram"]:
        for item in suggestions[model_key]:
            assert isinstance(item, tuple)
            assert len(item) == 2
