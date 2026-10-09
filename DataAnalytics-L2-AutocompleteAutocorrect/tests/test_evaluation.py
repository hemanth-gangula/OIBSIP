"""
Tests for evaluation metrics and CSV output functions.
"""
import sys
import os
import tempfile
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import (
    evaluate_autocomplete,
    evaluate_autocorrect,
    save_autocomplete_predictions,
    save_autocorrect_results,
)


# ------------------------------------------------------------------
# Helpers / fixtures
# ------------------------------------------------------------------

def mock_autocomplete_fn(context: str):
    """Always predicts 'the' as top suggestion."""
    return [("the", 0.9), ("a", 0.05), ("an", 0.03)]


def mock_correct_fn(word: str):
    """Always corrects to 'spelling'."""
    return [("spelling", 0.8), ("spelling2", 0.1)]


# ------------------------------------------------------------------
# Autocomplete evaluation tests
# ------------------------------------------------------------------

def test_evaluate_autocomplete_required_keys():
    test_cases = [("hello world", "the"), ("once upon", "a")]
    result = evaluate_autocomplete("test_model", mock_autocomplete_fn, test_cases)
    for key in ["model_name", "top1_accuracy", "top3_accuracy", "mrr", "n_evaluated"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_autocomplete_top1_range():
    test_cases = [("hello", "the"), ("world", "is")]
    result = evaluate_autocomplete("test", mock_autocomplete_fn, test_cases)
    assert 0.0 <= result["top1_accuracy"] <= 1.0


def test_evaluate_autocomplete_correct_top1():
    """mock_autocomplete_fn always returns 'the' first — test cases with 'the' as expected."""
    test_cases = [("hello", "the"), ("world", "the")]
    result = evaluate_autocomplete("test", mock_autocomplete_fn, test_cases)
    assert result["top1_accuracy"] == 1.0


def test_evaluate_autocomplete_n_evaluated():
    test_cases = [("a", "b"), ("c", "d"), ("e", "f")]
    result = evaluate_autocomplete("test", mock_autocomplete_fn, test_cases)
    assert result["n_evaluated"] == 3


def test_evaluate_autocomplete_empty():
    result = evaluate_autocomplete("test", mock_autocomplete_fn, [])
    assert result["n_evaluated"] == 0
    assert result["top1_accuracy"] == 0.0


def test_evaluate_autocomplete_mrr_range():
    test_cases = [("hello", "the")]
    result = evaluate_autocomplete("test", mock_autocomplete_fn, test_cases)
    assert 0.0 <= result["mrr"] <= 1.0


# ------------------------------------------------------------------
# Autocorrect evaluation tests
# ------------------------------------------------------------------

def test_evaluate_autocorrect_required_keys():
    test_cases = [("speling", "spelling"), ("recieve", "receive")]
    result = evaluate_autocorrect("test", mock_correct_fn, test_cases)
    for key in ["corrector_name", "top1_accuracy", "topk_accuracy", "n_evaluated", "n_correct_top1"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_autocorrect_top1_range():
    test_cases = [("speling", "spelling")]
    result = evaluate_autocorrect("test", mock_correct_fn, test_cases)
    assert 0.0 <= result["top1_accuracy"] <= 1.0


def test_evaluate_autocorrect_correct_top1():
    test_cases = [("speling", "spelling"), ("recieve", "spelling")]
    result = evaluate_autocorrect("test", mock_correct_fn, test_cases)
    assert result["top1_accuracy"] == 1.0


def test_evaluate_autocorrect_n_evaluated():
    test_cases = [("a", "b"), ("c", "d")]
    result = evaluate_autocorrect("test", mock_correct_fn, test_cases)
    assert result["n_evaluated"] == 2


# ------------------------------------------------------------------
# CSV output tests
# ------------------------------------------------------------------

def test_save_autocomplete_predictions_creates_file():
    rows = [
        {"context": "hello world", "model": "bigram", "rank": 1, "predicted_word": "the", "score": 0.9},
        {"context": "hello world", "model": "trigram", "rank": 1, "predicted_word": "a", "score": 0.7},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "preds.csv")
        save_autocomplete_predictions(rows, path)
        assert os.path.exists(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert len(written) == 2
        assert written[0]["context"] == "hello world"


def test_save_autocorrect_results_creates_file():
    rows = [
        {
            "misspelled_word": "speling", "expected_word": "spelling",
            "predicted_word": "spelling", "correct": True,
            "approach": "EditDistance", "edit_distance": 1,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.csv")
        save_autocorrect_results(rows, path)
        assert os.path.exists(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert len(written) == 1
        assert written[0]["misspelled_word"] == "speling"
