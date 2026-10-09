"""
evaluation.py — Metrics for autocomplete and autocorrect evaluation.

Metrics defined:
  Autocomplete:
    - top1_accuracy:  fraction of test cases where the correct next word
                      appears as the #1 prediction.
    - top3_accuracy:  fraction where the correct word is in the top-3.
    - mrr:            Mean Reciprocal Rank — average of 1/rank when the
                      correct word is found in the prediction list.

  Autocorrect:
    - top1_accuracy:  fraction of test cases where the top-1 correction
                      matches the expected word.
    - topk_accuracy:  fraction where the expected word is in the top-k
                      predictions (k=3 by default).
"""

import csv
import os
from typing import Callable, Dict, List, Tuple


def evaluate_autocomplete(
    model_name: str,
    predict_fn: Callable[[str], List[Tuple[str, float]]],
    test_cases: List[Tuple[str, str]],
) -> Dict:
    """
    Evaluate an autocomplete model on a list of (context, expected_next_word) pairs.

    Args:
        model_name: Human-readable label for the model (e.g. "bigram").
        predict_fn: Callable that accepts a context string and returns a list
                    of (predicted_word, score) tuples sorted by score desc.
        test_cases: List of (context, expected_next_word) tuples.

    Returns:
        Dictionary with keys:
            model_name    (str)
            top1_accuracy (float, 0.0-1.0)
            top3_accuracy (float, 0.0-1.0)
            mrr           (float, 0.0-1.0)
            n_evaluated   (int)
    """
    top1, top3, rr_sum = 0, 0, 0.0
    n = 0
    for context, expected in test_cases:
        preds = predict_fn(context)
        pred_words = [w for w, _ in preds]
        n += 1
        if pred_words and pred_words[0] == expected:
            top1 += 1
        if expected in pred_words[:3]:
            top3 += 1
        if expected in pred_words:
            rank = pred_words.index(expected) + 1
            rr_sum += 1.0 / rank
    if n == 0:
        return {
            "model_name": model_name,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mrr": 0.0,
            "n_evaluated": 0,
        }
    return {
        "model_name": model_name,
        "top1_accuracy": top1 / n,
        "top3_accuracy": top3 / n,
        "mrr": rr_sum / n,
        "n_evaluated": n,
    }


def evaluate_autocorrect(
    corrector_name: str,
    correct_fn: Callable[[str], list],
    test_cases: List[Tuple[str, str]],
    k: int = 3,
) -> Dict:
    """
    Evaluate a spelling corrector on a list of (misspelled, correct) pairs.

    Args:
        corrector_name: Human-readable label (e.g. "EditDistance").
        correct_fn:     Callable that accepts a misspelled word string and
                        returns a list of (candidate, score) tuples or
                        (candidate, edit_dist, freq) tuples.
        test_cases:     List of (misspelled_word, correct_word) tuples.
        k:              Top-k for topk_accuracy (default 3).

    Returns:
        Dictionary with keys:
            corrector_name   (str)
            top1_accuracy    (float)
            topk_accuracy    (float)
            n_evaluated      (int)
            n_correct_top1   (int)
    """
    top1, topk = 0, 0
    n = 0
    for misspelled, expected in test_cases:
        results = correct_fn(misspelled)
        # Each result may be (word, score) or (word, edit_dist, freq)
        candidates = [r[0] for r in results]
        n += 1
        if candidates and candidates[0] == expected:
            top1 += 1
        if expected in candidates[:k]:
            topk += 1
    if n == 0:
        return {
            "corrector_name": corrector_name,
            "top1_accuracy": 0.0,
            "topk_accuracy": 0.0,
            "n_evaluated": 0,
            "n_correct_top1": 0,
        }
    return {
        "corrector_name": corrector_name,
        "top1_accuracy": top1 / n,
        "topk_accuracy": topk / n,
        "n_evaluated": n,
        "n_correct_top1": top1,
    }


def save_autocomplete_predictions(
    results: List[Dict], path: str
) -> None:
    """
    Save autocomplete prediction results to a CSV file.

    Args:
        results: List of dicts, each with keys:
                   context, model, rank, predicted_word, score
        path:    Destination file path (directories will be created if needed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["context", "model", "rank", "predicted_word", "score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_autocorrect_results(results: List[Dict], path: str) -> None:
    """
    Save autocorrect evaluation results to a CSV file.

    Args:
        results: List of dicts, each with keys:
                   misspelled_word, expected_word, predicted_word,
                   correct, approach, edit_distance
        path:    Destination file path (directories will be created if needed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "misspelled_word", "expected_word", "predicted_word",
        "correct", "approach", "edit_distance",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
