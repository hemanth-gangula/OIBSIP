"""
Tests for autocorrect modules: parser, edit distance, and correctors.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autocorrect import (
    parse_spell_errors,
    edit_distance,
    edits1,
    edits2,
    EditDistanceCorrector,
    ProbabilisticCorrector,
)


SAMPLE_VOCAB = {
    "spelling": 500, "receive": 300, "language": 250,
    "the": 10000, "and": 8000, "test": 100,
}


def test_parse_spell_errors_basic():
    content = "spelling: speling, speeling\nreceive: recieve, recive*3\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = f.name
    try:
        pairs = parse_spell_errors(tmp_path)
        misspelled_words = [m for m, _ in pairs]
        correct_words = [c for _, c in pairs]
        assert "speling" in misspelled_words
        assert "speeling" in misspelled_words
        assert "recieve" in misspelled_words
        assert "recive" in misspelled_words  # frequency suffix stripped
        assert all(c in ["spelling", "receive"] for c in correct_words)
    finally:
        os.unlink(tmp_path)


def test_parse_spell_errors_strips_frequency():
    content = "four: fore*5, fours, fuore\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = f.name
    try:
        pairs = parse_spell_errors(tmp_path)
        misspelled = [m for m, _ in pairs]
        assert "fore" in misspelled  # *5 stripped
        assert "fore*5" not in misspelled
    finally:
        os.unlink(tmp_path)


def test_edit_distance_identical():
    assert edit_distance("hello", "hello") == 0


def test_edit_distance_single_deletion():
    assert edit_distance("cat", "ca") == 1


def test_edit_distance_single_insertion():
    assert edit_distance("ca", "cat") == 1


def test_edit_distance_single_substitution():
    assert edit_distance("cat", "bat") == 1


def test_edit_distance_known_pair():
    assert edit_distance("speling", "spelling") == 1


def test_edits1_nonempty():
    result = edits1("test")
    assert len(result) > 0
    assert isinstance(result, set)


def test_edits1_contains_known_deletion():
    result = edits1("tests")
    assert "test" in result


def test_edit_distance_corrector_known_word():
    corrector = EditDistanceCorrector(SAMPLE_VOCAB)
    result = corrector.correct("spelling")
    assert len(result) == 1
    assert result[0][0] == "spelling"
    assert result[0][1] == 0  # edit distance 0


def test_edit_distance_corrector_misspelling():
    corrector = EditDistanceCorrector(SAMPLE_VOCAB)
    result = corrector.correct("speling")
    assert len(result) > 0
    candidates = [r[0] for r in result]
    assert "spelling" in candidates


def test_probabilistic_corrector_known_word():
    corrector = ProbabilisticCorrector(SAMPLE_VOCAB)
    result = corrector.correct("spelling")
    assert len(result) == 1
    assert result[0][0] == "spelling"
    assert result[0][1] == 1.0


def test_probabilistic_corrector_misspelling():
    corrector = ProbabilisticCorrector(SAMPLE_VOCAB)
    result = corrector.correct("speling")
    assert len(result) > 0
    candidates = [r[0] for r in result]
    assert "spelling" in candidates


def test_probabilistic_corrector_sorted_by_probability():
    corrector = ProbabilisticCorrector(SAMPLE_VOCAB)
    result = corrector.correct("speling")
    probs = [p for _, p in result]
    assert probs == sorted(probs, reverse=True)
