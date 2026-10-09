# Validation Report

**Generated:** Final validation pass  
**Project:** Autocomplete and Autocorrect Data Analytics  
**Location:** `OIBSIP/DataAnalytics-L2-AutocompleteAutocorrect/`

---

## 1. Dataset Files (Unchanged)

| File | Size (bytes) | Status |
|------|-------------|--------|
| `data/big.txt` | 6,488,666 | ✅ OK — exceeds 6 MB threshold |
| `data/spell-errors.txt` | 451,530 | ✅ OK — exceeds 400 KB threshold |

Both original dataset files are intact and unmodified.

---

## 2. Corpus Statistics (from notebook execution)

| Metric | Value |
|--------|-------|
| Total tokens | 1,105,285 |
| Vocabulary size | 29,157 unique words |
| Spell-error pairs parsed | 39,627 |

---

## 3. Source Files

| File | Size (bytes) | Status |
|------|-------------|--------|
| `src/__init__.py` | 0 | ✅ OK (package marker) |
| `src/text_prediction.py` | 2,218 | ✅ OK |
| `src/autocomplete.py` | 6,689 | ✅ OK |
| `src/autocorrect.py` | 8,963 | ✅ OK |
| `src/evaluation.py` | 5,512 | ✅ OK |
| `notebooks/Autocomplete_Autocorrect_Analysis.ipynb` | 512,294 | ✅ OK (executed) |
| `tests/test_autocomplete.py` | 2,996 | ✅ OK |
| `tests/test_autocorrect.py` | 3,599 | ✅ OK |
| `tests/test_evaluation.py` | 5,110 | ✅ OK |
| `README.md` | 10,468 | ✅ OK |
| `requirements.txt` | 124 | ✅ OK |
| `.gitignore` | 142 | ✅ OK |
| `LICENSE` | 1,076 | ✅ OK |

---

## 4. Pytest Results

```
platform win32 -- Python 3.14.6, pytest-9.1.1
38 tests collected
38 passed in 0.37s
```

**All 38 tests passed. No failures, errors, or skips.**

Test coverage:
- `tests/test_autocomplete.py` — 12 tests (tokenization, vocab, bigram, trigram, suggestions API)
- `tests/test_autocorrect.py` — 14 tests (spell-error parsing, edit distance, edits1, both correctors)
- `tests/test_evaluation.py` — 12 tests (autocomplete metrics, autocorrect metrics, CSV save functions)

---

## 5. Notebook Execution Status

The notebook `notebooks/Autocomplete_Autocorrect_Analysis.ipynb` was executed in full. All output cells contain rendered results. Evidence of execution:

- Notebook file size: 512,294 bytes (contains cell outputs)
- All 7 interactive HTML chart files exist and are populated (~4.9 MB each, including Plotly JS bundle)
- Both CSV prediction files exist with actual model output

---

## 6. Autocomplete Evaluation

**Test contexts evaluated:** 12 unique contexts (exceeds required minimum of 10)

Sample contexts tested: `the`, `it was`, `in the`, `he said`, `machine learning`, `she looked`,
`a great`, `at the`, `one of`, `the man`, `for the`, `with a`

| Model | Top-1 Accuracy | Top-3 Accuracy | MRR | N (eval pairs) |
|-------|---------------|---------------|-----|----------------|
| Bigram | 0.305 | 0.570 | 0.422 | 200 |
| Trigram | 0.650 | 0.800 | 0.720 | 200 |

Evaluation used held-out next-word pairs from the corpus (train/test split applied).

---

## 7. Autocorrect Evaluation

**Unique misspelled words evaluated:** 50 (exceeds required minimum of 20)  
**Total rows in CSV:** 100 (50 words × 2 approaches)  
**Test pairs sourced from:** `data/spell-errors.txt` (Norvig's spelling-error corpus)

| Approach | Top-1 Accuracy | Top-3 Accuracy | N |
|----------|---------------|---------------|---|
| EditDistance | 0.480 | 0.580 | 50 |
| Probabilistic | 0.480 | 0.580 | 50 |

Both approaches share the same candidate-generation logic (edit distance enumeration) but differ in ranking: EditDistance ranks by minimum edit distance first, then corpus frequency; Probabilistic ranks purely by corpus unigram probability P(word).

---

## 8. Output Files

### Interactive HTML Charts (Plotly)

| File | Size (bytes) | Status |
|------|-------------|--------|
| `outputs/figures/top_20_words.html` | 4,863,745 | ✅ OK |
| `outputs/figures/word_frequency_distribution.html` | 4,962,506 | ✅ OK |
| `outputs/figures/autocomplete_model_comparison.html` | 4,863,720 | ✅ OK |
| `outputs/figures/autocomplete_prediction_scores.html` | 4,871,985 | ✅ OK |
| `outputs/figures/autocorrect_performance_comparison.html` | 4,863,753 | ✅ OK |
| `outputs/figures/correction_examples.html` | 4,864,366 | ✅ OK |
| `outputs/figures/edit_distance_heatmap.html` | 4,863,617 | ✅ OK |

All 7 charts generated. Each file includes the self-contained Plotly JS bundle and retains full interactivity when opened in a browser.

### Prediction CSV Files

| File | Size (bytes) | Rows | Status |
|------|-------------|------|--------|
| `outputs/predictions/autocomplete_predictions.csv` | 2,371 | 72 (12 contexts × 2 models × 3 predictions) | ✅ OK |
| `outputs/predictions/autocorrect_results.csv` | 4,716 | 100 (50 words × 2 approaches) | ✅ OK |

CSV columns:
- `autocomplete_predictions.csv`: `context, model, rank, predicted_word, score`
- `autocorrect_results.csv`: `misspelled_word, expected_word, predicted_word, correct, approach, edit_distance`

---

## 9. Unresolved Issues

None. All required files are present, all tests pass, all output files exist with non-trivial content, and both CSVs contain actual model results derived from the corpus.

---

## 10. Summary

| Check | Result |
|-------|--------|
| Required source files present | ✅ 15/15 |
| Output HTML charts present | ✅ 7/7 |
| Output CSV predictions present | ✅ 2/2 |
| Validation report present | ✅ |
| Data files intact (size check) | ✅ |
| Pytest suite | ✅ 38/38 passed |
| Notebook executed | ✅ |
| Autocomplete contexts ≥ 10 | ✅ 12 |
| Autocorrect pairs ≥ 20 | ✅ 50 |
