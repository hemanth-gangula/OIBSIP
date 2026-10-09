# Web Application Validation Report

Generated as part of FEAT-003 — Deployment configuration, Flask app tests, and documentation.

---

## Files Created

| Path | Description |
|------|-------------|
| `app.py` | Flask application and all API routes |
| `templates/base.html` | Jinja2 base layout template |
| `templates/index.html` | Home / overview page |
| `templates/autocomplete.html` | Autocomplete playground |
| `templates/autocorrect.html` | Autocorrect playground |
| `templates/analytics.html` | Data analytics dashboard |
| `templates/models.html` | NLP model comparison |
| `templates/about.html` | About / evaluation methodology |
| `public/css/main.css` | Main stylesheet |
| `public/js/main.js` | Shared JavaScript utilities |
| `public/js/autocomplete.js` | Autocomplete playground JavaScript |
| `public/js/autocorrect.js` | Autocorrect playground JavaScript |
| `public/js/analytics.js` | Analytics dashboard JavaScript (Plotly) |
| `vercel.json` | Vercel build and routing configuration |
| `.python-version` | Python 3.12 runtime pin for Vercel |
| `requirements.txt` | Updated with flask, flask-cors, gunicorn |
| `tests/test_app.py` | Flask test-client test suite |
| `outputs/reports/webapp_validation_report.md` | This file |

---

## Test Results

Command: `python -m pytest tests/ -v`

```
platform win32 -- Python 3.14.6, pytest-9.1.1
=========== test session starts ============
collected 55 items

tests/test_app.py::TestPageRoutes::test_index_ok                       PASSED
tests/test_app.py::TestPageRoutes::test_autocomplete_page_ok           PASSED
tests/test_app.py::TestPageRoutes::test_autocorrect_page_ok            PASSED
tests/test_app.py::TestPageRoutes::test_analytics_page_ok              PASSED
tests/test_app.py::TestPageRoutes::test_models_page_ok                 PASSED
tests/test_app.py::TestPageRoutes::test_about_page_ok                  PASSED
tests/test_app.py::TestPageRoutes::test_health_ok                      PASSED
tests/test_app.py::TestPageRoutes::test_nonexistent_404                PASSED
tests/test_app.py::TestApiAutocomplete::test_valid_request_returns_bigram_key PASSED
tests/test_app.py::TestApiAutocomplete::test_missing_body_returns_400  PASSED
tests/test_app.py::TestApiAutocomplete::test_empty_text_returns_400    PASSED
tests/test_app.py::TestApiAutocomplete::test_text_too_long_returns_400 PASSED
tests/test_app.py::TestApiAutocorrect::test_valid_word_returns_200     PASSED
tests/test_app.py::TestApiAutocorrect::test_missing_body_returns_400   PASSED
tests/test_app.py::TestApiAutocorrect::test_multi_word_returns_400     PASSED
tests/test_app.py::TestApiMetrics::test_returns_200_with_expected_keys PASSED
tests/test_app.py::TestApiAnalytics::test_returns_200_with_top_20_words PASSED
tests/test_autocomplete.py::test_tokenize_lowercase                    PASSED
tests/test_autocomplete.py::test_tokenize_no_empty_strings             PASSED
tests/test_autocomplete.py::test_tokenize_strips_punctuation           PASSED
tests/test_autocomplete.py::test_build_vocab_counts                    PASSED
tests/test_autocomplete.py::test_build_vocab_sorted_by_frequency       PASSED
tests/test_autocomplete.py::test_bigram_predict_returns_at_most_top_k  PASSED
tests/test_autocomplete.py::test_bigram_predict_sorted_descending      PASSED
tests/test_autocomplete.py::test_bigram_predict_unseen_word_returns_empty PASSED
tests/test_autocomplete.py::test_trigram_predict_returns_at_most_top_k PASSED
tests/test_autocomplete.py::test_trigram_predict_unseen_context_fallback PASSED
tests/test_autocomplete.py::test_get_autocomplete_suggestions_keys     PASSED
tests/test_autocomplete.py::test_get_autocomplete_suggestions_types    PASSED
tests/test_autocorrect.py::test_parse_spell_errors_basic               PASSED
tests/test_autocorrect.py::test_parse_spell_errors_strips_frequency    PASSED
tests/test_autocorrect.py::test_edit_distance_identical                PASSED
tests/test_autocorrect.py::test_edit_distance_single_deletion          PASSED
tests/test_autocorrect.py::test_edit_distance_single_insertion         PASSED
tests/test_autocorrect.py::test_edit_distance_single_substitution      PASSED
tests/test_autocorrect.py::test_edit_distance_known_pair               PASSED
tests/test_autocorrect.py::test_edits1_nonempty                        PASSED
tests/test_autocorrect.py::test_edits1_contains_known_deletion         PASSED
tests/test_autocorrect.py::test_edit_distance_corrector_known_word     PASSED
tests/test_autocorrect.py::test_edit_distance_corrector_misspelling    PASSED
tests/test_autocorrect.py::test_probabilistic_corrector_known_word     PASSED
tests/test_autocorrect.py::test_probabilistic_corrector_misspelling    PASSED
tests/test_autocorrect.py::test_probabilistic_corrector_sorted_by_probability PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_required_keys     PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_top1_range        PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_correct_top1      PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_n_evaluated       PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_empty             PASSED
tests/test_evaluation.py::test_evaluate_autocomplete_mrr_range         PASSED
tests/test_evaluation.py::test_evaluate_autocorrect_required_keys      PASSED
tests/test_evaluation.py::test_evaluate_autocorrect_top1_range         PASSED
tests/test_evaluation.py::test_evaluate_autocorrect_correct_top1       PASSED
tests/test_evaluation.py::test_evaluate_autocorrect_n_evaluated        PASSED
tests/test_evaluation.py::test_save_autocomplete_predictions_creates_file PASSED
tests/test_evaluation.py::test_save_autocorrect_results_creates_file   PASSED

=========== 55 passed in 11.70s ============
```

**Result: 55 passed, 0 failed** (38 existing + 17 new Flask app tests)

---

## Model Loading Verification

Command: `python -c "import app; m=app.load_models(); print(...)"`

```
INFO:app:Models loaded: vocab=29157, bigrams=127241, trigrams=656441
vocab: 29157   bigram: 127241   trigram: 656441
```

Models loaded successfully from `data/processed/model_data.json`.

---

## Vercel Configuration

File: `vercel.json`

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python",
      "config": {
        "excludeFiles": [
          "notebooks/**",
          "tests/**",
          ".pytest_cache/**",
          "outputs/figures/**",
          "__pycache__/**",
          ".git/**",
          "data/big.txt",
          ".agents/**"
        ]
      }
    }
  ],
  "routes": [
    { "src": "/(css|js)/(.*)", "dest": "/public/$1/$2" },
    { "src": "/(.*)", "dest": "/app.py" }
  ]
}
```

Status: **Valid JSON** — `builds` and `routes` keys present.

---

## Bundle Size Note

The `outputs/figures/` directory contains 7 interactive Plotly HTML charts (~4.6 MB each, ~32 MB total). These are excluded from the Vercel deployment bundle via `excludeFiles: ["outputs/figures/**"]`.

The web application serves equivalent charts as lightweight Plotly JSON via `GET /api/analytics`, rendered client-side in the browser. The large HTML chart artifacts are retained in the repository for local use and the Jupyter notebook workflow, but are not deployed to Vercel.

`data/big.txt` (~6.5 MB) is also excluded from the Vercel bundle; the models are served pre-computed from `data/processed/model_data.json` which is committed to the repository via `.gitignore` exception.

---

## .python-version

Content: `3.12` — pins the Vercel serverless runtime to Python 3.12.

---

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| `.python-version` contains exactly `3.12` | ✅ |
| `vercel.json` is valid JSON with `builds` and `routes` | ✅ |
| `requirements.txt` contains flask, flask-cors, gunicorn | ✅ |
| `python -m pytest tests/ -v` passes all 55 tests | ✅ |
| `README.md` contains `## Web Application` section | ✅ |
| `outputs/reports/webapp_validation_report.md` exists | ✅ |
