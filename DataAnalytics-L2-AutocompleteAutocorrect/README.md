# Autocomplete and Autocorrect — NLP Data Analytics

> A Python NLP project implementing autocomplete (bigram + trigram language models) and autocorrect (edit-distance + probabilistic spelling correction), with interactive Plotly visualisations and full evaluation metrics.

Part of the **OIBSIP** (Oasis Infobyte Internship Program) Data Analytics Level 2 portfolio.

---

## Web Application

A Flask web application wraps all NLP models with a professional, responsive interface served on Vercel.

### Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/autocomplete` | Next-word suggestions from bigram/trigram models |
| POST | `/api/autocorrect` | Spelling correction from edit-distance/probabilistic models |
| GET | `/api/metrics` | Previously reported baseline evaluation results |
| GET | `/api/analytics` | Corpus vocabulary statistics and top-20 words |
| GET | `/api/download/<filename>` | Download prediction CSV outputs |
| GET | `/health` | Health check; returns `{"status":"ok"}` |

### Deploy to Vercel

```bash
vercel login
vercel --prod
```

> **Note:** `data/processed/model_data.json` is included in the repository (committed via `.gitignore` exception). If it is missing, regenerate it by running the notebook end-to-end before deploying.

---

## Overview

This project builds two related Natural Language Processing systems:

1. **Autocomplete** — predicts the most likely next word(s) given a text prefix, using bigram and trigram n-gram language models trained on a large English corpus.
2. **Autocorrect** — detects and corrects misspelled words using edit-distance candidate generation ranked by two distinct strategies: edit-distance ordering and Norvig-style probability.

The project includes exploratory data analysis, interactive Plotly charts, model evaluation with standard NLP metrics, and a full test suite.

---

## Objectives

- Build and compare bigram and trigram autocomplete models.
- Implement and evaluate two autocorrect strategies: edit-distance ranking and probabilistic (Norvig-style) ranking.
- Produce interactive visualisations of corpus statistics and model performance.
- Evaluate models honestly using held-out test cases and standard metrics (Top-1 accuracy, Top-3 accuracy, MRR).

---

## Datasets

| File | Description | Source |
|------|-------------|--------|
| `data/big.txt` | Large English running-text corpus (~6.5 MB). Sourced from Peter Norvig's spelling-correction article resources. Contains public-domain texts including Project Gutenberg works. | https://www.norvig.com/big.txt |
| `data/spell-errors.txt` | Collection of documented spelling errors (~450 KB). Format: `correct_word: misp1, misp2*N, ...`. Sourced from Peter Norvig's n-grams page. | https://www.norvig.com/ngrams/ |

> These datasets are publicly available and were selected for this implementation. They are not proprietary or officially mandated by any internship provider.

---

## Project Structure

```
DataAnalytics-L2-AutocompleteAutocorrect/
├── data/
│   ├── big.txt                    # Peter Norvig corpus (public domain)
│   ├── spell-errors.txt           # Norvig spelling errors collection
│   └── processed/                 # Derived vocabulary CSV
├── notebooks/
│   └── Autocomplete_Autocorrect_Analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── text_prediction.py         # Corpus loading, tokenization, n-grams
│   ├── autocomplete.py            # BigramModel, TrigramModel
│   ├── autocorrect.py             # EditDistanceCorrector, ProbabilisticCorrector
│   └── evaluation.py              # Metric functions, CSV export
├── outputs/
│   ├── figures/                   # Plotly HTML charts (7 files)
│   ├── predictions/               # autocomplete_predictions.csv, autocorrect_results.csv
│   └── reports/                   # validation_report.md, model_comparison_summary.csv
├── tests/
│   ├── test_autocomplete.py
│   ├── test_autocorrect.py
│   └── test_evaluation.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `plotly`, `notebook`, `jupyterlab`, `kaleido`, `pytest`, `ipywidgets`.

---

## Running the Notebook

```bash
jupyter notebook notebooks/Autocomplete_Autocorrect_Analysis.ipynb
```

Or to execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Autocomplete_Autocorrect_Analysis.ipynb
```

---

## Running the Tests

```bash
python -m pytest tests/ -q
```

---

## Usage Examples

### Autocomplete

```python
import sys
sys.path.insert(0, 'src')
from text_prediction import load_corpus, tokenize, build_vocab
from autocomplete import BigramModel, TrigramModel, get_autocomplete_suggestions

corpus = load_corpus('data/big.txt')
tokens = tokenize(corpus)

bigram = BigramModel().fit(tokens)
trigram = TrigramModel().fit(tokens)

suggestions = get_autocomplete_suggestions('machine learning', bigram, trigram, top_k=3)
print(suggestions['bigram'])   # [(word, score), ...]
print(suggestions['trigram'])  # [(word, score), ...]
```

### Autocorrect

```python
from text_prediction import build_vocab
from autocorrect import EditDistanceCorrector, ProbabilisticCorrector

vocab = build_vocab(tokens)

ed = EditDistanceCorrector(vocab)
print(ed.correct('speling'))   # [(candidate, edit_dist, freq), ...]

prob = ProbabilisticCorrector(vocab)
print(prob.correct('recieve')) # [(candidate, probability), ...]
```

---

## Algorithms

### Autocomplete

| Model | Context | Formula |
|-------|---------|-------|
| **Bigram** | Previous 1 word | P(w₂\|w₁) = count(w₁, w₂) / count(w₁) |
| **Trigram** | Previous 2 words | P(w₃\|w₁,w₂) = count(w₁,w₂,w₃) / count(w₁,w₂). Falls back to bigram if context unseen. |

### Autocorrect

| Approach | Strategy |
|----------|----------|
| **EditDistanceCorrector** | Generate candidates within edit distance 1 or 2; rank by edit distance ascending, then corpus frequency descending. |
| **ProbabilisticCorrector** | Norvig-style: generate edit-1 candidates; if none, edit-2. Rank by P(w) = freq(w) / total_words. Strictly prefers edit-1 over edit-2. |

**Difference:** EditDistanceCorrector mixes edit-1 and edit-2 candidates, sorted by distance first. ProbabilisticCorrector strictly prefers edit-1 over edit-2 and ranks by normalised probability (interpretable as a likelihood), not raw frequency.

---

## Evaluation Methods

### Autocomplete
- **Top-1 Accuracy**: fraction of test contexts where the #1 prediction matches the actual next word.
- **Top-3 Accuracy**: fraction where the actual next word appears in the top 3 predictions.
- **MRR (Mean Reciprocal Rank)**: average of 1/rank when the correct word is found.
- **Test set**: held-out last 10% of corpus tokens, evaluated on the most frequent bigrams/trigrams.

### Autocorrect
- **Top-1 Accuracy**: fraction of test cases where the #1 correction matches the expected word.
- **Top-k Accuracy (k=3)**: fraction where the expected word is in the top-3 corrections.
- **Test set**: pairs from `spell-errors.txt` where the correct word appears in the corpus vocabulary.

---

## Results

> See the executed notebook and generated CSVs for actual computed scores.

- Autocomplete evaluation: `outputs/predictions/autocomplete_predictions.csv`
- Autocorrect evaluation: `outputs/predictions/autocorrect_results.csv`
- Model comparison summary: `outputs/reports/model_comparison_summary.csv`
- Validation report: `outputs/reports/validation_report.md`

---

## Interactive Charts

All charts are saved as interactive HTML files (open in any browser):

| Chart | File |
|-------|------|
| Top 20 Most Frequent Words | [outputs/figures/top_20_words.html](outputs/figures/top_20_words.html) |
| Word Frequency Distribution (Zipf's Law) | [outputs/figures/word_frequency_distribution.html](outputs/figures/word_frequency_distribution.html) |
| Autocomplete Model Comparison | [outputs/figures/autocomplete_model_comparison.html](outputs/figures/autocomplete_model_comparison.html) |
| Autocomplete Prediction Scores | [outputs/figures/autocomplete_prediction_scores.html](outputs/figures/autocomplete_prediction_scores.html) |
| Autocorrect Performance Comparison | [outputs/figures/autocorrect_performance_comparison.html](outputs/figures/autocorrect_performance_comparison.html) |
| Correction Examples | [outputs/figures/correction_examples.html](outputs/figures/correction_examples.html) |
| Edit Distance Distribution | [outputs/figures/edit_distance_heatmap.html](outputs/figures/edit_distance_heatmap.html) |

---

## Key Findings

1. **Zipf's Law holds in natural language corpora.** The top ~100 words (function words such as "the", "of", "and") account for a disproportionately large share of all tokens. The rank–frequency relationship follows a power-law distribution, as visible in the log-log word frequency chart.

2. **Trigram models provide more specific — but sparser — predictions than bigrams.** Trigrams condition on two words rather than one, yielding higher-confidence predictions where training data is available. However, many trigram contexts are unseen even in a large corpus, triggering bigram fallback. Whether trigrams outperform bigrams on a given test set depends on the overlap between test contexts and training n-grams.

3. **Edit-distance-1 corrections cover most common misspellings.** The majority of misspellings in the Norvig dataset are within one edit (insertion, deletion, substitution, or transposition) of the correct word. Both correction approaches achieve their highest accuracy on these near-miss misspellings; performance drops for more severe errors.

---

## Limitations

- N-gram models do not model long-range dependencies; a neural language model (LSTM, Transformer) would better capture sentence-level context.
- Autocomplete quality depends heavily on the corpus domain: models trained on literary text may perform poorly on technical or conversational text.
- The autocorrect vocabulary is limited to words in `big.txt`; proper nouns, technical terms, and rare words will not be suggested.
- Evaluation is performed on a subset of the available data; real-world performance may differ.
- No smoothing (e.g. Laplace, Kneser-Ney) is applied to n-gram probabilities, so unseen contexts get zero probability.

---

## Future Improvements

- Add Laplace or Kneser-Ney smoothing to handle unseen n-grams more gracefully.
- Implement a character-level or subword model for autocorrect to handle out-of-vocabulary words.
- Train and compare a neural language model (e.g. a simple LSTM) against the n-gram baselines.
- Build a simple web interface (Streamlit or Gradio) for interactive demos.
- Extend the evaluation to a larger, domain-specific test set.

---

## Reproducibility

All results are produced by executing the notebook end-to-end from the original data files. No numbers in this README are hardcoded; all metrics are computed at runtime. To reproduce:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/Autocomplete_Autocorrect_Analysis.ipynb
```

---

## License

MIT License — see [LICENSE](LICENSE).
