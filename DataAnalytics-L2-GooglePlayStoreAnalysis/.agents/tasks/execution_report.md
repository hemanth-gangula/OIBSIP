# Google Play Store Analysis — Execution Report

**Date:** 2025  
**Script:** `.agents/tasks/run_analysis.py`  
**Status:** COMPLETED SUCCESSFULLY — zero errors

---

## 1. Execution Summary

The full analysis pipeline ran to completion without any errors. All output files were generated from the actual input datasets.

---

## 2. Input Datasets

| File | Rows | Columns |
|------|------|---------|
| `data/googleplaystore.csv` | 10,841 | 13 |
| `data/googleplaystore_user_reviews.csv` | 64,295 | 5 |

---

## 3. Cleaning Results

| Step | Result |
|------|--------|
| Duplicate app rows removed | 483 |
| Malformed category rows removed (Category starts with digit) | 1 |
| Apps after cleaning | 10,357 |
| Unique categories | 33 |
| Reviews after NaN + duplicate drop | 29,692 |

---

## 4. Key Statistics (from actual data)

| Metric | Value |
|--------|-------|
| Total cleaned apps | 10,357 |
| Total cleaned reviews | 29,692 |
| Unique app categories | 33 |
| Overall average rating | 4.19 / 5.0 |
| Free apps | 92.6% |
| Paid apps | 7.4% |

### Sentiment Distribution (TextBlob, thresholds ±0.05)

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 17,873 | 60.2% |
| Neutral | 6,443 | 21.7% |
| Negative | 5,376 | 18.1% |

### Size vs. Installs Correlation
- Raw correlation: 0.1689  
- Log10(Installs) correlation: 0.3336 (modest positive relationship)

---

## 5. Generated Output Files

### Cleaned Datasets — `outputs/cleaned_data/`

| File | Size |
|------|------|
| `cleaned_apps.csv` | 1,556,908 bytes (~1.5 MB) |
| `cleaned_reviews.csv` | 4,910,433 bytes (~4.7 MB) |
| `merged_data.csv` | 9,314,001 bytes (~8.9 MB) — shape: (49,692 × 20) |

### Figures — `outputs/figures/`

| File | Size | Description |
|------|------|-------------|
| `category_distribution.png` | 125,338 bytes | Horizontal bar chart — app count per category |
| `ratings_analysis.png` | 140,833 bytes | Rating histogram + avg rating by category |
| `size_vs_installs.png` | 408,996 bytes | Scatter plot coloured by rating, with trend line |
| `pricing_analysis.png` | 70,233 bytes | Pie chart (free/paid) + price histogram |
| `revenue_estimate.png` | 64,774 bytes | Top 10 categories by theoretical gross revenue |
| `sentiment_distribution.png` | 80,735 bytes | Bar chart + polarity histogram |
| `sentiment_by_category.png` | 131,722 bytes | Stacked % bar chart by category (≥50 reviews) |
| `interactive_category_landscape.html` | 4,885,059 bytes | Plotly bubble scatter: count vs rating vs installs |
| `interactive_top_categories.html` | 4,864,382 bytes | Plotly bar: top 20 categories coloured by avg rating |

**Total: 7 PNG figures + 2 interactive HTML files**

---

## 6. Errors Encountered

None. The script ran cleanly on the first attempt.

---

## 7. Sentiment Analysis Notes

- **Library**: TextBlob
- **Thresholds**: Positive > 0.05, Negative < -0.05, Neutral in between
- **Corpus**: punkt_tab, brown, wordnet, averaged_perceptron_tagger_eng (all pre-downloaded)
- **Input**: 29,692 deduplicated reviews with non-null `Translated_Review`
- **All reviews classified** — no Unknown entries in final output

---

## 8. Data-Driven Insights

1. **Market is dominated by free apps (92.6%)** — paid apps are a small niche; developers relying on paid downloads alone face a difficult market.
2. **Average rating is high (4.19)** — the distribution is left-skewed, meaning users tend to rate apps highly or not at all; ratings below 4.0 stand out negatively.
3. **60% of reviews are positive** — TextBlob sentiment confirms generally favourable user sentiment across the platform; categories with high negative sentiment are outliers worth investigating.
4. **Size has a modest positive correlation with installs (log r = 0.33)** — larger apps (games, productivity suites) tend to accumulate more downloads, but size alone is not a strong predictor.
5. **Family and Game categories have the most apps** — highly saturated; new entrants should consider differentiation or less competitive verticals.

---

## 9. Files Verified

All files confirmed to exist and contain data:

```
outputs/cleaned_data/cleaned_apps.csv       ✓ 10,357 rows
outputs/cleaned_data/cleaned_reviews.csv    ✓ 29,692 rows
outputs/cleaned_data/merged_data.csv        ✓ 49,692 rows
outputs/figures/category_distribution.png  ✓
outputs/figures/ratings_analysis.png       ✓
outputs/figures/size_vs_installs.png       ✓
outputs/figures/pricing_analysis.png       ✓
outputs/figures/revenue_estimate.png       ✓
outputs/figures/sentiment_distribution.png ✓
outputs/figures/sentiment_by_category.png  ✓
outputs/figures/interactive_category_landscape.html  ✓
outputs/figures/interactive_top_categories.html      ✓
```
