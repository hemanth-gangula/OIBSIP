# Unveiling the Android App Market: Google Play Store Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![OASIS Infobyte](https://img.shields.io/badge/OASIS%20Infobyte-Level%202-green.svg)](https://oasisinfobyte.com/)

## 📋 Project Overview

**OASIS Infobyte Data Analytics Internship – Level 2**

A comprehensive analysis of the Google Play Store ecosystem using real-world app and review data. The project cleans messy raw data, performs exploratory analysis across 33 app categories, runs TextBlob sentiment analysis on ~30,000 user reviews, and produces actionable insights for developers planning to launch an Android application.

---

## 🎯 Objective

- Clean and validate real-world messy app data.
- Explore category saturation, ratings, app size, installs, and pricing.
- Classify user review sentiment (Positive / Neutral / Negative).
- Derive data-driven insights and business recommendations.

---

## 📊 Dataset

| File | Rows (raw) | Description |
|---|---|---|
| `googleplaystore.csv` | 10,841 | App attributes: category, rating, installs, price, size, etc. |
| `googleplaystore_user_reviews.csv` | 64,295 | User review text with pre-labelled sentiment scores |

**Source:** [Kaggle – Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)

After cleaning: **10,357 apps** and **29,692 reviews**.

---

## 🛠️ Technology Stack

| Library | Purpose |
|---|---|
| pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations |
| Matplotlib | Static charts |
| Seaborn | Statistical visualisations |
| Plotly | Interactive HTML charts |
| TextBlob | Sentiment analysis (NLP) |
| Jupyter Notebook | Interactive analysis environment |

---

## 📁 Folder Structure

```
DataAnalytics-L2-GooglePlayStoreAnalysis/
├── data/
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── notebooks/
│   └── Google_Play_Store_Analysis.ipynb   ← 15-section analysis notebook
├── outputs/
│   ├── cleaned_data/
│   │   ├── cleaned_apps.csv               ← 10,357 rows
│   │   ├── cleaned_reviews.csv            ← 29,692 rows
│   │   └── merged_data.csv                ← reviews joined to categories
│   └── figures/
│       ├── category_distribution.png
│       ├── rating_distribution.png
│       ├── average_rating_by_category.png
│       ├── size_vs_installs.png
│       ├── free_vs_paid.png
│       ├── paid_app_price_distribution.png
│       ├── sentiment_distribution.png
│       ├── sentiment_by_category.png
│       └── interactive_ratings_by_category.html
├── run_analysis.py         ← standalone execution script
├── build_notebook.py       ← notebook builder with embedded outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

```bash
# 1. Navigate to the project folder
cd DataAnalytics-L2-GooglePlayStoreAnalysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TextBlob corpora (first time only)
python -m textblob.download_corpora
```

---

## 💻 How to Run

### Option A – Jupyter Notebook (interactive)
```bash
jupyter notebook notebooks/Google_Play_Store_Analysis.ipynb
```
Run all cells from top to bottom. Output files are saved automatically.

### Option B – Standalone script (non-interactive)
```bash
python run_analysis.py
```
Runs the complete analysis and saves all outputs to `outputs/`.

---

## 🧹 Data Cleaning Methodology

| Step | Action | Rows Affected |
|---|---|---|
| Remove duplicate apps | `drop_duplicates()` | −483 |
| Drop malformed row (`Category = '1.9'`) | Filter | −1 |
| Rating: convert to numeric, invalidate outside [1, 5] | `pd.to_numeric` + range check | 1,474 set to NaN |
| Reviews: convert to numeric | `pd.to_numeric` | — |
| Size: parse `19M` → 19.0, `500k` → 0.49, `Varies` → NaN | Custom `parse_size()` | 1,526 → NaN |
| Installs: strip `+` and `,`, convert to int | `str.replace` + `pd.to_numeric` | — |
| Price: strip `$`, convert to float | `str.replace` + `pd.to_numeric` | — |
| Type: normalise to Free / Paid | String strip + filter | 1 → NaN |
| Last Updated: parse to `datetime` | `pd.to_datetime` | — |
| Remove duplicate reviews | `drop_duplicates()` | −33,616 |
| Remove reviews with missing text | `dropna()` | −26,868 |

**Final cleaned datasets:** 10,357 apps · 29,692 reviews.

---

## 📈 Analysis Summary (15 Sections)

| # | Section | Highlights |
|---|---|---|
| 1 | Project Overview | Objective, business problem, tools |
| 2 | Dataset Description | Column definitions, expected types |
| 3 | Import Libraries & Load Data | Robust path handling, error messages |
| 4 | Initial Inspection | Shape, dtypes, missing values, duplicates |
| 5 | Data Cleaning | All transformations documented |
| 6 | Category Analysis | 33 categories; FAMILY leads with 1,943 apps |
| 7 | Ratings Analysis | Mean 4.19; EVENTS highest, DATING lowest |
| 8 | Size vs Installs | Pearson r = 0.334 (log scale); weak link |
| 9 | Pricing Analysis | 92.6 % free; paid median $2.99 |
| 10 | Sentiment Analysis | 60.2 % Positive, 18.1 % Negative (TextBlob) |
| 11 | Sentiment by Category | COMICS 84 % positive; SOCIAL 49 % non-positive |
| 12 | Interactive Plotly Chart | Bubble chart: category landscape |
| 13 | Key Insights | 5 data-driven findings |
| 14 | Business Recommendations | 6 actionable strategies |
| 15 | Conclusion & Limitations | Data quality, temporal, and model constraints |

---

## 🔍 Key Findings (from actual data)

### 1. Market Saturation
FAMILY (1,943), GAME (1,121), and TOOLS (843) account for **37.7 %** of all apps.
New entrants face intense competition in these categories.
Niche categories (EVENTS, BEAUTY, COMICS) offer lower entry barriers.

### 2. Ratings Benchmark
- Overall mean rating: **4.19** (median 4.30) — strongly left-skewed.
- Highest-rated category: **EVENTS** (avg 4.44, n=45).
- Lowest-rated category: **DATING** (avg 3.97, n=159).
- 14.1 % of apps have no rating at all.

### 3. Free App Dominance
- **92.6 %** of apps are free.
- Paid app median price: **$2.99** · mean: $13.96 (outliers inflate mean).
- FAMILY generates the highest theoretical gross estimate (Price × Installs ≈ $185 M).
  > ⚠️ This is a rough upper bound only — it does not reflect actual revenue.
  > It excludes refunds, Google's 30 % platform fee, taxes, and the fact that installs ≠ purchases.

### 4. App Size vs Installs
- Pearson r (Size_MB vs log₁₀ Installs) = **0.334** — weak positive association.
- App size is not the primary driver of download success.

### 5. User Sentiment
- **60.2 %** Positive · **21.7 %** Neutral · **18.1 %** Negative (TextBlob, ±0.05 threshold).
- Most positive category: **COMICS** (84.4 %).
- Most-negative-leaning categories: **SOCIAL** and **GAME** (~49 % non-positive).

---

## 💡 Business Recommendations

1. **Avoid saturated categories.** Target EVENTS, EDUCATION, or niche verticals with fewer competitors and strong ratings.
2. **Target a 4.2+ rating.** Apps below 4.0 risk poor store visibility. Invest in UX before launch.
3. **Default to free with monetisation.** 92.6 % of apps are free; use in-app purchases or ads.
4. **App size matters less than quality.** Weak size–install correlation means feature completeness outweighs size optimisation.
5. **Enter HEALTH_AND_FITNESS or EDUCATION.** Both show >73 % positive sentiment and large user bases.
6. **Differentiate in SOCIAL and GAME.** ~49 % non-positive sentiment signals unmet user needs — an opportunity for quality entrants.

---

## ⚠️ Limitations

| Limitation | Impact |
|---|---|
| Installs ≠ purchases | Revenue estimates are theoretical only |
| Dataset is a snapshot | Current market dynamics may differ |
| 41.8 % of review text is missing | Sentiment analysis covers ~46 % of raw reviews |
| TextBlob lexicon-based | May misclassify sarcasm and domain jargon |
| 14.1 % apps lack ratings | Category averages exclude unrated apps |
| Unequal category sizes | Comparisons between small and large categories may mislead |

---

## 📂 Generated Outputs

### `outputs/cleaned_data/`
| File | Rows | Size |
|---|---|---|
| `cleaned_apps.csv` | 10,357 | ~1.5 MB |
| `cleaned_reviews.csv` | 29,692 | ~4.8 MB |
| `merged_data.csv` | 28,250 | ~5.5 MB |

### `outputs/figures/`
| File | Description |
|---|---|
| `category_distribution.png` | Horizontal bar — app count per category |
| `rating_distribution.png` | Histogram + category avg ratings |
| `average_rating_by_category.png` | Sorted horizontal bar |
| `size_vs_installs.png` | Scatter (colour = rating, trend line) |
| `free_vs_paid.png` | Pie + paid price histogram |
| `paid_app_price_distribution.png` | Bar chart — free vs paid counts |
| `sentiment_distribution.png` | Bar + pie of TextBlob sentiment |
| `sentiment_by_category.png` | Stacked horizontal bar |
| `interactive_ratings_by_category.html` | Plotly bubble chart (open in browser) |

---

## 🙏 Acknowledgements

- **OASIS Infobyte** for the internship programme.
- **Kaggle** and dataset contributors for the Google Play Store data.
- Open-source Python data science community.

---

*Last updated: October 2026*
