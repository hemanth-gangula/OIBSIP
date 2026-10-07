# Google Play Store Analytics
### Oasis Infobyte — Data Analytics Internship | Level 2 | Task 4

---

## Project Overview

**Title:** Unveiling the Android App Market — Google Play Store Analysis

**Objective:** Perform a comprehensive analysis of the Google Play Store ecosystem — cleaning messy real-world data, exploring app categories, analysing ratings and pricing trends, and conducting VADER-powered sentiment analysis on user reviews.

**Organisation:** Oasis Infobyte | Data Analytics Internship | Level 2 | Task 4

---

## Features

| Feature | Description |
|---------|-------------|
| **Two separate upload flows** | Apps dataset and Reviews dataset are loaded and analysed independently |
| **Data cleaning** | Fixes Installs (`'10,000+'`), Size (`'19M'`), Price (`'$2.99'`), Rating outliers, duplicates, nulls |
| **Category analysis** | Bar chart of app distribution; identifies most and least saturated categories |
| **Ratings analysis** | Distribution histogram + average rating per category |
| **Size vs Installs** | Scatter plot with Pearson correlation and plain-English interpretation |
| **Pricing analysis** | Free/Paid donut, price buckets, estimated revenue by category |
| **VADER sentiment** | Every review is classified fresh (Positive / Negative / Neutral) — original column used only for validation |
| **Sentiment by category** | App → Category mapping links Reviews to Apps dataset when both are uploaded |
| **Interactive Plotly charts** | One Plotly chart per dashboard — zoom, hover, filter |
| **Developer insights** | Five data-driven insights generated from actual data |
| **Jupyter Notebook** | Full reproducible analysis document |
| **Vercel deployment** | Production-ready Flask app |

---

## Tech Stack

- **Python 3.11**
- **Flask 3.0** — web framework
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — static charts
- **Plotly** — interactive visualisations
- **VADER (vaderSentiment)** — NLP sentiment analysis
- **Jupyter Notebook** — analysis documentation

---

## Dataset Sources

Both datasets are publicly available on Kaggle:

| Dataset | Kaggle Link |
|---------|------------|
| Google Play Store Apps | https://www.kaggle.com/datasets/lava18/google-play-store-apps |
| Google Play Store User Reviews | https://www.kaggle.com/datasets/lava18/google-play-store-apps |

> **Raw datasets are intentionally excluded from this repository.**  
> Download them from the Kaggle links above and upload via the application interface.

Expected filenames:
- `googleplaystore.csv`
- `googleplaystore_user_reviews.csv`

---

## Dashboard Descriptions

### Apps Dashboard
Triggered by uploading `googleplaystore.csv`. Shows:
- KPI cards (total apps, categories, average rating, free/paid split)
- Data cleaning summary (original rows, duplicates removed, dtype corrections, null counts)
- Category distribution bar chart
- Interactive Plotly category explorer
- Rating distribution + average rating by category
- Size vs Installs scatter plot with Pearson correlation
- Free/Paid donut chart
- Paid app price distribution
- Estimated revenue by category (with disclaimer)
- Five developer insights

### Reviews Dashboard
Triggered by uploading `googleplaystore_user_reviews.csv`. Shows:
- KPI cards (total reviews, positive/negative/neutral counts and percentages, average compound score)
- Data cleaning summary
- VADER methodology note
- Sentiment donut + bar chart
- VADER compound score histogram
- Sentiment by category (only when Apps dataset is also loaded)
- Interactive Plotly sentiment chart
- VADER vs original label agreement (validation check)
- Three sentiment insights

---

## Sentiment Methodology

**Library:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

VADER is a rule-based NLP model optimised for short social-media texts. Each review text is passed through the `SentimentIntensityAnalyzer` to produce a compound score (−1 to +1):

| Score | Label |
|-------|-------|
| ≥ 0.05 | Positive |
| ≤ −0.05 | Negative |
| −0.05 to 0.05 | Neutral |

The existing `Sentiment` column in the Kaggle dataset is used **only** as a validation reference — all dashboard classifications are produced fresh by VADER.

---

## Data Cleaning Methodology

### Apps Dataset
| Column | Issue | Fix |
|--------|-------|-----|
| `Installs` | `'10,000+'` string | Strip `,` and `+`, cast to float |
| `Size` | `'19M'`, `'512k'` string | Convert MB/KB to float (MB) |
| `Price` | `'$2.99'` string | Strip `$`, cast to float |
| `Rating` | Non-numeric, out-of-range | Coerce to numeric, set invalid to NaN |
| `Reviews` | String | Cast to integer |
| Duplicates | Exact row duplicates + multi-row same App | Drop exact dupes; keep highest-reviews entry per App |
| Nulls | Missing App or Category | Dropped |

### Reviews Dataset
| Step | Action |
|------|--------|
| Exact duplicates | Dropped |
| Null review text | Dropped |
| Empty strings | Dropped |
| Null App name | Dropped |

---

## Revenue Estimate Disclaimer

Revenue estimates shown in the Apps dashboard are **indicative only**.

Formula: `Price_USD × Installs`

The `Installs` column in the Kaggle dataset stores the lower bound of a reported range (e.g. `'10,000+'` → `10,000`). Actual install counts are likely significantly higher. This figure does not represent real company revenue and should not be used for financial decisions.

---

## Visualisations

| Chart | Type | Dashboard |
|-------|------|-----------|
| App distribution by category | Horizontal bar (seaborn) | Apps |
| Interactive category explorer | Plotly bar | Apps |
| Rating distribution | Histogram (matplotlib) | Apps |
| Average rating by category | Horizontal bar (seaborn) | Apps |
| Size vs Installs | Scatter (matplotlib) | Apps |
| Free vs Paid | Donut (matplotlib) | Apps |
| Price distribution | Bar (seaborn) | Apps |
| Revenue by category | Horizontal bar (seaborn) | Apps |
| Sentiment distribution | Donut (matplotlib) | Reviews |
| Sentiment counts | Bar (matplotlib) | Reviews |
| Compound score histogram | Histogram (matplotlib) | Reviews |
| Sentiment by category | Stacked horizontal bar (seaborn) | Reviews |
| Interactive sentiment | Plotly stacked bar / pie | Reviews |

---

## Jupyter Notebook

**File:** `Google_Play_Store_Analysis.ipynb`

Sections:
1. Project introduction
2. Import libraries
3. Load Apps dataset
4. Load User Reviews dataset
5. Inspect datasets
6. Data cleaning — Apps
7. Category analysis
8. Ratings analysis
9. Size vs Installs
10. Pricing analysis
11. Revenue estimation
12. Data cleaning — Reviews
13. VADER sentiment analysis
14. Sentiment by category
15. Interactive Plotly visualisations
16. Developer insights
17. Conclusion

---

## Local Setup

### Requirements
- Python 3.10+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/<your-username>/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-GooglePlayStoreAnalysis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### Jupyter Notebook

```bash
jupyter notebook Google_Play_Store_Analysis.ipynb
```

Place `googleplaystore.csv` and `googleplaystore_user_reviews.csv` in the same directory before running.

---

## Project Structure

```
DataAnalytics-L2-GooglePlayStoreAnalysis/
│
├── app.py                          # Flask application
├── data_processor.py               # Apps dataset loading, cleaning, analysis
├── sentiment_analyzer.py           # VADER sentiment analysis
├── visualizations.py               # matplotlib/seaborn/Plotly chart generation
├── requirements.txt                # Pinned Python dependencies
├── vercel.json                     # Vercel deployment config
├── README.md
├── Google_Play_Store_Analysis.ipynb
│
├── templates/
│   ├── index.html                  # Landing page (two separate upload buttons)
│   ├── apps_dashboard.html         # Apps dataset dashboard
│   └── reviews_dashboard.html      # Reviews dataset dashboard
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── upload.js
│
└── uploads/                        # Temporary runtime only — gitignored
```

---

## GitHub Repository

**Repository:** `OIBSIP`

**Path:** `DataAnalytics-L2-GooglePlayStoreAnalysis/`

> This project does not modify any other task in the repository.

---

## Deployment

### Vercel

This app is configured for Vercel serverless deployment via `vercel.json`.

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Live Demo:** *(add URL after deployment)*

### Vercel Limitations

- Serverless functions have execution time limits (~10 s for the free tier). For large datasets, consider pre-processing or chunked uploads.
- The `/tmp` directory is used for uploads in serverless environments. Persistent storage is not available between requests.
- Large matplotlib figures may approach memory limits on the free tier.

---

## Limitations

- Revenue estimates use lower-bound install counts and are indicative only.
- Sentiment analysis accuracy depends on review text quality; short or ambiguous reviews may be classified as Neutral.
- Category sentiment requires both datasets to be uploaded in the same server session.
- The Kaggle dataset was collected in 2018 — current Play Store data will differ.

---

*Built for the Oasis Infobyte Data Analytics Internship — Level 2 — Task 4.*
