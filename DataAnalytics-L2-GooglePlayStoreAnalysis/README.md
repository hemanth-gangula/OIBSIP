# 📊 Unveiling the Android App Market – Google Play Store Analysis

### Oasis Infobyte – Data Analytics Internship | Level 2 – Task 4

---

## 🌐 Live Demo

> **Try the deployed application right now — no installation required.**

**🔗 [https://google-play-store-analytics-amber.vercel.app/](https://google-play-store-analytics-amber.vercel.app/)**

Upload your datasets, run the full analysis, and explore the interactive dashboard live.

---

## 🎯 Objective

Perform a comprehensive exploratory data analysis of the real Google Play Store ecosystem. The application processes the actual Kaggle datasets uploaded by the user at runtime — no data is hardcoded, fabricated, or stored inside this repository.

---

## ⚡ Quick Start for Evaluators

> The fastest way to see the full analysis — no local setup required.

1. **Download** both datasets from the [Dataset Sources](#-dataset-sources) section below.
2. **Open** the Live Demo: [https://google-play-store-analytics-amber.vercel.app/](https://google-play-store-analytics-amber.vercel.app/)
3. **Upload** `googleplaystore.csv` in the **Apps Dataset** upload box.
4. **Upload** `googleplaystore_user_reviews.csv` in the **User Reviews Dataset** upload box.
5. **Click** ⚡ **Run Full Analysis**.
6. **Explore** the complete dashboard — charts, KPIs, sentiment analysis, and developer insights all generated from your data.

---

## 📦 Dataset Sources

> **The raw datasets are intentionally excluded from this repository to keep it lightweight and to follow good data-management practices. Download links are provided below.**

Both datasets are from the publicly available **Google Play Store Apps** collection on Kaggle, referenced by the Oasis Infobyte Task 4 requirements.

| Dataset | File | Download |
|---|---|---|
| Google Play Store Apps | `googleplaystore.csv` | [Kaggle – Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps) |
| Google Play Store User Reviews | `googleplaystore_user_reviews.csv` | [Kaggle – Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps) |

**Download steps:**
1. Visit the Kaggle link above (free account required).
2. Click **Download** to get the archive — both CSV files are included.
3. Extract `googleplaystore.csv` and `googleplaystore_user_reviews.csv`.
4. Upload them through the application's upload interface (Live Demo or local).

> The application accepts `.csv`, `.xlsx`, and `.xls` formats. No manual file placement needed.

---

## 📁 Project Structure

```
DataAnalytics-L2-GooglePlayStoreAnalysis/
│
├── app.py                          ← Flask backend (routes, upload, analysis)
├── data_processor.py               ← Data cleaning & all analysis logic
├── sentiment_analyzer.py           ← VADER + TextBlob NLP sentiment engine
├── visualizations.py               ← Matplotlib/Seaborn + 4 Plotly charts
├── requirements.txt                ← Python dependencies
├── vercel.json                     ← Vercel deployment configuration
├── README.md                       ← This file
│
├── Google_Play_Store_Analysis.ipynb  ← Standalone Jupyter Notebook (full analysis)
│
├── templates/
│   ├── index.html                  ← Upload interface
│   └── dashboard.html              ← Full analytics dashboard
│
└── static/
    ├── css/style.css               ← Professional dark-theme stylesheet
    └── js/upload.js                ← Drag-drop + chunked upload handler
```

> 📓 **The Jupyter Notebook** (`Google_Play_Store_Analysis.ipynb`) provides a fully standalone, reproducible version of the entire Task 4 analysis — separate from the web application.

---

## ✨ Features & Analysis Sections

| Feature | Description |
|---|---|
| **Dual Dataset Upload** | Separate upload areas for Apps and User Reviews datasets |
| **CSV & Excel Support** | `.csv`, `.xlsx`, `.xls` with auto-encoding detection |
| **Dataset Validation** | Column-level validation with clear error messages |
| **Data Cleaning Pipeline** | Installs, Price, Size, Rating, Type normalisation |
| **Category Analysis** | App distribution, bar chart, saturated category identification |
| **Ratings Analysis** | Distribution histogram, average by category |
| **Size & Installs Analysis** | Scatter plot, Pearson correlation, trend line |
| **Pricing Analysis** | Free vs Paid donut, price distribution, estimated revenue |
| **Sentiment Analysis** | VADER/TextBlob NLP — Positive / Negative / Neutral |
| **Sentiment by Category** | Merged dataset analysis, most positive/negative categories |
| **Interactive Plotly Charts** | Bubble, violin, installs bar, sunburst — 4 interactive charts |
| **Developer Insights** | 5 data-driven insights computed from the uploaded data |
| **Dynamic KPIs** | All KPI values computed at runtime — never hardcoded |
| **Jupyter Notebook** | Full standalone analysis notebook included |

---

## 🧹 Data Cleaning Methodology

| Column | Raw Issue | Fix Applied |
|---|---|---|
| `Installs` | `"10,000+"` string | Strip `,` and `+`, cast to `int` |
| `Price` | `"$4.99"` string | Strip `$`, cast to `float` |
| `Size` | `"19M"`, `"1.5k"` string | Parse unit, convert to `float` MB |
| `Rating` | Values outside 1–5 range | Set to `NaN` |
| `Reviews` | Mixed types | Cast to `numeric` |
| `Type` | Unexpected values (e.g. `"0"`) | Normalise to `Free` / `Paid` only |
| `Category` | Whitespace, mixed case | Strip + uppercase |
| Duplicate rows | Same `App` name repeated | Keep first occurrence |
| Null reviews | `NaN` review text | Drop rows |
| Sentiment labels | Non-standard values | Normalise to `Positive` / `Negative` / `Neutral` |

---

## 📂 Category Analysis

- Counts apps per category across the full dataset
- Horizontal bar chart — top 20 categories by app count
- Identifies the **5 most saturated categories** by competition volume

---

## ⭐ Ratings Analysis

- Overall average rating across all rated apps
- Distribution histogram with mean marker line
- Average rating per category (horizontal bar chart, top 20)
- Rating bucket breakdown: 1–2 · 2–3 · 3–4 · 4–4.5 · 4.5–5

---

## 📦 Size & Installs Analysis

- Scatter plot: App Size (MB) vs Install Count
- Pearson correlation coefficient with interpretation
- Log-scale trend line overlay
- Based on all apps with valid size and install data

---

## 💰 Pricing Analysis

- Free vs Paid donut chart with counts and percentages
- Paid app price distribution (bucketed by price range)
- Estimated revenue by category (top 10 paid categories)

---

## 📈 Revenue Estimation Methodology

```
Estimated Revenue = App Price (USD) × Install Count
```

**Assumptions & disclaimer:**
- Each install is treated as one purchase (upper-bound estimate)
- This is **NOT** actual Google Play Store revenue
- Ignores refunds, regional pricing, free trials, and in-app purchases

---

## 😊 Sentiment Analysis

- Primary engine: **VADER** (`vaderSentiment`) — fast, no inference needed
- Fallback: **TextBlob** if VADER is unavailable
- If the dataset already contains valid labels (≥ 80% coverage), they are used directly
- Classification thresholds: compound ≥ 0.05 = Positive · ≤ −0.05 = Negative
- Reports count, percentage, and average polarity score per class

---

## 📊 Sentiment by Category

- Merges Apps and User Reviews datasets on the `App` column
- Computes Positive / Negative / Neutral review counts per category
- Identifies the **top 5 most positive** and **top 5 most negative** categories
- Stacked horizontal bar chart (top 15 categories by review volume)

---

## 🔬 Interactive Visualisations (Plotly)

| Chart | Description | Analytical Value |
|---|---|---|
| Category Bubble | App count vs avg rating; bubble = median installs | Spot low-competition, high-satisfaction gaps |
| Rating Violin | Free vs Paid rating distribution with IQR box | Pricing strategy signal |
| Top Installs Bar | Total installs by category (top 10) | Market size overview |
| Sunburst | Type → Category → Rating bucket | Full landscape drill-down |

---

## 💡 Developer Insights

Five data-driven insights are generated dynamically from the uploaded data:

1. **Best Category to Enter** — balances average rating against competition count
2. **Free vs Paid Monetisation Strategy** — based on the actual market split
3. **Optimal App Size for Maximum Installs** — median size of top-quartile install apps
4. **Rating Benchmark to Beat** — launch target above market average
5. **Highest User Satisfaction Category** — highest positive review percentage

All insights are computed at runtime. No generic or hardcoded conclusions.

---

## 🛠 Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Flask 3.0 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Visualisation | Matplotlib 3.9, Seaborn 0.13, Plotly 5.22 |
| NLP / Sentiment | VADER (vaderSentiment 3.3), TextBlob 0.18 |
| Excel Support | openpyxl 3.1, xlrd 2.0 |
| Frontend | HTML5, CSS3, Vanilla JS (all inlined — no external dependencies) |
| Interactive Charts | Plotly.js (CDN) |
| Notebook | Jupyter Notebook |
| Deployment | Vercel (serverless Python) |

---

## 📤 Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | UTF-8, Latin-1, CP1252 auto-detected |
| Excel (modern) | `.xlsx` | openpyxl engine |
| Excel (legacy) | `.xls` | xlrd engine |

---

## ⚙️ Local Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hemanth-gangula/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-GooglePlayStoreAnalysis

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run Locally

```bash
python app.py
```

Open your browser at:

```
http://localhost:5000
```

Then upload both datasets and click **⚡ Run Full Analysis**.

---

## 📓 Jupyter Notebook

The notebook `Google_Play_Store_Analysis.ipynb` provides a fully standalone, reproducible version of the Task 4 analysis — independent of the web application.

**To run:**
```bash
jupyter notebook Google_Play_Store_Analysis.ipynb
```

Update the `APPS_PATH` and `REVIEWS_PATH` variables in Cell 2 to point to your downloaded dataset files.

---

## ⚠️ Important Note on Datasets

The raw datasets are **intentionally excluded** from this repository to keep the repository lightweight and to follow good data-management practices. Dataset source and download links are provided in the [Dataset Sources](#-dataset-sources) section above.

The application never hardcodes, fabricates, or ships data. Every analysis result is produced at runtime from the files you upload.

---

## 📸 Screenshots

> After running the analysis, add screenshots here:
> - Upload interface with file preview
> - Overview KPI dashboard
> - Category distribution chart
> - Ratings analysis
> - Size vs Installs scatter plot
> - Sentiment analysis (counts + donut)
> - Sentiment by category (stacked bar)
> - Interactive Plotly charts
> - Developer insights

---

## 📜 License

This project is created for educational purposes as part of the **Oasis Infobyte Data Analytics Internship**.
