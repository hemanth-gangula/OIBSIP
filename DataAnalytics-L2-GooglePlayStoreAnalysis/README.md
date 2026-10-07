# 📊 Google Play Store Analytics
### Oasis Infobyte – Data Analytics Internship | Level 2 – Task 4
**"Unveiling the Android App Market"**

---

## 🎯 Objective

Analyse the real Google Play Store ecosystem by uploading the actual datasets through a professional web interface. The application dynamically performs all required Task 4 analyses without any hardcoded or fabricated data.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Dual Dataset Upload** | Separate upload areas for Apps and User Reviews datasets |
| **CSV & Excel Support** | `.csv`, `.xlsx`, `.xls` formats with auto-encoding detection |
| **Dataset Validation** | Column-level validation with clear error messages |
| **Data Cleaning Pipeline** | Installs, Price, Size, Rating, Type normalization |
| **Category Analysis** | App distribution, bar chart, saturated category identification |
| **Ratings Analysis** | Distribution histogram, average by category |
| **Size & Installs Analysis** | Scatter plot, Pearson correlation, trend line |
| **Pricing Analysis** | Free vs Paid donut, price distribution, estimated revenue |
| **Sentiment Analysis** | VADER/TextBlob NLP classification (Positive/Negative/Neutral) |
| **Sentiment by Category** | Merged dataset analysis, most positive/negative categories |
| **Interactive Plotly Charts** | Bubble chart, violin plot, install bar chart, sunburst |
| **Developer Insights** | ≥5 data-driven insights from actual uploaded data |
| **Dynamic KPIs** | All KPI values computed at runtime — never hardcoded |

---

## 📁 Dataset Requirements

> **Important:** The datasets are **NOT** included in this repository. You must upload them through the application's upload interface.

### Dataset A – Google Play Store Apps
**Required columns:**
- `App` — App name
- `Category` — App category
- `Rating` — User rating (1.0–5.0)
- `Reviews` — Number of reviews
- `Size` — App size (e.g. "19M", "1.5k")
- `Installs` — Install count (e.g. "10,000+")
- `Type` — Free or Paid
- `Price` — Price (e.g. "$4.99" or "0")

### Dataset B – Google Play Store User Reviews
**Required columns:**
- `App` — App name
- `Translated_Review` — English review text
- `Sentiment` — Pre-labeled sentiment (Positive/Negative/Neutral)
- `Sentiment_Polarity` — Polarity score
- `Sentiment_Subjectivity` — Subjectivity score

---

## 📤 Upload Instructions

1. Start the application (`python app.py`)
2. Open `http://localhost:5000` in your browser
3. Drag & drop (or click to browse) the **Apps dataset** into the first upload box
4. Drag & drop (or click to browse) the **User Reviews dataset** into the second upload box
5. Both datasets will be validated and previewed immediately
6. Click **"⚡ Run Full Analysis"** to generate the complete dashboard

---

## 🗂 Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | UTF-8, Latin-1, CP1252 auto-detected |
| Excel (new) | `.xlsx` | openpyxl engine |
| Excel (legacy) | `.xls` | xlrd engine |

---

## 🧹 Data Cleaning Methodology

| Column | Issue | Fix |
|---|---|---|
| `Installs` | "10,000+" → string | Remove `,` and `+`, cast to `int` |
| `Price` | "$4.99" → string | Remove `$`, cast to `float` |
| `Size` | "19M", "1.5k" → string | Parse unit, convert to `float` MB |
| `Rating` | Out-of-range (>5 or <1) | Set to `NaN` |
| `Reviews` | Mixed types | Cast to `numeric` |
| `Type` | Unexpected values | Normalise to Free/Paid only |
| `Category` | Whitespace, case | Strip + uppercase |
| Duplicates | Same App name | Keep first occurrence |
| Null Reviews | NaN text | Drop rows |
| Sentiment | Non-standard labels | Normalise to Positive/Negative/Neutral |

---

## 📂 Category Analysis

- Counts apps per category
- Horizontal bar chart (top 20 categories)
- Identifies the 5 most saturated categories by app count

---

## ⭐ Rating Analysis

- Overall average rating across all rated apps
- Distribution histogram with mean line
- Average rating per category (horizontal bar chart)
- Rating bucket breakdown (1–2, 2–3, 3–4, 4–4.5, 4.5–5)

---

## 📦 Size & Installs Analysis

- Scatter plot: App Size (MB) vs Install Count
- Pearson correlation coefficient
- Log-scale trend line
- Interpretation of correlation strength

---

## 💰 Pricing Analysis

- Free vs Paid donut chart
- Paid app price distribution bar chart (bucketed by price range)
- Estimated revenue by category (top 10 paid categories)

---

## 📈 Revenue Estimation Methodology

```
Estimated Revenue = App Price (USD) × Install Count
```

**Assumptions:**
- Each install is treated as a single purchase
- This is a rough upper-bound estimate
- Does NOT represent actual Google Play revenue
- Ignores refunds, regional pricing, in-app purchases

---

## 😊 Sentiment Analysis

- Uses **VADER** (primary) or **TextBlob** (fallback)
- Classifies each review as **Positive**, **Negative**, or **Neutral**
- Threshold: VADER compound ≥ 0.05 = Positive, ≤ -0.05 = Negative
- If dataset labels cover ≥80% of rows, existing labels are used as primary
- Reports count, percentage, and average polarity per class

---

## 📊 Sentiment by Category

- Merges the Apps and User Reviews datasets on the `App` column
- Computes positive/negative/neutral review counts per category
- Identifies the **top 5 most positive** and **top 5 most negative** categories
- Stacked horizontal bar chart (top 15 categories)

---

## 🔬 Interactive Visualisations (Plotly)

| Tab | Chart | Value |
|---|---|---|
| Category Bubble | App count vs avg rating, bubble = median installs | Identify opportunity gaps |
| Rating Violin | Free vs Paid rating distributions | Pricing strategy insight |
| Top Installs | Total installs by category (bar) | Market size overview |
| Sunburst | Type → Category → Rating bucket | Full landscape drill-down |

---

## 💡 Developer Insights

At least 5 data-driven insights are generated dynamically:

1. **Best Category to Enter** – balances avg rating vs competition count
2. **Free vs Paid Strategy** – based on actual market split in dataset
3. **Optimal App Size** – median size of top 25% install apps
4. **Rating Benchmark** – suggests a launch target above market average
5. **Highest Satisfaction Category** – highest positive review percentage

All insights are computed from the uploaded data. No generic conclusions.

---

## 🛠 Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Flask 3.0 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Visualisation | Matplotlib 3.9, Seaborn 0.13, Plotly 5.22 |
| NLP / Sentiment | VADER (vaderSentiment 3.3), TextBlob 0.18 |
| Excel Support | openpyxl, xlrd |
| Frontend | HTML5, CSS3, Vanilla JS |
| Charts (interactive) | Plotly.js (CDN) |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd DataAnalytics-L2-GooglePlayStoreAnalysis

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download TextBlob corpora
python -m textblob.download_corpora
```

---

## ▶️ How to Run

```bash
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

---

## 📸 Screenshots

> Add screenshots here after first run:
> - Upload interface
> - Overview KPIs
> - Category analysis chart
> - Sentiment analysis
> - Interactive Plotly charts
> - Developer insights

---

## 🌐 Live Demo

> Add your deployment URL here (e.g. Render, Heroku, Railway)

---

## ⚠️ Important Note on Datasets

The Google Play Store datasets are **not included** in this repository and are **never hardcoded**.

You must upload them through the application's upload interface. The application only processes files provided at runtime by the user.

Recommended datasets (publicly available on Kaggle):
- `googleplaystore.csv` – Apps dataset
- `googleplaystore_user_reviews.csv` – User Reviews dataset

---

## 📜 License

This project is created for educational purposes as part of the Oasis Infobyte Data Analytics Internship.
