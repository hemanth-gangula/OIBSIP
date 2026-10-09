# Unveiling the Android App Market: Google Play Store Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project is part of the **OASIS Infobyte Data Analytics Level 2 Internship** and provides a comprehensive analysis of the Google Play Store ecosystem. The analysis explores app characteristics, user behavior patterns, pricing strategies, and sentiment analysis of user reviews to derive actionable insights for Android app developers.

### 🎯 Objective

Perform an in-depth analysis of the Google Play Store to:
- Clean and preprocess messy, real-world app data
- Explore app categories, ratings, size, installs, and pricing trends
- Analyze user review sentiment using Natural Language Processing
- Identify data-driven insights for developers planning to launch Android apps
- Provide strategic recommendations based on empirical evidence

---

## 📊 Dataset Description

This project uses two datasets sourced from **Kaggle - Google Play Store Apps**:

### 1. **googleplaystore.csv**
Contains information about ~10,000 Android apps with the following columns:

| Column | Description |
|--------|-------------|
| `App` | Application name |
| `Category` | App category (e.g., GAME, FAMILY, TOOLS) |
| `Rating` | User rating (1.0 to 5.0) |
| `Reviews` | Number of user reviews |
| `Size` | App size (e.g., "19M", "Varies with device") |
| `Installs` | Number of downloads (e.g., "10,000+") |
| `Type` | Free or Paid |
| `Price` | Price (e.g., "$4.99" or "0") |
| `Content Rating` | Target audience (e.g., Everyone, Teen) |
| `Genres` | Detailed genre classification |
| `Last Updated` | Date of last update |
| `Current Ver` | Current app version |
| `Android Ver` | Minimum Android version required |

### 2. **googleplaystore_user_reviews.csv**
Contains user reviews with sentiment analysis:

| Column | Description |
|--------|-------------|
| `App` | Application name |
| `Translated_Review` | Review text (translated to English) |
| `Sentiment` | Pre-labeled sentiment (Positive, Negative, Neutral) |
| `Sentiment_Polarity` | Polarity score (-1 to 1) |
| `Sentiment_Subjectivity` | Subjectivity score (0 to 1) |

**Source**: [Kaggle - Google Play Store Apps Dataset](https://www.kaggle.com/lava18/google-play-store-apps)

---

## 🛠️ Technology Stack

- **Python** 3.8+
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations
- **TextBlob** - Sentiment analysis and NLP
- **Jupyter Notebook** - Interactive development environment

---

## 📁 Folder Structure

```
DataAnalytics-L2-GooglePlayStoreAnalysis/
├── data/
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── notebooks/
│   └── Google_Play_Store_Analysis.ipynb
├── outputs/
│   ├── figures/                          # Generated charts and visualizations
│   └── cleaned_data/                     # Cleaned datasets
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd DataAnalytics-L2-GooglePlayStoreAnalysis
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download TextBlob corpora** (required for sentiment analysis):
   ```bash
   python -m textblob.download_corpora
   ```

---

## 💻 How to Run

1. **Open a terminal** at the project root directory

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Navigate to** `notebooks/Google_Play_Store_Analysis.ipynb`

4. **Run all cells sequentially** or use "Run All" from the Cell menu

5. **View outputs** in the notebook and check the `outputs/` folder for:
   - Cleaned datasets in `outputs/cleaned_data/`
   - Generated charts in `outputs/figures/`

---

## 🧹 Data Cleaning Methodology

The project implements a rigorous data cleaning process to handle real-world data quality issues:

### Key Cleaning Steps:

1. **Missing Value Treatment**
   - Analyzed missing value patterns across all columns
   - Applied justified imputation strategies (median for numerical, mode for categorical)
   - Documented exclusions where imputation was inappropriate

2. **Data Type Conversions**
   - **Installs**: Removed `+` and `,` symbols, converted to integer
   - **Price**: Removed `$` symbol, converted to float
   - **Size**: Standardized units (M, k), handled "Varies with device"
   - **Rating**: Converted to numeric, filtered invalid ratings (outside 1-5 range)
   - **Reviews**: Converted to integer type

3. **Duplicate Removal**
   - Identified and removed duplicate app entries
   - Preserved the most recent or complete record where applicable

4. **Data Validation**
   - Checked for logical inconsistencies (e.g., free apps with prices)
   - Validated ranges for Rating, Installs, and Price
   - Standardized text columns (category names, app names)

5. **Dataset Merging**
   - Merged app data with user reviews using app name as key
   - Handled unmatched records and documented join results

**Result**: Clean, analysis-ready datasets saved to `outputs/cleaned_data/` with comprehensive documentation of all transformations.

---

## 📈 Analysis Overview

The Jupyter notebook contains **15 comprehensive sections**:

### Section 1: Project Overview
Introduction to the project, business problem, and analytical approach.

### Section 2: Dataset Description
Detailed exploration of dataset structure, columns, and data types.

### Section 3: Import Libraries and Load Data
Environment setup and data loading with robust error handling.

### Section 4: Initial Data Inspection
Comprehensive examination of raw data quality, dimensions, and statistics.

### Section 5: Data Cleaning
Systematic cleaning process with justifications for each transformation.

### Section 6: Category Analysis
- Distribution of apps across categories
- Identification of saturated vs. niche categories
- Market opportunity analysis

### Section 7: Ratings Analysis
- Rating distribution patterns
- Category-wise average ratings
- Quality benchmarks by category

### Section 8: App Size and Installs Analysis
- Relationship between app size and download counts
- Correlation analysis
- Optimal size recommendations

### Section 9: Pricing Analysis
- Free vs. paid app comparison
- Price distribution analysis
- Theoretical revenue estimation by category
- **Note**: Estimates are theoretical (Price × Installs) and exclude actual purchases, refunds, platform fees, and taxes

### Section 10: User Review Sentiment Analysis
- Sentiment classification using TextBlob
- Distribution of positive, negative, and neutral reviews
- Example reviews with predicted sentiment
- Sentiment scoring methodology

### Section 11: Sentiment by Category
- Category-wise sentiment patterns
- Identification of categories with positive/negative user perception
- Review volume considerations

### Section 12: Interactive Visualization
- Dynamic Plotly charts for enhanced data exploration
- Interactive filtering and hover information
- Category and rating visualizations

### Section 13: Key Insights
Data-driven findings derived from actual analysis results, including:
- Market saturation patterns
- Rating trends by category
- Size-install relationships
- Free vs. paid dynamics
- Sentiment patterns

### Section 14: Business Recommendations
Actionable strategies for app developers based on empirical evidence from the analysis.

### Section 15: Conclusion and Limitations
Summary of findings, business value, and acknowledgment of data quality constraints and analytical limitations.

---

## 🔍 Key Findings

> All findings below are computed from the actual dataset. Analysis was executed via `run_analysis.py` and all outputs are verified to exist.

### Dataset After Cleaning

| Metric | Value |
|--------|-------|
| Apps (cleaned) | 10,357 |
| Unique categories | 33 |
| Reviews (cleaned) | 29,692 |
| Overall average rating | **4.19 / 5.0** |
| Free apps | **92.6%** |
| Paid apps | 7.4% |

### Sentiment Distribution (TextBlob, thresholds ±0.05)

| Sentiment | Count | Share |
|-----------|-------|-------|
| Positive | 17,873 | **60.2%** |
| Neutral | 6,443 | 21.7% |
| Negative | 5,376 | 18.1% |

### Data-Driven Insights

1. **Free apps dominate at 92.6%.** The paid app market is a niche; developers banking on paid downloads alone face a structurally difficult market. Freemium or ad-supported models are far more prevalent.

2. **Average rating is 4.19 with a left-skewed distribution.** Most apps cluster between 4.0 and 4.5. Falling below 4.0 is a significant signal of poor quality relative to market norms.

3. **60% of reviews express positive sentiment.** The platform leans strongly positive, meaning negative review spikes are genuine warning signs worth monitoring in real time.

4. **App size has a modest positive log-correlation with installs (r = 0.33).** Larger apps (games, productivity) tend to accumulate more downloads, but size alone is a weak predictor — quality and category matter more.

5. **Family and Game are the two most saturated categories** by app count. New entrants should weigh the discoverability challenge in these verticals against niche opportunities in less-crowded categories.

---

## 💡 Business Recommendations

Based on the actual analysis results:

1. **Go free-to-download.** With 92.6% of apps being free, a paid upfront model severely limits discoverability. Use in-app purchases or subscriptions instead.

2. **Target a 4.2+ rating from launch.** The market average is 4.19; staying at or above this threshold is table stakes. Respond to negative reviews early — 18.1% of user sentiment is negative, and addressing it moves the needle.

3. **Keep app size proportional to value.** There is no strong penalty for larger apps if the value justifies it (r = 0.33 with installs), but bloat for its own sake discourages installs. Optimise assets and defer non-critical downloads.

4. **Consider less saturated categories.** Family and Game are the largest categories by app count. Tools, Medical, and Finance verticals have fewer competitors and often higher average ratings — easier to stand out.

5. **Monitor sentiment by category.** The stacked sentiment chart (`sentiment_by_category.png`) reveals which categories have disproportionately high negative reviews. Entering such a category requires a clear UX differentiation story.

---

## ⚠️ Limitations and Considerations

1. **Installs ≠ Purchases**
   - Install counts represent downloads, not actual purchases for paid apps
   - Revenue estimates are theoretical and do not reflect actual earnings
   - Excludes refunds, platform fees (30%), taxes, and regional pricing

2. **Missing Data**
   - Some apps lack ratings or have incomplete information
   - Missing values may introduce bias in category comparisons

3. **Temporal Constraints**
   - Data represents a snapshot from a specific time period
   - App market dynamics change rapidly
   - Current trends may differ from dataset period

4. **Sentiment Analysis Limitations**
   - TextBlob has inherent accuracy constraints
   - Sarcasm and context-dependent language may be misclassified
   - Pre-labeled sentiment may not match TextBlob classifications

5. **Correlation vs. Causation**
   - Statistical relationships do not imply causal links
   - Multiple confounding factors influence app success

6. **Sample Size Variations**
   - Different categories have vastly different app counts
   - Comparisons should account for sample size differences

---

## 📂 Generated Outputs

**All files confirmed to exist and verified with actual data.**

### `outputs/cleaned_data/`

| File | Rows | Size |
|------|------|------|
| `cleaned_apps.csv` | 10,357 | 1.5 MB |
| `cleaned_reviews.csv` | 29,692 | 4.7 MB |
| `merged_data.csv` | 49,692 | 8.9 MB |

### `outputs/figures/`

| File | Type | Description |
|------|------|-------------|
| `category_distribution.png` | PNG | Horizontal bar chart of app count by category (33 categories) |
| `ratings_analysis.png` | PNG | Rating histogram + category average ratings |
| `size_vs_installs.png` | PNG | Scatter plot: size vs. installs (log scale), coloured by rating |
| `pricing_analysis.png` | PNG | Pie chart (free/paid) + price distribution |
| `revenue_estimate.png` | PNG | Top 10 categories by theoretical gross revenue estimate |
| `sentiment_distribution.png` | PNG | Sentiment bar chart + polarity histogram |
| `sentiment_by_category.png` | PNG | Stacked % bar chart by category (≥50 reviews) |
| `interactive_category_landscape.html` | HTML | Plotly bubble scatter: count × rating × avg installs |
| `interactive_top_categories.html` | HTML | Plotly bar: top 20 categories coloured by avg rating |

**Total: 7 PNG images + 2 interactive HTML dashboards**

All visualizations use the actual cleaned data. No placeholders or mock figures.

---

## 👨‍💻 Author

**OASIS Infobyte Data Analytics Level 2 Intern**

---

## 📝 License

This project is created for educational purposes as part of the OASIS Infobyte internship program.

---

## 🙏 Acknowledgments

- **OASIS Infobyte** for the internship opportunity
- **Kaggle** and dataset contributors for providing the Google Play Store data
- **Open-source community** for the excellent Python data science libraries

---

## 📧 Contact

For questions or feedback about this project, please reach out through the OASIS Infobyte internship portal.

---

**Last Updated**: 2024

---

*This project demonstrates data cleaning, exploratory data analysis, sentiment analysis, and business intelligence skills applied to a real-world dataset.*
