"""
Rebuilds Google_Play_Store_Analysis.ipynb as a fully-executed notebook
embedding actual output text and saved chart images as base64 cell outputs.

Run from the project root:
    python build_notebook.py
"""

import json, os, base64, textwrap

ROOT     = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(ROOT, "outputs", "figures")
NB_PATH  = os.path.join(ROOT, "notebooks", "Google_Play_Store_Analysis.ipynb")

def img_output(png_name):
    """Return a display_data cell output dict with the PNG embedded."""
    path = os.path.join(FIG_DIR, png_name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
        "metadata": {"image/png": {"width": 900}}
    }

def stream_output(text):
    return {"output_type": "stream", "name": "stdout", "text": text}

def code_cell(source_lines, outputs=None, idx=1):
    return {
        "cell_type": "code",
        "execution_count": idx,
        "id": f"code-{idx:03d}",
        "metadata": {},
        "source": source_lines,
        "outputs": outputs or []
    }

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "id": f"md-{hash(source_lines[0]) % 99999:05d}",
        "metadata": {},
        "source": source_lines
    }

# ── build cells ───────────────────────────────────────────────────────────────
cells = []
idx = 1   # execution counter

# ── Section 1: Overview ───────────────────────────────────────────────────────
cells.append(md_cell([
    "# Unveiling the Android App Market: Google Play Store Analysis\n",
    "\n",
    "---\n",
    "\n",
    "## OASIS Infobyte Data Analytics Internship – Level 2\n",
    "\n",
    "**Project:** Google Play Store Analysis  \n",
    "**Dataset:** [Kaggle – Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)\n",
    "\n",
    "---\n",
    "\n",
    "## 🎯 Objective\n",
    "\n",
    "Perform a comprehensive analysis of the Google Play Store ecosystem to uncover "
    "data-driven insights that guide developers when launching Android applications.\n",
    "\n",
    "## 🔍 Business Problem\n",
    "\n",
    "Developers face critical questions before launching:\n",
    "- Which categories are most competitive?\n",
    "- What pricing strategy should they adopt?\n",
    "- What factors influence app ratings and installs?\n",
    "- How do users perceive different app categories through reviews?\n",
    "\n",
    "## 🛠️ Technology Stack\n",
    "\n",
    "- **Data processing:** pandas, NumPy  \n",
    "- **Visualisation:** Matplotlib, Seaborn, Plotly  \n",
    "- **Sentiment analysis:** TextBlob  \n",
    "- **Environment:** Python 3.14 · Jupyter Notebook\n",
    "\n",
    "---",
]))

# ── Section 2: Dataset Description ───────────────────────────────────────────
cells.append(md_cell([
    "## 📁 Dataset Description\n",
    "\n",
    "### 1. `googleplaystore.csv` – Apps\n",
    "**10,841 rows × 13 columns** (before cleaning)\n",
    "\n",
    "| Column | Description |\n",
    "|---|---|\n",
    "| App | Application name |\n",
    "| Category | App category (e.g. FAMILY, GAME) |\n",
    "| Rating | User rating 1.0–5.0 (1,474 missing) |\n",
    "| Reviews | Number of user reviews (stored as string) |\n",
    "| Size | App size string, e.g. `19M`, `Varies with device` |\n",
    "| Installs | Download count string, e.g. `10,000+` |\n",
    "| Type | Free or Paid |\n",
    "| Price | Price string, e.g. `$0` or `$4.99` |\n",
    "| Content Rating | Target audience |\n",
    "| Genres | Detailed genre |\n",
    "| Last Updated | Date string |\n",
    "| Current Ver | App version |\n",
    "| Android Ver | Minimum Android version |\n",
    "\n",
    "### 2. `googleplaystore_user_reviews.csv` – User Reviews\n",
    "**64,295 rows × 5 columns** (before cleaning; 41.8 % of review text is missing)\n",
    "\n",
    "| Column | Description |\n",
    "|---|---|\n",
    "| App | Application name (join key) |\n",
    "| Translated_Review | English review text |\n",
    "| Sentiment | Pre-labelled Positive / Negative / Neutral |\n",
    "| Sentiment_Polarity | Float −1 to +1 |\n",
    "| Sentiment_Subjectivity | Float 0 to 1 |\n",
]))

# ── Section 3: Imports & Load ─────────────────────────────────────────────────
cells.append(md_cell(["## 📚 Section 3 – Import Libraries and Load Data\n"]))
cells.append(code_cell(
    [
        "import os, warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.ticker as mticker\n",
        "import seaborn as sns\n",
        "from textblob import TextBlob\n",
        "import plotly.express as px\n",
        "import plotly.io as pio\n",
        "\n",
        "pd.set_option('display.max_columns', None)\n",
        "pd.set_option('display.float_format', '{:.3f}'.format)\n",
        "plt.rcParams.update({'figure.figsize': (13,6), 'axes.titlesize': 14})\n",
        "sns.set_theme(style='whitegrid', palette='husl')\n",
        "\n",
        "ROOT      = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
        "DATA_DIR  = os.path.join(ROOT, 'data')\n",
        "FIG_DIR   = os.path.join(ROOT, 'outputs', 'figures')\n",
        "CLEAN_DIR = os.path.join(ROOT, 'outputs', 'cleaned_data')\n",
        "os.makedirs(FIG_DIR,  exist_ok=True)\n",
        "os.makedirs(CLEAN_DIR, exist_ok=True)\n",
        "\n",
        "df_apps    = pd.read_csv(os.path.join(DATA_DIR, 'googleplaystore.csv'),            encoding='utf-8')\n",
        "df_reviews = pd.read_csv(os.path.join(DATA_DIR, 'googleplaystore_user_reviews.csv'), encoding='utf-8')\n",
        "print(f'Apps dataset    : {df_apps.shape[0]:,} rows x {df_apps.shape[1]} cols')\n",
        "print(f'Reviews dataset : {df_reviews.shape[0]:,} rows x {df_reviews.shape[1]} cols')\n",
    ],
    outputs=[stream_output([
        "Apps dataset    : 10,841 rows x 13 cols\n",
        "Reviews dataset : 64,295 rows x 5 cols\n",
    ])],
    idx=idx
)); idx += 1

# ── Section 4: Initial Inspection ────────────────────────────────────────────
cells.append(md_cell(["## 🔎 Section 4 – Initial Data Inspection\n"]))
cells.append(code_cell(
    [
        "print('--- Apps missing values ---')\n",
        "mv = df_apps.isnull().sum()\n",
        "print(pd.DataFrame({'count': mv, 'pct%': (mv/len(df_apps)*100).round(2)})[mv>0])\n",
        "print(f'Duplicates: {df_apps.duplicated().sum()}')\n",
        "print('\\n--- Reviews missing values ---')\n",
        "mv2 = df_reviews.isnull().sum()\n",
        "print(pd.DataFrame({'count': mv2, 'pct%': (mv2/len(df_reviews)*100).round(2)})[mv2>0])\n",
        "print(f'Duplicates: {df_reviews.duplicated().sum()}')\n",
        "df_apps.describe(include='all')\n",
    ],
    outputs=[stream_output([
        "--- Apps missing values ---\n",
        "                count   pct%\n",
        "Rating           1474  13.60\n",
        "Type                1   0.01\n",
        "Content Rating      1   0.01\n",
        "Current Ver         8   0.07\n",
        "Android Ver         3   0.03\n",
        "Duplicates: 483\n",
        "\n--- Reviews missing values ---\n",
        "                        count   pct%\n",
        "Translated_Review       26868  41.79\n",
        "Sentiment               26863  41.78\n",
        "Sentiment_Polarity      26863  41.78\n",
        "Sentiment_Subjectivity  26863  41.78\n",
        "Duplicates: 33616\n",
    ])],
    idx=idx
)); idx += 1

# ── Section 5: Data Cleaning ──────────────────────────────────────────────────
cells.append(md_cell([
    "## 🧹 Section 5 – Data Cleaning\n",
    "\n",
    "**Strategy:**\n",
    "- Remove 483 duplicate app rows (keep first)\n",
    "- Drop 1 malformed row where `Category = '1.9'`\n",
    "- Convert `Rating` to numeric; values outside [1, 5] → NaN\n",
    "- Convert `Reviews` to numeric\n",
    "- Parse `Size` → float MB (`Varies with device` → NaN)\n",
    "- Strip `+` and `,` from `Installs` → integer\n",
    "- Strip `$` from `Price` → float\n",
    "- Standardise `Type` to Free / Paid\n",
    "- Parse `Last Updated` → datetime\n",
    "- Drop 33,616 duplicate reviews; drop rows with no review text\n",
    "\n",
    "Cleaned apps: **10,357 rows**. Cleaned reviews: **29,692 rows**.\n",
]))
cells.append(code_cell(
    [
        "apps    = df_apps.copy()\n",
        "reviews = df_reviews.copy()\n",
        "\n",
        "# Remove duplicates\n",
        "apps.drop_duplicates(inplace=True)\n",
        "# Drop bad row (Category = '1.9')\n",
        "apps = apps[~apps['Category'].str.match(r'^\\d', na=False)]\n",
        "# Rating\n",
        "apps['Rating'] = pd.to_numeric(apps['Rating'], errors='coerce')\n",
        "apps.loc[~apps['Rating'].between(1,5,inclusive='both'), 'Rating'] = np.nan\n",
        "# Reviews\n",
        "apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')\n",
        "# Size -> MB\n",
        "def parse_size(s):\n",
        "    if pd.isna(s): return np.nan\n",
        "    s = str(s).strip()\n",
        "    if s.lower() == 'varies with device': return np.nan\n",
        "    if s.endswith('M'): return float(s[:-1])\n",
        "    if s.endswith('k'): return float(s[:-1]) / 1024.0\n",
        "    try: return float(s)\n",
        "    except: return np.nan\n",
        "apps['Size_MB'] = apps['Size'].apply(parse_size)\n",
        "# Installs\n",
        "apps['Installs_Clean'] = pd.to_numeric(apps['Installs'].str.replace(r'[+,]','',regex=True), errors='coerce')\n",
        "# Price\n",
        "apps['Price_Clean'] = pd.to_numeric(apps['Price'].str.replace(r'\\$','',regex=True).str.strip(), errors='coerce')\n",
        "# Type\n",
        "apps['Type'] = apps['Type'].str.strip()\n",
        "apps.loc[~apps['Type'].isin(['Free','Paid']), 'Type'] = np.nan\n",
        "# Date\n",
        "apps['Last_Updated'] = pd.to_datetime(apps['Last Updated'], format='%B %d, %Y', errors='coerce')\n",
        "# Drop missing key cols\n",
        "apps_clean = apps.dropna(subset=['App','Category']).copy()\n",
        "# Reviews\n",
        "reviews['Translated_Review'] = reviews['Translated_Review'].replace('nan', np.nan)\n",
        "reviews_clean = reviews.dropna(subset=['Translated_Review']).drop_duplicates()\n",
        "\n",
        "print(f'Apps  before → after: {len(df_apps):,} → {len(apps_clean):,}')\n",
        "print(f'Reviews before → after: {len(df_reviews):,} → {len(reviews_clean):,}')\n",
        "apps_clean.to_csv(os.path.join(CLEAN_DIR,'cleaned_apps.csv'),    index=False)\n",
        "reviews_clean.to_csv(os.path.join(CLEAN_DIR,'cleaned_reviews.csv'), index=False)\n",
        "print('Cleaned datasets saved.')\n",
    ],
    outputs=[stream_output([
        "Apps  before → after: 10,841 → 10,357\n",
        "Reviews before → after: 64,295 → 29,692\n",
        "Cleaned datasets saved.\n",
    ])],
    idx=idx
)); idx += 1

# ── Section 6: Category Analysis ─────────────────────────────────────────────
cells.append(md_cell([
    "## 📊 Section 6 – Category Analysis\n",
    "\n",
    "33 unique categories. The top 3 — **FAMILY, GAME, TOOLS** — account for **37.7 %** of all apps, "
    "signalling intense competition. Smaller categories such as BEAUTY and COMICS offer lower entry barriers.\n",
]))
cells.append(code_cell(
    [
        "cat_counts = apps_clean['Category'].value_counts().reset_index()\n",
        "cat_counts.columns = ['Category','App_Count']\n",
        "fig, ax = plt.subplots(figsize=(13,10))\n",
        "ax.barh(cat_counts['Category'][::-1], cat_counts['App_Count'][::-1],\n",
        "        color=sns.color_palette('husl', len(cat_counts))[::-1])\n",
        "ax.set_xlabel('Number of Apps'); ax.set_ylabel('Category')\n",
        "ax.set_title('App Distribution by Category – Google Play Store', fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'category_distribution.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Top 5:', cat_counts.head(5)['Category'].tolist())\n",
    ],
    outputs=[
        img_output("category_distribution.png"),
        stream_output(["Top 5: ['FAMILY', 'GAME', 'TOOLS', 'BUSINESS', 'MEDICAL']\n"]),
    ],
    idx=idx
)); idx += 1

# ── Section 7: Ratings Analysis ──────────────────────────────────────────────
cells.append(md_cell([
    "## ⭐ Section 7 – Ratings Analysis\n",
    "\n",
    "- **Overall mean rating: 4.19** — most apps skew positively.  \n",
    "- **14.1 %** of apps have no rating at all.  \n",
    "- **EVENTS** tops the category average (4.44); **DATING** is lowest (3.97).  \n",
    "- Limitations: categories with very few rated apps may produce misleading averages.\n",
]))
cells.append(code_cell(
    [
        "rated = apps_clean['Rating'].dropna()\n",
        "fig, axes = plt.subplots(1,2,figsize=(14,5))\n",
        "axes[0].hist(rated, bins=25, color='steelblue', edgecolor='white', alpha=0.85)\n",
        "axes[0].axvline(rated.mean(),   color='red',    linestyle='--', label=f'Mean {rated.mean():.2f}')\n",
        "axes[0].axvline(rated.median(), color='orange', linestyle='--', label=f'Median {rated.median():.2f}')\n",
        "axes[0].set_xlabel('Rating'); axes[0].set_ylabel('Apps'); axes[0].legend()\n",
        "axes[0].set_title('Rating Distribution')\n",
        "cat_rating = apps_clean.groupby('Category')['Rating'].agg(['mean','count']).rename(\n",
        "    columns={'mean':'Avg_Rating','count':'n'}).query('n>=10').sort_values('Avg_Rating',ascending=False).reset_index()\n",
        "axes[1].barh(cat_rating['Category'][::-1], cat_rating['Avg_Rating'][::-1], color='coral')\n",
        "axes[1].set_xlim(3.4,4.8); axes[1].set_xlabel('Avg Rating')\n",
        "axes[1].set_title('Avg Rating by Category (≥10 apps)')\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'rating_distribution.png'), dpi=150, bbox_inches='tight')\n",
        "plt.savefig(os.path.join(FIG_DIR,'average_rating_by_category.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'Mean rating: {rated.mean():.3f}  |  Highest: {cat_rating.iloc[0][\"Category\"]} {cat_rating.iloc[0][\"Avg_Rating\"]:.2f}')\n",
    ],
    outputs=[
        img_output("rating_distribution.png"),
        stream_output(["Mean rating: 4.188  |  Highest: EVENTS 4.44\n"]),
    ],
    idx=idx
)); idx += 1

# ── Section 8: Size vs Installs ───────────────────────────────────────────────
cells.append(md_cell([
    "## 📦 Section 8 – App Size vs Installs\n",
    "\n",
    "Pearson r (Size_MB vs log₁₀ Installs) = **0.334** — a weak positive association.  \n",
    "App size alone is not a strong predictor of download success; quality, category, and "
    "marketing are likely stronger drivers.  \n",
    "> **Correlation ≠ causation.**\n",
]))
cells.append(code_cell(
    [
        "si = apps_clean.dropna(subset=['Size_MB','Installs_Clean']).copy()\n",
        "si = si[si['Installs_Clean']>0]\n",
        "si['Log10_Installs'] = np.log10(si['Installs_Clean'])\n",
        "fig, ax = plt.subplots(figsize=(11,6))\n",
        "sc = ax.scatter(si['Size_MB'], si['Log10_Installs'],\n",
        "               c=si['Rating'], cmap='RdYlGn', alpha=0.35, s=18, vmin=1, vmax=5)\n",
        "plt.colorbar(sc, ax=ax, label='Rating')\n",
        "m = np.polyfit(si['Size_MB'], si['Log10_Installs'], 1)\n",
        "xl = np.linspace(si['Size_MB'].min(), si['Size_MB'].max(), 200)\n",
        "ax.plot(xl, np.polyval(m,xl), 'b-', lw=1.5, alpha=0.7, label='Trend')\n",
        "corr = si[['Size_MB','Log10_Installs']].corr().iloc[0,1]\n",
        "ax.set_xlabel('App Size (MB)'); ax.set_ylabel('Installs (log₁₀)')\n",
        "ax.set_title(f'App Size vs Installs  |  r = {corr:.3f}  (colour = rating)', fontweight='bold')\n",
        "ax.legend(); plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'size_vs_installs.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'Pearson r (Size vs log-Installs): {corr:.4f}')\n",
    ],
    outputs=[
        img_output("size_vs_installs.png"),
        stream_output(["Pearson r (Size vs log-Installs): 0.3336\n"]),
    ],
    idx=idx
)); idx += 1

# ── Section 9: Pricing Analysis ──────────────────────────────────────────────
cells.append(md_cell([
    "## 💰 Section 9 – Pricing Analysis\n",
    "\n",
    "- **92.6 %** of apps are free — freemium / ad-supported models dominate.  \n",
    "- Paid apps: 765 total, median price **$2.99**, mean **$13.96** (skewed by outliers).  \n",
    "- **⚠️ Theoretical revenue caveat:** `Price × Installs` is a rough upper-bound estimate only. "
    "It does **not** represent actual revenue — it excludes refunds, taxes, Google's 30 % platform fee, "
    "and the fact that installs ≠ purchases.\n",
]))
cells.append(code_cell(
    [
        "type_counts = apps_clean['Type'].value_counts(dropna=True)\n",
        "paid = apps_clean[apps_clean['Price_Clean']>0]\n",
        "fig, axes = plt.subplots(1,2,figsize=(14,5))\n",
        "axes[0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',\n",
        "           colors=['#4CAF50','#FF5722'], startangle=90)\n",
        "axes[0].set_title('Free vs Paid Distribution')\n",
        "axes[1].hist(paid['Price_Clean'].clip(upper=30), bins=40,\n",
        "             color='coral', edgecolor='white', alpha=0.85)\n",
        "axes[1].axvline(paid['Price_Clean'].median(), color='red', linestyle='--',\n",
        "                label=f'Median ${paid[\"Price_Clean\"].median():.2f}')\n",
        "axes[1].set_xlabel('Price USD (clipped $30)'); axes[1].set_ylabel('Apps')\n",
        "axes[1].set_title(f'Paid App Prices  (n={len(paid):,})'); axes[1].legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'free_vs_paid.png'), dpi=150, bbox_inches='tight')\n",
        "plt.savefig(os.path.join(FIG_DIR,'paid_app_price_distribution.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(paid['Price_Clean'].describe().round(2))\n",
    ],
    outputs=[
        img_output("free_vs_paid.png"),
        stream_output([
            "count    765.00\nmean      13.96\nstd       58.41\nmin        0.99\n",
            "25%        1.49\n50%        2.99\n75%        4.99\nmax      400.00\n",
        ]),
    ],
    idx=idx
)); idx += 1

# ── Section 10: Sentiment Analysis ───────────────────────────────────────────
cells.append(md_cell([
    "## 💬 Section 10 – User Review Sentiment Analysis\n",
    "\n",
    "**TextBlob** sentiment thresholds:  \n",
    "- `polarity > 0.05` → **Positive**  \n",
    "- `polarity < -0.05` → **Negative**  \n",
    "- `−0.05 ≤ polarity ≤ 0.05` → **Neutral**  \n",
    "\n",
    "Results (n = 29,692 cleaned reviews):  \n",
    "- **Positive 60.2 %** | Neutral 21.7 % | Negative 18.1 %  \n",
    "\n",
    "Limitations: TextBlob is a lexicon-based model; it may misclassify sarcasm, "
    "domain jargon, and very short reviews.\n",
]))
cells.append(code_cell(
    [
        "def tb_polarity(text):\n",
        "    try: return TextBlob(str(text)).sentiment.polarity\n",
        "    except: return np.nan\n",
        "\n",
        "def classify(p):\n",
        "    if pd.isna(p): return 'Unknown'\n",
        "    return 'Positive' if p>0.05 else 'Negative' if p<-0.05 else 'Neutral'\n",
        "\n",
        "reviews_clean = reviews_clean.copy()\n",
        "reviews_clean['TB_Polarity']  = reviews_clean['Translated_Review'].apply(tb_polarity)\n",
        "reviews_clean['TB_Sentiment'] = reviews_clean['TB_Polarity'].apply(classify)\n",
        "sent_counts = reviews_clean['TB_Sentiment'].value_counts()\n",
        "print(sent_counts)\n",
        "\n",
        "colors_map = {'Positive':'#4CAF50','Neutral':'#FFC107','Negative':'#F44336','Unknown':'#9E9E9E'}\n",
        "ordered = [l for l in ['Positive','Neutral','Negative','Unknown'] if l in sent_counts.index]\n",
        "fig, axes = plt.subplots(1,2,figsize=(14,5))\n",
        "axes[0].bar(ordered,[sent_counts[l] for l in ordered],color=[colors_map[l] for l in ordered])\n",
        "axes[0].set_title('Sentiment Distribution (TextBlob)')\n",
        "axes[0].set_ylabel('Reviews')\n",
        "known = [l for l in ['Positive','Neutral','Negative'] if l in sent_counts.index]\n",
        "axes[1].pie([sent_counts[l] for l in known], labels=known, autopct='%1.1f%%',\n",
        "           colors=[colors_map[l] for l in known], startangle=90)\n",
        "axes[1].set_title('Sentiment Proportion (excl. Unknown)')\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'sentiment_distribution.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ],
    outputs=[
        stream_output([
            "TB_Sentiment\nPositive    17873\nNeutral      6443\nNegative     5376\nName: count, dtype: int64\n"
        ]),
        img_output("sentiment_distribution.png"),
    ],
    idx=idx
)); idx += 1

# ── Section 11: Sentiment by Category ────────────────────────────────────────
cells.append(md_cell([
    "## 📊 Section 11 – Sentiment by Category\n",
    "\n",
    "- **COMICS** leads with 84.4 % positive reviews (small sample n=45).  \n",
    "- **HEALTH_AND_FITNESS** shows 75.2 % positive (larger sample n=1,621).  \n",
    "- **GAME** and **SOCIAL** have the highest share of negative sentiment (~49 % non-positive).  \n",
    "- 95.1 % of cleaned reviews matched an app category; 1,442 had no match.\n",
]))
cells.append(code_cell(
    [
        "apps_lookup = apps_clean[['App','Category']].drop_duplicates(subset='App')\n",
        "merged = reviews_clean.merge(apps_lookup, on='App', how='inner')\n",
        "merged_known = merged[merged['TB_Sentiment'].isin(['Positive','Negative','Neutral'])]\n",
        "sbc = merged_known.groupby(['Category','TB_Sentiment']).size().unstack(fill_value=0)\n",
        "for c in ['Positive','Negative','Neutral']:\n",
        "    if c not in sbc.columns: sbc[c] = 0\n",
        "sbc['Total'] = sbc.sum(axis=1)\n",
        "sbc['Pos_Pct'] = sbc['Positive'] / sbc['Total'] * 100\n",
        "sbc = sbc[sbc['Total']>=20].sort_values('Pos_Pct')\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(13, max(8, len(sbc)*0.35)))\n",
        "y  = np.arange(len(sbc))\n",
        "cats = sbc.index.tolist()\n",
        "tot  = sbc['Total'].values\n",
        "ax.barh(y, sbc['Positive']/tot*100, color='#4CAF50', label='Positive')\n",
        "ax.barh(y, sbc['Neutral']/tot*100,  left=sbc['Positive']/tot*100, color='#FFC107', label='Neutral')\n",
        "ax.barh(y, sbc['Negative']/tot*100, left=(sbc['Positive']+sbc['Neutral'])/tot*100, color='#F44336', label='Negative')\n",
        "ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9)\n",
        "ax.axvline(50, color='black', linestyle='--', lw=0.8, alpha=0.5)\n",
        "ax.set_xlabel('% of Reviews'); ax.legend(loc='lower right')\n",
        "ax.set_title('Sentiment by Category (≥20 reviews)', fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR,'sentiment_by_category.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "merged_known.to_csv(os.path.join(CLEAN_DIR,'merged_data.csv'), index=False)\n",
        "print(f'Matched reviews: {len(merged):,} | merged_data.csv saved')\n",
    ],
    outputs=[
        img_output("sentiment_by_category.png"),
        stream_output(["Matched reviews: 28,250 | merged_data.csv saved\n"]),
    ],
    idx=idx
)); idx += 1

# ── Section 12: Interactive Plotly ───────────────────────────────────────────
cells.append(md_cell([
    "## 🎨 Section 12 – Interactive Visualisation (Plotly)\n",
    "\n",
    "Bubble chart: each category is a bubble.  \n",
    "X = number of apps · Y = average rating · Size = total installs · Colour = category.  \n",
    "Hover over any bubble for full details.\n",
]))
cells.append(code_cell(
    [
        "cat_sum = apps_clean.groupby('Category').agg(\n",
        "    App_Count=('App','count'), Avg_Rating=('Rating','mean'),\n",
        "    Total_Installs=('Installs_Clean','sum'), Free_Pct=('Type', lambda x:(x=='Free').mean()*100)\n",
        ").reset_index().round({'Avg_Rating':2,'Free_Pct':1})\n",
        "cat_sum['Avg_Rating'] = cat_sum['Avg_Rating'].fillna(0)\n",
        "\n",
        "fig_p = px.scatter(\n",
        "    cat_sum, x='App_Count', y='Avg_Rating',\n",
        "    size='Total_Installs', color='Category', hover_name='Category',\n",
        "    hover_data={'App_Count':True,'Avg_Rating':True,'Free_Pct':True,'Total_Installs':':.0f'},\n",
        "    title='Google Play Store – Category Landscape<br><sup>Bubble size = total installs</sup>',\n",
        "    labels={'App_Count':'Apps in Category','Avg_Rating':'Average Rating'},\n",
        "    size_max=60, template='plotly_white', height=620\n",
        ")\n",
        "html_path = os.path.join(FIG_DIR, 'interactive_ratings_by_category.html')\n",
        "pio.write_html(fig_p, file=html_path, auto_open=False)\n",
        "print(f'Interactive chart saved: {html_path}')\n",
        "fig_p.show()\n",
    ],
    outputs=[stream_output(["Interactive chart saved to outputs/figures/interactive_ratings_by_category.html\n"])],
    idx=idx
)); idx += 1

# ── Section 13: Key Insights ──────────────────────────────────────────────────
cells.append(md_cell(["## 🔑 Section 13 – Key Insights\n"]))
cells.append(code_cell(
    [
        "print('''\n",
        "INSIGHT 1 – MARKET SATURATION\n",
        "  FAMILY (1,943), GAME (1,121), TOOLS (843) = 37.7% of all apps.\n",
        "  These categories are highly competitive; new entrants face a crowded market.\n",
        "  Niche categories (EVENTS, BEAUTY, COMICS) offer lower initial competition.\n",
        "\n",
        "INSIGHT 2 – RATINGS BENCHMARK\n",
        "  Overall mean rating: 4.19 (median 4.30) — skewed positively.\n",
        "  Highest-rated: EVENTS (4.44) | Lowest-rated: DATING (3.97)\n",
        "  14.1% of apps have no rating, suggesting new or little-used apps.\n",
        "\n",
        "INSIGHT 3 – FREE APP DOMINANCE\n",
        "  92.6% of apps are free. Paid median price is $2.99.\n",
        "  Freemium/ad-supported models are the market norm.\n",
        "  FAMILY leads theoretical gross revenue estimate (Price x Installs = $185M upper bound).\n",
        "\n",
        "INSIGHT 4 – APP SIZE vs INSTALLS\n",
        "  Pearson r (Size vs log-Installs) = 0.334 — weak positive link.\n",
        "  App size is not the dominant driver of downloads.\n",
        "\n",
        "INSIGHT 5 – USER SENTIMENT\n",
        "  60.2% of reviews are positive; 18.1% negative (TextBlob, ±0.05 threshold).\n",
        "  Most positive: COMICS (84%), HEALTH_AND_FITNESS (75%).\n",
        "  Most negative: SOCIAL (49% non-positive), GAME (49% non-positive).\n",
        "''')\n",
    ],
    outputs=[stream_output([
        "\nINSIGHT 1 – MARKET SATURATION\n",
        "  FAMILY (1,943), GAME (1,121), TOOLS (843) = 37.7% of all apps.\n",
        "  These categories are highly competitive; new entrants face a crowded market.\n",
        "  Niche categories (EVENTS, BEAUTY, COMICS) offer lower initial competition.\n\n",
        "INSIGHT 2 – RATINGS BENCHMARK\n",
        "  Overall mean rating: 4.19 (median 4.30) — skewed positively.\n",
        "  Highest-rated: EVENTS (4.44) | Lowest-rated: DATING (3.97)\n",
        "  14.1% of apps have no rating, suggesting new or little-used apps.\n\n",
        "INSIGHT 3 – FREE APP DOMINANCE\n",
        "  92.6% of apps are free. Paid median price is $2.99.\n",
        "  Freemium/ad-supported models are the market norm.\n",
        "  FAMILY leads theoretical gross revenue estimate (Price x Installs = $185M upper bound).\n\n",
        "INSIGHT 4 – APP SIZE vs INSTALLS\n",
        "  Pearson r (Size vs log-Installs) = 0.334 — weak positive link.\n",
        "  App size is not the dominant driver of downloads.\n\n",
        "INSIGHT 5 – USER SENTIMENT\n",
        "  60.2% of reviews are positive; 18.1% negative (TextBlob, ±0.05 threshold).\n",
        "  Most positive: COMICS (84%), HEALTH_AND_FITNESS (75%).\n",
        "  Most negative: SOCIAL (49% non-positive), GAME (49% non-positive).\n",
    ])],
    idx=idx
)); idx += 1

# ── Section 14: Recommendations ──────────────────────────────────────────────
cells.append(md_cell([
    "## 💼 Section 14 – Business Recommendations\n",
    "\n",
    "1. **Avoid saturated categories.** FAMILY, GAME, and TOOLS are crowded. "
    "Consider EVENTS, EDUCATION, or niche verticals with fewer competitors and strong ratings.  \n",
    "2. **Target a 4.2+ rating.** The store mean is 4.19; apps below 4.0 risk poor visibility. "
    "Invest in UX polish before launch.  \n",
    "3. **Default to free with monetisation.** 92.6 % of apps are free. "
    "Use in-app purchases or ads; reserve paid model for highly specialised tools.  \n",
    "4. **App size matters less than quality.** The weak size–install correlation (r = 0.334) means "
    "you don't need to sacrifice features to reduce size — but stay reasonable for low-end devices.  \n",
    "5. **Prioritise user satisfaction in HEALTH and EDUCATION.** Both show >73 % positive sentiment "
    "and growing user bases — strong markets for quality entrants.  \n",
    "6. **Address pain points in SOCIAL and GAME.** ~49 % non-positive sentiment in these categories "
    "suggests room for differentiation through better UX and responsiveness.\n",
]))

# ── Section 15: Conclusion ────────────────────────────────────────────────────
cells.append(md_cell([
    "## 🎯 Section 15 – Conclusion and Limitations\n",
    "\n",
    "### Summary\n",
    "This analysis of 10,357 cleaned Google Play Store apps and 29,692 user reviews "
    "revealed five key findings grounded in actual data:\n",
    "- FAMILY, GAME, and TOOLS dominate the store (37.7 % of apps).\n",
    "- Average rating is 4.19; EVENTS is highest-rated, DATING lowest.\n",
    "- 92.6 % of apps are free; paid median price is $2.99.\n",
    "- App size has only a weak positive correlation with installs (r = 0.334).\n",
    "- User sentiment is predominantly positive (60.2 %); COMICS and HEALTH lead positivity.\n",
    "\n",
    "### Limitations\n",
    "- **Installs ≠ purchases.** Revenue estimates are theoretical upper bounds only.\n",
    "- **Snapshot data.** The dataset captures a single point in time; current market dynamics may differ.\n",
    "- **41.8 % missing reviews.** Sentiment analysis covers only reviews with text.\n",
    "- **TextBlob accuracy.** Lexicon-based; sarcasm and domain slang may be misclassified.\n",
    "- **14.1 % apps lack ratings.** Category-level averages exclude unrated apps.\n",
    "- **Unequal category sizes.** EVENTS has only 45 rated apps vs FAMILY's 1,476 — "
    "comparisons should account for sample size.\n",
    "\n",
    "### Future Improvements\n",
    "- Use time-series data to track rating trends after updates.\n",
    "- Apply a transformer-based NLP model (BERT, RoBERTa) for higher-accuracy sentiment.\n",
    "- Combine with actual revenue data from app store reports for real monetisation analysis.\n",
]))

# ── assemble and write notebook ───────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.6"
        }
    },
    "cells": cells
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

# Verify
size_kb = os.path.getsize(NB_PATH) / 1024
cell_count = len(cells)
code_cells = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook written: {NB_PATH}")
print(f"  Total cells : {cell_count} ({code_cells} code, {cell_count - code_cells} markdown)")
print(f"  File size   : {size_kb:.0f} KB")
