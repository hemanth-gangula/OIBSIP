"""
Google Play Store Analysis – Full Execution Script
OASIS Infobyte Data Analytics Level 2

Run from the project root:
    python run_analysis.py

Generates:
  outputs/cleaned_data/cleaned_apps.csv
  outputs/cleaned_data/cleaned_reviews.csv
  outputs/cleaned_data/merged_data.csv
  outputs/figures/*.png  (8 static charts)
  outputs/figures/interactive_ratings_by_category.html
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
import plotly.io as pio

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
FIG_DIR    = os.path.join(ROOT, "outputs", "figures")
CLEAN_DIR  = os.path.join(ROOT, "outputs", "cleaned_data")

os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

APPS_PATH    = os.path.join(DATA_DIR, "googleplaystore.csv")
REVIEWS_PATH = os.path.join(DATA_DIR, "googleplaystore_user_reviews.csv")

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (13, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})
sns.set_theme(style="whitegrid", palette="husl")

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ saved {name}  [{size_kb:.0f} KB]")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1 – LOAD DATA")
print("="*60)

for p in (APPS_PATH, REVIEWS_PATH):
    if not os.path.exists(p):
        sys.exit(f"ERROR: required file not found: {p}")

df_apps    = pd.read_csv(APPS_PATH,    encoding="utf-8")
df_reviews = pd.read_csv(REVIEWS_PATH, encoding="utf-8")

print(f"Apps dataset    : {df_apps.shape[0]:,} rows × {df_apps.shape[1]} cols")
print(f"Reviews dataset : {df_reviews.shape[0]:,} rows × {df_reviews.shape[1]} cols")
print("\nApps columns   :", df_apps.columns.tolist())
print("Reviews columns:", df_reviews.columns.tolist())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – INITIAL INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2 – INITIAL INSPECTION")
print("="*60)

print("\n--- Apps missing values ---")
mv = df_apps.isnull().sum()
mv_pct = (df_apps.isnull().mean() * 100).round(2)
print(pd.DataFrame({"count": mv, "pct": mv_pct})[mv > 0].to_string())

print(f"\nApps duplicates : {df_apps.duplicated().sum()}")
print(f"\n--- Reviews missing values ---")
mv2 = df_reviews.isnull().sum()
mv2_pct = (df_reviews.isnull().mean() * 100).round(2)
print(pd.DataFrame({"count": mv2, "pct": mv2_pct})[mv2 > 0].to_string())
print(f"\nReviews duplicates: {df_reviews.duplicated().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3 – DATA CLEANING")
print("="*60)

apps    = df_apps.copy()
reviews = df_reviews.copy()

apps_before    = len(apps)
reviews_before = len(reviews)

# ---------- Apps ----------

# 1. Remove full duplicates
dups = apps.duplicated().sum()
apps.drop_duplicates(inplace=True)
print(f"Apps: removed {dups} duplicate rows")

# 2. Drop the known malformed row where Category is a number (e.g. "1.9")
bad = apps["Category"].str.match(r"^\d", na=False)
print(f"Apps: removing {bad.sum()} rows where Category starts with a digit")
apps = apps[~bad]

# 3. Rating → numeric; invalidate values outside [1, 5]
apps["Rating"] = pd.to_numeric(apps["Rating"], errors="coerce")
out_of_range = (~apps["Rating"].between(1.0, 5.0, inclusive="both")) & apps["Rating"].notna()
apps.loc[out_of_range, "Rating"] = np.nan
print(f"Apps: {out_of_range.sum()} out-of-range ratings set to NaN")

# 4. Reviews → numeric
apps["Reviews"] = pd.to_numeric(apps["Reviews"], errors="coerce")

# 5. Size → float MB
def parse_size(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s.lower() == "varies with device":
        return np.nan
    if s.endswith("M"):
        try: return float(s[:-1])
        except ValueError: return np.nan
    if s.endswith("k"):
        try: return float(s[:-1]) / 1024.0
        except ValueError: return np.nan
    try: return float(s)
    except ValueError: return np.nan

apps["Size_MB"] = apps["Size"].apply(parse_size)
print(f"Apps: Size_MB non-null = {apps['Size_MB'].notna().sum()}, "
      f"null (varies/other) = {apps['Size_MB'].isna().sum()}")

# 6. Installs → numeric
apps["Installs_Clean"] = (
    apps["Installs"]
    .str.replace(r"[+,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
print(f"Apps: Installs_Clean non-null = {apps['Installs_Clean'].notna().sum()}")

# 7. Price → numeric
apps["Price_Clean"] = (
    apps["Price"]
    .str.replace(r"\$", "", regex=True)
    .str.strip()
    .pipe(pd.to_numeric, errors="coerce")
)
print(f"Apps: paid apps (price > 0) = {(apps['Price_Clean'] > 0).sum()}, "
      f"free = {(apps['Price_Clean'] == 0).sum()}")

# 8. Type: keep only Free / Paid
apps["Type"] = apps["Type"].str.strip()
apps.loc[~apps["Type"].isin(["Free", "Paid"]), "Type"] = np.nan

# 9. Last Updated → datetime
apps["Last_Updated"] = pd.to_datetime(
    apps["Last Updated"], format="%B %d, %Y", errors="coerce"
)

# 10. Drop rows missing App name or Category (essential keys)
apps_clean = apps.dropna(subset=["App", "Category"]).copy()
print(f"\nApps before → after cleaning: {apps_before} → {len(apps_clean)}")

# ---------- Reviews ----------

# Replace string 'nan' with actual NaN
reviews["Translated_Review"] = reviews["Translated_Review"].replace("nan", np.nan)
reviews_clean = reviews.dropna(subset=["Translated_Review"]).drop_duplicates()
print(f"Reviews before → after cleaning: {reviews_before} → {len(reviews_clean)}")

# ---------- Save cleaned ----------
apps_clean.to_csv(os.path.join(CLEAN_DIR, "cleaned_apps.csv"), index=False)
reviews_clean.to_csv(os.path.join(CLEAN_DIR, "cleaned_reviews.csv"), index=False)
print("\n✓ cleaned_apps.csv saved")
print("✓ cleaned_reviews.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4 – CATEGORY ANALYSIS")
print("="*60)

cat_counts = apps_clean["Category"].value_counts().reset_index()
cat_counts.columns = ["Category", "App_Count"]
print(f"Total unique categories: {len(cat_counts)}")
print("\nTop 10 by app count:")
print(cat_counts.head(10).to_string(index=False))

# Horizontal bar chart – all categories
fig, ax = plt.subplots(figsize=(13, 10))
colors = sns.color_palette("husl", len(cat_counts))
ax.barh(cat_counts["Category"][::-1], cat_counts["App_Count"][::-1], color=colors[::-1])
ax.set_xlabel("Number of Apps", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_title("App Distribution by Category – Google Play Store", fontsize=15, fontweight="bold")
for i, (cat, cnt) in enumerate(zip(cat_counts["Category"][::-1], cat_counts["App_Count"][::-1])):
    ax.text(cnt + 10, i, str(cnt), va="center", fontsize=8)
savefig("category_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – RATINGS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5 – RATINGS ANALYSIS")
print("="*60)

rated = apps_clean["Rating"].dropna()
print(f"Apps with ratings    : {len(rated):,}  ({len(rated)/len(apps_clean)*100:.1f}%)")
print(f"Overall mean rating  : {rated.mean():.3f}")
print(f"Overall median rating: {rated.median():.3f}")

# Chart 1: rating distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(rated, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(rated.mean(),   color="red",    linestyle="--", linewidth=1.5,
           label=f"Mean {rated.mean():.2f}")
ax.axvline(rated.median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"Median {rated.median():.2f}")
ax.set_xlabel("Rating (1–5)")
ax.set_ylabel("Number of Apps")
ax.set_title("Distribution of Google Play Store App Ratings", fontsize=14, fontweight="bold")
ax.legend()
savefig("rating_distribution.png")

# Chart 2: average rating by category (min 10 apps with rating)
cat_rating = (
    apps_clean.groupby("Category")["Rating"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "Avg_Rating", "count": "n"})
    .query("n >= 10")
    .sort_values("Avg_Rating", ascending=False)
    .reset_index()
)
print(f"\nCategories with ≥10 rated apps: {len(cat_rating)}")
print("\nTop 5 highest avg rating:")
print(cat_rating.head(5)[["Category", "Avg_Rating", "n"]].to_string(index=False))
print("\nBottom 5 avg rating:")
print(cat_rating.tail(5)[["Category", "Avg_Rating", "n"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 9))
palette = sns.color_palette("RdYlGn", len(cat_rating))
bars = ax.barh(cat_rating["Category"][::-1], cat_rating["Avg_Rating"][::-1],
               color=palette)
ax.set_xlim(3.4, 4.8)
ax.set_xlabel("Average Rating")
ax.set_title("Average App Rating by Category\n(categories with ≥10 rated apps)",
             fontsize=14, fontweight="bold")
# Annotate values
for bar, val in zip(bars, cat_rating["Avg_Rating"][::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9)
savefig("average_rating_by_category.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – APP SIZE vs INSTALLS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6 – APP SIZE vs INSTALLS")
print("="*60)

size_inst = apps_clean.dropna(subset=["Size_MB", "Installs_Clean"]).copy()
size_inst = size_inst[size_inst["Installs_Clean"] > 0]
size_inst["Log10_Installs"] = np.log10(size_inst["Installs_Clean"])

print(f"Apps with both Size_MB and Installs: {len(size_inst):,}")

corr_raw = size_inst[["Size_MB", "Installs_Clean"]].corr().iloc[0, 1]
corr_log = size_inst[["Size_MB", "Log10_Installs"]].corr().iloc[0, 1]
print(f"Pearson corr (Size_MB vs Installs)        : {corr_raw:.4f}")
print(f"Pearson corr (Size_MB vs log10(Installs)) : {corr_log:.4f}")
print("Note: correlation ≠ causation.")

fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(
    size_inst["Size_MB"], size_inst["Log10_Installs"],
    c=size_inst["Rating"], cmap="RdYlGn", alpha=0.35, s=18,
    vmin=1, vmax=5
)
plt.colorbar(sc, ax=ax, label="Rating")
# Trend line
m = np.polyfit(size_inst["Size_MB"], size_inst["Log10_Installs"], 1)
xline = np.linspace(size_inst["Size_MB"].min(), size_inst["Size_MB"].max(), 200)
ax.plot(xline, np.polyval(m, xline), "b-", linewidth=1.5, alpha=0.7, label="Trend line")
ax.set_xlabel("App Size (MB)")
ax.set_ylabel("Installs (log₁₀ scale)")
ax.set_title(
    f"App Size vs Installs  |  Pearson r(log) = {corr_log:.3f}\n"
    "(colour = app rating; correlation ≠ causation)",
    fontsize=13, fontweight="bold"
)
ax.legend()

# Y-axis labels as real install counts
yticks = [3, 4, 5, 6, 7, 8]
ax.set_yticks(yticks)
ax.set_yticklabels([f"10{'^'+str(t)}" if t>1 else "1" for t in yticks])
savefig("size_vs_installs.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PRICING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 7 – PRICING ANALYSIS")
print("="*60)

type_counts = apps_clean["Type"].value_counts(dropna=True)
print("App type breakdown:"); print(type_counts.to_string())

# Chart 1: free vs paid pie
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].pie(
    type_counts.values, labels=type_counts.index,
    autopct="%1.1f%%", startangle=90,
    colors=["#4CAF50", "#FF5722"],
    textprops={"fontsize": 12}
)
axes[0].set_title("Free vs Paid App Distribution", fontsize=13, fontweight="bold")

# Chart 2: paid app price histogram
paid = apps_clean[apps_clean["Price_Clean"] > 0]
print(f"\nPaid apps: {len(paid):,}")
print("\nPaid price summary:")
print(paid["Price_Clean"].describe().round(2).to_string())

axes[1].hist(paid["Price_Clean"].clip(upper=30), bins=40,
             color="coral", edgecolor="white", alpha=0.85)
axes[1].axvline(paid["Price_Clean"].median(), color="red", linestyle="--", linewidth=1.5,
                label=f"Median ${paid['Price_Clean'].median():.2f}")
axes[1].set_xlabel("Price (USD, clipped at $30)")
axes[1].set_ylabel("Number of Paid Apps")
axes[1].set_title(f"Paid App Price Distribution  (n={len(paid):,})", fontsize=13, fontweight="bold")
axes[1].legend()
savefig("free_vs_paid.png")

# Separate clear free-vs-paid bar chart
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(type_counts.index, type_counts.values, color=["#4CAF50", "#FF5722"], width=0.5)
for i, v in enumerate(type_counts.values):
    ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Apps")
ax.set_title("Free vs Paid Apps – Google Play Store", fontsize=13, fontweight="bold")
savefig("paid_app_price_distribution.png")

# Theoretical revenue estimate – top categories
print("\n--- THEORETICAL GROSS REVENUE ESTIMATE ---")
print("IMPORTANT: Price × Installs is a rough upper bound only.")
print("It excludes refunds, platform fees (30%), taxes, and actual purchase rates.")
paid_inst = paid.dropna(subset=["Installs_Clean"]).copy()
paid_inst = paid_inst[paid_inst["Installs_Clean"] > 0]
paid_inst["Theoretical_Gross"] = paid_inst["Price_Clean"] * paid_inst["Installs_Clean"]
rev_cat = paid_inst.groupby("Category")["Theoretical_Gross"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 categories by theoretical gross (Price × Installs):")
print(rev_cat.apply(lambda x: f"${x/1e6:.2f}M").to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 8 – SENTIMENT ANALYSIS (TextBlob)")
print("="*60)
print("Thresholds: polarity > 0.05 → Positive | < -0.05 → Negative | else Neutral")

def tb_polarity(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return np.nan

def classify(p):
    if pd.isna(p):   return "Unknown"
    if p >  0.05:    return "Positive"
    if p < -0.05:    return "Negative"
    return "Neutral"

print(f"Running TextBlob on {len(reviews_clean):,} reviews …")
reviews_clean = reviews_clean.copy()
reviews_clean["TB_Polarity"]  = reviews_clean["Translated_Review"].apply(tb_polarity)
reviews_clean["TB_Sentiment"] = reviews_clean["TB_Polarity"].apply(classify)

sent_counts = reviews_clean["TB_Sentiment"].value_counts()
print("\nTextBlob sentiment distribution:")
print(sent_counts.to_string())
total_known = sent_counts.drop("Unknown", errors="ignore").sum()
for label in ["Positive", "Negative", "Neutral"]:
    n = sent_counts.get(label, 0)
    print(f"  {label}: {n:,}  ({n/total_known*100:.1f}% of non-unknown)")

# Chart: sentiment distribution
colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107",
              "Negative": "#F44336", "Unknown": "#9E9E9E"}
ordered = [l for l in ["Positive", "Neutral", "Negative", "Unknown"] if l in sent_counts.index]
vals    = [sent_counts[l] for l in ordered]
cols    = [colors_map[l] for l in ordered]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(ordered, vals, color=cols, edgecolor="white", linewidth=0.5)
for i, v in enumerate(vals):
    axes[0].text(i, v + 100, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Number of Reviews")
axes[0].set_title("Review Sentiment Distribution (TextBlob)", fontsize=13, fontweight="bold")

# Pie of known sentiments
known_labels = [l for l in ["Positive", "Neutral", "Negative"] if l in sent_counts.index]
known_vals   = [sent_counts[l] for l in known_labels]
known_cols   = [colors_map[l] for l in known_labels]
axes[1].pie(known_vals, labels=known_labels, autopct="%1.1f%%",
            colors=known_cols, startangle=90, textprops={"fontsize": 12})
axes[1].set_title("Sentiment Proportion\n(excluding Unknown)", fontsize=13, fontweight="bold")
savefig("sentiment_distribution.png")

# Print example reviews
print("\n--- Example reviews by predicted sentiment ---")
for label in ["Positive", "Negative", "Neutral"]:
    sample = reviews_clean[reviews_clean["TB_Sentiment"] == label]["Translated_Review"].head(2)
    print(f"\n{label}:")
    for rev in sample:
        print(f"  • {str(rev)[:120]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – SENTIMENT BY CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 9 – SENTIMENT BY CATEGORY")
print("="*60)

# Join reviews with apps to get Category
apps_lookup = apps_clean[["App", "Category"]].drop_duplicates(subset="App")
merged = reviews_clean.merge(apps_lookup, on="App", how="inner")
print(f"Reviews matched to a category: {len(merged):,} "
      f"({len(merged)/len(reviews_clean)*100:.1f}% of cleaned reviews)")
print(f"Unmatched reviews: {len(reviews_clean) - len(merged):,}")

# Sentiment proportions per category (only Positive/Negative/Neutral)
merged_known = merged[merged["TB_Sentiment"].isin(["Positive", "Negative", "Neutral"])]
sent_by_cat  = (
    merged_known.groupby(["Category", "TB_Sentiment"])
    .size()
    .unstack(fill_value=0)
    .assign(Total=lambda d: d.sum(axis=1))
)
# Normalise
for col in ["Positive", "Negative", "Neutral"]:
    if col not in sent_by_cat.columns:
        sent_by_cat[col] = 0
sent_by_cat["Positive_Pct"] = sent_by_cat["Positive"] / sent_by_cat["Total"] * 100
sent_by_cat = sent_by_cat[sent_by_cat["Total"] >= 20].sort_values("Positive_Pct", ascending=False)

print(f"\nCategories with ≥20 reviews after join: {len(sent_by_cat)}")
print("\nTop 5 most positive:")
print(sent_by_cat[["Positive", "Negative", "Neutral", "Total", "Positive_Pct"]].head(5).to_string())
print("\nBottom 5 (most negative):")
print(sent_by_cat[["Positive", "Negative", "Neutral", "Total", "Positive_Pct"]].tail(5).to_string())

# Chart: stacked horizontal bar
plot_df = sent_by_cat.sort_values("Positive_Pct")
fig, ax = plt.subplots(figsize=(13, max(8, len(plot_df) * 0.35)))
cats   = plot_df.index.tolist()
pos    = plot_df["Positive"].values
neu    = plot_df["Neutral"].values
neg    = plot_df["Negative"].values
totals = plot_df["Total"].values

y = np.arange(len(cats))
ax.barh(y, pos/totals*100, color="#4CAF50", label="Positive")
ax.barh(y, neu/totals*100, left=pos/totals*100, color="#FFC107", label="Neutral")
ax.barh(y, neg/totals*100, left=(pos+neu)/totals*100, color="#F44336", label="Negative")
ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9)
ax.set_xlabel("Percentage of Reviews (%)")
ax.set_title("Sentiment Distribution by App Category\n(categories with ≥20 matched reviews)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.axvline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
savefig("sentiment_by_category.png")

# Save merged dataset
merged_out = merged_known.copy()
merged_out.to_csv(os.path.join(CLEAN_DIR, "merged_data.csv"), index=False)
print("✓ merged_data.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – INTERACTIVE PLOTLY CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 10 – INTERACTIVE PLOTLY CHART")
print("="*60)

# Build a summary table: category | app_count | avg_rating | free_pct | total_installs
cat_summary = apps_clean.groupby("Category").agg(
    App_Count      = ("App", "count"),
    Avg_Rating     = ("Rating", "mean"),
    Total_Installs = ("Installs_Clean", "sum"),
    Free_Pct       = ("Type", lambda x: (x == "Free").mean() * 100),
).reset_index().round({"Avg_Rating": 2, "Free_Pct": 1})
cat_summary["Avg_Rating"].fillna(0, inplace=True)

fig_plotly = px.scatter(
    cat_summary,
    x="App_Count",
    y="Avg_Rating",
    size="Total_Installs",
    color="Category",
    hover_name="Category",
    hover_data={"App_Count": True, "Avg_Rating": True,
                "Free_Pct": True, "Total_Installs": ":.0f"},
    title="Google Play Store – Category Landscape<br>"
          "<sup>Bubble size = total installs; colour = category</sup>",
    labels={
        "App_Count":      "Number of Apps",
        "Avg_Rating":     "Average Rating",
        "Total_Installs": "Total Installs",
        "Free_Pct":       "% Free Apps",
    },
    size_max=60,
    template="plotly_white",
)
fig_plotly.update_layout(
    xaxis_title="Number of Apps in Category",
    yaxis_title="Average Rating (1–5)",
    legend_title="Category",
    height=620,
    font=dict(size=12),
)
html_path = os.path.join(FIG_DIR, "interactive_ratings_by_category.html")
pio.write_html(fig_plotly, file=html_path, auto_open=False)
size_kb = os.path.getsize(html_path) / 1024
print(f"  ✓ saved interactive_ratings_by_category.html  [{size_kb:.0f} KB]")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 11 – KEY INSIGHTS (from actual data)")
print("="*60)

top3_cats  = cat_counts.head(3)["Category"].tolist()
top_rated  = cat_rating.iloc[0]
low_rated  = cat_rating.iloc[-1]
free_pct   = (apps_clean["Type"] == "Free").mean() * 100
pos_pct    = sent_counts.get("Positive", 0) / total_known * 100
neg_pct    = sent_counts.get("Negative", 0) / total_known * 100

print(f"""
INSIGHT 1 — Market Saturation
  The three most saturated categories are: {', '.join(top3_cats)}.
  They account for {cat_counts.head(3)['App_Count'].sum():,} of
  {len(apps_clean):,} total apps ({cat_counts.head(3)['App_Count'].sum()/len(apps_clean)*100:.1f}%).
  New developers entering these categories face intense competition.

INSIGHT 2 — Ratings Quality Benchmark
  Highest-rated category : {top_rated['Category']} (avg {top_rated['Avg_Rating']:.2f}, n={int(top_rated['n'])})
  Lowest-rated  category : {low_rated['Category']} (avg {low_rated['Avg_Rating']:.2f}, n={int(low_rated['n'])})
  Overall play-store mean: {rated.mean():.2f}
  Overall apps with no rating at all: {apps_clean['Rating'].isna().sum():,} ({apps_clean['Rating'].isna().mean()*100:.1f}%)

INSIGHT 3 — Free vs Paid Dominance
  {free_pct:.1f}% of apps are free. Paid apps represent only {100-free_pct:.1f}% of the store.
  Paid app median price: ${paid['Price_Clean'].median():.2f}  |  mean: ${paid['Price_Clean'].mean():.2f}
  Implication: freemium or ad-supported models dominate the market.

INSIGHT 4 — App Size and Install Behaviour
  Pearson r (Size_MB vs log10 Installs) = {corr_log:.3f}
  The weak correlation suggests app size is not the primary driver of installs.
  Apps of all sizes can achieve high download counts; quality and category matter more.

INSIGHT 5 — User Sentiment
  TextBlob classified {pos_pct:.1f}% of reviews as Positive and {neg_pct:.1f}% as Negative.
  Most positive sentiment category : {sent_by_cat['Positive_Pct'].idxmax()}
  Most negative sentiment category : {sent_by_cat['Positive_Pct'].idxmin()}
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 12 – OUTPUT FILE VERIFICATION")
print("="*60)

expected = {
    "cleaned_data/cleaned_apps.csv":     CLEAN_DIR,
    "cleaned_data/cleaned_reviews.csv":  CLEAN_DIR,
    "cleaned_data/merged_data.csv":      CLEAN_DIR,
    "figures/category_distribution.png":       FIG_DIR,
    "figures/rating_distribution.png":         FIG_DIR,
    "figures/average_rating_by_category.png":  FIG_DIR,
    "figures/size_vs_installs.png":            FIG_DIR,
    "figures/free_vs_paid.png":                FIG_DIR,
    "figures/paid_app_price_distribution.png": FIG_DIR,
    "figures/sentiment_distribution.png":      FIG_DIR,
    "figures/sentiment_by_category.png":       FIG_DIR,
    "figures/interactive_ratings_by_category.html": FIG_DIR,
}

all_ok = True
for rel, base in expected.items():
    fullpath = os.path.join(base, os.path.basename(rel))
    exists   = os.path.exists(fullpath)
    size_kb  = os.path.getsize(fullpath) / 1024 if exists else 0
    status   = "✓" if exists and size_kb > 0 else "✗ MISSING"
    print(f"  {status}  outputs/{rel}  [{size_kb:.0f} KB]")
    if not (exists and size_kb > 0):
        all_ok = False

print()
if all_ok:
    print("✓ All output files verified – analysis complete.")
else:
    print("✗ Some files are missing – check errors above.")

print("\n" + "="*60)
print("RUN COMPLETE")
print("="*60)
