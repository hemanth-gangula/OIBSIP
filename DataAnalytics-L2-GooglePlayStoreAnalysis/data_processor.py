"""
data_processor.py
-----------------
Handles loading, cleaning, and analysis of the Google Play Store Apps dataset.
All analysis values are derived from the uploaded real dataset — no hardcoding.
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Required column sets for validation
# ---------------------------------------------------------------------------
APPS_REQUIRED_COLS = {"App", "Category", "Rating", "Reviews", "Size", "Installs", "Type", "Price"}
REVIEWS_REQUIRED_COLS = {"App", "Translated_Review", "Sentiment"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_apps_dataset(df: pd.DataFrame) -> dict:
    """Return {'valid': bool, 'missing': list} for the Apps dataset."""
    missing = [c for c in APPS_REQUIRED_COLS if c not in df.columns]
    return {"valid": len(missing) == 0, "missing": missing}


def validate_reviews_dataset(df: pd.DataFrame) -> dict:
    """Return {'valid': bool, 'missing': list} for the Reviews dataset."""
    missing = [c for c in REVIEWS_REQUIRED_COLS if c not in df.columns]
    return {"valid": len(missing) == 0, "missing": missing}


def load_file(filepath: str) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    if filepath.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    return pd.read_csv(filepath, on_bad_lines="skip")


# ---------------------------------------------------------------------------
# Apps dataset — cleaning
# ---------------------------------------------------------------------------

def _parse_installs(val) -> float:
    """Convert '10,000+' → 10000.0, handle edge cases."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "").replace("+", "").strip()
    if s in ("Free", "Varies with device", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_size(val) -> float:
    """Convert '19M' → 19.0 (MB), '512k' → 0.5 (MB), 'Varies' → NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ("varies with device", "varies", ""):
        return np.nan
    s_upper = s.upper()
    try:
        if s_upper.endswith("M"):
            return float(s_upper[:-1])
        if s_upper.endswith("K"):
            return float(s_upper[:-1]) / 1024
        return float(s)
    except ValueError:
        return np.nan


def _parse_price(val) -> float:
    """Convert '$2.99' → 2.99, '0' or 'Free' → 0.0."""
    if pd.isna(val):
        return 0.0
    s = str(val).replace("$", "").strip()
    if s.lower() in ("free", "0", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def clean_apps_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the Apps dataset.
    Returns (cleaned_df, summary_dict).
    """
    summary = {}
    summary["original_rows"] = len(df)

    # Drop exact duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    summary["duplicates_removed"] = before_dedup - len(df)

    # Keep only the first occurrence of each App name (highest reviews tiebreak)
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Reviews", ascending=False).drop_duplicates(subset="App", keep="first")

    # Fix Installs
    df["Installs"] = df["Installs"].apply(_parse_installs)

    # Fix Size
    df["Size_MB"] = df["Size"].apply(_parse_size)

    # Fix Price
    df["Price_USD"] = df["Price"].apply(_parse_price)

    # Fix Rating — must be 0–5
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df.loc[(df["Rating"] < 0) | (df["Rating"] > 5), "Rating"] = np.nan

    # Normalize Type column
    df["Type"] = df["Type"].str.strip().str.capitalize()
    df.loc[~df["Type"].isin(["Free", "Paid"]), "Type"] = np.nan

    # Handle nulls — drop rows where critical fields are missing
    null_before = df.isnull().sum().to_dict()
    df = df[df["App"].notna() & df["Category"].notna()]

    summary["null_handling"] = {
        k: int(v) for k, v in null_before.items()
        if v > 0 and k in ["Rating", "Installs", "Size_MB", "Price_USD", "Type", "Reviews"]
    }
    summary["final_rows"] = len(df)
    summary["dtype_corrections"] = [
        "Installs: string '10,000+' → float",
        "Size: string '19M' → float (MB)",
        "Price: string '$2.99' → float (USD)",
        "Rating: coerced to numeric, out-of-range set to NaN",
        "Reviews: coerced to integer",
    ]

    df = df.reset_index(drop=True)
    return df, summary


# ---------------------------------------------------------------------------
# Apps dataset — analysis
# ---------------------------------------------------------------------------

def category_analysis(df: pd.DataFrame) -> dict:
    """Distribution of apps across categories."""
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    return {
        "categories": counts["Category"].tolist(),
        "counts": counts["Count"].tolist(),
        "most_saturated": counts.head(5)["Category"].tolist(),
        "least_represented": counts.tail(5)["Category"].tolist(),
        "total_categories": int(counts.shape[0]),
    }


def ratings_analysis(df: pd.DataFrame) -> dict:
    """Rating distribution and average rating by category."""
    rated = df[df["Rating"].notna()]

    # Distribution buckets
    bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    hist_counts, _ = np.histogram(rated["Rating"], bins=bins)

    # Average by category
    avg_by_cat = (
        rated.groupby("Category")["Rating"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_by_cat.columns = ["Category", "AvgRating"]

    return {
        "rating_bins": labels,
        "rating_counts": hist_counts.tolist(),
        "overall_mean": round(float(rated["Rating"].mean()), 2),
        "overall_median": round(float(rated["Rating"].median()), 2),
        "avg_by_category": avg_by_cat["Category"].tolist(),
        "avg_by_category_vals": avg_by_cat["AvgRating"].tolist(),
    }


def size_installs_analysis(df: pd.DataFrame) -> dict:
    """Scatter data and Pearson correlation for size vs installs."""
    sub = df[df["Size_MB"].notna() & df["Installs"].notna()].copy()
    sub = sub[sub["Installs"] > 0]  # log scale requires > 0

    corr = float(sub["Size_MB"].corr(sub["Installs"]))

    if abs(corr) < 0.2:
        strength = "very weak"
    elif abs(corr) < 0.4:
        strength = "weak"
    elif abs(corr) < 0.6:
        strength = "moderate"
    elif abs(corr) < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if corr >= 0 else "negative"
    interpretation = (
        f"The Pearson correlation of {corr:.3f} indicates a {strength} {direction} "
        f"relationship between app size and install count. "
        f"This suggests that size {'tends to accompany' if corr > 0 else 'does not strongly predict'} "
        f"higher installs, though correlation does not imply causation."
    )

    # Sample for scatter (max 2000 points for performance)
    sample = sub.sample(min(2000, len(sub)), random_state=42)
    return {
        "size_data": sample["Size_MB"].round(2).tolist(),
        "installs_data": sample["Installs"].tolist(),
        "correlation": round(corr, 4),
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "sample_size": len(sample),
    }


def pricing_analysis(df: pd.DataFrame) -> dict:
    """Free vs paid, price distribution, and revenue estimate by category."""
    type_counts = df["Type"].value_counts()
    free_count = int(type_counts.get("Free", 0))
    paid_count = int(type_counts.get("Paid", 0))

    paid_df = df[df["Type"] == "Paid"].copy()
    paid_df = paid_df[paid_df["Price_USD"] > 0]

    # Price buckets for paid apps
    price_bins = [0, 0.99, 2.99, 4.99, 9.99, 19.99, 999]
    price_labels = ["<$1", "$1–$3", "$3–$5", "$5–$10", "$10–$20", ">$20"]
    paid_df["PriceBucket"] = pd.cut(
        paid_df["Price_USD"], bins=price_bins, labels=price_labels, right=True
    )
    price_dist = paid_df["PriceBucket"].value_counts().sort_index()

    # Revenue estimate per category
    # Formula: Price_USD * Installs (midpoint of install range)
    # Installs column is already numeric (lower bound of range)
    # We use lower bound as conservative estimate and document this clearly
    revenue_df = paid_df[paid_df["Installs"].notna() & paid_df["Category"].notna()].copy()
    revenue_df["RevenueEstimate"] = revenue_df["Price_USD"] * revenue_df["Installs"]
    rev_by_cat = (
        revenue_df.groupby("Category")["RevenueEstimate"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    rev_by_cat.columns = ["Category", "Revenue"]

    return {
        "free_count": free_count,
        "paid_count": paid_count,
        "free_pct": round(free_count / (free_count + paid_count) * 100, 1) if (free_count + paid_count) > 0 else 0,
        "paid_pct": round(paid_count / (free_count + paid_count) * 100, 1) if (free_count + paid_count) > 0 else 0,
        "price_buckets": price_labels,
        "price_bucket_counts": [int(price_dist.get(l, 0)) for l in price_labels],
        "revenue_categories": rev_by_cat["Category"].tolist(),
        "revenue_estimates": rev_by_cat["Revenue"].round(0).tolist(),
        "revenue_disclaimer": (
            "Revenue estimates are indicative only. "
            "Installs represent the lower-bound of reported ranges (e.g. '10,000+' → 10,000). "
            "Actual installs may be significantly higher. "
            "This does not represent real company revenue."
        ),
    }


def developer_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generate at least 3 data-driven insights from the real dataset
    for a developer planning to launch a new app.
    """
    insights = []

    # Insight 1: Best category by rating (min 50 apps)
    cat_stats = df[df["Rating"].notna()].groupby("Category").agg(
        count=("App", "count"),
        avg_rating=("Rating", "mean"),
    ).reset_index()
    top_cat = cat_stats[cat_stats["count"] >= 50].sort_values("avg_rating", ascending=False).iloc[0]
    insights.append({
        "title": "Highest-Rated Category",
        "icon": "⭐",
        "body": (
            f"'{top_cat['Category']}' has the highest average rating of "
            f"{top_cat['avg_rating']:.2f} among categories with at least 50 apps. "
            f"Developers in this space benefit from a receptive audience."
        ),
    })

    # Insight 2: Free vs paid install gap
    inst = df[df["Installs"].notna() & df["Type"].notna()]
    med_installs = inst.groupby("Type")["Installs"].median()
    if "Free" in med_installs and "Paid" in med_installs:
        ratio = med_installs["Free"] / max(med_installs["Paid"], 1)
        insights.append({
            "title": "Free Apps Have Significantly More Installs",
            "icon": "📲",
            "body": (
                f"Median installs for free apps ({int(med_installs['Free']):,}) are "
                f"{ratio:.0f}× higher than paid apps ({int(med_installs['Paid']):,}). "
                f"A freemium or ad-supported model maximises reach."
            ),
        })

    # Insight 3: Optimal size range
    sub = df[df["Size_MB"].notna() & df["Installs"].notna()]
    sub = sub[sub["Installs"] > 0]
    size_buckets = pd.cut(sub["Size_MB"], bins=[0, 10, 25, 50, 100, 999],
                          labels=["<10 MB", "10–25 MB", "25–50 MB", "50–100 MB", ">100 MB"])
    sub = sub.copy()
    sub["SizeBucket"] = size_buckets
    best_size = sub.groupby("SizeBucket", observed=True)["Installs"].median().idxmax()
    insights.append({
        "title": "Optimal App Size",
        "icon": "📦",
        "body": (
            f"Apps in the '{best_size}' range achieve the highest median install count. "
            f"Keeping your app lean while delivering core value can improve conversion from "
            f"listing page to install."
        ),
    })

    # Insight 4: Most saturated vs opportunity gap
    cat_counts = df["Category"].value_counts()
    rated_cats = df[df["Rating"].notna()].groupby("Category")["Rating"].mean()
    opportunity = pd.DataFrame({"count": cat_counts, "avg_rating": rated_cats}).dropna()
    opportunity["score"] = opportunity["avg_rating"] / opportunity["count"]
    opp_cat = opportunity.sort_values("score", ascending=False).index[0]
    insights.append({
        "title": "Opportunity Gap",
        "icon": "🎯",
        "body": (
            f"'{opp_cat}' has a high average rating but relatively few apps, "
            f"suggesting an underserved niche. Entering a less saturated category "
            f"with quality content can yield outsized visibility."
        ),
    })

    # Insight 5: Rating vs reviews relationship
    has_reviews = df[df["Reviews"].notna() & df["Rating"].notna() & (df["Reviews"] > 0)]
    top_reviewed = has_reviews[has_reviews["Reviews"] > has_reviews["Reviews"].quantile(0.75)]
    low_reviewed = has_reviews[has_reviews["Reviews"] <= has_reviews["Reviews"].quantile(0.25)]
    avg_top = top_reviewed["Rating"].mean()
    avg_low = low_reviewed["Rating"].mean()
    insights.append({
        "title": "Review Volume Correlates with Ratings",
        "icon": "💬",
        "body": (
            f"Apps in the top 25% by review count average a rating of {avg_top:.2f}, "
            f"compared to {avg_low:.2f} for the bottom 25%. "
            f"Encouraging user reviews — through in-app prompts at high-satisfaction moments "
            f"— can reinforce a positive ratings loop."
        ),
    })

    return insights
