"""
sentiment_analyzer.py
---------------------
Performs VADER sentiment analysis on Google Play Store user reviews.
The existing 'Sentiment' column in the Kaggle dataset is used ONLY for
validation/comparison — all classifications are produced fresh by VADER.
"""

import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_reviews_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the User Reviews dataset.
    Returns (cleaned_df, summary_dict).
    """
    summary = {}
    summary["original_rows"] = len(df)

    # Drop fully duplicated rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    summary["duplicates_removed"] = before_dedup - len(df)

    # Drop rows where review text is null/empty
    before_null = len(df)
    df = df[df["Translated_Review"].notna()]
    df = df[df["Translated_Review"].astype(str).str.strip().str.lower() != "nan"]
    df = df[df["Translated_Review"].astype(str).str.strip() != ""]
    summary["null_reviews_removed"] = before_null - len(df)

    # Drop rows with no App name
    df = df[df["App"].notna()]

    summary["final_rows"] = len(df)
    df = df.reset_index(drop=True)
    return df, summary


# ---------------------------------------------------------------------------
# VADER sentiment classification
# ---------------------------------------------------------------------------

def _classify_compound(compound: float) -> str:
    """Map VADER compound score to Positive / Negative / Neutral."""
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def run_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER to every review text and add columns:
        vader_compound, vader_sentiment
    Returns the enriched DataFrame.
    """
    texts = df["Translated_Review"].astype(str).tolist()
    compounds = []
    sentiments = []

    for text in texts:
        scores = _analyzer.polarity_scores(text)
        compound = scores["compound"]
        compounds.append(round(compound, 4))
        sentiments.append(_classify_compound(compound))

    df = df.copy()
    df["vader_compound"] = compounds
    df["vader_sentiment"] = sentiments
    return df


# ---------------------------------------------------------------------------
# Aggregate sentiment metrics
# ---------------------------------------------------------------------------

def sentiment_distribution(df: pd.DataFrame) -> dict:
    """Count and percentage breakdown of VADER sentiment labels."""
    counts = df["vader_sentiment"].value_counts()
    total = len(df)

    labels = ["Positive", "Negative", "Neutral"]
    result = {}
    for label in labels:
        c = int(counts.get(label, 0))
        result[label] = {
            "count": c,
            "pct": round(c / total * 100, 1) if total > 0 else 0.0,
        }
    result["total"] = total
    return result


def sentiment_by_category(reviews_df: pd.DataFrame, apps_df: pd.DataFrame) -> dict | None:
    """
    Map reviews to app categories using the Apps dataset's App→Category mapping.
    Returns per-category sentiment breakdown, or None if mapping is not possible.
    """
    if apps_df is None or apps_df.empty:
        return None

    # Build the mapping: App (name) → Category
    cat_map = apps_df[["App", "Category"]].drop_duplicates("App").set_index("App")["Category"]

    reviews_df = reviews_df.copy()
    reviews_df["Category"] = reviews_df["App"].map(cat_map)

    mapped = reviews_df[reviews_df["Category"].notna()]
    if mapped.empty:
        return None

    pivot = (
        mapped.groupby(["Category", "vader_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all three columns exist
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total"] = pivot["Positive"] + pivot["Negative"] + pivot["Neutral"]
    pivot["PositivePct"] = (pivot["Positive"] / pivot["Total"].replace(0, np.nan) * 100).round(1)
    pivot["NegativePct"] = (pivot["Negative"] / pivot["Total"].replace(0, np.nan) * 100).round(1)

    pivot = pivot.sort_values("PositivePct", ascending=False).reset_index(drop=True)

    return {
        "categories": pivot["Category"].tolist(),
        "positive_counts": pivot["Positive"].tolist(),
        "negative_counts": pivot["Negative"].tolist(),
        "neutral_counts": pivot["Neutral"].tolist(),
        "positive_pct": pivot["PositivePct"].fillna(0).tolist(),
        "negative_pct": pivot["NegativePct"].fillna(0).tolist(),
        "most_positive": pivot.iloc[0]["Category"] if len(pivot) > 0 else "N/A",
        "most_negative": pivot.sort_values("NegativePct", ascending=False).iloc[0]["Category"] if len(pivot) > 0 else "N/A",
        "mapped_reviews": int(len(mapped)),
        "unmapped_reviews": int(len(reviews_df) - len(mapped)),
    }


def sentiment_insights(df: pd.DataFrame) -> list[dict]:
    """Return observation-level insights from the sentiment analysis."""
    insights = []

    dist = sentiment_distribution(df)
    pos_pct = dist["Positive"]["pct"]
    neg_pct = dist["Negative"]["pct"]
    neu_pct = dist["Neutral"]["pct"]

    insights.append({
        "title": "Overall Sentiment",
        "icon": "😊",
        "body": (
            f"{pos_pct}% of reviews are positive, {neg_pct}% are negative, "
            f"and {neu_pct}% are neutral. "
            + ("The Play Store user base leans positive overall."
               if pos_pct > 50
               else "A significant portion of users express concern — common in competitive markets.")
        ),
    })

    # Average compound score
    avg_compound = round(float(df["vader_compound"].mean()), 4)
    insights.append({
        "title": "Average Sentiment Score",
        "icon": "📊",
        "body": (
            f"The mean VADER compound score is {avg_compound} (scale: -1 to +1). "
            f"Scores above 0.05 are positive; below -0.05 are negative. "
            f"This dataset's mean is {'above' if avg_compound > 0 else 'below'} neutral."
        ),
    })

    # Most reviewed app sentiment
    top_app = (
        df.groupby("App")
        .agg(count=("vader_sentiment", "count"), pos=("vader_sentiment", lambda x: (x == "Positive").sum()))
        .reset_index()
    )
    top_app["pos_pct"] = top_app["pos"] / top_app["count"]
    top_app_row = top_app.sort_values("count", ascending=False).iloc[0]
    insights.append({
        "title": "Most Reviewed App",
        "icon": "🔍",
        "body": (
            f"'{top_app_row['App']}' has the most reviews in this dataset ({int(top_app_row['count']):,}), "
            f"with {top_app_row['pos_pct']*100:.1f}% classified as positive by VADER."
        ),
    })

    return insights
