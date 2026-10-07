"""
data_processor.py
-----------------
Handles all data loading, validation, cleaning, and analysis for
the Google Play Store Analytics application.
"""

import pandas as pd
import numpy as np
import re
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
APPS_REQUIRED_COLS = ['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price']
REVIEWS_REQUIRED_COLS = ['App', 'Translated_Review', 'Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity']

PALETTE = {
    'primary':   '#6C63FF',
    'secondary': '#3ECFCF',
    'accent':    '#FF6584',
    'bg':        '#0F0F1A',
    'card':      '#1A1A2E',
    'text':      '#E0E0E0',
    'positive':  '#2ECC71',
    'negative':  '#E74C3C',
    'neutral':   '#F39C12',
}

# ─────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────

def load_dataframe(file_storage):
    """Read a Flask FileStorage object into a pandas DataFrame (CSV or Excel)."""
    filename = file_storage.filename.lower()
    try:
        if filename.endswith('.csv'):
            content = file_storage.read()
            # Try common encodings
            for enc in ('utf-8', 'latin-1', 'cp1252'):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    return df, None
                except UnicodeDecodeError:
                    continue
            return None, "Could not decode CSV file. Try saving as UTF-8."
        elif filename.endswith(('.xlsx', '.xls')):
            content = file_storage.read()
            engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
            df = pd.read_excel(io.BytesIO(content), engine=engine)
            return df, None
        else:
            return None, f"Unsupported file format: {filename}. Use .csv, .xlsx, or .xls"
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_apps_dataset(df):
    """Validate that the Apps dataset contains required columns."""
    missing = [c for c in APPS_REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "Apps dataset validated successfully."


def validate_reviews_dataset(df):
    """Validate that the Reviews dataset contains required columns."""
    missing = [c for c in REVIEWS_REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "User Reviews dataset validated successfully."


def get_dataset_info(df):
    """Return basic metadata about a DataFrame."""
    preview = df.head(5).fillna('').astype(str).to_dict(orient='records')
    return {
        'rows': int(len(df)),
        'cols': int(len(df.columns)),
        'columns': list(df.columns),
        'preview': preview,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


# ─────────────────────────────────────────────
# CLEANING – APPS DATASET
# ─────────────────────────────────────────────

def _clean_size(val):
    """Convert size string like '19M', '1.5k' to float MB."""
    if pd.isna(val) or str(val).strip() in ('', 'Varies with device', 'nan'):
        return np.nan
    val = str(val).strip()
    if val.endswith('M') or val.endswith('m'):
        try:
            return float(val[:-1])
        except ValueError:
            return np.nan
    if val.endswith('k') or val.endswith('K'):
        try:
            return float(val[:-1]) / 1024
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def _clean_installs(val):
    """Convert installs string like '10,000+' to integer."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace(',', '').replace('+', '').replace(' ', '').strip()
    if val in ('', 'nan', 'Free'):
        return np.nan
    try:
        return int(float(val))
    except ValueError:
        return np.nan


def _clean_price(val):
    """Convert price string like '$4.99' to float."""
    if pd.isna(val):
        return 0.0
    val = str(val).replace('$', '').replace(' ', '').strip()
    if val in ('', 'nan', '0'):
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def clean_apps_dataset(df):
    """
    Full cleaning pipeline for the Apps dataset.
    Returns (cleaned_df, cleaning_log dict).
    """
    log = {}
    original_rows = len(df)
    df = df.copy()

    # 1. Remove duplicates
    df.drop_duplicates(subset=['App'], keep='first', inplace=True)
    log['duplicates_removed'] = original_rows - len(df)

    # 2. Drop rows where App name is null
    df.dropna(subset=['App'], inplace=True)
    log['null_app_names_dropped'] = original_rows - log['duplicates_removed'] - len(df)

    # 3. Clean Size → numeric MB
    df['Size_MB'] = df['Size'].apply(_clean_size)
    log['size_nulls_after_clean'] = int(df['Size_MB'].isna().sum())

    # 4. Clean Installs → integer
    df['Installs_Numeric'] = df['Installs'].apply(_clean_installs)
    log['installs_nulls_after_clean'] = int(df['Installs_Numeric'].isna().sum())

    # 5. Clean Price → float
    df['Price_Numeric'] = df['Price'].apply(_clean_price)
    log['price_nulls_after_clean'] = int(df['Price_Numeric'].isna().sum())

    # 6. Clean Rating → numeric, drop out-of-range
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    invalid_ratings = ((df['Rating'] < 1) | (df['Rating'] > 5)).sum()
    df.loc[(df['Rating'] < 1) | (df['Rating'] > 5), 'Rating'] = np.nan
    log['invalid_ratings_nulled'] = int(invalid_ratings)

    # 7. Clean Reviews → numeric
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    log['reviews_nulls'] = int(df['Reviews'].isna().sum())

    # 8. Normalise Type column
    df['Type'] = df['Type'].astype(str).str.strip()
    df.loc[~df['Type'].isin(['Free', 'Paid']), 'Type'] = np.nan

    # 9. Normalise Category
    df['Category'] = df['Category'].astype(str).str.strip().str.upper()
    df = df[df['Category'] != 'NAN']

    log['final_rows'] = int(len(df))
    log['columns_added'] = ['Size_MB', 'Installs_Numeric', 'Price_Numeric']

    return df, log


# ─────────────────────────────────────────────
# CLEANING – REVIEWS DATASET
# ─────────────────────────────────────────────

def clean_reviews_dataset(df):
    """
    Full cleaning pipeline for the User Reviews dataset.
    Returns (cleaned_df, cleaning_log dict).
    """
    log = {}
    original_rows = len(df)
    df = df.copy()

    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)
    log['duplicates_removed'] = original_rows - len(df)

    # 2. Drop null reviews
    df.dropna(subset=['Translated_Review'], inplace=True)
    log['null_reviews_dropped'] = original_rows - log['duplicates_removed'] - len(df)

    # 3. Numeric columns
    df['Sentiment_Polarity']    = pd.to_numeric(df['Sentiment_Polarity'],    errors='coerce')
    df['Sentiment_Subjectivity']= pd.to_numeric(df['Sentiment_Subjectivity'],errors='coerce')

    # 4. Normalise Sentiment labels
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.capitalize()
    df.loc[~df['Sentiment'].isin(['Positive', 'Negative', 'Neutral']), 'Sentiment'] = 'Neutral'

    log['final_rows'] = int(len(df))
    return df, log


# ─────────────────────────────────────────────
# ANALYSIS – CATEGORY
# ─────────────────────────────────────────────

def analyse_categories(df):
    counts = (
        df['Category']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Category', 'count': 'App_Count', 'Category': 'Category'})
    )
    # pandas ≥ 2.0 value_counts already names the column 'count'
    if 'count' in counts.columns:
        counts = counts.rename(columns={'count': 'App_Count'})
    if 'Category' not in counts.columns:
        counts.columns = ['Category', 'App_Count']

    top5 = counts.head(5)['Category'].tolist()
    return {
        'category_counts': counts.to_dict(orient='records'),
        'top5_saturated': top5,
        'total_categories': int(counts.shape[0]),
    }


# ─────────────────────────────────────────────
# ANALYSIS – RATINGS
# ─────────────────────────────────────────────

def analyse_ratings(df):
    rated = df.dropna(subset=['Rating'])
    avg_overall = round(float(rated['Rating'].mean()), 3)

    avg_by_cat = (
        rated.groupby('Category')['Rating']
        .mean()
        .round(3)
        .reset_index()
        .sort_values('Rating', ascending=False)
        .rename(columns={'Rating': 'Avg_Rating'})
    )

    dist_bins  = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    dist_labels= ['1–2', '2–3', '3–4', '4–4.5', '4.5–5']
    dist = pd.cut(rated['Rating'], bins=dist_bins, labels=dist_labels, include_lowest=True)
    dist_counts = dist.value_counts().sort_index().reset_index()
    dist_counts.columns = ['Range', 'Count']

    return {
        'avg_overall': avg_overall,
        'total_rated': int(len(rated)),
        'avg_by_category': avg_by_cat.to_dict(orient='records'),
        'distribution': dist_counts.to_dict(orient='records'),
    }


# ─────────────────────────────────────────────
# ANALYSIS – SIZE & INSTALLS
# ─────────────────────────────────────────────

def analyse_size_installs(df):
    sub = df.dropna(subset=['Size_MB', 'Installs_Numeric']).copy()
    sub = sub[sub['Installs_Numeric'] > 0]

    if len(sub) < 2:
        return {'correlation': None, 'note': 'Insufficient data for correlation.', 'sample': []}

    corr = round(float(sub['Size_MB'].corr(sub['Installs_Numeric'])), 4)

    # Sample up to 500 points for scatter
    sample = sub[['App', 'Size_MB', 'Installs_Numeric', 'Category']].sample(
        min(500, len(sub)), random_state=42
    )

    if corr > 0.3:
        note = "Positive correlation: larger apps tend to have more installs."
    elif corr < -0.3:
        note = "Negative correlation: smaller apps tend to have more installs."
    else:
        note = "Weak correlation: app size has little direct influence on install count."

    return {
        'correlation': corr,
        'note': note,
        'scatter_data': sample.to_dict(orient='records'),
        'total_points': int(len(sub)),
    }


# ─────────────────────────────────────────────
# ANALYSIS – PRICING
# ─────────────────────────────────────────────

def analyse_pricing(df):
    free_count = int((df['Type'] == 'Free').sum())
    paid_count = int((df['Type'] == 'Paid').sum())
    total      = free_count + paid_count or 1

    paid_df = df[df['Type'] == 'Paid'].copy()
    paid_df = paid_df.dropna(subset=['Price_Numeric'])
    paid_df = paid_df[paid_df['Price_Numeric'] > 0]

    # Price distribution bins
    bins   = [0, 1, 2, 5, 10, 50, 500]
    labels = ['$0–1', '$1–2', '$2–5', '$5–10', '$10–50', '$50+']
    paid_df['Price_Range'] = pd.cut(paid_df['Price_Numeric'], bins=bins, labels=labels, include_lowest=True)
    price_dist = paid_df['Price_Range'].value_counts().sort_index().reset_index()
    price_dist.columns = ['Range', 'Count']

    # Revenue estimation: Price × Installs (assumption: each install = one purchase)
    paid_df['Est_Revenue'] = paid_df['Price_Numeric'] * paid_df['Installs_Numeric'].fillna(0)
    rev_by_cat = (
        paid_df.groupby('Category')['Est_Revenue']
        .sum()
        .reset_index()
        .sort_values('Est_Revenue', ascending=False)
        .head(10)
    )
    rev_by_cat['Est_Revenue'] = rev_by_cat['Est_Revenue'].round(0).astype(int)

    return {
        'free_count':  free_count,
        'paid_count':  paid_count,
        'free_pct':    round(free_count / total * 100, 1),
        'paid_pct':    round(paid_count / total * 100, 1),
        'price_distribution': price_dist.to_dict(orient='records'),
        'revenue_by_category': rev_by_cat.to_dict(orient='records'),
        'revenue_formula': (
            "Estimated Revenue = Price × Install Count  "
            "(Assumption: each install represents one purchase. "
            "This is a rough upper-bound estimate, not actual company revenue.)"
        ),
    }


# ─────────────────────────────────────────────
# ANALYSIS – SENTIMENT BY CATEGORY
# ─────────────────────────────────────────────

def analyse_sentiment_by_category(apps_df, reviews_df):
    """Merge apps and reviews on 'App', then aggregate sentiment per category."""
    merged = reviews_df.merge(
        apps_df[['App', 'Category']].drop_duplicates(subset=['App']),
        on='App', how='inner'
    )

    if merged.empty:
        return {
            'merged_rows': 0,
            'by_category': [],
            'most_positive': [],
            'most_negative': [],
            'note': 'No matching apps found between datasets.',
        }

    pivot = (
        merged.groupby(['Category', 'Sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot['Total']       = pivot['Positive'] + pivot['Negative'] + pivot['Neutral']
    pivot['Positive_Pct']= (pivot['Positive'] / pivot['Total'].replace(0, 1) * 100).round(1)
    pivot['Negative_Pct']= (pivot['Negative'] / pivot['Total'].replace(0, 1) * 100).round(1)

    pivot_sorted_pos = pivot.sort_values('Positive_Pct', ascending=False)
    pivot_sorted_neg = pivot.sort_values('Negative_Pct', ascending=False)

    return {
        'merged_rows': int(len(merged)),
        'by_category': pivot.sort_values('Total', ascending=False).head(20).to_dict(orient='records'),
        'most_positive': pivot_sorted_pos.head(5)[['Category', 'Positive_Pct']].to_dict(orient='records'),
        'most_negative': pivot_sorted_neg.head(5)[['Category', 'Negative_Pct']].to_dict(orient='records'),
    }


# ─────────────────────────────────────────────
# DEVELOPER INSIGHTS
# ─────────────────────────────────────────────

def generate_developer_insights(apps_df, reviews_df, pricing_data, rating_data, category_data):
    """Generate at least 3 data-driven insights from the actual uploaded data."""
    insights = []

    # ── Insight 1: Best category to enter ──
    try:
        avg_by_cat = pd.DataFrame(rating_data['avg_by_category'])
        cat_counts = pd.DataFrame(category_data['category_counts'])
        cat_counts.columns = ['Category', 'App_Count']
        merged = avg_by_cat.merge(cat_counts, on='Category')
        # Balance: high rating, moderate competition
        merged['Score'] = merged['Avg_Rating'] - (merged['App_Count'] / merged['App_Count'].max()) * 0.5
        best = merged.sort_values('Score', ascending=False).iloc[0]
        insights.append({
            'title': '🎯 Best Category to Enter',
            'body': (
                f"<strong>{best['Category']}</strong> offers a strong balance of high average rating "
                f"({best['Avg_Rating']:.2f}★) with manageable competition ({int(best['App_Count'])} apps). "
                f"Apps in this category receive positive user reception, making it a strategic entry point."
            ),
        })
    except Exception:
        pass

    # ── Insight 2: Free vs Paid monetisation ──
    try:
        insights.append({
            'title': '💰 Free vs Paid Monetisation Strategy',
            'body': (
                f"<strong>{pricing_data['free_pct']}%</strong> of apps on the Play Store are free, "
                f"vs only <strong>{pricing_data['paid_pct']}%</strong> paid. "
                f"Given the dominance of free apps, a freemium model or in-app purchase strategy "
                f"is likely to achieve higher install volume than a paid upfront approach. "
                f"Reserve paid pricing for niche productivity or utility apps with clear unique value."
            ),
        })
    except Exception:
        pass

    # ── Insight 3: Optimal app size ──
    try:
        sub = apps_df.dropna(subset=['Size_MB', 'Installs_Numeric']).copy()
        sub = sub[sub['Installs_Numeric'] > 0]
        q75 = sub['Installs_Numeric'].quantile(0.75)
        top_installs = sub[sub['Installs_Numeric'] >= q75]
        median_size = round(float(top_installs['Size_MB'].median()), 1)
        insights.append({
            'title': '📦 Optimal App Size for Maximum Installs',
            'body': (
                f"Apps in the top 25% of installs have a median size of "
                f"<strong>{median_size} MB</strong>. "
                f"Keeping your app lean near this target helps avoid install friction, "
                f"especially for users on limited mobile data or lower-end devices."
            ),
        })
    except Exception:
        pass

    # ── Insight 4: Rating threshold ──
    try:
        avg = rating_data['avg_overall']
        insights.append({
            'title': '⭐ Rating Benchmark to Beat',
            'body': (
                f"The overall average Play Store rating in this dataset is <strong>{avg}★</strong>. "
                f"To stand out, target a minimum launch rating of <strong>{round(avg + 0.2, 1)}★</strong> "
                f"by prioritising UX polish, responding to early reviews, and resolving critical bugs "
                f"before your public launch."
            ),
        })
    except Exception:
        pass

    # ── Insight 5: Sentiment-driven category ──
    try:
        reviews_clean = reviews_df.copy()
        merged = reviews_clean.merge(
            apps_df[['App', 'Category']].drop_duplicates(subset=['App']),
            on='App', how='inner'
        )
        if not merged.empty:
            pos_rate = (
                merged.groupby('Category')
                .apply(lambda x: (x['Sentiment'] == 'Positive').sum() / max(len(x), 1))
                .sort_values(ascending=False)
            )
            top_cat = pos_rate.index[0]
            top_pct = round(float(pos_rate.iloc[0]) * 100, 1)
            insights.append({
                'title': '😊 Highest User Satisfaction Category',
                'body': (
                    f"<strong>{top_cat}</strong> has the highest proportion of positive user reviews "
                    f"(<strong>{top_pct}%</strong> positive sentiment). "
                    f"Launching in this category means entering a market where users are generally happy "
                    f"and more likely to leave good ratings organically."
                ),
            })
    except Exception:
        pass

    return insights[:5]  # Return up to 5 insights
