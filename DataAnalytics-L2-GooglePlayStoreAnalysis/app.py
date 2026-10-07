"""
app.py
------
Flask application — Google Play Store Analytics
Oasis Infobyte | Data Analytics | Level 2 | Task 4

Two separate upload flows:
  /              → Landing page (two upload buttons)
  /upload-apps   → Validate, clean, analyse Apps dataset → Apps dashboard
  /upload-reviews→ Validate, clean, analyse Reviews dataset → Reviews dashboard

All dashboard charts are served as Plotly JSON (fully interactive).
matplotlib/seaborn are retained in visualizations.py for Jupyter Notebook use only.
"""

import os
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash,
)
from werkzeug.utils import secure_filename

import pandas as pd

from data_processor import (
    load_file,
    validate_apps_dataset,
    validate_reviews_dataset,
    clean_apps_dataset,
    category_analysis,
    ratings_analysis,
    size_installs_analysis,
    pricing_analysis,
    developer_insights,
)
from sentiment_analyzer import (
    clean_reviews_dataset,
    run_vader_sentiment,
    sentiment_distribution,
    sentiment_by_category,
    sentiment_insights,
)
from visualizations import (
    # Apps dashboard — 7 Plotly charts
    chart_category_bar,
    chart_rating_distribution,
    chart_avg_rating_by_category,
    chart_size_vs_installs,
    chart_free_vs_paid,
    chart_price_distribution,
    chart_revenue_by_category,
    # Reviews dashboard — 4 Plotly charts
    chart_sentiment_donut,
    chart_sentiment_bar,
    chart_compound_histogram,
    chart_sentiment_by_category,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_CONTENT_MB = 50

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "gplay-oasis-analytics-2024-task4")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


@app.template_filter("format_number")
def format_number(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return value


# In-memory cache — Apps df is stored so Reviews dashboard can map categories
_cache: dict = {"apps_df": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save(file_storage) -> str:
    dest = UPLOAD_FOLDER / secure_filename(file_storage.filename)
    file_storage.save(str(dest))
    return str(dest)


def _err(msg: str):
    flash(msg, "error")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ── Apps upload ─────────────────────────────────────────────────────────────

@app.route("/upload-apps", methods=["POST"])
def upload_apps():
    if "apps_file" not in request.files:
        return _err("No file attached. Please select the Apps dataset CSV.")
    f = request.files["apps_file"]
    if f.filename == "":
        return _err("No file selected.")
    if not _allowed(f.filename):
        return _err("Unsupported file type. Please upload a CSV or Excel file.")

    try:
        path = _save(f)
    except Exception as e:
        return _err(f"Upload failed: {e}")

    try:
        df_raw = load_file(path)
    except Exception as e:
        return _err(f"Could not read file: {e}")

    if df_raw.empty:
        return _err("The uploaded file is empty.")

    v = validate_apps_dataset(df_raw)
    if not v["valid"]:
        return _err(
            f"This doesn't look like the Google Play Store Apps dataset. "
            f"Missing columns: {', '.join(v['missing'])}. "
            "Please upload googleplaystore.csv."
        )

    try:
        df_clean, clean_summary = clean_apps_dataset(df_raw)
    except Exception as e:
        return _err(f"Data cleaning failed: {e}")

    _cache["apps_df"] = df_clean

    try:
        cat_data   = category_analysis(df_clean)
        rat_data   = ratings_analysis(df_clean)
        size_data  = size_installs_analysis(df_clean)
        price_data = pricing_analysis(df_clean)
        insights   = developer_insights(df_clean)
    except Exception as e:
        return _err(f"Analysis error: {e}")

    try:
        p_category    = chart_category_bar(cat_data)
        p_rating_dist = chart_rating_distribution(rat_data)
        p_avg_rating  = chart_avg_rating_by_category(rat_data)
        p_scatter     = chart_size_vs_installs(size_data)
        p_free_paid   = chart_free_vs_paid(price_data)
        p_price_dist  = chart_price_distribution(price_data)
        p_revenue     = chart_revenue_by_category(price_data)
    except Exception as e:
        return _err(f"Chart generation failed: {e}")

    kpis = {
        "total_apps":       clean_summary["final_rows"],
        "total_categories": cat_data["total_categories"],
        "avg_rating":       rat_data["overall_mean"],
        "free_pct":         price_data["free_pct"],
        "paid_pct":         price_data["paid_pct"],
        "free_count":       price_data["free_count"],
        "paid_count":       price_data["paid_count"],
    }

    return render_template(
        "apps_dashboard.html",
        clean_summary  = clean_summary,
        kpis           = kpis,
        cat_data       = cat_data,
        rat_data       = rat_data,
        size_data      = size_data,
        price_data     = price_data,
        insights       = insights,
        # All charts as Plotly JSON
        p_category    = p_category,
        p_rating_dist = p_rating_dist,
        p_avg_rating  = p_avg_rating,
        p_scatter     = p_scatter,
        p_free_paid   = p_free_paid,
        p_price_dist  = p_price_dist,
        p_revenue     = p_revenue,
    )


# ── Reviews upload ───────────────────────────────────────────────────────────

@app.route("/upload-reviews", methods=["POST"])
def upload_reviews():
    if "reviews_file" not in request.files:
        return _err("No file attached. Please select the Reviews dataset CSV.")
    f = request.files["reviews_file"]
    if f.filename == "":
        return _err("No file selected.")
    if not _allowed(f.filename):
        return _err("Unsupported file type. Please upload a CSV or Excel file.")

    try:
        path = _save(f)
    except Exception as e:
        return _err(f"Upload failed: {e}")

    try:
        df_raw = load_file(path)
    except Exception as e:
        return _err(f"Could not read file: {e}")

    if df_raw.empty:
        return _err("The uploaded file is empty.")

    v = validate_reviews_dataset(df_raw)
    if not v["valid"]:
        return _err(
            f"This doesn't look like the Google Play Store User Reviews dataset. "
            f"Missing columns: {', '.join(v['missing'])}. "
            "Please upload googleplaystore_user_reviews.csv."
        )

    try:
        df_clean, clean_summary = clean_reviews_dataset(df_raw)
    except Exception as e:
        return _err(f"Data cleaning failed: {e}")

    if len(df_clean) == 0:
        return _err("No valid reviews remain after cleaning. Please check your file.")

    try:
        df_sent = run_vader_sentiment(df_clean)
    except Exception as e:
        return _err(f"Sentiment analysis failed: {e}")

    try:
        dist      = sentiment_distribution(df_sent)
        apps_df   = _cache.get("apps_df")
        cat_data  = sentiment_by_category(df_sent, apps_df)
        s_insights = sentiment_insights(df_sent)
    except Exception as e:
        return _err(f"Sentiment aggregation error: {e}")

    try:
        p_sent_donut   = chart_sentiment_donut(dist)
        p_sent_bar     = chart_sentiment_bar(dist)
        p_compound     = chart_compound_histogram(df_sent)
        p_sent_by_cat  = chart_sentiment_by_category(cat_data) if cat_data else None
    except Exception as e:
        return _err(f"Chart generation failed: {e}")

    # Validation: VADER vs original Sentiment column
    validation_comparison = None
    if "Sentiment" in df_clean.columns:
        merged = df_sent[["vader_sentiment", "Sentiment"]].copy()
        merged["Sentiment"] = merged["Sentiment"].str.strip().str.capitalize()
        merged = merged[merged["Sentiment"].isin(["Positive", "Negative", "Neutral"])]
        if len(merged) > 0:
            agreement = (merged["vader_sentiment"] == merged["Sentiment"]).mean()
            validation_comparison = {
                "agreement_pct": round(float(agreement) * 100, 1),
                "note": (
                    "VADER classifications were compared against the dataset's original "
                    "Sentiment column as a validation check. Agreement does not affect "
                    "the VADER-derived results shown in this dashboard."
                ),
            }

    kpis = {
        "total_reviews":  clean_summary["final_rows"],
        "positive_count": dist["Positive"]["count"],
        "negative_count": dist["Negative"]["count"],
        "neutral_count":  dist["Neutral"]["count"],
        "positive_pct":   dist["Positive"]["pct"],
        "negative_pct":   dist["Negative"]["pct"],
        "neutral_pct":    dist["Neutral"]["pct"],
        "avg_compound":   round(float(df_sent["vader_compound"].mean()), 4),
        "apps_linked":    cat_data is not None,
    }

    return render_template(
        "reviews_dashboard.html",
        clean_summary         = clean_summary,
        kpis                  = kpis,
        dist                  = dist,
        cat_data              = cat_data,
        s_insights            = s_insights,
        validation_comparison = validation_comparison,
        # All charts as Plotly JSON
        p_sent_donut   = p_sent_donut,
        p_sent_bar     = p_sent_bar,
        p_compound     = p_compound,
        p_sent_by_cat  = p_sent_by_cat,
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(_):
    flash(f"File too large. Maximum is {MAX_CONTENT_MB} MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(_):
    flash("An unexpected server error occurred. Please try again.", "error")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
