"""
sentiment_analyzer.py
---------------------
Sentiment analysis for the Google Play Store User Reviews dataset.

Strategy (fast & accurate):
  1. The real Kaggle dataset already has Sentiment labels (Positive/Negative/Neutral).
     If ≥ 80 % of rows carry valid labels we use them directly (zero NLP cost).
  2. For any rows with missing/invalid labels we fill them via VADER compound score
     derived from the pre-computed Sentiment_Polarity column (also in the dataset)
     — no per-row NLP inference needed, so the pipeline stays sub-second.
  3. If the polarity column is also missing we fall back to VADER on a capped
     sample (≤ 5 000 rows) so the route never times out.
"""

import pandas as pd
import numpy as np

# Optional NLP imports — used only for fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _label_from_polarity(score):
    """Map a numeric polarity score → Positive / Negative / Neutral."""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def _vader_single(text):
    score = _vader.polarity_scores(str(text))['compound']
    return _label_from_polarity(score), round(score, 4)


def _textblob_single(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    return _label_from_polarity(polarity), round(polarity, 4)


# ─────────────────────────────────────────────
# MAIN API
# ─────────────────────────────────────────────

def analyse_sentiment(reviews_df):
    """
    Analyse sentiment for the full reviews DataFrame.
    Returns a dict (no DataFrames) PLUS 'enriched_df' key with NLP columns added.
    """
    df = reviews_df.copy()

    # ── Step 1: check existing label coverage ──────────────────────────────
    valid_labels = {'Positive', 'Negative', 'Neutral'}
    existing_mask = df['Sentiment'].isin(valid_labels)
    coverage = existing_mask.mean()          # fraction with valid labels

    # ── Step 2: fill NLP_Score from Sentiment_Polarity (already numeric) ───
    if 'Sentiment_Polarity' in df.columns:
        df['NLP_Score'] = pd.to_numeric(df['Sentiment_Polarity'], errors='coerce').fillna(0.0)
        score_source = 'Sentiment_Polarity column'
    else:
        df['NLP_Score'] = 0.0
        score_source = 'default (no polarity column)'

    # ── Step 3: decide primary sentiment column ────────────────────────────
    if coverage >= 0.80:
        # Dataset labels are reliable – use them directly
        df['NLP_Sentiment'] = df['Sentiment'].where(existing_mask,
                              df['NLP_Score'].apply(_label_from_polarity))
        primary_col  = 'NLP_Sentiment'
        label_source = f'Dataset labels ({coverage*100:.1f}% valid) + polarity fallback'
        engine_used  = 'Dataset labels (pre-labeled)'
    else:
        # Labels unreliable – derive from polarity column if available
        if 'Sentiment_Polarity' in df.columns and df['NLP_Score'].abs().max() > 0:
            df['NLP_Sentiment'] = df['NLP_Score'].apply(_label_from_polarity)
            primary_col  = 'NLP_Sentiment'
            label_source = f'Computed from Sentiment_Polarity (coverage was {coverage*100:.1f}%)'
            engine_used  = 'Sentiment_Polarity score'
        else:
            # Last resort: run VADER/TextBlob on capped sample
            CAP = 5000
            sample_idx = df.index[:CAP]
            labels = []
            scores = []
            if VADER_AVAILABLE:
                for txt in df.loc[sample_idx, 'Translated_Review'].astype(str):
                    lbl, sc = _vader_single(txt)
                    labels.append(lbl); scores.append(sc)
                engine_used = 'VADER'
            elif TEXTBLOB_AVAILABLE:
                for txt in df.loc[sample_idx, 'Translated_Review'].astype(str):
                    lbl, sc = _textblob_single(txt)
                    labels.append(lbl); scores.append(sc)
                engine_used = 'TextBlob'
            else:
                labels = ['Neutral'] * len(sample_idx)
                scores = [0.0]      * len(sample_idx)
                engine_used = 'none (fallback neutral)'

            df['NLP_Sentiment'] = 'Neutral'
            df['NLP_Score']     = 0.0
            df.loc[sample_idx, 'NLP_Sentiment'] = labels
            df.loc[sample_idx, 'NLP_Score']     = scores
            # Fill rest from existing labels where available
            rest_mask = ~df.index.isin(sample_idx) & existing_mask
            df.loc[rest_mask, 'NLP_Sentiment'] = df.loc[rest_mask, 'Sentiment']
            primary_col  = 'NLP_Sentiment'
            label_source = f'VADER on first {CAP} rows + dataset labels for rest'

    # ── Step 4: counts & percentages ──────────────────────────────────────
    counts  = df[primary_col].value_counts()
    total   = max(len(df), 1)

    pos = int(counts.get('Positive', 0))
    neg = int(counts.get('Negative', 0))
    neu = int(counts.get('Neutral',  0))

    avg_polarity = (
        df.groupby(primary_col)['NLP_Score']
        .mean().round(4).to_dict()
    )

    return {
        'engine_used':    engine_used,
        'label_source':   label_source,
        'primary_col':    primary_col,
        'total_reviews':  total,
        'positive_count': pos,
        'negative_count': neg,
        'neutral_count':  neu,
        'positive_pct':   round(pos / total * 100, 1),
        'negative_pct':   round(neg / total * 100, 1),
        'neutral_pct':    round(neu / total * 100, 1),
        'avg_polarity':   avg_polarity,
        'enriched_df':    df,       # carries NLP_Sentiment + NLP_Score
    }
