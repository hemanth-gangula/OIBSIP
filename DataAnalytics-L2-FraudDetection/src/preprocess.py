"""
preprocess.py
-------------
Data loading, cleaning, feature engineering, scaling,
stratified train/test split, and SMOTE oversampling.

All operations are performed on the real creditcard.csv dataset.
SMOTE is applied ONLY to training data to prevent data leakage.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# -- Paths -------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -- 1. Load dataset ---------------------------------------------------------
def load_data(path=DATA_PATH):
    """Load the Credit Card Fraud dataset and print inspection summary."""
    print("Loading dataset from: " + path)
    df = pd.read_csv(path)

    print("\n-- Dataset Inspection ------------------------------------------")
    print("  Shape          : {:,} rows x {:} columns".format(df.shape[0], df.shape[1]))
    print("  Missing values : {:}".format(df.isnull().sum().sum()))
    print("  Data types     : {:}".format(df.dtypes.value_counts().to_dict()))

    vc = df["Class"].value_counts()
    n_total = len(df)
    print("\n-- Class Distribution ------------------------------------------")
    print("  Legitimate (0) : {:,}  ({:.4f}%)".format(vc[0], vc[0]/n_total*100))
    print("  Fraudulent (1) : {:,}  ({:.4f}%)".format(vc[1], vc[1]/n_total*100))
    print("  Imbalance ratio: 1:{:}".format(int(vc[0]/vc[1])))

    return df


# -- 2. Feature engineering --------------------------------------------------
def engineer_features(df):
    """
    Derive time-of-day features from the raw Time column.
    Time is in seconds elapsed since the first transaction.
    The dataset covers ~48 hours so we map it to hour-of-day (0-23).
    """
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600 % 24).astype(int)
    return df


# -- 3. Split features / target ----------------------------------------------
def split_features_target(df):
    """
    Drop Time (raw seconds; we keep the derived Hour) and return X, y.
    Amount and Hour are retained as-is; scaling happens next.
    """
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[feature_cols].copy()
    y = df["Class"].copy()
    return X, y, feature_cols


# -- 4. Stratified train/test split ------------------------------------------
def make_split(X, y, test_size=0.20, random_state=42):
    """
    Stratified split preserves the original class ratio in both sets.
    test_size=0.20 -> 20% of the data held out for final evaluation.
    SMOTE is NEVER applied to the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print("\n-- Train / Test Split ------------------------------------------")
    print("  Train size : {:,}  (fraud={:,})".format(len(X_train), y_train.sum()))
    print("  Test  size : {:,}  (fraud={:,})".format(len(X_test), y_test.sum()))
    return X_train, X_test, y_train, y_test


# -- 5. Scale Amount and Hour ------------------------------------------------
def scale_features(X_train, X_test, feature_cols):
    """
    StandardScaler is fit ONLY on training data, then applied to both sets.
    This prevents any test-set information leaking into the scaler parameters.
    Only Amount and Hour are scaled -- V1-V28 are already standardised.
    """
    cols_to_scale = ["Amount", "Hour"]
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print("\n  Scaler saved -> " + scaler_path)

    return X_train, X_test, scaler


# -- 6. SMOTE oversampling (training data only) ------------------------------
def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE (Synthetic Minority Oversampling TEchnique) creates synthetic
    fraud samples in the TRAINING set only.

    WHY SMOTE:
    The dataset has a 1:577 imbalance. Training on raw data causes the model
    to predict 'legitimate' for every transaction and still achieve 99.83%
    accuracy -- a numerically impressive but useless model. SMOTE rebalances
    the training distribution so the model learns genuine fraud patterns.

    WHY NOT ON TEST SET:
    The test set must represent real-world distribution. Oversampling it
    would inflate recall artificially and produce optimistic metrics that
    would never hold in production.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    before = y_train.value_counts()
    after  = pd.Series(y_resampled).value_counts()
    print("\n-- SMOTE Oversampling (training only) --------------------------")
    print("  Before -> Legit: {:,}  | Fraud: {:,}".format(before[0], before[1]))
    print("  After  -> Legit: {:,}  | Fraud: {:,}".format(after[0], after[1]))
    print("  Total training samples after SMOTE: {:,}".format(len(X_resampled)))

    return X_resampled, y_resampled


# -- 7. Full pipeline --------------------------------------------------------
def run_preprocessing(path=DATA_PATH):
    """
    End-to-end preprocessing pipeline.
    Returns:
        X_train_sm, y_train_sm  -- SMOTE-balanced training data
        X_test,     y_test      -- unmodified test data (real distribution)
        feature_cols            -- ordered list of feature column names
        scaler                  -- fitted StandardScaler
    """
    df               = load_data(path)
    df               = engineer_features(df)
    X, y, feat_cols  = split_features_target(df)
    X_tr, X_te, y_tr, y_te = make_split(X, y)
    X_tr, X_te, scaler      = scale_features(X_tr, X_te, feat_cols)
    X_tr_sm, y_tr_sm        = apply_smote(X_tr, y_tr)

    # Save feature column list for consistent API input ordering
    feat_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print("\n  Feature list saved -> " + feat_path)

    # Save class stats for the web app dashboard
    vc = df["Class"].value_counts()
    stats = {
        "total_transactions":  int(len(df)),
        "legitimate_count":    int(vc[0]),
        "fraud_count":         int(vc[1]),
        "fraud_pct":           round(float(vc[1] / len(df) * 100), 4),
        "legit_pct":           round(float(vc[0] / len(df) * 100), 4),
        "imbalance_ratio":     int(vc[0] / vc[1]),
        "time_span_hours":     48,
        "fraud_mean_amount":   round(float(df[df["Class"]==1]["Amount"].mean()), 2),
        "fraud_max_amount":    round(float(df[df["Class"]==1]["Amount"].max()), 2),
        "fraud_median_amount": round(float(df[df["Class"]==1]["Amount"].median()), 2),
        "legit_mean_amount":   round(float(df[df["Class"]==0]["Amount"].mean()), 2),
        "legit_max_amount":    round(float(df[df["Class"]==0]["Amount"].max()), 2),
        "legit_median_amount": round(float(df[df["Class"]==0]["Amount"].median()), 2),
    }
    stats_path = os.path.join(MODEL_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  Dataset stats saved -> " + stats_path)
    print("\n-- Preprocessing complete --------------------------------------\n")

    return X_tr_sm, y_tr_sm, X_te, y_te, feat_cols, scaler


# -- CLI entry point ---------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
