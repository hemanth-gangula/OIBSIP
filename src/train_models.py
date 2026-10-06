"""
train_models.py
---------------
Trains Logistic Regression and Random Forest classifiers on the
SMOTE-balanced training set and saves:
  - models/logistic_regression.pkl
  - models/random_forest.pkl
  - models/model_metrics.json   (real metrics computed from test set)
  - models/lr_coefficients.json
  - models/rf_feature_importance.json

All metrics are computed on the HELD-OUT test set (real distribution).
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
import joblib

from preprocess import run_preprocessing

# -- Paths -------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -- Evaluation helper -------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    """Compute full metrics on the real test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model":         name,
        "precision":     round(float(precision_score(y_test, y_pred)), 6),
        "recall":        round(float(recall_score(y_test, y_pred)), 6),
        "f1_score":      round(float(f1_score(y_test, y_pred)), 6),
        "roc_auc":       round(float(roc_auc_score(y_test, y_prob)), 6),
        "avg_precision": round(float(average_precision_score(y_test, y_prob)), 6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    print("\n-- " + name + " " + "-"*(50-len(name)))
    print("  Precision  : {:.4f}".format(metrics["precision"]))
    print("  Recall     : {:.4f}".format(metrics["recall"]))
    print("  F1 Score   : {:.4f}".format(metrics["f1_score"]))
    print("  ROC-AUC    : {:.4f}".format(metrics["roc_auc"]))
    print("  Avg Prec   : {:.4f}".format(metrics["avg_precision"]))
    print("  Confusion Matrix:")
    print("    TN={}  FP={}".format(tn, fp))
    print("    FN={}  TP={}".format(fn, tp))
    return metrics


# -- 1. Logistic Regression --------------------------------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test, feature_cols):
    print("\n====== Training Logistic Regression ======")
    t0 = time.time()

    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight=None,   # class imbalance already handled by SMOTE
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    elapsed = time.time() - t0
    print("  Training time: {:.1f}s".format(elapsed))

    metrics = evaluate_model("Logistic Regression", lr, X_test, y_test)
    metrics["training_time_seconds"] = round(elapsed, 2)

    lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    joblib.dump(lr, lr_path)
    print("  Model saved -> " + lr_path)

    # Save coefficients for feature importance analysis
    coef_df = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": lr.coef_[0].tolist()
    }).sort_values("coefficient", key=abs, ascending=False)

    coef_path = os.path.join(MODEL_DIR, "lr_coefficients.json")
    coef_df.to_json(coef_path, orient="records", indent=2)
    print("  Coefficients saved -> " + coef_path)

    return lr, metrics


# -- 2. Random Forest --------------------------------------------------------
def train_random_forest(X_train, y_train, X_test, y_test, feature_cols):
    print("\n====== Training Random Forest ======")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,   # SMOTE handles balance
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print("  Training time: {:.1f}s".format(elapsed))

    metrics = evaluate_model("Random Forest", rf, X_test, y_test)
    metrics["training_time_seconds"] = round(elapsed, 2)

    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    joblib.dump(rf, rf_path)
    print("  Model saved -> " + rf_path)

    # Save feature importances
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf.feature_importances_.tolist()
    }).sort_values("importance", ascending=False)

    fi_path = os.path.join(MODEL_DIR, "rf_feature_importance.json")
    fi_df.to_json(fi_path, orient="records", indent=2)
    print("  Feature importances saved -> " + fi_path)

    return rf, metrics


# -- 3. Save comparison metrics ----------------------------------------------
def save_model_metrics(lr_metrics, rf_metrics):
    """Bundle both models' metrics into a single JSON for the web app."""
    all_metrics = {
        "logistic_regression": lr_metrics,
        "random_forest":       rf_metrics,
        "comparison": {
            "best_roc_auc":   "Random Forest" if rf_metrics["roc_auc"]   > lr_metrics["roc_auc"]   else "Logistic Regression",
            "best_recall":    "Random Forest" if rf_metrics["recall"]    > lr_metrics["recall"]    else "Logistic Regression",
            "best_precision": "Random Forest" if rf_metrics["precision"] > lr_metrics["precision"] else "Logistic Regression",
            "best_f1":        "Random Forest" if rf_metrics["f1_score"]  > lr_metrics["f1_score"]  else "Logistic Regression",
        }
    }

    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n  All metrics saved -> " + metrics_path)

    print("\n-- Model Comparison --------------------------------------------")
    print("{:<18} {:>22} {:>16}".format("Metric", "Logistic Regression", "Random Forest"))
    print("-" * 58)
    for m in ["precision", "recall", "f1_score", "roc_auc"]:
        print("  {:<16} {:>22.4f} {:>16.4f}".format(m, lr_metrics[m], rf_metrics[m]))

    return all_metrics


# -- Main --------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  FRAUD DETECTION -- MODEL TRAINING PIPELINE")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feat_cols, scaler = run_preprocessing()

    # Convert numpy arrays from SMOTE back to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feat_cols)
    X_test_df  = pd.DataFrame(X_test,  columns=feat_cols)

    lr_model, lr_metrics = train_logistic_regression(
        X_train_df, y_train, X_test_df, y_test, feat_cols)

    rf_model, rf_metrics = train_random_forest(
        X_train_df, y_train, X_test_df, y_test, feat_cols)

    save_model_metrics(lr_metrics, rf_metrics)

    print("\n====== Training Complete ======")
    print("  Models and artifacts saved to: " + MODEL_DIR)


if __name__ == "__main__":
    main()
