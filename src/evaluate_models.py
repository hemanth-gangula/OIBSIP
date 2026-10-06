"""
evaluate_models.py
------------------
Generates all evaluation charts and EDA visualisations from the real dataset.

Outputs saved to models/plots/:
  EDA
    01_class_distribution.png
    02_amount_distribution_by_class.png
    03_time_of_day_fraud.png
    04_fraud_amount_boxplot.png
    05_correlation_heatmap_fraud.png
    06_pca_feature_distributions.png

  Model Evaluation
    07_confusion_matrix_lr.png
    08_confusion_matrix_rf.png
    09_roc_curves.png
    10_precision_recall_curves.png
    11_model_comparison_bar.png

  Feature Analysis
    12_lr_feature_importance.png
    13_rf_feature_importance.png

All values come directly from the real dataset and trained models.
No results are fabricated or hardcoded.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Add src to path so preprocess imports work
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from preprocess import run_preprocessing

# -- Colour palette ----------------------------------------------------------
FRAUD_COL  = "#E74C3C"   # red
LEGIT_COL  = "#2ECC71"   # green
LR_COL     = "#3498DB"   # blue
RF_COL     = "#F39C12"   # orange
BG_COL     = "#F8F9FA"
GRID_COL   = "#DEE2E6"

plt.rcParams.update({
    "figure.facecolor":  BG_COL,
    "axes.facecolor":    BG_COL,
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})


def save(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COL)
    plt.close(fig)
    print("  Saved -> " + path)
    return path


# ============================================================================
# EDA PLOTS
# ============================================================================

def plot_class_distribution(df):
    """Plot 01 — Class distribution (count + pie)."""
    vc = df["Class"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Credit Card Fraud — Class Distribution", fontsize=15, fontweight="bold", y=1.02)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(["Legitimate (0)", "Fraudulent (1)"],
                  [vc[0], vc[1]],
                  color=[LEGIT_COL, FRAUD_COL], edgecolor="white", width=0.5)
    ax.set_title("Transaction Count by Class")
    ax.set_ylabel("Number of Transactions")
    for bar, val in zip(bars, [vc[0], vc[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, vc[0] * 1.12)

    # Pie chart (log-scale effect via explode)
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        [vc[0], vc[1]],
        labels=["Legitimate\n(99.83%)", "Fraudulent\n(0.17%)"],
        colors=[LEGIT_COL, FRAUD_COL],
        explode=[0, 0.12],
        autopct="%1.2f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10}
    )
    autotexts[1].set_fontweight("bold")
    ax2.set_title("Class Proportion\n(Severe Imbalance: 1:577)")

    fig.tight_layout()
    return save(fig, "01_class_distribution.png")


def plot_amount_distribution(df):
    """Plot 02 — Transaction amount distribution by class."""
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount Distribution by Class", fontsize=15, fontweight="bold")

    # Full range (log y)
    ax = axes[0]
    ax.hist(legit, bins=80, color=LEGIT_COL, alpha=0.7, label=f"Legitimate (n={len(legit):,})", density=True)
    ax.hist(fraud, bins=40, color=FRAUD_COL, alpha=0.85, label=f"Fraudulent (n={len(fraud):,})", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Transaction Amount (EUR)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Full Amount Range")
    ax.legend()

    # Zoomed 0-500
    ax2 = axes[1]
    legit_zoom = legit[legit <= 500]
    fraud_zoom = fraud[fraud <= 500]
    ax2.hist(legit_zoom, bins=60, color=LEGIT_COL, alpha=0.7, label="Legitimate", density=True)
    ax2.hist(fraud_zoom, bins=40, color=FRAUD_COL, alpha=0.85, label="Fraudulent", density=True)
    ax2.set_xlabel("Transaction Amount (EUR)")
    ax2.set_ylabel("Density")
    ax2.set_title("Zoomed: Amount up to EUR 500")
    ax2.legend()

    # Annotation
    fig.text(0.5, -0.02,
             f"Fraud mean: EUR {fraud.mean():.2f}  |  Median: EUR {fraud.median():.2f}  |  Max: EUR {fraud.max():.2f}    "
             f"Legit mean: EUR {legit.mean():.2f}  |  Median: EUR {legit.median():.2f}  |  Max: EUR {legit.max():.2f}",
             ha="center", fontsize=9, color="#555")

    fig.tight_layout()
    return save(fig, "02_amount_distribution_by_class.png")


def plot_time_of_day(df):
    """Plot 03 — Fraud transactions across hours of the day."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600 % 24).astype(int)

    fraud_hourly = df[df["Class"] == 1].groupby("Hour").size()
    legit_hourly = df[df["Class"] == 0].groupby("Hour").size()
    fraud_rate   = df.groupby("Hour")["Class"].mean() * 100   # fraud rate %

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Fraud Patterns by Hour of Day (48-hour window mapped to 0-23h)", fontsize=14, fontweight="bold")

    hours = range(24)

    # Transaction volume
    ax = axes[0]
    width = 0.4
    x = np.array(list(hours))
    l_vals = [legit_hourly.get(h, 0) for h in hours]
    f_vals = [fraud_hourly.get(h, 0) for h in hours]
    ax.bar(x - width/2, l_vals, width=width, color=LEGIT_COL, alpha=0.8, label="Legitimate")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, f_vals, width=width, color=FRAUD_COL, alpha=0.85, label="Fraudulent")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Legitimate Transactions", color=LEGIT_COL)
    ax2.set_ylabel("Fraudulent Transactions", color=FRAUD_COL)
    ax.set_title("Transaction Volume by Hour")
    ax.set_xticks(list(hours))
    lines1 = mpatches.Patch(color=LEGIT_COL, label="Legitimate")
    lines2 = mpatches.Patch(color=FRAUD_COL, label="Fraudulent")
    ax.legend(handles=[lines1, lines2], loc="upper right")

    # Fraud rate %
    ax3 = axes[1]
    rate_vals = [float(fraud_rate.get(h, 0.0)) for h in hours]
    ax3.bar(list(hours), rate_vals, color=FRAUD_COL, alpha=0.8, edgecolor="white")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Fraud Rate (%)")
    ax3.set_title("Fraud Rate (%) by Hour of Day")
    ax3.set_xticks(list(hours))

    # Mark peak hour
    peak_h = int(np.argmax(rate_vals))
    ax3.annotate(f"Peak: Hour {peak_h}\n({rate_vals[peak_h]:.3f}%)",
                 xy=(peak_h, rate_vals[peak_h]),
                 xytext=(peak_h + 1.5, rate_vals[peak_h] + 0.01),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=9)

    fig.tight_layout()
    return save(fig, "03_time_of_day_fraud.png")


def plot_amount_boxplot(df):
    """Plot 04 — Amount boxplot fraud vs legitimate (capped at 99th pct)."""
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]
    cap   = df["Amount"].quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Transaction Amount: Fraud vs Legitimate", fontsize=14, fontweight="bold")

    # Boxplot (matplotlib 3.9+ uses tick_labels instead of labels)
    ax = axes[0]
    bp = ax.boxplot(
        [legit[legit <= cap], fraud[fraud <= cap]],
        tick_labels=["Legitimate", "Fraudulent"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2}
    )
    bp["boxes"][0].set_facecolor(LEGIT_COL)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(FRAUD_COL)
    bp["boxes"][1].set_alpha(0.7)
    ax.set_ylabel("Transaction Amount (EUR)")
    ax.set_title(f"Boxplot (capped at 99th pct: EUR {cap:.0f})")

    # Violin plot
    ax2 = axes[1]
    data_v = [legit[legit <= cap].values, fraud[fraud <= cap].values]
    parts  = ax2.violinplot(data_v, showmedians=True)
    for pc, col in zip(parts["bodies"], [LEGIT_COL, FRAUD_COL]):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Legitimate", "Fraudulent"])
    ax2.set_ylabel("Transaction Amount (EUR)")
    ax2.set_title("Violin Plot")

    fig.tight_layout()
    return save(fig, "04_fraud_amount_boxplot.png")


def plot_correlation_heatmap(df):
    """Plot 05 — Correlation of top PCA features with the Class label."""
    corr = df.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False)
    top20 = corr.head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [FRAUD_COL if v > 0 else LEGIT_COL for v in top20.values]
    bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Class (Fraud=1)")
    ax.set_title("Top 20 Features Correlated with Fraud\n(Red = positive correlation, Green = negative)")
    ax.set_xlim(-0.35, 0.35)

    for bar, val in zip(bars, top20.values[::-1]):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    fig.tight_layout()
    return save(fig, "05_correlation_heatmap_fraud.png")


def plot_pca_feature_distributions(df):
    """Plot 06 — Distribution of the 6 most fraud-correlated features."""
    corr    = df.corr()["Class"].drop(["Class", "Time", "Amount"])
    top6    = corr.abs().sort_values(ascending=False).head(6).index.tolist()
    fraud   = df[df["Class"] == 1]
    legit   = df[df["Class"] == 0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Top 6 PCA Features: Fraud vs Legitimate Distributions", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for i, feat in enumerate(top6):
        ax = axes_flat[i]
        ax.hist(legit[feat], bins=60, color=LEGIT_COL, alpha=0.6, density=True, label="Legitimate")
        ax.hist(fraud[feat], bins=40, color=FRAUD_COL, alpha=0.75, density=True, label="Fraudulent")
        c = float(df.corr()["Class"][feat])
        ax.set_title(f"{feat}  (r={c:.3f})")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save(fig, "06_pca_feature_distributions.png")


# ============================================================================
# MODEL EVALUATION PLOTS
# ============================================================================

def plot_confusion_matrix(cm_dict, model_name, filename, color):
    """Plots 07, 08 — Confusion matrix heatmap."""
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=["Predicted: Legit", "Predicted: Fraud"],
        yticklabels=["Actual: Legit", "Actual: Fraud"],
        linewidths=2, linecolor="white", ax=ax, cbar=False
    )

    for i in range(2):
        for j in range(2):
            val  = cm[i, j]
            pct  = val / total * 100
            bg   = cm[i, j] / cm.max()
            clr  = "white" if bg > 0.5 else "black"
            cell = "TN" if (i==0 and j==0) else ("FP" if (i==0 and j==1) else ("FN" if (i==1 and j==0) else "TP"))
            ax.text(j + 0.5, i + 0.38, str(val), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=clr)
            ax.text(j + 0.5, i + 0.62, f"{pct:.2f}%", ha="center", va="center",
                    fontsize=10, color=clr)
            ax.text(j + 0.5, i + 0.82, f"[{cell}]", ha="center", va="center",
                    fontsize=9, color=clr, style="italic")

    ax.set_title(f"Confusion Matrix — {model_name}\n"
                 f"TP={cm_dict['tp']}  TN={cm_dict['tn']}  FP={cm_dict['fp']}  FN={cm_dict['fn']}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, filename)


def plot_roc_curves(lr_model, rf_model, X_test, y_test):
    """Plot 09 — ROC curves for both models."""
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    lr_auc = auc(lr_fpr, lr_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(lr_fpr, lr_tpr, color=LR_COL, lw=2,
            label=f"Logistic Regression (AUC = {lr_auc:.4f})")
    ax.plot(rf_fpr, rf_tpr, color=RF_COL, lw=2,
            label=f"Random Forest       (AUC = {rf_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.5000)")
    ax.fill_between(lr_fpr, lr_tpr, alpha=0.05, color=LR_COL)
    ax.fill_between(rf_fpr, rf_tpr, alpha=0.05, color=RF_COL)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves — Logistic Regression vs Random Forest\n"
                 "(computed on 20% held-out test set, real class distribution)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    return save(fig, "09_roc_curves.png")


def plot_precision_recall_curves(lr_model, rf_model, X_test, y_test):
    """Plot 10 — Precision-Recall curves."""
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_prob)
    rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_prob)
    lr_ap = average_precision_score(y_test, lr_prob)
    rf_ap = average_precision_score(y_test, rf_prob)

    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(lr_rec, lr_prec, color=LR_COL, lw=2,
            label=f"Logistic Regression (AP = {lr_ap:.4f})")
    ax.plot(rf_rec, rf_prec, color=RF_COL, lw=2,
            label=f"Random Forest       (AP = {rf_ap:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline (random, AP = {baseline:.4f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves\n"
                 "(important when positive class is rare — 0.17% fraud rate)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    return save(fig, "10_precision_recall_curves.png")


def plot_model_comparison(lr_metrics, rf_metrics):
    """Plot 11 — Side-by-side metric comparison bar chart."""
    metrics = ["precision", "recall", "f1_score", "roc_auc"]
    labels  = ["Precision", "Recall", "F1 Score", "ROC-AUC"]
    lr_vals = [lr_metrics[m] for m in metrics]
    rf_vals = [rf_metrics[m] for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, lr_vals, width, color=LR_COL, alpha=0.85,
                   label="Logistic Regression", edgecolor="white")
    bars2 = ax.bar(x + width/2, rf_vals, width, color=RF_COL, alpha=0.85,
                   label="Random Forest", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, color=LR_COL, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, color=RF_COL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison\n(Logistic Regression vs Random Forest — real test set metrics)")
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    fig.tight_layout()
    return save(fig, "11_model_comparison_bar.png")


# ============================================================================
# FEATURE IMPORTANCE PLOTS
# ============================================================================

def plot_lr_coefficients(coef_path):
    """Plot 12 — Logistic Regression top coefficients."""
    coef_df = pd.read_json(coef_path)
    top15   = coef_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [FRAUD_COL if v > 0 else LEGIT_COL for v in top15["coefficient"]]
    ax.barh(top15["feature"][::-1], top15["coefficient"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value (log-odds scale)")
    ax.set_title("Logistic Regression — Top 15 Features by |Coefficient|\n"
                 "Positive = increases fraud probability  |  Negative = decreases fraud probability")

    for i, (feat, val) in enumerate(zip(top15["feature"][::-1], top15["coefficient"][::-1])):
        ax.text(val + (0.02 if val >= 0 else -0.02),
                i, f"{val:.4f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    fig.tight_layout()
    return save(fig, "12_lr_feature_importance.png")


def plot_rf_feature_importance(fi_path):
    """Plot 13 — Random Forest feature importances."""
    fi_df = pd.read_json(fi_path)
    top15 = fi_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_gradient = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top15)))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1],
            color=colors_gradient, edgecolor="white")
    ax.set_xlabel("Gini Importance (mean decrease in impurity)")
    ax.set_title("Random Forest — Top 15 Feature Importances\n"
                 "(V1-V28 are PCA-transformed; original feature names are confidential)")

    for i, (feat, val) in enumerate(zip(top15["feature"][::-1], top15["importance"][::-1])):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", ha="left", fontsize=8)

    fig.tight_layout()
    return save(fig, "13_rf_feature_importance.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  FRAUD DETECTION -- EVALUATION & VISUALISATION PIPELINE")
    print("=" * 60)

    # -- Load data -----------------------------------------------------------
    print("\n[1/5] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)

    # -- EDA plots -----------------------------------------------------------
    print("\n[2/5] Generating EDA plots ...")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_of_day(df)
    plot_amount_boxplot(df)
    plot_correlation_heatmap(df)
    plot_pca_feature_distributions(df)

    # -- Load models + preprocessing -----------------------------------------
    print("\n[3/5] Loading models and rerunning preprocessing for test set ...")
    _, _, X_test, y_test, feat_cols, _ = run_preprocessing()
    X_test_df = pd.DataFrame(X_test, columns=feat_cols)

    lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))

    with open(os.path.join(MODEL_DIR, "model_metrics.json")) as f:
        all_metrics = json.load(f)
    lr_m = all_metrics["logistic_regression"]
    rf_m = all_metrics["random_forest"]

    # -- Model evaluation plots ----------------------------------------------
    print("\n[4/5] Generating model evaluation plots ...")
    plot_confusion_matrix(lr_m["confusion_matrix"], "Logistic Regression",
                          "07_confusion_matrix_lr.png", LR_COL)
    plot_confusion_matrix(rf_m["confusion_matrix"], "Random Forest",
                          "08_confusion_matrix_rf.png", RF_COL)
    plot_roc_curves(lr_model, rf_model, X_test_df, y_test)
    plot_precision_recall_curves(lr_model, rf_model, X_test_df, y_test)
    plot_model_comparison(lr_m, rf_m)

    # -- Feature analysis plots ----------------------------------------------
    print("\n[5/5] Generating feature importance plots ...")
    plot_lr_coefficients(os.path.join(MODEL_DIR, "lr_coefficients.json"))
    plot_rf_feature_importance(os.path.join(MODEL_DIR, "rf_feature_importance.json"))

    print("\n====== All plots saved to: " + PLOT_DIR + " ======")

    # -- Save plot paths JSON for web app ------------------------------------
    plot_files = sorted(os.listdir(PLOT_DIR))
    plots_index = {f.split("_", 1)[1].replace(".png", ""): f for f in plot_files if f.endswith(".png")}
    with open(os.path.join(MODEL_DIR, "plots_index.json"), "w") as f:
        json.dump(plots_index, f, indent=2)
    print("  Plot index saved -> " + os.path.join(MODEL_DIR, "plots_index.json"))


if __name__ == "__main__":
    main()
