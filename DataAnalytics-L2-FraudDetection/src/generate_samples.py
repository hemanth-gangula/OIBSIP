"""
generate_samples.py
-------------------
Extracts real transactions from the dataset to use as demo samples
in the web application. Saves models/sample_transactions.json.

Samples: 5 legitimate + 5 fraudulent real transactions from the test set.
NO synthetic data — every row comes directly from creditcard.csv.
"""

import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Derive Hour feature (same as training pipeline)
df["Hour"] = (df["Time"] // 3600 % 24).astype(int)

feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
X = df[feature_cols]
y = df["Class"]

# Reproduce the exact same split (same random_state=42)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Pick real fraud and legit samples from the test set
fraud_rows = X_test[y_test == 1].head(5)
legit_rows = X_test[y_test == 0].head(5)

samples = []

for i, (idx, row) in enumerate(fraud_rows.iterrows()):
    entry = {
        "id":           f"FRAUD_SAMPLE_{i+1}",
        "true_label":   1,
        "description":  f"Real fraudulent transaction #{i+1} from test set",
        "features":     {col: round(float(row[col]), 6) for col in feature_cols}
    }
    samples.append(entry)

for i, (idx, row) in enumerate(legit_rows.iterrows()):
    entry = {
        "id":           f"LEGIT_SAMPLE_{i+1}",
        "true_label":   0,
        "description":  f"Real legitimate transaction #{i+1} from test set",
        "features":     {col: round(float(row[col]), 6) for col in feature_cols}
    }
    samples.append(entry)

out_path = os.path.join(MODEL_DIR, "sample_transactions.json")
with open(out_path, "w") as f:
    json.dump(samples, f, indent=2)

print(f"Saved {len(samples)} real sample transactions -> {out_path}")
print(f"  Fraud samples : {sum(1 for s in samples if s['true_label']==1)}")
print(f"  Legit samples : {sum(1 for s in samples if s['true_label']==0)}")
