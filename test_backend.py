"""
test_backend.py
---------------
End-to-end test of the FastAPI backend without running a live server.
Uses FastAPI's TestClient (built on httpx) to test all endpoints.
"""

import sys
import os
import json

# Make sure the backend can find its artifacts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app", "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

PASS = []
FAIL = []

def test(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  |  {detail}")

print("\n" + "="*60)
print("  FRAUD DETECTION API -- END-TO-END TESTS")
print("="*60)

# ── Health check ────────────────────────────────────────────────
print("\n[1] GET /")
r = client.get("/")
test("Health check 200", r.status_code == 200)
data = r.json()
test("Status online", data.get("status") == "online")
test("Has disclaimer", "disclaimer" in data)

# ── Stats ────────────────────────────────────────────────────────
print("\n[2] GET /api/stats")
r = client.get("/api/stats")
test("Stats 200", r.status_code == 200)
stats = r.json()["data"]
test("Total tx 284807",    stats["total_transactions"] == 284807)
test("Fraud count 492",    stats["fraud_count"]        == 492)
test("Legit count 284315", stats["legitimate_count"]   == 284315)
test("Fraud pct ~0.17",    abs(stats["fraud_pct"] - 0.1727) < 0.01)
test("Imbalance ratio 577",stats["imbalance_ratio"]    == 577)

# ── Metrics ──────────────────────────────────────────────────────
print("\n[3] GET /api/metrics")
r = client.get("/api/metrics")
test("Metrics 200", r.status_code == 200)
metrics = r.json()["data"]
rf = metrics["random_forest"]
lr = metrics["logistic_regression"]
test("RF precision > 0.8",  rf["precision"] > 0.8,  f"got {rf['precision']}")
test("RF recall > 0.8",     rf["recall"]    > 0.8,  f"got {rf['recall']}")
test("RF F1 > 0.8",         rf["f1_score"]  > 0.8,  f"got {rf['f1_score']}")
test("RF AUC > 0.96",       rf["roc_auc"]   > 0.96, f"got {rf['roc_auc']}")
test("LR recall > 0.9",     lr["recall"]    > 0.9,  f"got {lr['recall']}")
test("LR AUC > 0.96",       lr["roc_auc"]   > 0.96, f"got {lr['roc_auc']}")

# ── Features ─────────────────────────────────────────────────────
print("\n[4] GET /api/features")
r = client.get("/api/features")
test("Features 200", r.status_code == 200)
feat_data = r.json()
test("30 features",  feat_data["feature_count"] == 30)
test("Has Amount",   "Amount" in feat_data["features"])
test("Has Hour",     "Hour"   in feat_data["features"])
test("Has V14",      "V14"    in feat_data["features"])

# ── Predict RF (default values = typical legitimate) ─────────────
print("\n[5] POST /api/predict (default values — RF)")
r = client.post("/api/predict", json={
    "V1":0.0,"V2":0.0,"V3":0.0,"V4":0.0,"V5":0.0,
    "V6":0.0,"V7":0.0,"V8":0.0,"V9":0.0,"V10":0.0,
    "V11":0.0,"V12":0.0,"V13":0.0,"V14":0.0,"V15":0.0,
    "V16":0.0,"V17":0.0,"V18":0.0,"V19":0.0,"V20":0.0,
    "V21":0.0,"V22":0.0,"V23":0.0,"V24":0.0,"V25":0.0,
    "V26":0.0,"V27":0.0,"V28":0.0,
    "Amount":50.0,"Hour":14,"model":"random_forest"
})
test("Predict 200", r.status_code == 200)
pred = r.json()
test("Status ok",             pred["status"] == "ok")
test("Has primary",           "primary" in pred)
test("Prediction is 0 or 1",  pred["primary"]["prediction"] in [0,1])
test("Has probability",       0.0 <= pred["primary"]["fraud_probability"] <= 1.0)
test("Has risk_score 0-100",  0 <= pred["primary"]["risk_score"] <= 100)
test("Has risk_level",        pred["primary"]["risk_level"] in ["LOW","MEDIUM","HIGH","CRITICAL"])
test("Has disclaimer",        "disclaimer" in pred)
print(f"     prediction={pred['primary']['prediction']}  prob={pred['primary']['fraud_probability']:.4f}  risk={pred['primary']['risk_level']}")

# ── Predict with a real fraud sample ─────────────────────────────
print("\n[6] POST /api/predict with real fraud sample")
samples_path = os.path.join(os.path.dirname(__file__), "models", "sample_transactions.json")
with open(samples_path) as f:
    samples = json.load(f)

fraud_sample = next(s for s in samples if s["true_label"] == 1)
legit_sample = next(s for s in samples if s["true_label"] == 0)

# Fraud transaction
payload = {**fraud_sample["features"], "model": "random_forest"}
r = client.post("/api/predict", json=payload)
test("Fraud sample 200", r.status_code == 200)
pred = r.json()
print(f"     FRAUD sample -> prediction={pred['primary']['prediction']}  prob={pred['primary']['fraud_probability']:.4f}")
test("Fraud sample has result", pred["primary"]["prediction"] in [0,1])

# Legitimate transaction
payload = {**legit_sample["features"], "model": "random_forest"}
r = client.post("/api/predict", json=payload)
test("Legit sample 200", r.status_code == 200)
pred = r.json()
print(f"     LEGIT sample -> prediction={pred['primary']['prediction']}  prob={pred['primary']['fraud_probability']:.4f}")
test("Legit sample has result", pred["primary"]["prediction"] in [0,1])

# ── Both models ───────────────────────────────────────────────────
print("\n[7] POST /api/predict model=both")
r = client.post("/api/predict", json={
    **fraud_sample["features"], "model": "both"
})
test("Both models 200", r.status_code == 200)
pred = r.json()
test("Has secondary", pred.get("secondary") is not None)
test("Primary is RF",  pred["primary"]["model_used"] == "Random Forest")
test("Secondary is LR",pred["secondary"]["model_used"] == "Logistic Regression")
print(f"     RF:  pred={pred['primary']['prediction']} prob={pred['primary']['fraud_probability']:.4f}")
print(f"     LR:  pred={pred['secondary']['prediction']} prob={pred['secondary']['fraud_probability']:.4f}")

# ── Batch prediction ──────────────────────────────────────────────
print("\n[8] POST /api/predict/batch")
batch = [fraud_sample["features"], legit_sample["features"]]
r = client.post("/api/predict/batch", json=batch)
test("Batch 200", r.status_code == 200)
batch_resp = r.json()
test("Batch total=2",   batch_resp["total"]  == 2)
test("Has results",     len(batch_resp["results"]) == 2)
print(f"     Batch: {batch_resp['fraud_count']} fraud, {batch_resp['legit_count']} legit")

# ── Input validation ──────────────────────────────────────────────
print("\n[9] Input validation")
r = client.post("/api/predict", json={"Amount": -999, "Hour": 12, "model": "random_forest"})
test("Negative amount rejected", r.status_code == 422, f"got {r.status_code}")

r = client.post("/api/predict", json={"Amount": 50, "Hour": 99, "model": "random_forest"})
test("Invalid hour rejected", r.status_code == 422, f"got {r.status_code}")

# ── Sample transactions ───────────────────────────────────────────
print("\n[10] GET /api/sample-transactions")
r = client.get("/api/sample-transactions")
test("Samples 200", r.status_code == 200)
samples_resp = r.json()
test("Has 10 samples", len(samples_resp["samples"]) == 10)
test("Has fraud sample", any(s["true_label"]==1 for s in samples_resp["samples"]))
test("Has legit sample", any(s["true_label"]==0 for s in samples_resp["samples"]))

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED TESTS:")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("  ALL TESTS PASSED")
print("="*60)

sys.exit(0 if not FAIL else 1)
