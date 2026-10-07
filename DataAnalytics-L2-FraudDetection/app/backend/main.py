"""
main.py — FastAPI Fraud Detection Backend
------------------------------------------
Endpoints:
  GET  /                    health check
  GET  /api/stats           dataset statistics (from models/dataset_stats.json)
  GET  /api/metrics         model performance metrics (from models/model_metrics.json)
  GET  /api/features        feature column names
  POST /api/predict         single-transaction fraud prediction
  POST /api/predict/batch   batch prediction (up to 100 transactions)

All predictions use the real trained models (logistic_regression.pkl, random_forest.pkl).
No results are hardcoded or fabricated.
"""

import os
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import uvicorn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Load artifacts at startup (once)
# ---------------------------------------------------------------------------
print("[startup] Loading model artifacts from:", MODEL_DIR)

lr_model     = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf_model     = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)          # ordered list of 30 feature names

with open(os.path.join(MODEL_DIR, "dataset_stats.json")) as f:
    DATASET_STATS = json.load(f)

with open(os.path.join(MODEL_DIR, "model_metrics.json")) as f:
    MODEL_METRICS = json.load(f)

with open(os.path.join(MODEL_DIR, "rf_feature_importance.json")) as f:
    RF_FI = json.load(f)

with open(os.path.join(MODEL_DIR, "lr_coefficients.json")) as f:
    LR_COEF = json.load(f)

print(f"[startup] Loaded features: {len(FEATURE_COLS)} columns")
print("[startup] Models ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Machine learning API for Credit Card Fraud Detection. "
        "Trained on the ULB Credit Card Fraud Detection dataset (284,807 real transactions). "
        "FOR EDUCATIONAL/ANALYTICAL PURPOSES ONLY — not a production banking decision system."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend (localhost:3000 + Vercel deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "*",   # open for demo; restrict to specific domains in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    """
    A single credit card transaction for fraud prediction.
    V1-V28: PCA-transformed features (dimensionless, typically -5 to 5).
    Amount: transaction amount in EUR (0 to ~25,000).
    Hour:   hour of day 0-23 (derived from raw Time field).
    model:  which model to use for prediction.
    """
    V1:     float = Field(default=0.0, description="PCA feature V1")
    V2:     float = Field(default=0.0, description="PCA feature V2")
    V3:     float = Field(default=0.0, description="PCA feature V3")
    V4:     float = Field(default=0.0, description="PCA feature V4")
    V5:     float = Field(default=0.0, description="PCA feature V5")
    V6:     float = Field(default=0.0, description="PCA feature V6")
    V7:     float = Field(default=0.0, description="PCA feature V7")
    V8:     float = Field(default=0.0, description="PCA feature V8")
    V9:     float = Field(default=0.0, description="PCA feature V9")
    V10:    float = Field(default=0.0, description="PCA feature V10")
    V11:    float = Field(default=0.0, description="PCA feature V11")
    V12:    float = Field(default=0.0, description="PCA feature V12")
    V13:    float = Field(default=0.0, description="PCA feature V13")
    V14:    float = Field(default=0.0, description="PCA feature V14")
    V15:    float = Field(default=0.0, description="PCA feature V15")
    V16:    float = Field(default=0.0, description="PCA feature V16")
    V17:    float = Field(default=0.0, description="PCA feature V17")
    V18:    float = Field(default=0.0, description="PCA feature V18")
    V19:    float = Field(default=0.0, description="PCA feature V19")
    V20:    float = Field(default=0.0, description="PCA feature V20")
    V21:    float = Field(default=0.0, description="PCA feature V21")
    V22:    float = Field(default=0.0, description="PCA feature V22")
    V23:    float = Field(default=0.0, description="PCA feature V23")
    V24:    float = Field(default=0.0, description="PCA feature V24")
    V25:    float = Field(default=0.0, description="PCA feature V25")
    V26:    float = Field(default=0.0, description="PCA feature V26")
    V27:    float = Field(default=0.0, description="PCA feature V27")
    V28:    float = Field(default=0.0, description="PCA feature V28")
    Amount: float = Field(default=100.0, ge=0.0, le=30000.0, description="Transaction amount EUR")
    Hour:   int   = Field(default=12,    ge=0,   le=23,       description="Hour of day 0-23")
    model:  Literal["random_forest", "logistic_regression", "both"] = Field(
        default="random_forest",
        description="Model to use for prediction"
    )

    @field_validator("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                     "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                     "V21","V22","V23","V24","V25","V26","V27","V28", mode="before")
    @classmethod
    def clamp_pca(cls, v):
        """PCA features are typically in [-10, 10]; clamp extreme values."""
        v = float(v)
        if v < -30 or v > 30:
            raise ValueError("PCA feature value out of expected range [-30, 30]")
        return v


class PredictionResult(BaseModel):
    prediction:        int           # 0 = legitimate, 1 = fraud
    label:             str           # "Legitimate" or "FRAUD"
    fraud_probability: float         # probability of fraud (0.0 – 1.0)
    risk_score:        int           # 0-100 risk score
    risk_level:        str           # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    model_used:        str
    disclaimer:        str


class PredictionResponse(BaseModel):
    status:            str
    transaction_id:    Optional[str] = None
    primary:           PredictionResult
    secondary:         Optional[PredictionResult] = None
    disclaimer:        str


DISCLAIMER = (
    "EDUCATIONAL/ANALYTICAL DEMO ONLY. This prediction is produced by a machine learning model "
    "trained on a research dataset. It must NOT be used as the sole basis for any real banking, "
    "financial, or fraud-management decision. Always apply human oversight and comply with "
    "applicable regulations."
)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _make_prediction(tx: TransactionInput, model_name: str) -> PredictionResult:
    """Run one model on one transaction and return a structured result."""
    # Build feature vector in the exact order the model was trained on
    import pandas as pd
    raw = {col: getattr(tx, col, 0.0) for col in FEATURE_COLS}
    row = pd.DataFrame([raw], columns=FEATURE_COLS)

    # Scale Amount and Hour (same scaler used during training)
    row = row.copy()
    row[["Amount", "Hour"]] = scaler.transform(row[["Amount", "Hour"]])

    # Keep as DataFrame with column names to suppress sklearn warnings
    X = row  # shape (1, 30) with named columns

    model = rf_model if model_name == "random_forest" else lr_model
    pred  = int(model.predict(X)[0])
    prob  = float(model.predict_proba(X)[0][1])

    # Risk score 0-100
    risk_score = int(round(prob * 100))
    if risk_score < 20:
        risk_level = "LOW"
    elif risk_score < 50:
        risk_level = "MEDIUM"
    elif risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return PredictionResult(
        prediction        = pred,
        label             = "FRAUD" if pred == 1 else "Legitimate",
        fraud_probability = round(prob, 6),
        risk_score        = risk_score,
        risk_level        = risk_level,
        model_used        = model_name.replace("_", " ").title(),
        disclaimer        = DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {
        "service":     "Fraud Detection API",
        "status":      "online",
        "version":     "1.0.0",
        "models":      ["random_forest", "logistic_regression"],
        "disclaimer":  DISCLAIMER,
    }


@app.get("/api/stats", tags=["Data"])
def get_stats():
    """Return real dataset statistics derived from creditcard.csv."""
    return {"status": "ok", "data": DATASET_STATS}


@app.get("/api/metrics", tags=["Models"])
def get_metrics():
    """Return real model evaluation metrics from the held-out test set."""
    return {"status": "ok", "data": MODEL_METRICS}


@app.get("/api/features", tags=["Models"])
def get_features():
    """Return the ordered feature column list."""
    return {
        "status":       "ok",
        "feature_count": len(FEATURE_COLS),
        "features":     FEATURE_COLS,
        "scale_cols":   ["Amount", "Hour"],
        "pca_cols":     [c for c in FEATURE_COLS if c.startswith("V")],
    }


@app.get("/api/feature-importance", tags=["Models"])
def get_feature_importance():
    """Return RF feature importances and LR coefficients."""
    return {
        "status": "ok",
        "random_forest":        RF_FI[:15],
        "logistic_regression":  LR_COEF[:15],
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(tx: TransactionInput):
    """
    Predict whether a transaction is fraudulent.

    - model='random_forest'        → RF prediction only
    - model='logistic_regression'  → LR prediction only
    - model='both'                 → Both models; primary=RF, secondary=LR
    """
    try:
        if tx.model == "both":
            primary   = _make_prediction(tx, "random_forest")
            secondary = _make_prediction(tx, "logistic_regression")
        elif tx.model == "logistic_regression":
            primary   = _make_prediction(tx, "logistic_regression")
            secondary = None
        else:
            primary   = _make_prediction(tx, "random_forest")
            secondary = None

        return PredictionResponse(
            status         = "ok",
            primary        = primary,
            secondary      = secondary,
            disclaimer     = DISCLAIMER,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionInput]):
    """
    Batch prediction for up to 100 transactions.
    All use Random Forest by default for batch mode.
    """
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 transactions.")
    if len(transactions) == 0:
        raise HTTPException(status_code=400, detail="Empty batch.")

    results = []
    for i, tx in enumerate(transactions):
        try:
            result = _make_prediction(tx, "random_forest")
            results.append({
                "index":            i,
                "prediction":       result.prediction,
                "label":            result.label,
                "fraud_probability": result.fraud_probability,
                "risk_score":       result.risk_score,
                "risk_level":       result.risk_level,
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    fraud_count = sum(1 for r in results if r.get("prediction") == 1)
    return {
        "status":       "ok",
        "total":        len(transactions),
        "fraud_count":  fraud_count,
        "legit_count":  len(transactions) - fraud_count,
        "results":      results,
        "disclaimer":   DISCLAIMER,
    }


@app.get("/api/sample-transactions", tags=["Data"])
def get_sample_transactions():
    """
    Returns sample transaction vectors from the real dataset for UI demonstration.
    These are REAL transactions from the test set, not synthetic.
    Loaded from the pre-computed samples JSON.
    """
    samples_path = os.path.join(MODEL_DIR, "sample_transactions.json")
    if not os.path.exists(samples_path):
        raise HTTPException(
            status_code=404,
            detail="Sample transactions not generated yet. Run src/generate_samples.py first."
        )
    with open(samples_path) as f:
        samples = json.load(f)
    return {"status": "ok", "samples": samples}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
