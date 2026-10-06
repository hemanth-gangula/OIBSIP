# Scalability Discussion
## Scaling to ~1 Million Transactions per Hour

**Oasis Infobyte Data Analytics L2 Task 3 — Technical Architecture Note**

---

This document explains how the analytical fraud-detection pipeline in this project
would need to be re-architected to serve approximately **1 million transactions per
hour** (~278 transactions per second) in a production environment.

---

## Current Demo Architecture

```
creditcard.csv  →  Python script  →  sklearn model  →  JSON output
```

This works for analysis and education but handles a single batch offline — not
real-time traffic.

---

## Target Production Architecture

```
Card Transaction
      │
      ▼
┌──────────────────┐
│  API Gateway     │  ← rate limiting, auth, TLS termination
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────┐
│  Message Broker  │────►│  Feature Service     │
│  (Kafka/Kinesis) │     │  (Redis feature      │
└──────────────────┘     │   store, <1ms lookup)│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Model Serving Layer  │
                         │  (Triton + ONNX RF)   │
                         │  <5ms per inference   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼──────────────┐
                    ▼               ▼              ▼
              ┌─────────┐   ┌──────────┐   ┌──────────────┐
              │ Approve │   │  Review  │   │ Auto-Decline │
              └─────────┘   └──────────┘   └──────────────┘
```

---

## Component Breakdown

### 1. Ingestion — Apache Kafka

- **Why:** Kafka handles 278+ tx/sec with room to spare (benchmarked at millions/sec)
- Partition topics by card hash for per-customer ordering guarantees
- Retention: 7 days (allows replay for model retraining and audit)
- AWS Kinesis is the managed alternative if on AWS

### 2. Feature Service — Redis + Feast

- Pre-computed velocity features: tx count in last 1/5/60 minutes per card
- PCA transformation applied offline during card registration
- Sub-millisecond lookup latency (Redis is in-memory)
- Feature Store (Feast/Tecton) handles point-in-time correctness for retraining

### 3. Model Serving — Triton Inference Server + ONNX

| Metric | Value |
|--------|-------|
| Serialisation | Export RF to ONNX with `sklearn-onnx` |
| Inference latency | <5ms per transaction (p99) |
| Throughput | >10,000 predictions/sec per GPU instance |
| Scaling | Horizontal via Kubernetes HPA |

```python
# Export to ONNX (one-time, run after training)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 30]))]
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)
with open("models/random_forest.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### 4. Infrastructure Sizing (1M tx/hour)

| Component | Technology | Instance Type | Count |
|-----------|-----------|--------------|-------|
| API Gateway | Kong / AWS API GW | - | Auto |
| Kafka Brokers | Apache Kafka | m5.xlarge | 6 |
| Feature Store | Redis Cluster | r6g.large | 3 |
| Model Serving | Triton + ONNX | c5.2xlarge | 4-8 |
| Monitoring | Prometheus + Grafana | t3.medium | 1 |
| Audit DB | PostgreSQL RDS | db.r5.large | 1+2 replicas |

**Estimated cost (AWS, on-demand):** ~$3,000–$5,000/month
**With Reserved Instances (1yr):** ~$1,500–$2,500/month

### 5. Batching vs Streaming

| Pattern | When to Use | Latency |
|---------|-------------|---------|
| Real-time stream | Card-present, CNP high-value | <100ms |
| Micro-batch (50 tx) | E-commerce checkout | <100ms |
| Batch (nightly) | Account review, low-risk flagging | Hours |

### 6. Model Retraining

Fraud patterns evolve as adversaries adapt (**concept drift**):

```
Weekly pipeline:
  1. Collect new confirmed-fraud labels from dispute resolution
  2. Retrain RF on rolling 6-month window
  3. Shadow deploy: new model runs alongside live model (no production traffic)
  4. Compare metrics on live traffic for 48 hours
  5. Promote if AUC improves > 0.005 (Champion/Challenger pattern)
  6. Roll back automatically if precision drops > 5%
```

Tools: **MLflow** (experiment tracking) + **Airflow** (pipeline orchestration)

### 7. Monitoring & Alerting

| Signal | Tool | Threshold |
|--------|------|-----------|
| Data drift (PSI) | Evidently AI | PSI > 0.2 triggers alert |
| Concept drift | Model recall on labelled data | Recall drops > 5% → retrain |
| Inference latency | Prometheus | p99 > 50ms → scale out |
| FP rate spike | Custom metric | +20% FP → human review |
| System health | Grafana | Standard CPU/memory/disk |

### 8. Compliance & Explainability

PSD2 (EU) and similar regulations require explainable automated decisions:

```python
import shap

# SHAP values — per-prediction explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_single_transaction)
# → "V14 contributed +0.42 to fraud score, Amount contributed -0.12"
```

All predictions must be logged with:
- Transaction ID, timestamp, feature values
- Model version, prediction, probability
- SHAP feature contributions (for dispute resolution)
- Stored in immutable audit log (S3 + Athena for querying)

### 9. Latency Budget (card-present, 100ms SLA)

| Step | Budget |
|------|--------|
| Network (client → gateway) | 20ms |
| API Gateway processing | 5ms |
| Feature lookup (Redis) | 2ms |
| Model inference (ONNX) | 5ms |
| Decision + response | 3ms |
| Network (gateway → client) | 20ms |
| **Total** | **~55ms** |

Well within the 100ms SLA. Remaining 45ms budget allows for fallback logic,
logging, and network variance.

---

## Summary

The analytical pipeline in this project is fully production-capable in terms of
model quality (RF AUC=0.9683, precision=0.8247). The gap between demo and
production is not the model — it is the **serving infrastructure**. The
components above convert the offline batch pipeline into a real-time system
capable of handling any transaction volume a European card network can generate.

---

*Oasis Infobyte Data Analytics Internship — Level 2, Task 3*
