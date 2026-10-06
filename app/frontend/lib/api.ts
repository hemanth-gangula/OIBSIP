/**
 * api.ts — Typed API client for the FastAPI backend
 * All calls go to NEXT_PUBLIC_API_URL (default: http://localhost:8000)
 */

import type {
  DatasetStats, AllMetrics, PredictionResponse,
  TransactionInput, FeatureImportanceItem, SampleTransaction
} from "@/types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health:            () => get<{ status: string; version: string }>("/"),
  stats:             () => get<{ data: DatasetStats }>("/api/stats"),
  metrics:           () => get<{ data: AllMetrics }>("/api/metrics"),
  features:          () => get<{ features: string[]; feature_count: number }>("/api/features"),
  featureImportance: () => get<{
    random_forest: FeatureImportanceItem[];
    logistic_regression: FeatureImportanceItem[];
  }>("/api/feature-importance"),
  samples:           () => get<{ samples: SampleTransaction[] }>("/api/sample-transactions"),
  predict:           (tx: TransactionInput) =>
    post<PredictionResponse>("/api/predict", tx),
};

// Static fallback data — used when the backend is offline (e.g. Vercel preview without backend)
// All values are the REAL computed metrics, not fabricated.
export const STATIC_STATS: DatasetStats = {
  total_transactions:   284807,
  legitimate_count:     284315,
  fraud_count:          492,
  fraud_pct:            0.1727,
  legit_pct:            99.8273,
  imbalance_ratio:      577,
  time_span_hours:      48,
  fraud_mean_amount:    122.21,
  fraud_max_amount:     2125.87,
  fraud_median_amount:  9.25,
  legit_mean_amount:    88.29,
  legit_max_amount:     25691.16,
  legit_median_amount:  22.00,
};

export const STATIC_METRICS: AllMetrics = {
  logistic_regression: {
    model: "Logistic Regression",
    precision: 0.0557, recall: 0.9184, f1_score: 0.1050,
    roc_auc: 0.9706, avg_precision: 0.7280,
    training_time_seconds: 12.5,
    confusion_matrix: { tn: 55338, fp: 1526, fn: 8,  tp: 90 },
    classification_report: {},
  },
  random_forest: {
    model: "Random Forest",
    precision: 0.8247, recall: 0.8163, f1_score: 0.8205,
    roc_auc: 0.9683, avg_precision: 0.8669,
    training_time_seconds: 263.6,
    confusion_matrix: { tn: 56847, fp: 17, fn: 18, tp: 80 },
    classification_report: {},
  },
  comparison: {
    best_roc_auc:   "Logistic Regression",
    best_recall:    "Logistic Regression",
    best_precision: "Random Forest",
    best_f1:        "Random Forest",
  },
};
