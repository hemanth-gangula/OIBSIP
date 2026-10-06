// Shared TypeScript types for the Fraud Detection Dashboard

export interface DatasetStats {
  total_transactions: number;
  legitimate_count:   number;
  fraud_count:        number;
  fraud_pct:          number;
  legit_pct:          number;
  imbalance_ratio:    number;
  time_span_hours:    number;
  fraud_mean_amount:  number;
  fraud_max_amount:   number;
  fraud_median_amount:number;
  legit_mean_amount:  number;
  legit_max_amount:   number;
  legit_median_amount:number;
}

export interface ConfusionMatrix {
  tn: number; fp: number;
  fn: number; tp: number;
}

export interface ModelMetrics {
  model:                   string;
  precision:               number;
  recall:                  number;
  f1_score:                number;
  roc_auc:                 number;
  avg_precision:           number;
  training_time_seconds:   number;
  confusion_matrix:        ConfusionMatrix;
  classification_report:   Record<string, unknown>;
}

export interface AllMetrics {
  logistic_regression: ModelMetrics;
  random_forest:       ModelMetrics;
  comparison: {
    best_roc_auc:   string;
    best_recall:    string;
    best_precision: string;
    best_f1:        string;
  };
}

export interface PredictionResult {
  prediction:        number;
  label:             string;
  fraud_probability: number;
  risk_score:        number;
  risk_level:        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  model_used:        string;
  disclaimer:        string;
}

export interface PredictionResponse {
  status:         string;
  primary:        PredictionResult;
  secondary?:     PredictionResult;
  disclaimer:     string;
}

export interface TransactionInput {
  V1: number;  V2: number;  V3: number;  V4: number;
  V5: number;  V6: number;  V7: number;  V8: number;
  V9: number;  V10: number; V11: number; V12: number;
  V13: number; V14: number; V15: number; V16: number;
  V17: number; V18: number; V19: number; V20: number;
  V21: number; V22: number; V23: number; V24: number;
  V25: number; V26: number; V27: number; V28: number;
  Amount: number;
  Hour:   number;
  model:  "random_forest" | "logistic_regression" | "both";
}

export interface FeatureImportanceItem {
  feature:    string;
  importance?: number;
  coefficient?: number;
}

export interface SampleTransaction {
  id:          string;
  true_label:  number;
  description: string;
  features:    Record<string, number>;
}
