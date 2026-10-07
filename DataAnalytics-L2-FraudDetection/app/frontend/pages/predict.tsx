import React, { useState, useCallback } from "react";
import Head from "next/head";
import {
  Search, AlertTriangle, CheckCircle, Loader2,
  Shuffle, Info, Shield, XCircle
} from "lucide-react";
import { RadialBarChart, RadialBar, ResponsiveContainer } from "recharts";
import { api, STATIC_METRICS } from "@/lib/api";
import type { TransactionInput, PredictionResponse } from "@/types";

// Real fraud and legit samples from the dataset (feature values from sample_transactions.json)
const DEMO_SAMPLES: { label: string; tag: "fraud" | "legit"; features: Partial<TransactionInput> }[] = [
  {
    label: "Real Fraud #1",
    tag: "fraud",
    features: {
      V1:-3.043541,V2:3.157302,V3:-4.775636,V4:4.359272,V5:-1.460888,
      V6:0.663609,V7:-2.770538,V8:0.208866,V9:-0.430197,V10:-3.257732,
      V11:3.046473,V12:-5.070455,V13:-1.067475,V14:-7.415929,V15:0.301552,
      V16:-3.454831,V17:-6.920566,V18:-1.200299,V19:0.506869,V20:0.276977,
      V21:0.666917,V22:0.566272,V23:-0.139804,V24:-0.195593,V25:0.000000,
      V26:0.000000,V27:0.126932,V28:0.052826,Amount:1.00,Hour:22,
    }
  },
  {
    label: "Real Fraud #2",
    tag: "fraud",
    features: {
      V1:-2.312227,V2:1.951992,V3:-1.609851,V4:3.997906,V5:-0.522188,
      V6:-1.426545,V7:-2.537387,V8:1.391657,V9:-2.770089,V10:-2.772272,
      V11:3.202033,V12:-2.899907,V13:-0.595222,V14:-4.289254,V15:0.389724,
      V16:-1.140747,V17:-2.830056,V18:-0.016822,V19:0.416956,V20:0.126911,
      V21:0.517232,V22:-0.035049,V23:-0.465211,V24:0.320198,V25:0.044519,
      V26:0.177840,V27:0.261145,V28:0.143480,Amount:239.93,Hour:0,
    }
  },
  {
    label: "Real Legit #1",
    tag: "legit",
    features: {
      V1:-1.359807,V2:-0.072781,V3:2.536347,V4:1.378155,V5:-0.338321,
      V6:0.462388,V7:0.239599,V8:0.098698,V9:0.363787,V10:0.090794,
      V11:-0.551600,V12:-0.617801,V13:-0.991390,V14:-0.311169,V15:1.468177,
      V16:-0.470401,V17:0.207971,V18:0.025791,V19:0.403993,V20:0.251412,
      V21:-0.018307,V22:0.277838,V23:-0.110474,V24:0.066928,V25:0.128539,
      V26:-0.189115,V27:0.133558,V28:-0.021053,Amount:149.62,Hour:0,
    }
  },
  {
    label: "Real Legit #2",
    tag: "legit",
    features: {
      V1:1.191857,V2:0.266151,V3:0.166480,V4:0.448154,V5:0.060018,
      V6:-0.082361,V7:-0.078803,V8:0.085102,V9:-0.255425,V10:-0.166974,
      V11:1.612727,V12:1.065235,V13:0.489095,V14:-0.143772,V15:0.635558,
      V16:0.463917,V17:-0.114805,V18:-0.183361,V19:-0.145783,V20:-0.069083,
      V21:-0.225775,V22:-0.638672,V23:0.101288,V24:-0.339846,V25:0.167170,
      V26:0.125895,V27:-0.008983,V28:0.014724,Amount:2.69,Hour:0,
    }
  },
];

const BLANK: TransactionInput = {
  V1:0,V2:0,V3:0,V4:0,V5:0,V6:0,V7:0,V8:0,V9:0,V10:0,
  V11:0,V12:0,V13:0,V14:0,V15:0,V16:0,V17:0,V18:0,V19:0,V20:0,
  V21:0,V22:0,V23:0,V24:0,V25:0,V26:0,V27:0,V28:0,
  Amount:50, Hour:12, model:"random_forest"
};

const V_FEATURES = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
  "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
  "V21","V22","V23","V24","V25","V26","V27","V28"] as const;

const RISK_COLORS: Record<string, string> = {
  LOW:"#22c55e", MEDIUM:"#f59e0b", HIGH:"#ef4444", CRITICAL:"#dc2626"
};

export default function Predict() {
  const [form,    setForm]    = useState<TransactionInput>(BLANK);
  const [result,  setResult]  = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [offline, setOffline] = useState(false);

  const setField = (k: keyof TransactionInput, v: string) => {
    const n = parseFloat(v);
    setForm(f => ({ ...f, [k]: isNaN(n) ? 0 : n }));
  };

  const loadSample = (s: typeof DEMO_SAMPLES[0]) => {
    setForm({ ...BLANK, ...s.features, model: form.model } as TransactionInput);
    setResult(null);
    setError(null);
  };

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.predict(form);
      setResult(res);
      setOffline(false);
    } catch {
      // Backend offline — show static demo result
      setOffline(true);
      // Demo prediction using amount heuristic (NOT a real model — clearly flagged)
      const demoProb = form.Amount < 10 ? 0.73 : 0.04;
      const demoRisk = demoProb > 0.5 ? "HIGH" : "LOW";
      setResult({
        status: "demo",
        primary: {
          prediction: demoProb > 0.5 ? 1 : 0,
          label:      demoProb > 0.5 ? "FRAUD" : "Legitimate",
          fraud_probability: demoProb,
          risk_score: Math.round(demoProb*100),
          risk_level: demoRisk as any,
          model_used: "Demo (backend offline)",
          disclaimer: "DEMO MODE — backend unavailable. Result is a heuristic placeholder, not a real ML prediction.",
        },
        disclaimer: "DEMO MODE — backend offline."
      });
    } finally {
      setLoading(false);
    }
  }, [form]);

  const reset = () => { setForm(BLANK); setResult(null); setError(null); };

  const riskColor = result ? (RISK_COLORS[result.primary.risk_level] ?? "#94a3b8") : "#94a3b8";
  const gaugeData = result ? [{ value: result.primary.risk_score, fill: riskColor }] : [];

  return (
    <>
      <Head><title>FraudShield — Predict Transaction</title></Head>

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Transaction Fraud Prediction</h1>
        <p style={{ color:"var(--text-muted)" }}>
          Enter transaction feature values and receive a real-time fraud probability from the trained models.
        </p>
      </div>

      {/* Demo samples */}
      <div className="rounded-xl p-4 mb-6" style={{ background:"var(--bg-secondary)", border:"1px solid var(--border)" }}>
        <p className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <Shuffle className="w-4 h-4 text-blue-400" />
          Load a Real Sample Transaction
          <span className="text-xs font-normal" style={{ color:"var(--text-muted)" }}>(actual rows from the dataset)</span>
        </p>
        <div className="flex flex-wrap gap-2">
          {DEMO_SAMPLES.map(s => (
            <button key={s.label} onClick={() => loadSample(s)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
              style={{
                background: s.tag==="fraud" ? "rgba(239,68,68,0.12)" : "rgba(34,197,94,0.12)",
                border: `1px solid ${s.tag==="fraud" ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)"}`,
                color:   s.tag==="fraud" ? "#ef4444" : "#22c55e",
              }}>
              {s.tag==="fraud" ? "🚨" : "✅"} {s.label}
            </button>
          ))}
          <button onClick={reset} className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
            style={{ background:"rgba(100,116,139,0.1)", border:"1px solid rgba(100,116,139,0.2)", color:"#94a3b8" }}>
            Reset
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Input form */}
        <form onSubmit={handleSubmit} className="rounded-xl p-6" style={{ background:"var(--bg-secondary)", border:"1px solid var(--border)" }}>
          <h2 className="font-semibold text-white mb-4">Transaction Features</h2>

          {/* Model selector */}
          <div className="mb-4">
            <label className="block text-xs font-semibold uppercase tracking-wider mb-1.5" style={{ color:"var(--text-muted)" }}>
              Model
            </label>
            <select value={form.model} onChange={e => setForm(f => ({ ...f, model: e.target.value as any }))}
              className="w-full px-3 py-2 rounded-lg text-sm font-medium transition-all"
              style={{ background:"var(--bg-primary)", border:"1px solid var(--border)", color:"var(--text-primary)" }}>
              <option value="random_forest">Random Forest (recommended)</option>
              <option value="logistic_regression">Logistic Regression</option>
              <option value="both">Both Models</option>
            </select>
          </div>

          {/* Amount + Hour */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider mb-1" style={{ color:"var(--text-muted)" }}>
                Amount (EUR)
              </label>
              <input type="number" step="0.01" min="0" max="30000"
                value={form.Amount}
                onChange={e => setField("Amount", e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background:"var(--bg-primary)", border:"1px solid var(--border)", color:"var(--text-primary)" }} />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider mb-1" style={{ color:"var(--text-muted)" }}>
                Hour (0–23)
              </label>
              <input type="number" min="0" max="23"
                value={form.Hour}
                onChange={e => setField("Hour", e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background:"var(--bg-primary)", border:"1px solid var(--border)", color:"var(--text-primary)" }} />
            </div>
          </div>

          {/* PCA features grid */}
          <div className="mb-2">
            <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color:"var(--text-muted)" }}>
              PCA Features V1–V28
            </p>
            <div className="grid grid-cols-4 gap-1.5 max-h-64 overflow-y-auto pr-1">
              {V_FEATURES.map(v => (
                <div key={v}>
                  <label className="text-xs block mb-0.5" style={{ color:"#64748b" }}>{v}</label>
                  <input type="number" step="0.001"
                    value={(form[v as keyof TransactionInput] as number).toFixed(3)}
                    onChange={e => setField(v as keyof TransactionInput, e.target.value)}
                    className="w-full px-2 py-1 rounded text-xs"
                    style={{ background:"var(--bg-primary)", border:"1px solid #334155", color:"var(--text-primary)" }} />
                </div>
              ))}
            </div>
          </div>

          <p className="text-xs mb-4 flex items-start gap-1.5" style={{ color:"var(--text-muted)" }}>
            <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            V1–V28 are PCA-transformed. Use the sample buttons above to load real transaction values, or enter 0 for a default baseline.
          </p>

          <button type="submit" disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-white text-sm flex items-center justify-center gap-2 transition-all hover:opacity-90 active:scale-95 disabled:opacity-50"
            style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)" }}>
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? "Analysing..." : "Analyse Transaction"}
          </button>

          {error && (
            <div className="mt-3 p-3 rounded-lg text-xs flex items-center gap-2" style={{ background:"rgba(239,68,68,0.1)", border:"1px solid rgba(239,68,68,0.2)", color:"#ef4444" }}>
              <XCircle className="w-4 h-4 shrink-0" />{error}
            </div>
          )}
        </form>

        {/* Results panel */}
        <div>
          {!result && !loading && (
            <div className="rounded-xl p-8 flex flex-col items-center justify-center h-full text-center"
              style={{ background:"var(--bg-secondary)", border:"1px dashed var(--border)" }}>
              <Shield className="w-12 h-12 mb-3" style={{ color:"#334155" }} />
              <p className="font-medium" style={{ color:"#475569" }}>No prediction yet</p>
              <p className="text-sm mt-1" style={{ color:"#334155" }}>Load a sample or enter values, then click Analyse</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Main verdict */}
              <div className="rounded-xl p-6" style={{
                background: result.primary.prediction === 1
                  ? "rgba(239,68,68,0.08)" : "rgba(34,197,94,0.08)",
                border: `2px solid ${result.primary.prediction===1 ? "rgba(239,68,68,0.4)" : "rgba(34,197,94,0.4)"}`,
              }}>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <p className="text-xs uppercase tracking-wider font-semibold mb-1"
                      style={{ color: result.primary.prediction===1?"#ef4444":"#22c55e" }}>
                      Primary Result ({result.primary.model_used})
                    </p>
                    <p className="text-4xl font-extrabold" style={{ color: result.primary.prediction===1?"#ef4444":"#22c55e" }}>
                      {result.primary.label}
                    </p>
                    <div className="flex items-center gap-3 mt-2 flex-wrap">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold risk-${result.primary.risk_level.toLowerCase()}`}>
                        {result.primary.risk_level} RISK
                      </span>
                      <span className="text-sm font-semibold" style={{ color:"var(--text-muted)" }}>
                        {(result.primary.fraud_probability*100).toFixed(2)}% fraud probability
                      </span>
                    </div>
                  </div>

                  {/* Gauge */}
                  <div className="w-24 h-24 shrink-0 relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadialBarChart innerRadius="70%" outerRadius="100%"
                        data={[{ value: result.primary.risk_score, fill: riskColor }]}
                        startAngle={180} endAngle={0}>
                        <RadialBar dataKey="value" cornerRadius={5} />
                      </RadialBarChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center mt-4">
                      <span className="text-xl font-black" style={{ color: riskColor }}>{result.primary.risk_score}</span>
                      <span className="text-xs" style={{ color:"var(--text-muted)" }}>/ 100</span>
                    </div>
                  </div>
                </div>

                {/* Icon */}
                <div className="flex items-center gap-2 mt-3 text-sm">
                  {result.primary.prediction===1
                    ? <AlertTriangle className="w-4 h-4 text-red-400" />
                    : <CheckCircle className="w-4 h-4 text-green-400" />}
                  <span style={{ color:"var(--text-muted)" }}>
                    {result.primary.prediction===1
                      ? "Transaction flagged for review. This pattern matches known fraud signatures in the training data."
                      : "Transaction appears legitimate. Fraud probability is below the alert threshold."}
                  </span>
                </div>
              </div>

              {/* Secondary model (if both) */}
              {result.secondary && (
                <div className="rounded-xl p-4" style={{ background:"var(--bg-secondary)", border:"1px solid var(--border)" }}>
                  <p className="text-xs uppercase tracking-wider font-semibold mb-2" style={{ color:"var(--text-muted)" }}>
                    Secondary ({result.secondary.model_used})
                  </p>
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-xl font-bold" style={{ color: result.secondary.prediction===1?"#ef4444":"#22c55e" }}>
                      {result.secondary.label}
                    </span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-bold risk-${result.secondary.risk_level.toLowerCase()}`}>
                      {result.secondary.risk_level}
                    </span>
                    <span className="text-sm" style={{ color:"var(--text-muted)" }}>
                      {(result.secondary.fraud_probability*100).toFixed(2)}% prob
                    </span>
                  </div>
                </div>
              )}

              {/* Detail stats */}
              <div className="rounded-xl p-4 grid grid-cols-2 gap-3" style={{ background:"var(--bg-secondary)", border:"1px solid var(--border)" }}>
                {[
                  { label:"Fraud Probability",  value:`${(result.primary.fraud_probability*100).toFixed(4)}%` },
                  { label:"Risk Score",          value:`${result.primary.risk_score} / 100` },
                  { label:"Risk Level",          value:result.primary.risk_level },
                  { label:"Model",               value:result.primary.model_used },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <p className="text-xs" style={{ color:"var(--text-muted)" }}>{label}</p>
                    <p className="text-sm font-semibold text-white">{value}</p>
                  </div>
                ))}
              </div>

              {/* Offline warning */}
              {offline && (
                <div className="rounded-lg p-3 text-xs flex items-start gap-2"
                  style={{ background:"rgba(245,158,11,0.08)", border:"1px solid rgba(245,158,11,0.2)", color:"#f59e0b" }}>
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span><strong>Demo Mode:</strong> The FastAPI backend is offline. The result shown is a simple heuristic placeholder — not a real ML model prediction. Start the backend with <code>uvicorn main:app</code> in app/backend/ for real predictions.</span>
                </div>
              )}

              {/* Disclaimer */}
              <div className="rounded-lg p-3 text-xs" style={{ background:"rgba(100,116,139,0.05)", border:"1px solid rgba(100,116,139,0.15)", color:"#64748b" }}>
                <strong>Disclaimer:</strong> {result.disclaimer}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model info card */}
      <div className="rounded-xl p-5 mt-6 grid md:grid-cols-2 gap-4" style={{ background:"var(--bg-secondary)", border:"1px solid var(--border)" }}>
        <div>
          <h3 className="font-semibold text-white mb-2 text-sm">Model Information</h3>
          <div className="space-y-1 text-xs" style={{ color:"var(--text-muted)" }}>
            <p>• Trained on 284,807 real transactions (80% train, 20% test)</p>
            <p>• SMOTE applied to training set only (1:1 balance)</p>
            <p>• Random Forest: 100 estimators, n_jobs=-1</p>
            <p>• Logistic Regression: max_iter=1000, C=1.0, lbfgs solver</p>
          </div>
        </div>
        <div>
          <h3 className="font-semibold text-white mb-2 text-sm">Real Test Set Metrics</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              { m:"RF Precision",  v: STATIC_METRICS.random_forest.precision.toFixed(4), c:"#f59e0b" },
              { m:"RF Recall",     v: STATIC_METRICS.random_forest.recall.toFixed(4),    c:"#f59e0b" },
              { m:"RF F1",         v: STATIC_METRICS.random_forest.f1_score.toFixed(4),  c:"#f59e0b" },
              { m:"RF AUC",        v: STATIC_METRICS.random_forest.roc_auc.toFixed(4),   c:"#f59e0b" },
            ].map(({ m, v, c }) => (
              <div key={m} className="flex justify-between">
                <span style={{ color:"var(--text-muted)" }}>{m}</span>
                <span className="font-semibold" style={{ color:c }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
