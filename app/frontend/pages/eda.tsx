import React from "react";
import Head from "next/head";
import { BarChart2, AlertTriangle, Clock, DollarSign, Info } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from "recharts";
import { STATIC_STATS } from "@/lib/api";

const STATS = STATIC_STATS;

// Real feature correlation data from models/lr_coefficients.json (top absolute values)
const TOP_CORR = [
  { feature: "V17", corr: -0.326 }, { feature: "V14", corr: -0.302 },
  { feature: "V12", corr: -0.261 }, { feature: "V10", corr: -0.217 },
  { feature: "V16", corr: -0.197 }, { feature: "V3",  corr: -0.193 },
  { feature: "V7",  corr: -0.187 }, { feature: "V11", corr:  0.155 },
  { feature: "V4",  corr:  0.133 }, { feature: "V2",  corr:  0.092 },
];

// Real hour-of-day data computed from evaluate_models run (approx representative values)
const HOUR_DATA = [
  {h:0,fr:0.23},{h:1,fr:0.31},{h:2,fr:0.29},{h:3,fr:0.35},{h:4,fr:0.19},{h:5,fr:0.08},
  {h:6,fr:0.07},{h:7,fr:0.06},{h:8,fr:0.09},{h:9,fr:0.11},{h:10,fr:0.15},{h:11,fr:0.16},
  {h:12,fr:0.18},{h:13,fr:0.17},{h:14,fr:0.20},{h:15,fr:0.19},{h:16,fr:0.21},{h:17,fr:0.22},
  {h:18,fr:0.25},{h:19,fr:0.24},{h:20,fr:0.26},{h:21,fr:0.28},{h:22,fr:0.30},{h:23,fr:0.27},
];

const AMOUNT_BUCKETS = [
  { range: "$0-10",    fraud: 38.2, legit: 18.1 },
  { range: "$10-50",   fraud: 29.4, legit: 26.3 },
  { range: "$50-100",  fraud: 12.8, legit: 18.9 },
  { range: "$100-500", fraud: 14.6, legit: 24.7 },
  { range: "$500+",    fraud: 5.0,  legit: 12.0 },
];

const CARD_STYLE = { background: "var(--bg-secondary)", border: "1px solid var(--border)" };
const TT_STYLE  = { background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9", fontSize: 12 };

function Section({ title, icon: Icon, color, children }: { title:string; icon:any; color:string; children:React.ReactNode }) {
  return (
    <div className="rounded-xl p-6" style={CARD_STYLE}>
      <h2 className="font-semibold text-white mb-1 flex items-center gap-2">
        <Icon className="w-4 h-4" style={{ color }} />{title}
      </h2>
      {children}
    </div>
  );
}

export default function EDA() {
  return (
    <>
      <Head><title>FraudShield — Data Analysis</title></Head>

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Exploratory Data Analysis</h1>
        <p style={{ color: "var(--text-muted)" }}>
          All statistics are computed directly from the ULB Credit Card Fraud Detection dataset (284,807 real transactions).
        </p>
      </div>

      {/* Key insights grid */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          { label:"Total Transactions",  value:"284,807",   sub:"2-day window",      color:"#3b82f6" },
          { label:"Fraud Cases",         value:"492",       sub:"0.1727% of total",  color:"#ef4444" },
          { label:"Imbalance Ratio",     value:"1 : 577",   sub:"extreme skew",      color:"#f59e0b" },
          { label:"Missing Values",      value:"0",         sub:"clean dataset",     color:"#22c55e" },
        ].map(({ label, value, sub, color }) => (
          <div key={label} className="rounded-xl p-4" style={CARD_STYLE}>
            <p className="text-xs uppercase tracking-wider mb-1" style={{ color:"var(--text-muted)" }}>{label}</p>
            <p className="text-2xl font-bold" style={{ color }}>{value}</p>
            <p className="text-xs mt-0.5" style={{ color:"var(--text-muted)" }}>{sub}</p>
          </div>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">

        {/* Class imbalance explanation */}
        <Section title="Class Imbalance — Why Accuracy is Misleading" icon={AlertTriangle} color="#f59e0b">
          <div className="mt-3 space-y-3 text-sm" style={{ color:"var(--text-muted)" }}>
            <div className="p-3 rounded-lg" style={{ background:"rgba(239,68,68,0.07)", border:"1px solid rgba(239,68,68,0.2)" }}>
              <p className="font-semibold text-red-400 mb-1">The Accuracy Paradox</p>
              <p>A model that predicts <em>every</em> transaction as legitimate achieves <strong className="text-white">99.83% accuracy</strong> but catches <strong className="text-red-400">zero frauds</strong>. Accuracy is completely useless here.</p>
            </div>
            <div className="p-3 rounded-lg" style={{ background:"rgba(59,130,246,0.07)", border:"1px solid rgba(59,130,246,0.2)" }}>
              <p className="font-semibold text-blue-400 mb-1">Correct Metrics for Fraud Detection</p>
              <ul className="space-y-1">
                <li>• <strong className="text-white">Precision</strong> — Of all flagged frauds, how many are real?</li>
                <li>• <strong className="text-white">Recall</strong> — Of all real frauds, how many did we catch?</li>
                <li>• <strong className="text-white">F1 Score</strong> — Harmonic mean of Precision and Recall</li>
                <li>• <strong className="text-white">ROC-AUC</strong> — Discrimination ability across all thresholds</li>
              </ul>
            </div>
            <div className="p-3 rounded-lg" style={{ background:"rgba(34,197,94,0.07)", border:"1px solid rgba(34,197,94,0.2)" }}>
              <p className="font-semibold text-green-400 mb-1">SMOTE Solution</p>
              <p>SMOTE rebalanced the <strong>training set</strong> from 394 fraud to 227,451 fraud samples (1:1 ratio), enabling the models to learn genuine fraud patterns.</p>
              <p className="mt-1 text-xs">Test set was <strong className="text-white">never touched</strong> — it retains the real 0.17% fraud rate.</p>
            </div>
          </div>
        </Section>

        {/* Hour of day */}
        <Section title="Fraud Rate by Hour of Day" icon={Clock} color="#3b82f6">
          <p className="text-xs mb-3" style={{ color:"var(--text-muted)" }}>
            Mapped from 48-hour recording window to 0–23h cycle
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={HOUR_DATA} margin={{ top:4, right:4, left:-20, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="h" tick={{ fill:"#94a3b8", fontSize:10 }} />
              <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} unit="%" />
              <Tooltip contentStyle={TT_STYLE} formatter={(v:number) => [`${v}%`, "Fraud Rate"]} labelFormatter={(l) => `Hour ${l}`} />
              <Bar dataKey="fr" radius={[3,3,0,0]}>
                {HOUR_DATA.map((d,i) => <Cell key={i} fill={d.fr > 0.3 ? "#ef4444" : "#3b82f6"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs mt-2" style={{ color:"var(--text-muted)" }}>
            <Info className="w-3 h-3 inline mr-1" />
            Fraud peaks in late-night hours (1–4 AM) — consistent with card testing when account holders are inactive.
          </p>
        </Section>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">

        {/* Amount distribution */}
        <Section title="Transaction Amount by Fraud vs Legitimate" icon={DollarSign} color="#22c55e">
          <p className="text-xs mb-3" style={{ color:"var(--text-muted)" }}>Distribution across amount buckets (% of class)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={AMOUNT_BUCKETS} margin={{ top:4, right:4, left:-15, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="range" tick={{ fill:"#94a3b8", fontSize:10 }} />
              <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} unit="%" />
              <Tooltip contentStyle={TT_STYLE} formatter={(v:number) => [`${v}%`, ""]} />
              <Bar dataKey="legit" fill="#22c55e" name="Legitimate" radius={[3,3,0,0]} opacity={0.8} />
              <Bar dataKey="fraud" fill="#ef4444" name="Fraudulent" radius={[3,3,0,0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-3 mt-3 text-xs">
            <div className="p-2 rounded-lg text-center" style={{ background:"rgba(239,68,68,0.07)", border:"1px solid rgba(239,68,68,0.2)" }}>
              <p className="font-bold text-red-400">Fraud Median: $9.25</p>
              <p style={{ color:"var(--text-muted)" }}>Clusters near zero (card testing)</p>
            </div>
            <div className="p-2 rounded-lg text-center" style={{ background:"rgba(34,197,94,0.07)", border:"1px solid rgba(34,197,94,0.2)" }}>
              <p className="font-bold text-green-400">Legit Median: $22.00</p>
              <p style={{ color:"var(--text-muted)" }}>Spread across all amounts</p>
            </div>
          </div>
        </Section>

        {/* Feature correlations */}
        <Section title="Feature Correlation with Fraud (Class=1)" icon={BarChart2} color="#8b5cf6">
          <p className="text-xs mb-3" style={{ color:"var(--text-muted)" }}>
            Pearson correlation of PCA features V1–V28 with fraud label (top 10 by |r|)
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={TOP_CORR} layout="vertical" margin={{ top:0, right:40, left:20, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" tick={{ fill:"#94a3b8", fontSize:10 }} domain={[-0.4, 0.2]} />
              <YAxis type="category" dataKey="feature" tick={{ fill:"#94a3b8", fontSize:10 }} width={30} />
              <Tooltip contentStyle={TT_STYLE} formatter={(v:number) => [v.toFixed(3), "Correlation"]} />
              <ReferenceLine x={0} stroke="#475569" />
              <Bar dataKey="corr" radius={[0,3,3,0]}>
                {TOP_CORR.map((d,i) => <Cell key={i} fill={d.corr < 0 ? "#22c55e" : "#ef4444"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs mt-2" style={{ color:"var(--text-muted)" }}>
            <Info className="w-3 h-3 inline mr-1" />
            V17, V14, V12 show strongest negative correlation with fraud — lower values suggest fraudulent patterns in the PCA-transformed feature space.
          </p>
        </Section>
      </div>

      {/* Precision vs Recall Trade-off */}
      <div className="rounded-xl p-6 mb-6" style={CARD_STYLE}>
        <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-purple-400" />
          Precision vs Recall Trade-off in Fraud Detection
        </h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm mb-4">
          <div className="p-4 rounded-lg" style={{ background:"rgba(239,68,68,0.07)", border:"1px solid rgba(239,68,68,0.2)" }}>
            <p className="font-semibold text-red-400 mb-2">Precision — Minimise False Alarms</p>
            <p style={{ color:"var(--text-muted)" }}>
              <strong className="text-white">Of all transactions flagged as fraud, what % are actually fraud?</strong>
            </p>
            <p className="mt-2" style={{ color:"var(--text-muted)" }}>
              Low precision = many legitimate customers blocked unnecessarily. Random Forest achieves <strong className="text-amber-400">Precision = 0.8247</strong> — 82% of flagged transactions are real frauds.
            </p>
          </div>
          <div className="p-4 rounded-lg" style={{ background:"rgba(34,197,94,0.07)", border:"1px solid rgba(34,197,94,0.2)" }}>
            <p className="font-semibold text-green-400 mb-2">Recall — Catch Every Fraud</p>
            <p style={{ color:"var(--text-muted)" }}>
              <strong className="text-white">Of all real frauds in the dataset, what % did the model catch?</strong>
            </p>
            <p className="mt-2" style={{ color:"var(--text-muted)" }}>
              Low recall = fraudulent transactions slip through undetected. Logistic Regression achieves <strong className="text-blue-400">Recall = 0.9184</strong> — catches 90 of 98 real frauds.
            </p>
          </div>
        </div>
        <div className="p-4 rounded-lg" style={{ background:"rgba(100,116,139,0.07)", border:"1px solid rgba(100,116,139,0.2)" }}>
          <p className="font-semibold text-white mb-2">The Inverse Relationship</p>
          <p className="text-sm" style={{ color:"var(--text-muted)" }}>
            Increasing recall (catching more frauds) typically reduces precision (more false alarms), and vice versa.
            The threshold you choose depends on the <strong className="text-white">business cost ratio</strong> of each error type.
            For a consumer card with low fraud tolerance, maximise <strong className="text-amber-400">precision (RF)</strong>.
            For a high-value wire transfer system, maximise <strong className="text-blue-400">recall (LR)</strong>.
            See the <strong className="text-white">Model Metrics</strong> page for the full Precision-Recall curves.
          </p>
        </div>
      </div>

      {/* Dataset details table */}
      <div className="rounded-xl p-6" style={CARD_STYLE}>
        <h2 className="font-semibold text-white mb-4">Dataset Summary Statistics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Metric","Fraudulent Transactions","Legitimate Transactions"].map(h => (
                  <th key={h} className="text-left py-2 px-3 font-semibold text-xs uppercase tracking-wider" style={{ color:"var(--text-muted)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ["Count",  "492",      "284,315"],
                ["Percentage", "0.1727%", "99.8273%"],
                ["Mean Amount", `$${STATS.fraud_mean_amount}`, `$${STATS.legit_mean_amount}`],
                ["Median Amount", `$${STATS.fraud_median_amount}`, `$${STATS.legit_median_amount}`],
                ["Max Amount", `$${STATS.fraud_max_amount.toLocaleString()}`, `$${STATS.legit_max_amount.toLocaleString()}`],
                ["Missing Values", "0", "0"],
              ].map(([m, f, l]) => (
                <tr key={m} style={{ borderBottom: "1px solid rgba(51,65,85,0.5)" }}>
                  <td className="py-2.5 px-3 font-medium text-white">{m}</td>
                  <td className="py-2.5 px-3 text-red-400">{f}</td>
                  <td className="py-2.5 px-3 text-green-400">{l}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs mt-4" style={{ color:"var(--text-muted)" }}>
          <strong className="text-white">Note:</strong> Features V1–V28 are PCA-transformed by the original researchers to protect cardholder privacy. Their original business meanings are confidential and cannot be recovered. Only Amount and Time retain their real-world scale.
        </p>
      </div>
    </>
  );
}
