import React from "react";
import Head from "next/head";
import Link from "next/link";
import {
  Shield, Database, TrendingUp, AlertTriangle,
  ArrowRight, CheckCircle, BarChart2, Activity, Search
} from "lucide-react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import StatCard from "@/components/StatCard";
import MetricBadge from "@/components/MetricBadge";
import { STATIC_STATS, STATIC_METRICS } from "@/lib/api";

// Real values from the trained models
const STATS   = STATIC_STATS;
const METRICS = STATIC_METRICS;

const PIE_DATA = [
  { name: "Legitimate", value: STATS.legitimate_count, color: "#22c55e" },
  { name: "Fraudulent", value: STATS.fraud_count,      color: "#ef4444" },
];

const FEATURES = [
  { step: "1", title: "Real Dataset",    desc: "284,807 transactions, 0 missing values, 48-hour window." },
  { step: "2", title: "SMOTE Balancing", desc: "Training set rebalanced from 1:577 to 1:1 without touching test data." },
  { step: "3", title: "Two Models",      desc: "Logistic Regression for high recall; Random Forest for precision." },
  { step: "4", title: "Full Evaluation", desc: "Precision, Recall, F1, ROC-AUC on held-out real-distribution test set." },
];

export default function Dashboard() {
  return (
    <>
      <Head>
        <title>FraudShield — Dashboard</title>
      </Head>

      {/* Hero */}
      <section className="text-center mb-12 py-4">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-4"
          style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)", color: "#3b82f6" }}>
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          Oasis Infobyte — Data Analytics Level 2, Task 3
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4 leading-tight">
          Credit Card{" "}
          <span className="gradient-text">Fraud Detection</span>
        </h1>
        <p className="text-lg max-w-2xl mx-auto mb-8" style={{ color: "var(--text-muted)" }}>
          Machine learning pipeline trained on 284,807 real transactions.
          Detects fraudulent activity using Logistic Regression and Random Forest
          with SMOTE oversampling to handle severe class imbalance.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link href="/predict"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-white text-sm transition-all hover:opacity-90 active:scale-95"
            style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)" }}>
            Try Live Prediction <ArrowRight className="w-4 h-4" />
          </Link>
          <Link href="/performance"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all"
            style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
            View Model Metrics <BarChart2 className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Key stats row */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Transactions" value={STATS.total_transactions.toLocaleString()} sub="284,807 real transactions" icon={Database} color="blue" />
        <StatCard label="Fraudulent"         value={STATS.fraud_count.toLocaleString()}        sub={`${STATS.fraud_pct}% of total`}   icon={AlertTriangle} color="red" />
        <StatCard label="Legitimate"         value={STATS.legitimate_count.toLocaleString()}   sub={`${STATS.legit_pct.toFixed(2)}% of total`} icon={CheckCircle} color="green" />
        <StatCard label="Imbalance Ratio"    value={`1 : ${STATS.imbalance_ratio}`}            sub="1 fraud per 577 legit" icon={TrendingUp} color="amber" />
      </section>

      {/* Class distribution + Model comparison */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Pie chart */}
        <div className="rounded-xl p-6" style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
          <h2 className="font-semibold text-white mb-1">Class Distribution</h2>
          <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
            Severe imbalance — 1 fraud per 577 legitimate transactions
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={PIE_DATA} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value">
                {PIE_DATA.map((entry, i) => <Cell key={i} fill={entry.color} stroke="transparent" />)}
              </Pie>
              <Tooltip
                formatter={(v: number) => [v.toLocaleString(), ""]}
                contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }} />
              <Legend formatter={(v) => <span style={{ color: "#94a3b8", fontSize: 12 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-3 mt-2">
            <div className="rounded-lg p-3 text-center" style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)" }}>
              <p className="text-lg font-bold text-green-400">99.83%</p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>Legitimate</p>
            </div>
            <div className="rounded-lg p-3 text-center" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
              <p className="text-lg font-bold text-red-400">0.17%</p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>Fraudulent</p>
            </div>
          </div>
        </div>

        {/* Model comparison */}
        <div className="rounded-xl p-6" style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
          <h2 className="font-semibold text-white mb-1">Model Performance</h2>
          <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>Real metrics from 20% held-out test set</p>

          {/* Random Forest */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-amber-400">Random Forest</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">Recommended</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <MetricBadge label="Precision" value={METRICS.random_forest.precision} color="amber" />
              <MetricBadge label="Recall"    value={METRICS.random_forest.recall}    color="amber" />
              <MetricBadge label="F1"        value={METRICS.random_forest.f1_score}  color="amber" />
              <MetricBadge label="AUC"       value={METRICS.random_forest.roc_auc}   color="amber" />
            </div>
          </div>

          {/* Logistic Regression */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-blue-400">Logistic Regression</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">High Recall</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <MetricBadge label="Precision" value={METRICS.logistic_regression.precision} color="blue" />
              <MetricBadge label="Recall"    value={METRICS.logistic_regression.recall}    color="blue" />
              <MetricBadge label="F1"        value={METRICS.logistic_regression.f1_score}  color="blue" />
              <MetricBadge label="AUC"       value={METRICS.logistic_regression.roc_auc}   color="blue" />
            </div>
          </div>
        </div>
      </div>

      {/* Transaction amount stats */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Fraud Mean Amount"   value={`$${STATS.fraud_mean_amount}`}   sub="avg per fraudulent tx"  icon={Activity} color="red" />
        <StatCard label="Fraud Median Amount" value={`$${STATS.fraud_median_amount}`} sub="typical fraud amount"   icon={Activity} color="red" />
        <StatCard label="Legit Mean Amount"   value={`$${STATS.legit_mean_amount}`}   sub="avg per legitimate tx"  icon={Activity} color="green" />
        <StatCard label="Legit Max Amount"    value={`$${STATS.legit_max_amount.toLocaleString()}`} sub="largest single tx" icon={Activity} color="green" />
      </section>

      {/* Methodology */}
      <section className="rounded-xl p-6 mb-8" style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
        <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" /> Methodology
        </h2>
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
          {FEATURES.map(({ step, title, desc }) => (
            <div key={step} className="flex gap-3">
              <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 font-bold text-xs text-white"
                style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)" }}>
                {step}
              </div>
              <div>
                <p className="text-sm font-semibold text-white">{title}</p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Quick nav cards */}
      <section className="grid sm:grid-cols-3 gap-4">
        {[
          { href:"/eda",         icon: BarChart2, title:"Data Analysis", desc:"EDA, class imbalance, amount distributions, time patterns", color:"#3b82f6" },
          { href:"/performance", icon: Activity,  title:"Model Metrics",  desc:"Confusion matrices, ROC curves, Precision-Recall, feature importance", color:"#f59e0b" },
          { href:"/predict",     icon: Search,    title:"Live Prediction",desc:"Enter transaction features and get real-time fraud probability", color:"#8b5cf6" },
        ].map(({ href, icon: Icon, title, desc, color }) => (
          <Link key={href} href={href}
            className="rounded-xl p-5 card-hover flex flex-col gap-3 group"
            style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}>
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
              <Icon className="w-5 h-5" style={{ color }} />
            </div>
            <div>
              <h3 className="font-semibold text-white group-hover:text-blue-400 transition-colors">{title}</h3>
              <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{desc}</p>
            </div>
            <div className="flex items-center gap-1 text-xs font-medium" style={{ color }}>
              Explore <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>
        ))}
      </section>
    </>
  );
}
