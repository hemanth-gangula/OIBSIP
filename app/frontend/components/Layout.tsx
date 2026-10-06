import React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  Shield, LayoutDashboard, BarChart2,
  Activity, Search, AlertTriangle
} from "lucide-react";

const NAV = [
  { href: "/",            label: "Dashboard",   icon: LayoutDashboard },
  { href: "/eda",         label: "Data Analysis",icon: BarChart2 },
  { href: "/performance", label: "Model Metrics",icon: Activity },
  { href: "/predict",     label: "Predict",      icon: Search },
];

interface LayoutProps { children: React.ReactNode; }

export default function Layout({ children }: LayoutProps) {
  const router = useRouter();

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--bg-primary)" }}>
      {/* Top nav */}
      <header className="sticky top-0 z-50 border-b" style={{ background: "rgba(15,23,42,0.95)", borderColor: "var(--border)", backdropFilter: "blur(12px)" }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)" }}>
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold text-white text-sm leading-none block">FraudShield</span>
                <span className="text-xs leading-none" style={{ color: "var(--text-muted)" }}>Analytics Platform</span>
              </div>
            </Link>

            {/* Nav links */}
            <nav className="hidden md:flex items-center gap-1">
              {NAV.map(({ href, label, icon: Icon }) => {
                const active = router.pathname === href;
                return (
                  <Link key={href} href={href}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all"
                    style={{
                      color:      active ? "#fff" : "var(--text-muted)",
                      background: active ? "rgba(59,130,246,0.15)" : "transparent",
                    }}>
                    <Icon className="w-4 h-4" />
                    {label}
                  </Link>
                );
              })}
            </nav>

            {/* Status badge */}
            <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
              style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.25)", color: "#22c55e" }}>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              Demo Mode
            </div>
          </div>

          {/* Mobile nav */}
          <nav className="md:hidden flex gap-1 pb-3 overflow-x-auto">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = router.pathname === href;
              return (
                <Link key={href} href={href}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all"
                  style={{
                    color:      active ? "#fff" : "var(--text-muted)",
                    background: active ? "rgba(59,130,246,0.15)" : "transparent",
                  }}>
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Disclaimer banner */}
      <div className="border-b py-2 px-4 text-center text-xs flex items-center justify-center gap-2"
        style={{ background: "rgba(245,158,11,0.08)", borderColor: "rgba(245,158,11,0.2)", color: "#f59e0b" }}>
        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
        <span>
          <strong>Educational Demo Only</strong> — This application is for analytical and learning purposes.
          It is not a production banking system and must not be used for real fraud decisions.
        </span>
      </div>

      {/* Page content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t py-6 text-center text-xs" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
        <p>Oasis Infobyte Data Analytics Internship — Level 2, Task 3 &nbsp;|&nbsp; Credit Card Fraud Detection</p>
        <p className="mt-1">Dataset: ULB Machine Learning Group via Kaggle &nbsp;|&nbsp; Models: Logistic Regression + Random Forest</p>
      </footer>
    </div>
  );
}
