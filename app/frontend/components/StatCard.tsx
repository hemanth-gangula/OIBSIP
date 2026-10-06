import React from "react";
import { LucideIcon } from "lucide-react";
import clsx from "clsx";

interface StatCardProps {
  label:    string;
  value:    string | number;
  sub?:     string;
  icon?:    LucideIcon;
  color?:   "blue" | "green" | "red" | "amber";
  large?:   boolean;
}

const COLOR_MAP = {
  blue:  { bg: "rgba(59,130,246,0.1)",  border: "rgba(59,130,246,0.2)",  icon: "#3b82f6" },
  green: { bg: "rgba(34,197,94,0.1)",   border: "rgba(34,197,94,0.2)",   icon: "#22c55e" },
  red:   { bg: "rgba(239,68,68,0.1)",   border: "rgba(239,68,68,0.2)",   icon: "#ef4444" },
  amber: { bg: "rgba(245,158,11,0.1)",  border: "rgba(245,158,11,0.2)",  icon: "#f59e0b" },
};

export default function StatCard({ label, value, sub, icon: Icon, color = "blue", large }: StatCardProps) {
  const c = COLOR_MAP[color];
  return (
    <div
      className={clsx("rounded-xl p-5 card-hover", large && "p-6")}
      style={{ background: "var(--bg-secondary)", border: `1px solid var(--border)` }}>
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>
            {label}
          </p>
          <p className={clsx("font-bold truncate", large ? "text-3xl" : "text-2xl")} style={{ color: "var(--text-primary)" }}>
            {value}
          </p>
          {sub && <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{sub}</p>}
        </div>
        {Icon && (
          <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ml-3"
            style={{ background: c.bg, border: `1px solid ${c.border}` }}>
            <Icon className="w-5 h-5" style={{ color: c.icon }} />
          </div>
        )}
      </div>
    </div>
  );
}
