import React from "react";
import clsx from "clsx";

interface MetricBadgeProps {
  label: string;
  value: number;
  color?: "blue" | "green" | "red" | "amber" | "purple";
  format?: "pct" | "decimal" | "int";
}

const colors = {
  blue:   "bg-blue-500/10 border-blue-500/30 text-blue-400",
  green:  "bg-green-500/10 border-green-500/30 text-green-400",
  red:    "bg-red-500/10 border-red-500/30 text-red-400",
  amber:  "bg-amber-500/10 border-amber-500/30 text-amber-400",
  purple: "bg-purple-500/10 border-purple-500/30 text-purple-400",
};

export default function MetricBadge({ label, value, color = "blue", format = "decimal" }: MetricBadgeProps) {
  const display =
    format === "pct"     ? `${(value * 100).toFixed(1)}%` :
    format === "decimal" ? value.toFixed(4) :
    value.toLocaleString();

  return (
    <div className={clsx("flex flex-col items-center px-4 py-3 rounded-lg border text-center", colors[color])}>
      <span className="text-2xl font-bold">{display}</span>
      <span className="text-xs font-medium mt-0.5 opacity-80">{label}</span>
    </div>
  );
}
