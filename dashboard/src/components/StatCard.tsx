import React from "react";

export function StatCard({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string | number;
  sublabel?: string;
}) {
  return (
    <div
      className="rounded-lg p-4"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div
        className="text-xs font-medium uppercase tracking-wide"
        style={{ color: "var(--muted)" }}
      >
        {label}
      </div>
      <div
        className="mt-1.5 text-2xl font-semibold tabular-nums"
        style={{ color: "var(--ink)", letterSpacing: "-0.02em" }}
      >
        {value}
      </div>
      {sublabel && (
        <div className="mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
          {sublabel}
        </div>
      )}
    </div>
  );
}
