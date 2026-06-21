import React from "react";

export function ScopeSelector({
  scope,
  options,
  onChange,
}: {
  scope: string;
  options: { value: string; label: string }[];
  onChange: (scope: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
      <span>Scope</span>
      <select
        value={scope}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Memory scope"
        className="rounded px-2 py-1 text-xs"
        style={{
          background: "var(--surface-2)",
          color: "var(--ink)",
          border: "1px solid var(--border)",
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
