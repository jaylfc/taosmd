import React, { useState } from "react";

interface Props {
  data: { date: string; count: number }[];
}

const W = 100;
const H = 40;
const GAP = 0.6;

export function GrowthChart({ data }: Props) {
  const [range, setRange] = useState<7 | 30>(30);
  const slice = data.slice(-range);
  const max = Math.max(1, ...slice.map((d) => d.count));
  const total = slice.reduce((s, d) => s + d.count, 0);
  const bw = slice.length ? (W - GAP * (slice.length - 1)) / slice.length : 0;

  return (
    <div
      className="rounded-lg p-4"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold" style={{ color: "var(--ink)" }}>
            Memory growth
          </h2>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            {total} new in the last {range} days
          </p>
        </div>
        <div
          role="radiogroup"
          aria-label="Time range"
          className="inline-flex rounded"
          style={{ border: "1px solid var(--border)", overflow: "hidden" }}
        >
          {([7, 30] as const).map((r, i) => {
            const active = range === r;
            return (
              <button
                key={r}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => setRange(r)}
                className="px-2.5 py-1 text-xs font-medium transition-colors duration-150"
                style={{
                  background: active ? "var(--accent-dim)" : "transparent",
                  color: active ? "var(--accent)" : "var(--muted-bright)",
                  borderLeft: i === 0 ? "none" : "1px solid var(--border)",
                }}
              >
                {r}d
              </button>
            );
          })}
        </div>
      </div>
      {slice.length === 0 ? (
        <p className="py-8 text-center text-xs" style={{ color: "var(--muted)" }}>
          No memories yet
        </p>
      ) : (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="w-full"
          style={{ height: 120 }}
          role="img"
          aria-label={`${total} memories created over the last ${range} days`}
        >
          {slice.map((d, i) => {
            const h = (d.count / max) * (H - 1);
            return (
              <rect
                key={d.date}
                x={i * (bw + GAP)}
                y={H - h}
                width={bw}
                height={h}
                fill="var(--accent)"
                opacity={0.85}
              />
            );
          })}
        </svg>
      )}
    </div>
  );
}
