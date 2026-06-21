import React from "react";

interface Props {
  supported: number;
  unverified: number;
  flagged: number;
  hallucinationRate: number;
}

const SIZE = 140;
const STROKE = 16;
const R = (SIZE - STROKE) / 2;
const C = 2 * Math.PI * R;

function Arc({
  value,
  total,
  offset,
  color,
}: {
  value: number;
  total: number;
  offset: number;
  color: string;
}) {
  if (total <= 0 || value <= 0) return null;
  const len = (value / total) * C;
  return (
    <circle
      cx={SIZE / 2}
      cy={SIZE / 2}
      r={R}
      fill="none"
      stroke={color}
      strokeWidth={STROKE}
      strokeDasharray={`${len} ${C - len}`}
      strokeDashoffset={-offset}
    />
  );
}

function Legend({ color, label, value }: { color: string; label: string; value: number }) {
  return (
    <div
      className="flex items-center gap-2 text-xs"
      style={{ color: "var(--muted-bright)" }}
    >
      <span
        className="inline-block rounded-sm"
        style={{ width: 9, height: 9, background: color }}
        aria-hidden="true"
      />
      <span style={{ minWidth: 70 }}>{label}</span>
      <span className="tabular-nums" style={{ color: "var(--ink)" }}>
        {value}
      </span>
    </div>
  );
}

export function VerificationDonut({ supported, unverified, flagged, hallucinationRate }: Props) {
  const total = supported + unverified + flagged;
  const pctVerified = total > 0 ? Math.round((100 * supported) / total) : 0;
  const denom = total || 1;
  const oUnverified = (supported / denom) * C;
  const oFlagged = ((supported + unverified) / denom) * C;

  return (
    <div className="flex items-center gap-6">
      <div className="relative shrink-0" style={{ width: SIZE, height: SIZE }}>
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          role="img"
          aria-label={`${pctVerified} percent of ${total} facts verified against their source; ${flagged} flagged, ${unverified} unverified`}
          style={{ transform: "rotate(-90deg)" }}
        >
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            fill="none"
            stroke="var(--surface-2)"
            strokeWidth={STROKE}
          />
          <Arc value={supported} total={total} offset={0} color="var(--success)" />
          <Arc value={unverified} total={total} offset={oUnverified} color="var(--muted)" />
          <Arc value={flagged} total={total} offset={oFlagged} color="var(--warning)" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold tabular-nums" style={{ color: "var(--ink)" }}>
            {pctVerified}%
          </span>
          <span
            className="text-[10px] font-medium uppercase tracking-wide"
            style={{ color: "var(--muted)" }}
          >
            verified
          </span>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <Legend color="var(--success)" label="Verified" value={supported} />
        <Legend color="var(--muted)" label="Unverified" value={unverified} />
        <Legend color="var(--warning)" label="Flagged" value={flagged} />
        <div className="mt-1.5 text-xs" style={{ color: "var(--muted)" }}>
          Hallucination rate{" "}
          <span style={{ color: "var(--ink)" }}>{(hallucinationRate * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
