import React from "react";
import { relativeTime } from "../utils/time";

export function ActivityFeed({
  items,
}: {
  items: { kind: string; label: string; ts: number }[];
}) {
  return (
    <div>
      <h3
        className="mb-2.5 text-xs font-semibold uppercase tracking-wide"
        style={{ color: "var(--muted-bright)" }}
      >
        Recent activity
      </h3>
      {items.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          No activity yet
        </p>
      ) : (
        <ul className="flex flex-col gap-2.5" role="list">
          {items.map((it, i) => (
            <li key={i} className="flex items-center gap-2.5 text-sm">
              <span
                className="inline-block shrink-0 rounded-full"
                style={{ width: 7, height: 7, background: "var(--accent)" }}
                aria-hidden="true"
              />
              <span className="flex-1 truncate" style={{ color: "var(--ink)" }}>
                {it.label}
              </span>
              <span className="shrink-0 text-xs" style={{ color: "var(--muted)" }}>
                {relativeTime(it.ts)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
