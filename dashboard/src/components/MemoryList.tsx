import React from "react";
import type { Memory } from "../types";
import { relativeTime } from "../utils/time";

export function MemoryList({ items }: { items: Memory[] }) {
  return (
    <div>
      <h3
        className="mb-2.5 text-xs font-semibold uppercase tracking-wide"
        style={{ color: "var(--muted-bright)" }}
      >
        Recent memories
      </h3>
      {items.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          No memories in this scope yet
        </p>
      ) : (
        <ul className="flex flex-col gap-2" role="list">
          {items.map((m, i) => (
            <li
              key={i}
              className="rounded-lg p-3"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <p className="text-sm" style={{ color: "var(--ink)" }}>
                {m.text || <span style={{ color: "var(--muted)" }}>(no text)</span>}
              </p>
              <div
                className="mt-1.5 flex items-center gap-2 text-xs"
                style={{ color: "var(--muted)" }}
              >
                <span
                  className="rounded px-1.5 py-0.5"
                  style={{ background: "var(--surface)", color: "var(--muted-bright)" }}
                >
                  {m.agent || "unknown"}
                </span>
                <span>{m.kind}</span>
                <span className="ml-auto">{relativeTime(m.ts)}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
