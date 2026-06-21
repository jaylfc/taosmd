import React from "react";

export function TopList({
  title,
  items,
}: {
  title: string;
  items: { name: string; count: number }[];
}) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div>
      <h3
        className="mb-2.5 text-xs font-semibold uppercase tracking-wide"
        style={{ color: "var(--muted-bright)" }}
      >
        {title}
      </h3>
      {items.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          None yet
        </p>
      ) : (
        <div className="flex flex-col gap-2.5">
          {items.map((it) => (
            <div key={it.name} className="flex items-center gap-2.5">
              <span
                className="truncate text-xs"
                style={{ color: "var(--ink)", width: 110 }}
                title={it.name}
              >
                {it.name}
              </span>
              <div
                className="h-2 flex-1 overflow-hidden rounded-full"
                style={{ background: "var(--surface-2)" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{ width: `${(it.count / max) * 100}%`, background: "var(--accent)" }}
                />
              </div>
              <span
                className="tabular-nums text-xs"
                style={{ color: "var(--muted)", width: 36, textAlign: "right" }}
              >
                {it.count}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
